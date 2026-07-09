"""Data collection orchestration for PRICE2.

This module provides the DataCollector class, which drives the multi-step
process of collecting Ribo-seq run statistics, mapping reads to loci, and
assembling locus-level data structures for downstream deconvolution.
Intermediate results are persisted in an SQLite database located in the
working directory (``price.db``).
"""

import bisect
import logging
import os
import HTSeq
import pysam
from pyfaidx import Fasta
import sqlite3 as sql
from pickle import dumps, loads
import zlib
import pandas as pd
import numpy as np
import multiprocessing as mp
from collections import defaultdict

from price2 import multimap
from price2.reference_annotation import ReferenceAnnotation
from price2.ribo_seq_run import RiboSeqRun, ribo_seq_runs_from_bams
from price2.locus import Locus
from price2.genomic_region import GenomicRegion
from price2.config import Config

logger = logging.getLogger(__name__)


class DataCollector:
    """Orchestrate data collection for PRICE2 deconvolution.

    Builds loci from a reference annotation, persists Ribo-seq run
    statistics and read mappings to an SQLite database, and assembles
    per-locus data structures ready for ORF deconvolution.

    Attributes
    ----------
    loci_intervals : HTSeq.GenomicArray
        Stranded genomic array mapping intervals to their :class:`Locus`.
    loci_set : set[Locus]
        Full set of loci constructed from the reference annotation.
    chr_order : list[str] or None
        Chromosome names in BAM header order; ``None`` if no BAM found.
    """

    loci_intervals: HTSeq.GenomicArray
    loci_set: set[Locus]
    chr_order: list[str] | None

    def __init__(
        self,
        reference_annotation: ReferenceAnnotation,
        genome: Fasta,
        config: Config,
    ) -> None:
        """Initialise the DataCollector.

        Parameters
        ----------
        reference_annotation : ReferenceAnnotation
            Parsed reference annotation used to define loci.
        genome : pyfaidx.Fasta
            Indexed FASTA handle keyed by chromosome name.
        config : Config
            Run configuration.
        """
        self.config = config
        self.genome = genome
        self.reference_annotation = reference_annotation
        self.bam_dir = config.bam_dir
        self.db_path = f"{config.w_dir}/price.db"
        self.get_chromosome_order()
        self.make_loci(self.reference_annotation)

    def collect_runs(self) -> None:
        """Collect and persist Ribo-seq run statistics.

        Discovers BAM files in ``config.bam_dir`` (or uses the explicit
        list from ``config.bam_ids``), computes per-run statistics for any
        run not already stored in the database, and appends them to
        ``self.runs``.
        """
        logger.info("Collecting Ribo-seq runs...")
        if self.config.bam_ids:
            bam_ids = set(self.config.bam_ids)
        else:
            bam_ids = {
                f.split(".")[0] for f in os.listdir(self.bam_dir) if f.endswith(".bam")
            }

        if os.path.exists(self.db_path):

            db = sql.connect(self.db_path, timeout=60)
            cur = db.cursor()
            cur.execute("SELECT * FROM runs")
            stored_runs = cur.fetchall()
            run_ids = {run_id for run_id, _ in stored_runs}
            self.runs = [loads(run_blob) for _, run_blob in stored_runs]
            bam_ids = bam_ids - run_ids
        else:
            self.runs = []
            db = sql.connect(self.db_path, timeout=60)
            cur = db.cursor()

            cur.execute(
                """CREATE TABLE runs (
                        run_id text PRIMARY KEY,
                        run_blob blob
                        )"""
            )

        new_runs = ribo_seq_runs_from_bams(
            self.bam_dir,
            bam_ids,
            self.config.w_dir,
            self.reference_annotation,
            self.config.processes,
            self.config.log_level,
            high_quality_only=self.config.high_quality_runs_only,
        )
        self.runs += new_runs

        for run in new_runs:
            cur.execute("INSERT INTO runs VALUES (?, ?)", (run.id, dumps(run)))

        db.commit()
        db.close()
        logger.info("Collected %d Ribo-seq run(s).", len(self.runs))

    def collect_mappings(self) -> None:
        """Map reads from all runs to loci and persist results.

        Creates the ``reads`` and ``transcript_read_counts`` tables in the
        database if they do not already exist, then maps each unprocessed
        run's BAM against the loci.

        Parallelism is over *locus chunks*, not over runs: there are only
        as many runs as BAM files (typically < 10) but tens of thousands of
        loci, and the per-alignment work is the dominant cost.  Runs are
        processed one at a time so a run's rows still land in the database
        as a single transaction — ``processed_run_ids`` therefore keeps its
        all-or-nothing resume semantics — while all of
        ``config.processes`` work on that run's loci concurrently.

        The parent is the only database writer; workers return their blobs
        and spill multimapping alignments straight to disk (see
        :mod:`price2.multimap`).
        """
        logger.info("Collecting read mappings...")
        db = sql.connect(self.db_path)
        cur = db.cursor()
        cur.execute(
            """CREATE TABLE IF NOT EXISTS reads (
                    locus_id text NOT NULL,
                    run_id text NOT NULL,
                    reads_blob blob NOT NULL
                    )"""
        )

        cur.execute(
            """CREATE TABLE IF NOT EXISTS transcript_read_counts (
                         locus_id text NOT NULL,
                         run_id text NOT NULL,
                         transcript_read_counts_blob blob NOT NULL
                         )"""
        )
        db.commit()

        processed_run_ids = {
            run_id for run_id, in cur.execute("SELECT DISTINCT run_id FROM reads")
        }
        db.close()

        pending = [run for run in self.runs if run.id not in processed_run_ids]
        if not pending:
            logger.info("Read mappings already collected for all runs.")
            return
        if not self.loci_set:
            logger.warning("No loci to map reads against.")
            return

        record_multimap = self.config.multimap_em

        # Chunk the loci in BAM coordinate order so each worker's fetches
        # sweep a contiguous slab of the file instead of seeking randomly.
        chr_rank = {c: i for i, c in enumerate(self.chr_order or [])}
        loci = sorted(
            self.loci_set,
            key=lambda loc: (
                chr_rank.get(loc.iv.chrom, len(chr_rank)),
                loc.iv.start,
                loc.iv.strand,
            ),
        )

        spill_root = ""
        if record_multimap:
            # The derived EM tables are rebuilt from these spill files (see
            # build_multimap_index).  Spills of runs collected by an earlier,
            # interrupted pass are kept; the pending runs start clean.
            spill_root = multimap.init_spill(
                self.db_path, [loc.id for loc in loci]
            )
            for run in pending:
                multimap.reset_run_spill(spill_root, run.id)
                # Create it even if the run turns out to have no multimapping
                # alignments, so a later resume can tell "collected, nothing to
                # spill" apart from "never collected".
                os.makedirs(os.path.join(spill_root, run.id), exist_ok=True)
            missing = [
                run_id
                for run_id in processed_run_ids
                if not os.path.isdir(os.path.join(spill_root, run_id))
            ]
            if missing:
                logger.warning(
                    "runs %s have stored reads but no multimapping spill; "
                    "their alignments will be absent from the linkage index",
                    ", ".join(sorted(missing)),
                )

        # Read depth spans orders of magnitude between loci — the deepest
        # single locus of a human Ribo-seq run costs seconds on its own — so
        # a chunk must stay small enough that one hot locus cannot become the
        # critical path.  A handful of loci per chunk keeps enough genomic
        # locality for the BAM fetches to sweep the file, while leaving the
        # ``imap_unordered`` dispatch free to balance the rest.
        n_proc = max(1, self.config.processes)
        chunk_size = max(1, min(4, -(-len(loci) // (n_proc * 4))))
        bounds = [
            (i, min(i + chunk_size, len(loci)))
            for i in range(0, len(loci), chunk_size)
        ]

        # Workers read the loci through this module global: with the default
        # fork start method they inherit it copy-on-write, so the (large)
        # Locus objects are never pickled.
        global _WORKER_LOCI
        _WORKER_LOCI = loci

        try:
            for run in pending:
                tasks = [
                    (
                        run.id,
                        self.bam_dir,
                        lo,
                        hi,
                        chunk_idx,
                        os.path.join(spill_root, run.id) if record_multimap else "",
                    )
                    for chunk_idx, (lo, hi) in enumerate(bounds)
                ]
                self._collect_run(run.id, tasks, n_proc)
        finally:
            _WORKER_LOCI = []

        db = sql.connect(self.db_path)
        cur = db.cursor()
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_reads_locus_id ON reads(locus_id)"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_trc_locus_id "
            "ON transcript_read_counts(locus_id)"
        )
        db.commit()
        db.close()

        logger.info("Read mappings collected.")

    def _collect_run(self, run_id: str, tasks: list, n_proc: int) -> None:
        """Map one run's BAM against every locus chunk and store the result.

        Parameters
        ----------
        run_id : str
            Identifier of the Ribo-seq run being mapped.
        tasks : list of tuple
            One :func:`collect_mappings_chunk` argument tuple per locus chunk.
        n_proc : int
            Number of worker processes.
        """
        # Fork before opening the database: SQLite connections must not be
        # carried across fork(), and the workers have no use for one.
        # The context must be requested explicitly: importing
        # ``orf_activity_estimator`` sets the global start method to
        # ``forkserver``, under which workers would not inherit
        # ``_WORKER_LOCI``.
        try:
            pool = mp.get_context("fork").Pool(n_proc)
        except AssertionError:
            # A daemonic process may not spawn children (Process.start
            # asserts this); fall back to mapping the chunks in-process.
            # Only pool *creation* is guarded: an AssertionError raised while
            # consuming results would otherwise re-run chunks already stored.
            pool = None

        db = sql.connect(self.db_path, timeout=600)
        cur = db.cursor()
        cur.execute("PRAGMA busy_timeout = 600000")

        def store(result: tuple) -> None:
            reads_rows, trc_rows = result
            cur.executemany(
                "INSERT INTO reads (locus_id, run_id, reads_blob) "
                "VALUES (?, ?, ?)",
                reads_rows,
            )
            cur.executemany(
                "INSERT INTO transcript_read_counts ("
                "locus_id, run_id, transcript_read_counts_blob) "
                "VALUES (?, ?, ?)",
                trc_rows,
            )

        try:
            if pool is None:
                for task in tasks:
                    store(collect_mappings_chunk(task))
            else:
                with pool:
                    for result in pool.imap_unordered(
                        collect_mappings_chunk, tasks, chunksize=1
                    ):
                        store(result)
            db.commit()
        finally:
            db.close()
        logger.info("Mapped run %s (%d locus chunks).", run_id, len(tasks))

    def get_chromosome_order(self) -> None:
        """Set ``self.chr_order`` from the first BAM file found in ``bam_dir``.

        Reads the ``SQ`` header records of the first ``.bam`` file in
        ``self.bam_dir``.  ``self.chr_order`` is set to ``None`` when no BAM
        file is present.
        """
        self.chr_order = None
        for bam_file in os.listdir(self.bam_dir):
            if bam_file.endswith(".bam"):
                with pysam.AlignmentFile(
                    os.path.join(self.bam_dir, bam_file), "rb"
                ) as _bam:
                    self.chr_order = [sq["SN"] for sq in _bam.header["SQ"]]
                return

    def make_loci(
        self,
        reference_annotation: ReferenceAnnotation,
        distance: int = 50,
    ) -> None:
        """Build loci by merging nearby transcript intervals.

        Iterates over transcripts in ``reference_annotation``, marks their
        genomic intervals in a binary step-array, and then merges
        consecutive occupied intervals on the same strand that are at most
        ``distance`` bases apart into a single :class:`~price2.locus.Locus`.
        Populates ``self.loci_intervals`` and ``self.loci_set``.

        Parameters
        ----------
        reference_annotation : ReferenceAnnotation
            Parsed annotation whose transcripts define the locus boundaries.
        distance : int, optional
            Maximum gap (in bases) between two transcript intervals that are
            still merged into the same locus.  Default is 50.
        """
        loci_intervals_binary = HTSeq.GenomicArray(
            "auto", stranded=True, storage="step", typecode="b"
        )
        for transcript in reference_annotation.transcripts.values():
            loci_intervals_binary[transcript.iv] = True
        self.loci_intervals = HTSeq.GenomicArray(
            "auto", stranded=True, storage="step", typecode="O"
        )
        self.loci_set = set()

        connected_loci = {"+": [], "-": []}
        loci_counter = 0
        for iv, step in loci_intervals_binary.steps():
            if step:
                if not connected_loci[iv.strand]:
                    connected_loci[iv.strand].append(iv)
                    continue
                if (
                    connected_loci[iv.strand][-1].chrom == iv.chrom
                    and connected_loci[iv.strand][-1].end + distance > iv.start
                ):
                    connected_loci[iv.strand].append(iv)
                else:
                    connected_iv = HTSeq.GenomicInterval(
                        connected_loci[iv.strand][0].chrom,
                        connected_loci[iv.strand][0].start,
                        connected_loci[iv.strand][-1].end,
                        iv.strand,
                    )
                    locus = Locus(
                        connected_iv,
                        reference_annotation.transcript_intervals,
                        loci_counter,
                    )
                    loci_counter += 1
                    self.loci_intervals[connected_iv] = locus
                    self.loci_set.add(locus)
                    connected_loci[iv.strand] = [iv]
        for strand in ["+", "-"]:
            try:
                connected_iv = HTSeq.GenomicInterval(
                    connected_loci[strand][0].chrom,
                    connected_loci[strand][0].start,
                    connected_loci[strand][-1].end,
                    strand,
                )
                locus = Locus(
                    connected_iv,
                    reference_annotation.transcript_intervals,
                    loci_counter,
                )
                loci_counter += 1
                self.loci_intervals[connected_iv] = locus
                self.loci_set.add(locus)
            except IndexError:
                pass

    def collect_loci(self) -> None:
        """Persist pre-RGR locus skeletons to the ``loci`` table.

        Per-locus transcript filtering and ORF candidate generation
        (formerly performed here serially) now run inside the parallel
        deconvolution workers via :func:`build_rgrs`.  This method only
        records the locus skeletons and the run count needed to compute
        ``min_explained_reads`` downstream.
        """
        logger.info("Saving locus skeletons...")
        db = sql.connect(self.db_path)
        cur = db.cursor()

        cur.execute(
            """CREATE TABLE IF NOT EXISTS loci (
                    locus_id text PRIMARY KEY,
                    loc_blob blob
                    )"""
        )
        processed_loci_ids = {
            loc_id for loc_id, in cur.execute("SELECT locus_id FROM loci").fetchall()
        }
        loci_ids_to_process = {loc.id for loc in self.loci_set} - processed_loci_ids
        loc_dict = {loc.id: loc for loc in self.loci_set}

        for loc_id in loci_ids_to_process:
            locus = loc_dict[loc_id]
            cur.execute(
                "INSERT INTO loci VALUES (?, ?)", (locus.id, dumps(locus))
            )

        db.commit()
        db.close()
        logger.info("Saved %d locus skeletons.", len(loci_ids_to_process))


def build_rgrs(
    locus: Locus,
    db_path: str,
    genome: Fasta,
    config: Config,
    min_explained_reads: float,
) -> bool:
    """Filter a locus's transcripts by read support and build its RGRs.

    Greedily selects transcripts that jointly explain the most observed
    reads above ``min_explained_reads``, prunes the locus's transcript
    set and ``transcript_intervals`` accordingly, and calls
    :meth:`~price2.locus.Locus.make_rgrs` when transcripts remain.

    Parameters
    ----------
    locus : Locus
        Pre-RGR locus skeleton, mutated in place.
    db_path : str
        Path to ``price.db`` (read-only access for
        ``transcript_read_counts``).
    genome : pyfaidx.Fasta
        Worker-local indexed FASTA handle.
    config : Config
        Parsed PRICE configuration.
    min_explained_reads : float
        Threshold (count, not per-run) used to discard transcripts with
        insufficient read support.

    Returns
    -------
    bool
        ``True`` when the locus retained at least one transcript and
        RGRs were built; ``False`` for empty loci that downstream code
        should skip.
    """
    db = sql.connect(db_path, timeout=120)
    cur = db.cursor()
    cur.execute("PRAGMA busy_timeout = 120000")
    cur.execute(
        "SELECT * FROM transcript_read_counts WHERE locus_id = ?",
        (locus.id,),
    )

    transcript_read_counts: dict = {}
    for entry in cur.fetchall():
        for k, v in loads(zlib.decompress(entry[2])).items():
            try:
                transcript_read_counts[k] += v
            except KeyError:
                transcript_read_counts[k] = v
    db.close()

    tr_ids = [t.id for t in locus.transcripts]
    explaining_transcripts_reads_list = []

    if tr_ids and transcript_read_counts:
        n_tr = len(tr_ids)
        n_rs = len(transcript_read_counts)
        tr_to_col = {tr_id: i for i, tr_id in enumerate(tr_ids)}

        M = np.zeros((n_rs, n_tr), dtype=bool)
        counts = np.zeros(n_rs, dtype=np.float64)

        for row_i, (read_set, count) in enumerate(transcript_read_counts.items()):
            counts[row_i] = count
            for member in read_set:
                if member in tr_to_col:
                    M[row_i, tr_to_col[member]] = True

        while counts.sum() > 0:
            weighted = M.T @ counts
            best_col = int(np.argmax(weighted))
            best_score = weighted[best_col]
            if best_score == 0:
                break
            explaining_transcripts_reads_list.append(
                (tr_ids[best_col], best_score)
            )
            counts[M[:, best_col]] = 0.0

    transcripts_dict = {tr.id: tr for tr in locus.transcripts}

    locus.keep_transcripts = [
        transcripts_dict[tr_id]
        for tr_id, count in explaining_transcripts_reads_list
        if count > min_explained_reads
    ]

    locus.transcripts_number = len(locus.transcripts)
    locus.transcripts = locus.keep_transcripts

    new_tr_intervals = HTSeq.GenomicArray(
        list(locus.transcript_intervals.chrom_vectors.keys()), typecode="O"
    )
    for step_iv, step_set in locus.transcript_intervals.steps():
        new_step_set = set()
        for tr in step_set:
            if tr in locus.transcripts:
                new_step_set.add(tr)
        new_tr_intervals[step_iv] = new_step_set
    locus.transcript_intervals = new_tr_intervals

    if not locus.transcripts:
        return False

    locus.make_rgrs(genome, config)
    return True


#: Loci to map against, shared with the collection workers by fork.  Set by
#: :meth:`DataCollector.collect_mappings` before the pool is created.
_WORKER_LOCI: list[Locus] = []

#: Per-process cache of the open BAM handle, keyed by run id.  Chunks of one
#: run land on the same worker repeatedly, so the index is parsed once.
_WORKER_BAM: dict[str, pysam.AlignmentFile] = {}

_READS_COLUMNS = (
    ("is_first_iv", bool),
    ("start", np.uint32),
    ("end", np.uint32),
    ("untemplated_addition", bool),
    ("unique", bool),
    ("count", np.uint16),
)


def _worker_bam(bam_dir: str, run_id: str) -> pysam.AlignmentFile:
    """Return this process's open handle for ``run_id``, opening it once."""
    handle = _WORKER_BAM.get(run_id)
    if handle is None:
        for stale in _WORKER_BAM.values():
            stale.close()
        _WORKER_BAM.clear()
        handle = pysam.AlignmentFile(f"{bam_dir}/{run_id}.bam", "rb")
        _WORKER_BAM[run_id] = handle
    return handle


def _slow_path_transcripts(blocks: list, locus: Locus) -> list:
    """Resolve a read's compatible transcripts via :meth:`map_to_local`.

    The general fallback for reads the geometric fast paths do not cover
    (three or more blocks, or a locus whose annotation has abutting exons).

    Parameters
    ----------
    blocks : list of (int, int)
        The alignment's reference blocks, in chromosome order.
    locus : Locus
        Locus the read was fetched from.

    Returns
    -------
    list of Transcript
        Transcripts into whose exon structure the read maps cleanly.
    """
    bp_starts, bp_ends, bp_sets = locus.transcript_breakpoint_index
    n_bp = len(bp_starts)

    transcript_sets = []
    for start, end in blocks:
        i = bisect.bisect_right(bp_ends, start)
        while i < n_bp and bp_starts[i] < end:
            transcript_sets.append(bp_sets[i])
            i += 1
    if not transcript_sets:
        return []

    chrom = locus.iv.chrom
    strand = locus.iv.strand
    region = GenomicRegion(
        [HTSeq.GenomicInterval(chrom, s, e, strand) for s, e in blocks]
    )
    transcripts = []
    for transcript in set.intersection(*transcript_sets):
        try:
            transcript.exons.map_to_local(region)
            transcripts.append(transcript)
        except ValueError:
            continue
    return transcripts


def collect_mappings_chunk(data: tuple) -> tuple:
    """Map one run's reads against a contiguous chunk of loci.

    Designed to be called via :class:`multiprocessing.Pool`; the loci
    themselves are read from the :data:`_WORKER_LOCI` module global, which
    forked workers inherit without pickling.

    Most alignments never need a :class:`RiboSeqAlignment` or a
    :meth:`~price2.genomic_region.GenomicRegion.map_to_local` call.  A read
    that is a single block maps into exactly the transcripts common to
    every breakpoint step it touches, provided those steps cover it without
    a gap (a transcript's exons are separated by introns, so a gapless
    covered stretch lies inside one exon).  A read that is two blocks maps
    into a transcript iff its gap is one of that transcript's introns and
    its outer ends stay within the flanking exons — a lookup in
    :attr:`~price2.locus.Locus.transcript_junction_index`.  Everything else
    falls back to :func:`_slow_path_transcripts`.

    Parameters
    ----------
    data : tuple
        ``(run_id, bam_dir, lo, hi, chunk_idx, run_spill_dir)`` where
        ``lo``/``hi`` slice :data:`_WORKER_LOCI` and ``run_spill_dir`` is
        empty when multimapping linkage is not being recorded.

    Returns
    -------
    tuple
        ``(reads_rows, transcript_count_rows)`` — blob rows for the parent
        to insert.  Multimapping alignments are spilled to disk, not
        returned.
    """
    run_id, bam_dir, lo, hi, chunk_idx, run_spill_dir = data
    record_multimap = bool(run_spill_dir)

    reads_rows: list = []
    transcript_count_rows: list = []
    mm_qnames: list = []
    mm_loci: list = []
    mm_keys: list = []

    qname_hash = multimap.qname_hash
    group_key = multimap.group_key
    bisect_right = bisect.bisect_right
    sf = _worker_bam(bam_dir, run_id)

    for locus_idx in range(lo, hi):
        locus = _WORKER_LOCI[locus_idx]
        transcripts_counts: dict = defaultdict(int)
        mappings_dict: dict = defaultdict(int)

        is_minus = locus.iv.strand == "-"
        bp_starts, bp_ends, bp_sets = locus.transcript_breakpoint_index
        n_bp = len(bp_starts)
        junctions = locus.transcript_junction_index
        single_block_fast = not locus.has_abutting_exons
        # (first_step, last_step) -> frozenset of transcript ids, shared by
        # every single-block read covered by exactly that run of steps.
        step_memo: dict = {}

        for alignment in sf.fetch(locus.iv.chrom, locus.iv.start, locus.iv.end):
            if alignment.is_unmapped or alignment.is_reverse != is_minus:
                continue

            cigar = alignment.cigartuples
            if not cigar:
                continue
            # 5' untemplated addition: a 1-nt soft clip at the read's 5' end.
            ua = cigar[-1] == (4, 1) if is_minus else cigar[0] == (4, 1)

            blocks = alignment.get_blocks()
            n_blocks = len(blocks)

            if n_blocks == 1 and single_block_fast:
                start, end = blocks[0]
                i0 = bisect_right(bp_ends, start)
                if i0 >= n_bp or bp_starts[i0] > start:
                    continue  # read starts outside any exon
                i1 = i0
                while i1 + 1 < n_bp and bp_starts[i1 + 1] < end:
                    if bp_starts[i1 + 1] != bp_ends[i1]:
                        break  # gap between steps
                    i1 += 1
                if bp_ends[i1] < end:
                    continue  # read runs past the covered stretch
                transcripts_ids = step_memo.get((i0, i1))
                if transcripts_ids is None:
                    candidates = (
                        bp_sets[i0] if i0 == i1
                        else set.intersection(*bp_sets[i0:i1 + 1])
                    )
                    transcripts_ids = frozenset(tr.id for tr in candidates)
                    step_memo[(i0, i1)] = transcripts_ids
            elif n_blocks == 2:
                (a1, b1), (a2, b2) = blocks
                spliced = junctions.get((b1, a2))
                if not spliced:
                    continue  # gap is not an annotated intron here
                transcripts_ids = frozenset(
                    tr.id
                    for tr, donor_start, acceptor_end in spliced
                    if a1 >= donor_start and b2 <= acceptor_end
                )
            else:
                transcripts_ids = frozenset(
                    tr.id for tr in _slow_path_transcripts(blocks, locus)
                )

            if not transcripts_ids:
                continue

            try:
                unique = alignment.get_tag("NH") == 1
            except KeyError:
                unique = True

            ivs_tuple = tuple(blocks)
            mappings_dict[(ua, unique, ivs_tuple)] += 1
            transcripts_counts[transcripts_ids] += 1

            # Record this multimapping alignment so its read can be linked
            # to its other in-locus slots for the EM E-step.
            if record_multimap and not unique:
                mm_qnames.append(qname_hash(alignment.query_name))
                mm_loci.append(locus_idx)
                mm_keys.append(group_key(ivs_tuple, ua))

        reads_rows.append(
            (locus.id, run_id, zlib.compress(dumps(_reads_frame(mappings_dict))))
        )
        transcript_count_rows.append(
            (locus.id, run_id, zlib.compress(dumps(transcripts_counts)))
        )

    if record_multimap:
        multimap.write_spill(
            run_spill_dir, chunk_idx, mm_qnames, mm_loci, mm_keys
        )

    return reads_rows, transcript_count_rows


def _reads_frame(mappings_dict: dict) -> pd.DataFrame:
    """Build a locus's collapsed-reads DataFrame in the stored blob layout.

    One row per exonic interval; ``is_first_iv`` marks the first interval
    of each read, which is how :meth:`~price2.locus.Locus.get_reads_from_db`
    recovers read boundaries.

    Parameters
    ----------
    mappings_dict : dict
        ``{(untemplated_addition, unique, ivs_tuple): count}``.

    Returns
    -------
    pandas.DataFrame
        Columns ``is_first_iv``, ``start``, ``end``,
        ``untemplated_addition``, ``unique``, ``count``.
    """
    columns: dict = {name: [] for name, _ in _READS_COLUMNS}
    for (ua, unique, ivs_tuple), count in mappings_dict.items():
        is_first_iv = True
        for start, end in ivs_tuple:
            columns["is_first_iv"].append(is_first_iv)
            columns["start"].append(start)
            columns["end"].append(end)
            columns["untemplated_addition"].append(ua)
            columns["unique"].append(unique)
            columns["count"].append(count)
            is_first_iv = False
    return pd.DataFrame(
        {
            name: np.asarray(columns[name], dtype=dtype)
            for name, dtype in _READS_COLUMNS
        }
    )
