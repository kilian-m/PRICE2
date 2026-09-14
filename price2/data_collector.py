"""Data collection orchestration for PRICE2.

This module provides the DataCollector class, which drives the multi-step
process of collecting Ribo-seq run statistics, mapping reads to loci, and
assembling locus-level data structures for downstream deconvolution.
Intermediate results are persisted in an SQLite database located in the
working directory (``price.db``).
"""

import bisect
import logging
import math
import os
import HTSeq
import pysam
import pandas as pd
import numpy as np
import multiprocessing as mp
from collections import defaultdict

from price2 import database
from price2.bam import (
    cached_alignment_file,
    first_mapped_read_has_md,
    footprint,
    is_unique,
)
from price2 import multimap
from price2.reference_annotation import ReferenceAnnotation
from price2.ribo_seq_run import ribo_seq_runs_from_bams
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
    loci : list[Locus]
        The loci built from the reference annotation (:func:`build_loci`).
    chr_order : list[str] or None
        Chromosome names in BAM header order; ``None`` if no BAM found.
    """

    loci: list[Locus]
    chr_order: list[str] | None

    def __init__(
        self,
        reference_annotation: ReferenceAnnotation,
        config: Config,
    ) -> None:
        """Initialise the DataCollector.

        Parameters
        ----------
        reference_annotation : ReferenceAnnotation
            Parsed reference annotation used to define loci.
        config : Config
            Run configuration.
        """
        self.config = config
        self.reference_annotation = reference_annotation
        self.bam_dir = config.bam_dir
        self.db_path = config.layout.db_path
        self.get_chromosome_order()
        self.loci = build_loci(self.reference_annotation)

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
                os.path.splitext(f)[0]
                for f in os.listdir(self.bam_dir)
                if f.endswith(".bam")
            }

        # The database may already exist without holding any runs: a cold
        # start records its configuration fingerprints before collecting
        # anything (see :mod:`price2.run_state`).
        with database.connect(self.db_path, commit=True) as db:
            cur = db.cursor()
            database.create_collection_tables(cur)
            stored_runs = cur.execute("SELECT * FROM runs").fetchall()
        run_ids = {run_id for run_id, _ in stored_runs}
        self.runs = [
            database.unpickle_blob(run_blob) for _, run_blob in stored_runs
        ]
        bam_ids = bam_ids - run_ids

        if self.config.align_ends_type == "endtoend":
            for bam_id in sorted(bam_ids):
                if first_mapped_read_has_md(f"{self.bam_dir}/{bam_id}.bam") is False:
                    logger.warning(
                        "align_ends_type='endtoend' but BAM %s has no MD tag. "
                        "The untemplated addition (RT nucleotide) is recovered "
                        "from the 5'-terminal mismatch, which requires the MD "
                        "tag; without it no untemplated additions will be "
                        "detected. Re-map with STAR "
                        "'--outSAMattributes NH HI AS nM MD' (or add tags with "
                        "'samtools calmd -b in.bam ref.fa').",
                        bam_id,
                    )

        new_runs = ribo_seq_runs_from_bams(
            self.bam_dir,
            bam_ids,
            self.config.w_dir,
            self.reference_annotation,
            self.config.processes,
            high_quality_only=self.config.high_quality_runs_only,
            end_to_end=self.config.align_ends_type == "endtoend",
        )
        self.runs += new_runs

        with database.connect(self.db_path, commit=True) as db:
            db.executemany(
                "INSERT INTO runs VALUES (?, ?)",
                [(run.id, database.pickle_blob(run)) for run in new_runs],
            )
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
        processed_run_ids = self._processed_run_ids()
        pending = [run for run in self.runs if run.id not in processed_run_ids]
        if not pending:
            logger.info("Read mappings already collected for all runs.")
            return
        if not self.loci:
            logger.warning("No loci to map reads against.")
            return

        record_multimap = self.config.multimap_em
        # Without the EM outer loop there is no way to spread a multimapping
        # read over the loci it aligns to, and counting it at full weight in
        # each of them multi-counts it -- so discard those alignments instead.
        drop_multimap = not self.config.multimap_em
        if drop_multimap:
            logger.info(
                "multimap_em is disabled: multimapping alignments (NH > 1) "
                "are discarded, not stored."
            )

        loci = self._loci_in_bam_order()
        spill_root = ""
        if record_multimap:
            spill_root = self._prepare_spill(loci, pending, processed_run_ids)

        n_proc = max(1, self.config.processes)
        bounds = _locus_chunks(len(loci), n_proc)
        end_to_end = self.config.align_ends_type == "endtoend"
        for run in pending:
            tasks = [
                (
                    run.id,
                    self.bam_dir,
                    lo,
                    hi,
                    chunk_idx,
                    os.path.join(spill_root, run.id) if record_multimap else "",
                    end_to_end,
                    drop_multimap,
                )
                for chunk_idx, (lo, hi) in enumerate(bounds)
            ]
            self._map_run_reads(run.id, tasks, n_proc, loci)

        with database.connect(self.db_path, commit=True) as db:
            database.create_read_indexes(db.cursor())

        logger.info("Read mappings collected.")

    def _processed_run_ids(self) -> set[str]:
        """The runs whose reads are stored, creating the tables on a cold start."""
        with database.connect(self.db_path, commit=True) as db:
            cur = db.cursor()
            database.create_collection_tables(cur)
            return {
                run_id
                for run_id, in cur.execute("SELECT DISTINCT run_id FROM reads")
            }

    def _loci_in_bam_order(self) -> list[Locus]:
        """The loci in BAM coordinate order.

        Chunked in this order, each worker's fetches sweep a contiguous slab
        of the file instead of seeking randomly.
        """
        chr_rank = {c: i for i, c in enumerate(self.chr_order or [])}
        return sorted(
            self.loci,
            key=lambda loc: (
                chr_rank.get(loc.iv.chrom, len(chr_rank)),
                loc.iv.start,
                loc.iv.strand,
            ),
        )

    def _prepare_spill(
        self, loci: list[Locus], pending: list, processed_run_ids: set[str]
    ) -> str:
        """Set up the multimapping spill directory for the pending runs.

        The derived EM tables are rebuilt from the spill files (see
        :func:`price2.multimap.build_multimap_index`).  Spills of runs
        collected by an earlier, interrupted pass are kept; the pending runs
        start clean.

        Raises
        ------
        RuntimeError
            When a run has stored reads but no spill: its alignments would
            be absent from the linkage index, and the EM would silently
            treat them as unique.
        """
        spill_root = multimap.init_spill(self.db_path, [loc.id for loc in loci])
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
            raise RuntimeError(
                f"runs {', '.join(sorted(missing))} have stored reads but no "
                "multimapping spill, so their alignments cannot enter the "
                "linkage index; re-collect with warm_start=false"
            )
        return spill_root

    def _map_run_reads(
        self, run_id: str, tasks: list, n_proc: int, loci: list[Locus]
    ) -> None:
        """Map one run's BAM against every locus chunk and store the result.

        Parameters
        ----------
        run_id : str
            Identifier of the Ribo-seq run being mapped.
        tasks : list of tuple
            One :func:`collect_mappings_chunk` argument tuple per locus chunk.
        n_proc : int
            Number of worker processes.
        loci : list[Locus]
            The loci the chunks index into.
        """
        # Fork before opening the database: SQLite connections must not be
        # carried across fork(), and the workers have no use for one.
        # ``fork`` so that the workers inherit the loci through the
        # initializer's arguments without pickling them; the deconvolution
        # pool uses ``forkserver`` instead.
        try:
            pool = mp.get_context("fork").Pool(
                n_proc, initializer=_init_mapping_worker, initargs=(loci,)
            )
        except AssertionError:
            # A daemonic process may not spawn children (Process.start
            # asserts this); fall back to mapping the chunks in-process.
            # Only pool *creation* is guarded: an AssertionError raised while
            # consuming results would otherwise re-run chunks already stored.
            pool = None

        with database.connect(self.db_path, timeout=600, commit=True) as db:
            cur = db.cursor()

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

            if pool is None:
                _init_mapping_worker(loci)
                for task in tasks:
                    store(collect_mappings_chunk(task))
                # This process buffered the spill itself, so flush it here.
                multimap.flush_spill()
            else:
                # close()+join(), NOT terminate(): the workers hold this run's
                # buffered multimapping alignments and only write them from a
                # multiprocessing exit finalizer, which a terminated (SIGTERMed)
                # worker never reaches.  ``with pool:`` calls terminate() and
                # would silently drop the spill.  join() also guarantees every
                # worker has finished writing before the index is built.
                try:
                    for result in pool.imap_unordered(
                        collect_mappings_chunk, tasks, chunksize=1
                    ):
                        store(result)
                    pool.close()
                except BaseException:
                    pool.terminate()
                    raise
                finally:
                    pool.join()
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

    def collect_loci(self) -> None:
        """Persist pre-RGR locus skeletons to the ``loci`` table.

        Per-locus transcript filtering and ORF candidate generation
        (formerly performed here serially) now run inside the parallel
        deconvolution workers via :func:`price2.orf_candidates.build_rgrs`.
        This method only
        records the locus skeletons and the run count needed to compute
        ``min_explained_reads`` downstream.
        """
        logger.info("Saving locus skeletons...")
        with database.connect(self.db_path, commit=True) as db:
            cur = db.cursor()
            database.create_collection_tables(cur)
            stored = {
                loc_id for loc_id, in cur.execute("SELECT locus_id FROM loci")
            }
            new_loci = [loc for loc in self.loci if loc.id not in stored]
            cur.executemany(
                "INSERT INTO loci VALUES (?, ?)",
                [(loc.id, database.pickle_blob(loc)) for loc in new_loci],
            )
        logger.info("Saved %d locus skeletons.", len(new_loci))


def build_loci(
    reference_annotation: ReferenceAnnotation, distance: int = 50
) -> list[Locus]:
    """Build the loci of an annotation by merging nearby transcripts.

    Marks every transcript's span in a stranded step array and merges
    consecutive occupied stretches on the same strand and chromosome that
    are less than *distance* bases apart into one
    :class:`~price2.locus.Locus`.  The loci are numbered in the order they
    are closed, which is the order of the step array.

    Parameters
    ----------
    reference_annotation : ReferenceAnnotation
        Parsed annotation whose transcripts define the locus boundaries.
    distance : int, optional
        Two stretches closer than this are merged.

    Returns
    -------
    list[Locus]
        The loci, ``loc_0``, ``loc_1``, ... in that order.
    """
    occupied = HTSeq.GenomicArray("auto", stranded=True, storage="step", typecode="b")
    for transcript in reference_annotation.transcripts.values():
        occupied[transcript.iv] = True

    loci: list[Locus] = []
    pending: dict[str, list[HTSeq.GenomicInterval]] = {"+": [], "-": []}

    def close(strand: str) -> None:
        stretch = pending[strand]
        if stretch:
            iv = HTSeq.GenomicInterval(
                stretch[0].chrom, stretch[0].start, stretch[-1].end, strand
            )
            loci.append(
                Locus(iv, reference_annotation.transcript_intervals, len(loci))
            )
            pending[strand] = []

    for iv, step in occupied.steps():
        if not step:
            continue
        stretch = pending[iv.strand]
        if stretch and not (
            stretch[-1].chrom == iv.chrom and stretch[-1].end + distance > iv.start
        ):
            close(iv.strand)
        pending[iv.strand].append(iv)
    for strand in ("+", "-"):
        close(strand)
    return loci


def _locus_chunks(n_loci: int, n_proc: int) -> list[tuple[int, int]]:
    """Split ``range(n_loci)`` into the ``(lo, hi)`` chunks of the mapping tasks.

    Read depth spans orders of magnitude between loci — the deepest single
    locus of a human Ribo-seq run costs seconds on its own — so a chunk must
    stay small enough that one hot locus cannot become the critical path.  A
    handful of loci per chunk keeps enough genomic locality for the BAM
    fetches to sweep the file, while leaving the ``imap_unordered`` dispatch
    free to balance the rest.
    """
    chunk_size = max(1, min(4, math.ceil(n_loci / (n_proc * 4))))
    return [
        (i, min(i + chunk_size, n_loci)) for i in range(0, n_loci, chunk_size)
    ]


#: Loci to map against, indexed by the chunk bounds of the tasks.  Set in
#: every mapping worker by :func:`_init_mapping_worker`.
_WORKER_LOCI: list[Locus] = []


def _init_mapping_worker(loci: list[Locus]) -> None:
    """Give a mapping worker its loci (the pool's ``initializer``)."""
    global _WORKER_LOCI
    _WORKER_LOCI = loci

#: Layout of the collapsed-reads blob.  ``count`` is how many identical
#: mappings collapse into one key; on deep, non-deduplicated libraries a
#: single key can exceed 65,535, so it must be wider than ``uint16``.
_READS_COLUMNS = (
    ("is_first_iv", bool),
    ("start", np.uint32),
    ("end", np.uint32),
    ("untemplated_addition", bool),
    ("unique", bool),
    ("count", np.uint32),
)


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
    region = GenomicRegion(blocks, chrom=chrom, strand=strand)
    return [
        transcript
        for transcript in set.intersection(*transcript_sets)
        if transcript.exons.try_map_to_local(region) is not None
    ]


#: The transcript set of a read that maps into none.
_NO_TRANSCRIPTS: frozenset[str] = frozenset()


class _TranscriptResolver:
    """The transcripts of one locus that a read's reference blocks map into.

    Most reads never need a :class:`~price2.ribo_seq_alignment.RiboSeqAlignment`
    or a :meth:`~price2.genomic_region.GenomicRegion.map_to_local` call.  A
    read that is a single block maps into exactly the transcripts common to
    every breakpoint step it touches, provided those steps cover it without
    a gap (a transcript's exons are separated by introns, so a gapless
    covered stretch lies inside one exon; a locus with abutting exons loses
    this path, see :attr:`~price2.locus.Locus.has_abutting_exons`).  A read
    that is two blocks maps into a transcript iff its gap is one of that
    transcript's introns and its outer ends stay within the flanking exons
    -- a lookup in :attr:`~price2.locus.Locus.transcript_junction_index`.
    Everything else falls back to :func:`_slow_path_transcripts`.

    One resolver serves all reads of a locus: single-block reads covered by
    the same run of steps share one memoised transcript set.
    """

    __slots__ = (
        "locus",
        "bp_starts",
        "bp_ends",
        "bp_sets",
        "n_bp",
        "junctions",
        "single_block_fast",
        "_step_memo",
    )

    def __init__(self, locus: Locus) -> None:
        self.locus = locus
        self.bp_starts, self.bp_ends, self.bp_sets = locus.transcript_breakpoint_index
        self.n_bp = len(self.bp_starts)
        self.junctions = locus.transcript_junction_index
        self.single_block_fast = not locus.has_abutting_exons
        self._step_memo: dict[tuple[int, int], frozenset[str]] = {}

    def resolve(self, blocks: list[tuple[int, int]]) -> frozenset[str]:
        """The ids of the transcripts *blocks* (in chromosome order) map into."""
        if len(blocks) == 2:
            return self._two_blocks(blocks)
        if len(blocks) != 1 or not self.single_block_fast:
            return frozenset(tr.id for tr in _slow_path_transcripts(blocks, self.locus))
        # Single block, inlined: this is the path nearly every read takes.
        start, end = blocks[0]
        bp_starts, bp_ends, n_bp = self.bp_starts, self.bp_ends, self.n_bp
        i0 = bisect.bisect_right(bp_ends, start)
        if i0 >= n_bp or bp_starts[i0] > start:
            return _NO_TRANSCRIPTS  # read starts outside any exon
        i1 = i0
        while i1 + 1 < n_bp and bp_starts[i1 + 1] < end:
            if bp_starts[i1 + 1] != bp_ends[i1]:
                break  # gap between steps
            i1 += 1
        if bp_ends[i1] < end:
            return _NO_TRANSCRIPTS  # read runs past the covered stretch
        ids = self._step_memo.get((i0, i1))
        if ids is None:
            bp_sets = self.bp_sets
            candidates = (
                bp_sets[i0] if i0 == i1 else set.intersection(*bp_sets[i0 : i1 + 1])
            )
            ids = self._step_memo[(i0, i1)] = frozenset(tr.id for tr in candidates)
        return ids

    def _two_blocks(self, blocks: list[tuple[int, int]]) -> frozenset[str]:
        (a1, b1), (a2, b2) = blocks
        spliced = self.junctions.get((b1, a2))
        if not spliced:
            return _NO_TRANSCRIPTS  # gap is not an annotated intron here
        return frozenset(
            tr.id
            for tr, donor_start, acceptor_end in spliced
            if a1 >= donor_start and b2 <= acceptor_end
        )


def _map_locus_reads(
    alignments,
    locus: Locus,
    end_to_end: bool,
    drop_multimap: bool,
    record_multimap: bool,
) -> tuple[dict, dict, list[tuple[int, int]]]:
    """Tally one locus's alignments by footprint and by transcript set.

    Parameters
    ----------
    alignments : iterable of pysam.AlignedSegment
        The records fetched over the locus.
    locus : Locus
        The locus they were fetched from.
    end_to_end : bool
        Selects EndToEnd untemplated-addition detection
        (:func:`price2.bam.footprint`).
    drop_multimap : bool
        Discard alignments with ``NH > 1`` instead of counting them in
        every locus they align to.
    record_multimap : bool
        Also list the multimapping alignments for the linkage spill.

    Returns
    -------
    tuple
        ``(mappings, transcript_counts, multimappers)``: the collapsed
        footprints ``{(untemplated_addition, unique, blocks): count}`` (the
        input of :func:`_reads_frame`), the reads per compatible transcript
        set ``{frozenset of transcript ids: count}``, and the
        ``(query-name hash, group key)`` of every recorded multimapping
        alignment.
    """
    is_minus = locus.iv.strand == "-"
    resolver = _TranscriptResolver(locus)
    resolve = resolver.resolve
    qname_hash = multimap.qname_hash
    group_key = multimap.group_key

    mappings: dict = defaultdict(int)
    transcript_counts: dict = defaultdict(int)
    multimappers: list[tuple[int, int]] = []
    for alignment in alignments:
        if alignment.is_unmapped or alignment.is_reverse != is_minus:
            continue
        # Checked before the transcript mapping: the alignment is dropped
        # outright, so none of that work is needed.
        if drop_multimap and not is_unique(alignment):
            continue
        found = footprint(alignment, end_to_end)
        if found is None:
            continue
        blocks, ua = found
        transcript_ids = resolve(blocks)
        if not transcript_ids:
            continue
        # Multimapping alignments were skipped above when dropping them.
        unique = drop_multimap or is_unique(alignment)
        ivs_tuple = tuple(blocks)
        mappings[(ua, unique, ivs_tuple)] += 1
        transcript_counts[transcript_ids] += 1
        # Record this multimapping alignment so its read can be linked to
        # its other in-locus slots for the EM E-step.
        if record_multimap and not unique:
            multimappers.append(
                (qname_hash(alignment.query_name), group_key(ivs_tuple, ua))
            )
    return mappings, transcript_counts, multimappers


def collect_mappings_chunk(data: tuple) -> tuple:
    """Map one run's reads against a contiguous chunk of loci.

    Designed to be called via :class:`multiprocessing.Pool`; the loci
    themselves are read from the :data:`_WORKER_LOCI` module global, which
    :func:`_init_mapping_worker` sets in every worker.  Each locus's reads
    are tallied by :func:`_map_locus_reads`.

    Parameters
    ----------
    data : tuple
        ``(run_id, bam_dir, lo, hi, chunk_idx, run_spill_dir, end_to_end,
        drop_multimap)`` where ``lo``/``hi`` slice :data:`_WORKER_LOCI`,
        ``run_spill_dir`` is empty when multimapping linkage is not being
        recorded, ``end_to_end`` selects EndToEnd untemplated-addition
        detection, and ``drop_multimap`` discards alignments with ``NH > 1``
        instead of counting them in every locus they align to.

    Returns
    -------
    tuple
        ``(reads_rows, transcript_count_rows)`` — blob rows for the parent
        to insert.  Multimapping alignments are spilled to disk, not
        returned.
    """
    (
        run_id,
        bam_dir,
        lo,
        hi,
        chunk_idx,
        run_spill_dir,
        end_to_end,
        drop_multimap,
    ) = data
    record_multimap = bool(run_spill_dir)

    reads_rows: list = []
    transcript_count_rows: list = []
    mm_qnames: list = []
    mm_loci: list = []
    mm_keys: list = []

    # One handle per worker: chunks of one run land on the same worker
    # repeatedly, so the index is parsed once.
    sf = cached_alignment_file(f"{bam_dir}/{run_id}.bam", exclusive=True)

    for locus_idx in range(lo, hi):
        locus = _WORKER_LOCI[locus_idx]
        mappings, transcript_counts, multimappers = _map_locus_reads(
            sf.fetch(locus.iv.chrom, locus.iv.start, locus.iv.end),
            locus,
            end_to_end,
            drop_multimap,
            record_multimap,
        )
        reads_rows.append(
            (locus.id, run_id, database.compress_blob(_reads_frame(mappings)))
        )
        transcript_count_rows.append(
            (locus.id, run_id, database.compress_blob(transcript_counts))
        )
        for qname, key in multimappers:
            mm_qnames.append(qname)
            mm_loci.append(locus_idx)
            mm_keys.append(key)

    if record_multimap:
        multimap.write_spill(run_spill_dir, mm_qnames, mm_loci, mm_keys)

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
