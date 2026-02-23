"""Data collection orchestration for PRICE2.

This module provides the DataCollector class, which drives the multi-step
process of collecting Ribo-seq run statistics, mapping reads to loci, and
assembling locus-level data structures for downstream deconvolution.
Intermediate results are persisted in an SQLite database located in the
working directory (``price.db``).
"""

import os
import HTSeq
import sqlite3 as sql
from pickle import dumps, loads
import zlib
import pandas as pd
import numpy as np
from multiprocessing import Pool
from collections import defaultdict

from price2.reference_annotation import ReferenceAnnotation
from price2.ribo_seq_run import RiboSeqRun, ribo_seq_runs_from_bams
from price2.locus import Locus
from price2.ribo_seq_alignment import RiboSeqAlignment
from price2.config import Config


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
        genome: dict[str, HTSeq.Sequence],
        config: Config,
    ) -> None:
        """Initialise the DataCollector.

        Parameters
        ----------
        reference_annotation : ReferenceAnnotation
            Parsed reference annotation used to define loci.
        genome : dict[str, HTSeq.Sequence]
            Mapping from chromosome name to its nucleotide sequence.
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
        )
        self.runs += new_runs

        for run in new_runs:
            cur.execute("INSERT INTO runs VALUES (?, ?)", (run.id, dumps(run)))

        db.commit()
        db.close()

    def collect_mappings(self) -> None:
        """Map reads from all runs to loci and persist results.

        Creates the ``reads`` and ``transcript_read_counts`` tables in the
        database if they do not already exist, then dispatches
        :func:`collect_mappings_run` in parallel for any run whose mappings
        have not yet been stored.
        """
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

        try:
            with Pool(self.config.processes) as p:
                p.map(
                    collect_mappings_run,
                    [
                        (
                            run.id,
                            self.bam_dir,
                            self.db_path,
                            self.loci_set,
                        )
                        for run in self.runs
                        if run.id not in processed_run_ids
                    ],
                )
        except AssertionError:
            for run in self.runs:
                collect_mappings_run(
                    (
                        run.id,
                        self.bam_dir,
                        self.db_path,
                        self.loci_set,
                    )
                )

    def get_chromosome_order(self) -> None:
        """Set ``self.chr_order`` from the first BAM file found in ``bam_dir``.

        Reads the ``SQ`` header records of the first ``.bam`` file in
        ``self.bam_dir``.  ``self.chr_order`` is set to ``None`` when no BAM
        file is present.
        """
        self.chr_order = None
        for bam_file in os.listdir(self.bam_dir):
            if bam_file.endswith(".bam"):
                bam_reader = HTSeq.BAM_Reader(os.path.join(self.bam_dir, bam_file))
                self.chr_order = [x["SN"] for x in bam_reader.get_header_dict()["SQ"]]
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
        """Filter transcripts per locus and build read-generating regions.

        Loads transcript read-count data from the database, greedily selects
        transcripts that explain the most reads (above
        ``config.min_explained_reads_per_run`` per run), prunes the transcript
        set of each locus accordingly, and calls
        :meth:`~price2.locus.Locus.make_rgrs` to construct read-generating
        regions.  Serialises each non-empty locus to the ``loci`` table.
        """
        db = sql.connect(self.db_path)
        cur = db.cursor()
        r = cur.execute("""SELECT * FROM runs""")
        num_runs = len([x for x in r.fetchall()])
        min_explained_reads = self.config.min_explained_reads_per_run * num_runs

        loc_dict = {loc.id: loc for loc in self.loci_set}

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

        for loc_id in loci_ids_to_process:
            locus = loc_dict[loc_id]
            cur.execute(
                """SELECT * FROM transcript_read_counts
                        WHERE locus_id = ?""",
                (locus.id,),
            )

            # Aggregate transcript read counts across all runs.
            transcript_read_counts: dict = {}
            for entry in cur.fetchall():
                for k, v in loads(zlib.decompress(entry[2])).items():
                    try:
                        transcript_read_counts[k] += v
                    except KeyError:
                        transcript_read_counts[k] = v

            tr_ids = [t.id for t in locus.transcripts]

            rows = []
            for read_set, count in transcript_read_counts.items():
                row = [tr.id in read_set for tr in locus.transcripts]
                row.append(count)
                rows.append(row)

            df = pd.DataFrame(rows, columns=tr_ids + ["count"])

            explaining_transcripts_reads_list = []

            while df["count"].sum() > 0:
                t = df.drop(columns="count").multiply(df["count"], axis=0).sum()
                explaining_transcripts_reads_list.append((t.idxmax(), t.max()))
                df.loc[df[t.idxmax()], "count"] = 0

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

            if locus.transcripts:
                locus.make_rgrs(self.genome, self.config)
                cur.execute("INSERT INTO loci VALUES (?, ?)", (locus.id, dumps(locus)))

        db.commit()
        db.close()


def collect_mappings_run(
    data: tuple,
) -> None:
    """Map reads from a single BAM run to all loci and persist results.

    Designed to be called via :class:`multiprocessing.Pool`.  Accepts a
    single tuple argument so it is compatible with ``Pool.map``.

    Parameters
    ----------
    data : tuple
        A 4-tuple of ``(run_id, bam_dir, db_path, loci_set)`` where

        * ``run_id`` – identifier of the Ribo-seq run (BAM filename stem).
        * ``bam_dir`` – directory containing BAM files.
        * ``db_path`` – path to the SQLite database.
        * ``loci_set`` – set of :class:`~price2.locus.Locus` objects to map
          reads against.
    """
    run_id, bam_dir, db_path, loci_set = data

    br = HTSeq.BAM_Reader(f"{bam_dir}/{run_id}.bam")

    reads_rows: list = []
    transcript_count_rows: list = []

    for locus in loci_set:
        transcripts_counts = defaultdict(int)
        strand = locus.iv.strand
        mappings_dict = defaultdict(int)
        for alignment in br.fetch(locus.iv.chrom, locus.iv.start, locus.iv.end):
            if alignment.iv.strand != strand:
                continue
            transcript_sets = []
            rsa = RiboSeqAlignment(alignment)

            for query_iv in rsa.genomic_region.intervals:
                for subject_iv, transcript_set in locus.transcript_intervals[
                    query_iv
                ].steps():
                    transcript_sets.append(transcript_set)
            transcript_candidates = set.intersection(*transcript_sets)
            transcripts = []
            for tr in transcript_candidates:
                try:
                    tr.exons.map_to_local(rsa.genomic_region)
                    transcripts.append(tr)
                except ValueError:
                    continue

            if transcripts:
                transcripts_ids = frozenset(tr.id for tr in transcripts)
                ivs_tuple = tuple(
                    (iv.start, iv.end) for iv in rsa.genomic_region.intervals
                )
                mapping = (rsa.untemplated_addition, rsa.unique(), ivs_tuple)
                mappings_dict[mapping] += 1
                transcripts_counts[transcripts_ids] += 1
        read_list = []
        for entry, count in mappings_dict.items():
            rsa.untemplated_addition, uniq, ivs_tuple = entry
            is_first_iv = True
            for iv in ivs_tuple:
                read_list.append(
                    (
                        is_first_iv,
                        iv[0],
                        iv[1],
                        rsa.untemplated_addition,
                        uniq,
                        count,
                    )
                )
                is_first_iv = False
        df = pd.DataFrame(
            read_list,
            columns=[
                "is_first_iv",
                "start",
                "end",
                "untemplated_addition",
                "unique",
                "count",
            ],
        )
        df["is_first_iv"] = df["is_first_iv"].astype(bool)
        df["start"] = df["start"].astype(np.uint32)
        df["end"] = df["end"].astype(np.uint32)
        df["untemplated_addition"] = df["untemplated_addition"].astype(bool)
        df["unique"] = df["unique"].astype(bool)
        df["count"] = df["count"].astype(np.uint16)

        reads_rows.append((locus.id, run_id, zlib.compress(dumps(df))))
        transcript_count_rows.append(
            (locus.id, run_id, zlib.compress(dumps(transcripts_counts)))
        )

    db = sql.connect(db_path, timeout=60)
    cur = db.cursor()
    cur.executemany(
        """INSERT INTO reads (
                     locus_id,
                     run_id,
                     reads_blob
                     ) VALUES (?, ?, ?)""",
        reads_rows,
    )

    cur.executemany(
        """INSERT INTO transcript_read_counts (
                     locus_id,
                     run_id,
                     transcript_read_counts_blob
                     ) VALUES (?, ?, ?)""",
        transcript_count_rows,
    )
    db.commit()
    db.close()
