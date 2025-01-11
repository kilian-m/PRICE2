import os
import HTSeq
import sqlite3
from pickle import dumps, loads
import zlib
import pandas as pd
import numpy as np
from multiprocessing import Pool

from price2.reference_annotation import ReferenceAnnotation
from price2.ribo_seq_run import RiboSeqRun
from price2.locus import Locus
from price2.ribo_seq_alignment import RiboSeqAlignment


class DataCollector:
    loci_intervals: HTSeq.GenomicArray
    loci_set: set[Locus]
    chr_order: list[str]
    read_db: sqlite3.Connection

    def __init__(
        self,
        bam_dir: str,
        reference_annotation: ReferenceAnnotation,
        genome: dict[str : HTSeq.Sequence],
        runs: list[RiboSeqRun],
    ):

        self.runs = runs

        self.bam_dir = bam_dir
        self.get_chromosome_order()

        self.make_loci(reference_annotation)

    def collect_mappings(
        self,
        reads_db_path: str,
        transcript_read_counts_db_path: str,
        processes: int = 32,
    ):
        if os.path.exists(reads_db_path):
            os.remove(reads_db_path)

        read_db = sqlite3.connect(reads_db_path)

        cur = read_db.cursor()

        cur.execute(
            """CREATE TABLE IF NOT EXISTS reads (
                    locus_id text NOT NULL,
                    run_id text NOT NULL,
                    reads_blob blob NOT NULL
                    )"""
        )
        read_db.commit()
        read_db.close()

        if os.path.exists(transcript_read_counts_db_path):
            os.remove(transcript_read_counts_db_path)

        self.transcript_read_counts_db = sqlite3.connect(transcript_read_counts_db_path)
        cur = self.transcript_read_counts_db.cursor()

        cur.execute(
            """CREATE TABLE IF NOT EXISTS transcript_read_counts (
                         locus_id text NOT NULL,
                         run_id text NOT NULL,
                         transcript_read_counts_blob blob NOT NULL
                         )"""
        )
        self.transcript_read_counts_db.commit()
        self.transcript_read_counts_db.close()

        with Pool(processes) as p:
            p.map(
                collect_mappings_run,
                [
                    (
                        run,
                        self.bam_dir,
                        reads_db_path,
                        transcript_read_counts_db_path,
                        self.loci_set,
                    )
                    for run in self.runs
                ],
            )

    def make_rgrs(
        self,
        transcript_read_counts_db_path: str,
        genome: dict[str : HTSeq.Sequence],
        min_explained_reads: int = 50,
    ):
        conn = sqlite3.connect(transcript_read_counts_db_path)
        cur = conn.cursor()

        self.loci_with_transcripts = set()

        for locus in self.loci_set:
            cur.execute(
                """SELECT * FROM transcript_read_counts
                        WHERE locus_id = ?""",
                (locus.id,),
            )
            d = {}
            for entry in cur.fetchall():
                for k, v in loads(zlib.decompress(entry[2])).items():
                    try:
                        d[k] += v
                    except KeyError:
                        d[k] = v

            tr_ids = [t.id for t in locus.transcripts]

            l = []
            for k, v in d.items():
                l1 = [True if tr.id in k else False for tr in locus.transcripts]
                l1.append(v)
                l.append(l1)

            df = pd.DataFrame(l, columns=tr_ids + ["count"])

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

            locus.transcripts = locus.keep_transcripts

            if locus.transcripts:
                self.loci_with_transcripts.add(locus)
                locus.make_rgrs(genome)

    def get_chromosome_order(self):
        self.chr_order = None
        for bam_file in os.listdir(self.bam_dir):
            if bam_file.endswith(".bam"):
                if not self.chr_order:
                    bam_reader = HTSeq.BAM_Reader(os.path.join(self.bam_dir, bam_file))
                    self.chr_order = [
                        x["SN"] for x in bam_reader.get_header_dict()["SQ"]
                    ]
                return

    def make_loci(self, reference_annotation: ReferenceAnnotation):
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
                    and connected_loci[iv.strand][-1].end + 50 > iv.start
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

    def collect_loci(self, loci_db_path: str):
        if os.path.exists(loci_db_path):
            os.remove(
                loci_db_path,
            )
        loci_db = sqlite3.connect(loci_db_path)
        cur = loci_db.cursor()

        cur.execute(
            """CREATE TABLE loci (
                    locus_id text PRIMARY KEY,
                    loc_blob blob
                    )"""
        )

        for loc in self.loci_with_transcripts:
            cur.execute("INSERT INTO loci VALUES (?, ?)", (loc.id, dumps(loc)))

        loci_db.commit()
        loci_db.close()

    def collect_runs(self, run_db_path: str):
        if os.path.exists(run_db_path):
            os.remove(run_db_path)
        run_db = sqlite3.connect(run_db_path)
        cur = run_db.cursor()

        cur.execute(
            """CREATE TABLE runs (
                    run_id text PRIMARY KEY,
                    run_blob blob
                    )"""
        )

        for run in self.runs:
            cur.execute("INSERT INTO runs VALUES (?, ?)", (run.id, dumps(run)))

        run_db.commit()
        run_db.close()


def collect_mappings_run(data):

    run, bam_dir, reads_db_path, transcript_read_counts_db_path, loci_set = data

    br = HTSeq.BAM_Reader(f"{bam_dir}/{run.id}.bam")

    read_db = sqlite3.connect(reads_db_path)
    cur = read_db.cursor()

    transcript_read_counts_db = sqlite3.connect(transcript_read_counts_db_path)
    cur_tr = transcript_read_counts_db.cursor()

    for locus in loci_set:
        transcripts_counts = {}
        strand = locus.iv.strand
        mappings_dict = {}
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
                    tr.exons.induce(rsa.genomic_region)
                    transcripts.append(tr)
                except ValueError:
                    continue

            if transcripts:
                transcripts_ids = frozenset(tr.id for tr in transcripts)
                ivs_tuple = ((iv.start, iv.end) for iv in rsa.genomic_region.intervals)
                mapping = (rsa.untemplated_addition, rsa.unique(), ivs_tuple)
                if mapping not in mappings_dict:
                    mappings_dict[mapping] = 1
                else:
                    mappings_dict[mapping] += 1
                try:
                    transcripts_counts[transcripts_ids] += 1
                except KeyError:
                    transcripts_counts[transcripts_ids] = 1

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

        cur.execute(
            """INSERT INTO reads (
                         locus_id,
                         run_id,
                         reads_blob
                         ) VALUES (?, ?, ?)""",
            (locus.id, run.id, zlib.compress(dumps(df))),
        )

        read_db.commit()

        cur_tr.execute(
            """INSERT INTO transcript_read_counts (
                         locus_id,
                         run_id,
                         transcript_read_counts_blob
                         ) VALUES (?, ?, ?)""",
            (locus.id, run.id, zlib.compress(dumps(transcripts_counts))),
        )
        transcript_read_counts_db.commit()

    read_db.close()
    transcript_read_counts_db.close()
