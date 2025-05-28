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


class DataCollector:
    loci_intervals: HTSeq.GenomicArray
    loci_set: set[Locus]
    chr_order: list[str]
    read_db: sql.Connection

    def __init__(
        self,
        reference_annotation: ReferenceAnnotation,
        genome: dict[str : HTSeq.Sequence],
        config: dict,
    ):

        self.bam_dir = config["bam_dir"]
        self.get_chromosome_order()

        self.reference_annotation = reference_annotation
        self.db_path = f"{config['w_dir']}/price.db"
        self.make_loci(self.reference_annotation)
        self.config = config
        self.genome = genome

    def collect_runs(self):

        bam_ids = {
            f.split(".")[0] for f in os.listdir(self.bam_dir) if f.endswith(".bam")
        }

        if os.path.exists(self.db_path):

            db = sql.connect(self.db_path, timeout=60)
            cur = db.cursor()
            cur.execute("SELECT * FROM runs")
            tmp = [(run_id, run) for run_id, run in cur.fetchall()]
            run_ids = {run_id for run_id, _ in tmp}
            self.runs = [loads(run) for _, run in tmp]
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
            self.config["w_dir"],
            self.reference_annotation,
            self.config["processes"],
        )
        self.runs += new_runs

        for run in new_runs:
            cur.execute("INSERT INTO runs VALUES (?, ?)", (run.id, dumps(run)))

        db.commit()
        db.close()

    def collect_mappings(
        self,
    ):
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
        db.close()

        try:
            with Pool(self.config["processes"]) as p:
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

    def make_loci(self, reference_annotation: ReferenceAnnotation, distance: int = 50):
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

    def collect_loci(self):

        db = sql.connect(self.db_path)
        cur = db.cursor()
        r = cur.execute("""SELECT * FROM runs""")
        num_runs = len([x for x in r.fetchall()])
        min_explained_reads = self.config["min_explained_reads_per_run"] * num_runs

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

            locus.transcripts_number = len(locus.transcripts)
            locus.transcripts = locus.keep_transcripts

            ti = locus.transcript_intervals
            locus.transcript_intervals = HTSeq.GenomicArrayOfSets(
                "auto", stranded=True, storage="step"
            )
            for iv, val in ti.steps():
                if val:
                    for tr in val:
                        if tr in locus.keep_transcripts:
                            locus.transcript_intervals[iv] = val

            if locus.transcripts:
                locus.make_rgrs(self.genome)
                cur.execute("INSERT INTO loci VALUES (?, ?)", (locus.id, dumps(locus)))

        db.commit()
        db.close()


def collect_mappings_run(data):

    run_id, bam_dir, db_path, loci_set, worker_db_path = data

    br = HTSeq.BAM_Reader(f"{bam_dir}/{run_id}.bam")

    db = sql.connect(db_path)
    cur = db.cursor()

    cur.execute("SELECT locus_id FROM reads WHERE run_id = ?", (run_id,))

    processed_loc_ids = {loc_id for loc_id, in cur.fetchall()}
    db.close()

    loci_set = {loc for loc in loci_set if loc.id not in processed_loc_ids}

    l_reads = []
    l_transcript_read_counts = []

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
                    tr.exons.induce(rsa.genomic_region)
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

        l_reads.append((locus.id, run_id, zlib.compress(dumps(df))))
        l_transcript_read_counts.append(
            (locus.id, run_id, zlib.compress(dumps(transcripts_counts)))
        )

    db = sql.connect(worker_db_path, timeout=60)
    cur = db.cursor()
    cur.executemany(
        """INSERT INTO reads (
                     locus_id,
                     run_id,
                     reads_blob
                     ) VALUES (?, ?, ?)""",
        l_reads,
    )

    cur.executemany(
        """INSERT INTO transcript_read_counts (
                     locus_id,
                     run_id,
                     transcript_read_counts_blob
                     ) VALUES (?, ?, ?)""",
        l_transcript_read_counts,
    )
    db.commit()
    db.close()

    # db = sql.connect(db_path, timeout=60)
    # cur = db.cursor()
    # cur.execute(
    #    """INSERT INTO reads (
    #                 locus_id,
    #                 run_id,
    #                 reads_blob
    #                 ) VALUES (?, ?, ?)""",
    #    (locus.id, run_id, zlib.compress(dumps(df))),
    # )


#
# cur.execute(
#    """INSERT INTO transcript_read_counts (
#                 locus_id,
#                 run_id,
#                 transcript_read_counts_blob
#                 ) VALUES (?, ?, ?)""",
#    (locus.id, run_id, zlib.compress(dumps(transcripts_counts))),
# )
# db.commit()
# db.close()
