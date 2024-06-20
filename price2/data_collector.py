import os
import HTSeq
import sqlite3
from pickle import dumps
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

    def __init__(self, 
                 bam_dir: str,
                 reference_annotation: ReferenceAnnotation,
                 genome: dict[str: HTSeq.Sequence],
                 runs: list[RiboSeqRun]
                 ):
        
        self.runs = runs

        # get chromosome order
        self.chr_order = None
        self.bam_dir = bam_dir
        for bam_file in os.listdir(bam_dir):
            if bam_file.endswith(".bam"):
                if not self.chr_order:
                    bam_reader = HTSeq.BAM_Reader(os.path.join(bam_dir, bam_file))
                    self.chr_order = [x['SN'] for x in bam_reader.get_header_dict()['SQ']]
                break

        # make loci
        loci_intervals_binary = HTSeq.GenomicArray("auto", stranded=True, storage="step", typecode="b")
        for transcript in reference_annotation.transcripts.values():
            loci_intervals_binary[transcript.iv] = True
        self.loci_intervals = HTSeq.GenomicArray("auto", stranded=True, storage="step", typecode="O")
        self.loci_set = set()

        connected_loci = {'+':[], '-':[]}
        loci_counter = 0
        for iv, step in loci_intervals_binary.steps():
            if step:
                if not connected_loci[iv.strand]:
                    connected_loci[iv.strand].append(iv)
                    continue
                if connected_loci[iv.strand][-1].chrom == iv.chrom\
                        and connected_loci[iv.strand][-1].end + 50 > iv.start:
                    connected_loci[iv.strand].append(iv)
                else:
                    connected_iv = HTSeq.GenomicInterval(
                        connected_loci[iv.strand][0].chrom, 
                        connected_loci[iv.strand][0].start, 
                        connected_loci[iv.strand][-1].end, 
                        iv.strand
                        )
                    locus = Locus(connected_iv, reference_annotation.transcript_intervals, loci_counter)
                    loci_counter += 1
                    self.loci_intervals[connected_iv] = locus
                    self.loci_set.add(locus)
                    connected_loci[iv.strand] = [iv]
        for strand in ['+', '-']:
            try:
                connected_iv = HTSeq.GenomicInterval(
                    connected_loci[strand][0].chrom, 
                    connected_loci[strand][0].start, 
                    connected_loci[strand][-1].end, 
                    strand
                    )
                locus = Locus(connected_iv, reference_annotation.transcript_intervals, loci_counter)
                loci_counter += 1
                self.loci_intervals[connected_iv] = locus
                self.loci_set.add(locus)
            except IndexError:
                pass
        
        # make rgrs
        for locus in self.loci_set:
            locus.make_rgrs(genome)


    def collect_mappings(self, reads_db_path:str, processes:int=32):
        if os.path.exists(reads_db_path):
            os.remove(reads_db_path)

        self.read_db = sqlite3.connect(reads_db_path)

        self.cur = self.read_db.cursor()

        self.cur.execute('''CREATE TABLE IF NOT EXISTS reads (
                    locus_id text NOT NULL,
                    run_id text NOT NULL,
                    reads_blob blob NOT NULL
                    )''')
        self.read_db.commit()
        self.read_db.close()
        
        with Pool(processes) as p:
            p.map(collect_mappings_run, [(run, self.bam_dir, reads_db_path, self.loci_set) for run in self.runs])



        ## get read counts for each locus
        #self.read_db = sqlite3.connect(reads_db_path)
        #self.cur = self.read_db.cursor()
#
        #for loc in self.loci_set:
        #    temp = self.cur.execute('''SELECT reads_blob FROM reads
        #                     WHERE locus_id = ?''', (loc.id,))
        #    for run in temp:
        #        df = pd.read_pickle(run[0])
        #        loc.read_count += len(df)




    def collect_loci(self, loci_db_path:str):
        if os.path.exists(loci_db_path):
            os.remove(loci_db_path, )
        loci_db = sqlite3.connect(loci_db_path)
        cur = loci_db.cursor()


        cur.execute('''CREATE TABLE loci (
                    locus_id text PRIMARY KEY,
                    loc_blob blob
                    )''')

        for loc in self.loci_set:
            cur.execute('INSERT INTO loci VALUES (?, ?)', (loc.id, dumps(loc)))

        loci_db.commit()
        loci_db.close()


    def collect_runs(self, run_db_path:str):
        if os.path.exists(run_db_path):
            os.remove(run_db_path)
        run_db = sqlite3.connect(run_db_path)
        cur = run_db.cursor()

        cur.execute('''CREATE TABLE runs (
                    run_id text PRIMARY KEY,
                    run_blob blob
                    )''')

        for run in self.runs:
            cur.execute('INSERT INTO runs VALUES (?, ?)', (run.id, dumps(run)))

        run_db.commit()
        run_db.close()


        


def collect_mappings_run(data):

    run, bam_dir, reads_db_path, loci_set = data

    br = HTSeq.BAM_Reader(f'{bam_dir}/{run.id}.bam')

    read_db = sqlite3.connect(reads_db_path)
    cur = read_db.cursor()

    for locus in loci_set:
        strand = locus.iv.strand
        read_list = []
        read_counter = 0
        for alignment in br.fetch(locus.iv.chrom, locus.iv.start, locus.iv.end):
            if alignment.iv.strand != strand:
                continue
            rgrs = set()
            rsa = RiboSeqAlignment(alignment)
            for query_iv in rsa.genomic_region.intervals:
                for subject_iv, rgr_set in locus.rgr_intervals[query_iv].steps():
                    rgrs |= rgr_set
            if rgrs:
                for iv in rsa.genomic_region.intervals:
                    read_list.append((
                        read_counter,
                        iv.chrom,
                        iv.strand,
                        iv.start,
                        iv.end,
                        rsa.untemplated_addition,
                        rsa.unique(),
                    ))
                read_counter += 1
            for rgr in rgrs:
                try:
                    iv_on_rgr = rgr.genomic_region.induce(rsa.genomic_region)
                except ValueError:
                    continue
                if rgr.type == 'ORF':
                    frame = iv_on_rgr[0]%3
                else:
                    frame = None
                likelihood = run.cleavage_model.pmf(
                    len(rsa),
                    rsa.untemplated_addition,
                    frame,
                )
                rgr.read_count += likelihood
        
        df = pd.DataFrame(read_list, columns=['read_id','chrom', 'strand', 'start', 'end', 'untemplated_addition', 'unique'])
        df['read_id'] = df['read_id'].astype(np.uint32)
        df['start'] = df['start'].astype(np.uint32)
        df['end'] = df['end'].astype(np.uint32)
        df['chrom'] = df['chrom'].astype('category')
        df['strand'] = df['strand'].astype('category')


        cur.execute('''INSERT INTO reads (
                         locus_id,
                         run_id,
                         reads_blob
                         ) VALUES (?, ?, ?)''', (locus.id, run.id, zlib.compress(dumps(df))))
        

        read_db.commit()

    read_db.close()

