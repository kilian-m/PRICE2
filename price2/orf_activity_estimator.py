import HTSeq
import pysam
import numpy as np
import os

from price2.reference_annotation import ReferenceAnnotation
from price2.locus import Locus
from price2.cleavage_model import CleavageModel
from price2.ribo_seq_alignment import RiboSeqAlignment
from price2.ribo_seq_run import RiboSeqRun



class ORFActivityEstimator:
    ribo_seq_runs: list[RiboSeqRun]
    #cms: list[CleavageModel]
    #alignments: list[HTSeq.BAM_Reader]
    alignment: HTSeq.BAM_Reader
    overlap_likelihood_ratio_threshold: float
    delta_cutoff: float
    maxiter: int
    loci_intervals: HTSeq.GenomicArray
    loci_set: set[Locus]
    chr_order: list[str]
    

    def __init__(
            self,
            read_alignments_dir: str,
            reference_annotation: ReferenceAnnotation,
            overlap_likelihood_ratio_threshold: float=.99,
            delta_cutoff: float=1,
            maxiter: int=100,
            cleavage_model: CleavageModel=None,
    ) -> None:

        self.ribo_seq_runs = []
        self.chr_order = None
        # iterate bam files in read_alignments_dir
        for bam_file in os.listdir(read_alignments_dir):
            if bam_file.endswith(".bam"):
                if not self.chr_order:
                    bam_reader = HTSeq.BAM_Reader(os.path.join(read_alignments_dir, bam_file))
                    self.chr_order = [x['SN'] for x in bam_reader.get_header_dict()['SQ']]
                self.ribo_seq_runs.append(
                    RiboSeqRun(bam_file.split('.')[0], 
                               read_alignments_dir, 
                               reference_annotation,
                               cleavage_model=cleavage_model,
                               )
                )


        self.overlap_likelihood_ratio_threshold = overlap_likelihood_ratio_threshold
        self.delta_cutoff = delta_cutoff
        self.maxiter = maxiter
        loci_intervals_binary = HTSeq.GenomicArray("auto", stranded=True, storage="step", typecode="b")
        for transcript in reference_annotation.transcripts.values():
            loci_intervals_binary[transcript.iv] = True
        self.loci_intervals = HTSeq.GenomicArray("auto", stranded=True, storage="step", typecode="O")
        self.loci_set = set()
        
        connected_loci = {'+':[], '-':[]}
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
                    locus = Locus(connected_iv, reference_annotation.transcript_intervals)
                    self.loci_intervals[connected_iv] = locus
                    self.loci_set.add(locus)
                    connected_loci[iv.strand] = [iv]
        for strand in ['+', '-']:
            connected_iv = HTSeq.GenomicInterval(
                connected_loci[strand][0].chrom, 
                connected_loci[strand][0].start, 
                connected_loci[strand][-1].end, 
                strand
                )
            locus = Locus(connected_iv, reference_annotation.transcript_intervals)
            self.loci_intervals[connected_iv] = locus
            self.loci_set.add(locus)
                

    def estimate_activity(self, genome) -> None:
        # iterate chromosomes
        for chr in self.chr_order:
            next_chr = False
            current_loci = {'+': None, '-': None}

            ivs_steps = {
                '+': self.loci_intervals[HTSeq.GenomicInterval(chr, 0, 1e10, "+")].steps(),
                '-': self.loci_intervals[HTSeq.GenomicInterval(chr, 0, 1e10, "-")].steps(),
            }

            # iterate loci
            while True:
                for strand in ['+', '-']:
                    if not current_loci[strand]:
                        try:
                            while not (tmp:=next(ivs_steps[strand])[1]):
                                pass
                            current_loci[strand] = tmp
                            current_loci[strand].activate(genome, self.ribo_seq_runs)
                        except StopIteration:
                            continue
                        
                # iterate ribo_seq_runs
                for run in self.ribo_seq_runs:

                    # iterate alignments
                    for alignment in run.bam_reader:

                        if alignment.optional_field('NH') > 1:
                            continue
                        aln = RiboSeqAlignment(alignment)
                        strand = aln.alignment.iv.strand
                        if not current_loci[strand]:
                            continue

                        if aln.alignment.iv.chrom != chr: # end of chromosome (in this run)
                            next_chr = True
                            break
                        
                        if current_loci[strand].iv.overlaps(aln.alignment.iv):
                            current_loci[strand].add_ribo_seq_alignment(aln, run)
                        elif current_loci[strand].iv.end < aln.alignment.iv.start:
                            break

                if next_chr:
                    if current_loci['+']:
                        current_loci['+'].run_orf_deconvolution(self.ribo_seq_runs)
                    if current_loci['-']:
                        current_loci['-'].run_orf_deconvolution(self.ribo_seq_runs)
                    break
                
                if not current_loci['+'] and not current_loci['-']:
                    break
                elif not current_loci['+']:
                    strand = '-'
                elif not current_loci['-']:
                    strand = '+'
                elif current_loci['+'].iv.end < current_loci['-'].iv.end:
                    strand = '+'
                else:
                    strand = '-'
                current_loci[strand].run_orf_deconvolution(self.ribo_seq_runs)
                current_loci[strand] = None
