# assign reads to compatibility groups
# for each locus: compute ORF activities


import HTSeq

from price2.reference_annotation import ReferenceAnnotation
from price2.genomic_features import Locus
from price2.ribo_seq_alignment import RiboSeqAlignment

class CompatibilityGroup:
    def __init__(self) -> None:
        self.length = 0
        self.read_count = 0


class AssociationGroup:
    def __init__(self) -> None:
        self.read_count = 0


class ORFActivityEstimator:

    def __init__(
            self,
            reference_annotation: ReferenceAnnotation,
            loci_set: set[Locus],
            read_alignment_file_path: str,
            overlap_likelihood_ratio_threshold: float = .99,
            delta_cutoff: float=1,
            maxiter: int=100,
            ) -> None:
        self.reference_annotation = reference_annotation
        self.loci_set = loci_set


    def create_compatibility_groups(
            self,
            orf_intervals: HTSeq.GenomicArrayOfSets,
            ) -> None:
        self.compatibility_groups = {}
        for locus in loci_set:
            iv = locus.interval
            for step_iv, step_set in orf_intervals[iv].steps():
                fr_set  = frozenset(step_set)
                if not fr_set in self.compatibility_groups:
                    cg = CompatibilityGroup()
                    self.compatibility_groups[fr_set] = cg
                    locus.compatibility_groups[fr_set] = cg
                locus.compatibility_groups[fr_set].length += step_iv.length


    def fill_compatibility_groups(bam_file_path: str) -> None:
        reads = HTSeq.BAM_Reader(bam_file_path)
        for read in reads:
            aln = RiboSeqAlignment(read)
            if not aln.unique:
                continue
            if aln.is_spliced(): # TODO: remove this
                continue
            