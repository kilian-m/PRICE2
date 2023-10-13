import HTSeq

from .genomic_region import GenomicRegion
from .genomic_features import Transcript
from .reference_annotation import ReferenceAnnotation


class RiboSeqAlignment():

    alignment: HTSeq._HTSeq.SAM_Alignment
    untemplated_addition: bool
    matching_length: int
    unique: bool
    genomic_region: GenomicRegion

    def __init__(self, sa: HTSeq._HTSeq.SAM_Alignment):
        self.alignment = sa
        self.unique = self.alignment.optional_field('NH') == 1
        self.untemplated_addition = self.has_untemplated_addition()
        self.matching_length = sum([x.size for x in self.alignment.cigar if x.type=='M'])\
            - self.untemplated_addition
        
        intervals = []
        for cigar_op in self.alignment.cigar:
            if cigar_op.type == 'M':
                intervals.append(cigar_op.ref_iv)
        
        if self.alignment.iv.strand == '-':
            intervals = intervals[::-1]

        self.genomic_region = GenomicRegion(intervals)

    
    # for cleavage model estimation
    def is_countable(self, reference_annotation: ReferenceAnnotation, max_considered_len: int=50, min_considered_len: int=15,
    min_dist_to_start: int=30, min_dist_to_end: int=30) -> bool:
        if not self.unique:
            return False
        if self.close_to_any_tis(reference_annotation, 30):
            return False
        if not min_considered_len <= self.matching_length <= max_considered_len:
            return False
        return True
        

    def dist_to_cds_start(self, transcript: Transcript) -> int:
        interval = transcript.cds.induce(self.genomic_region)
        if not interval:
            return None
        return interval[0]


    def dist_to_cds_end(self, transcript: Transcript) -> int:
        interval = transcript.cds.induce(self.genomic_region)
        if not interval:
            return None
        return transcript.cds.length - interval[1]


    def close_to_any_tis(self, reference_annotation: ReferenceAnnotation, distance: int=30) -> bool:
        
        gi = HTSeq.GenomicInterval(
            self.alignment.iv.chrom,
            self.alignment.iv.start - distance,
            self.alignment.iv.start + distance,
            self.alignment.iv.strand,
        )
        try:
            if len(list(reference_annotation.ti_sites_new[gi].steps()))>1:
                return True
        except IndexError:
            pass
        return False
    
    def overlaps_ti(self, reference_annotation: ReferenceAnnotation) -> bool:
        try:
            if len(l:=list(reference_annotation.ti_sites_new[self.alignment.iv].steps()))>1:
                return l
        except IndexError:
            pass
        return False

    #def ti_site_position(self, reference_annotation: ReferenceAnnotation) -> int:
    #    #try:
    #    for step in reference_annotation.ti_sites[self.alignment.iv].steps():
    #        if step[1]:
    #            return step[0].start - self.alignment.iv.start
    #    #except IndexError:
    #    #    pass

    def get_frames(self, reference_annotation) -> tuple:
        transcript_candidates = reference_annotation.collect_coding_transcripts(self.genomic_region)
        if len(transcript_candidates) == 0:
            return None
        frames = [0,0,0]
        for transcript in transcript_candidates:
            frame = transcript.cds.induce(self.genomic_region)[0]%3
            frames[frame] += 1
        return tuple(frames)


    def get_frame(self, reference_annotation) -> int:
        frames = self.get_frames(reference_annotation)
        if frames[0] and not frames[1] and not frames[2]:
            return 0
        if not frames[0] and frames[1] and not frames[2]:
            return 1
        if not frames[0] and not frames[1] and frames[2]:
            return 2
        return None


    def has_untemplated_addition(self) -> bool:
        if self.alignment.iv.strand == '+':
            return self.alignment.optional_field('MD')[0] == '0'
        elif self.alignment.iv.strand == '-':
            return self.alignment.optional_field('MD')[-1] == '0'\
                and self.alignment.optional_field('MD')[-2] in 'ACGT'
        
        
    def is_spliced(self) -> bool:
        if len(self.alignment.cigar) >= 3:
            for cigar_op in self.alignment.cigar:
                if cigar_op.type == 'N':
                    return True
        return False
            
