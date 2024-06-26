import HTSeq

from .genomic_features import ReadGeneratingRegion, Transcript
from .ribo_seq_alignment import RiboSeqAlignment
from .ribo_seq_run import RiboSeqRun

  

class CompatibilityGroup:
    def __init__(self, length: int=0) -> None:
        self.length = length
        self.read_count = 0

    
    def __repr__(self) -> str:
        return f'CG:length:{self.length}, read_count:{self.read_count}'
    

    def add_length_to_cgs(cgs_dict: dict[(frozenset, frozenset, frozenset, frozenset):'CompatibilityGroup'],
                   rgr_length: list[(ReadGeneratingRegion, int)],
                   step_length: int) -> None:
        nones, zeros, ones, twos = set(), set(), set(), set()
        for rgr, length in rgr_length:
            if rgr.type=='NOISE':
                nones.add(rgr)
                continue
            l = length%3
            if l == 0:
                zeros.add(rgr)
            elif l == 1:
                ones.add(rgr)
            elif l == 2:
                twos.add(rgr)

        nones, zeros, ones, twos = frozenset(nones), frozenset(zeros), frozenset(ones), frozenset(twos)
        if (nones, zeros, ones, twos) in cgs_dict:
            cgs_dict[(nones, zeros, ones, twos)].length += step_length
        elif (nones, ones, twos, zeros) in cgs_dict:
            cgs_dict[(nones, ones, twos, zeros)].length += step_length
        elif (nones, twos, zeros, ones) in cgs_dict:
            cgs_dict[(nones, twos, zeros, ones)].length += step_length
        else:
            cgs_dict[(nones, zeros, ones, twos)] = CompatibilityGroup(length=step_length)


class EquivalenceGroup:
    length: int
    read_count: int

    def __init__(self, 
                 ) -> None:
        self.length = 0
        self.read_count = 0

class Locus:
    # permanent properties
    iv: HTSeq.GenomicInterval
    read_count: int
    read_counts: dict[RiboSeqRun:int]
    rgr_set = set[ReadGeneratingRegion]
    rgr_intervals: HTSeq.GenomicArrayOfSets
    transcripts: set[Transcript]
    overlap_likelihood_ratio_threshold: float = .99

    cgs: dict[RiboSeqRun:dict[frozenset[ReadGeneratingRegion]: CompatibilityGroup]]
    egs: dict[EquivalenceGroup: EquivalenceGroup]

    def __init__(self, iv: HTSeq.GenomicInterval, transcript_intervals: HTSeq.GenomicArrayOfSets, loci_number: int) -> None:
        self.iv = iv
        self.read_count = 0
        self.read_counts = dict()
        self.rgr_set = set()
        self.rgr_intervals = HTSeq.GenomicArrayOfSets("auto", stranded=True, storage="step")
        self.uncounted_reads = 0
        self.times = dict()
        self.id = f'loc_{loci_number}'

        self.transcript_intervals = HTSeq.GenomicArrayOfSets("auto", stranded=True, storage="step")

        for iv, val in transcript_intervals[self.iv].steps():
            self.transcript_intervals[iv] = val

        self.transcripts = set()

        #for iv, value in transcript_intervals[self.iv].steps():
        for iv, value in self.transcript_intervals.steps():
            self.transcripts |= value

        self.lls = [] # for plotting


    def __repr__(self) -> str:
        return f'Locus({self.iv})'

    
    def make_rgrs(self, genome: dict[str: HTSeq._HTSeq.Sequence],) -> None:
        for transcript in self.transcripts:
            noise = ReadGeneratingRegion(
                'NOISE',
                transcript,
                (0, len(transcript.exons)),
            )
            if not noise in self.rgr_set:
                self.rgr_set.add(noise)
                for iv in noise.genomic_region.intervals:
                    self.rgr_intervals[iv] += noise
            
            seq = transcript.exons.get_sequence(genome)
            for orf_iv_on_transcript in find_orfs(seq):
                orf = ReadGeneratingRegion(
                    'ORF',
                    transcript,
                    orf_iv_on_transcript,
                )
                if not orf in self.rgr_set:
                    self.rgr_set.add(orf)
                    for iv in orf.genomic_region.intervals:
                        self.rgr_intervals[iv] += orf
        
        l = list(self.rgr_set)
        for rgr in self.rgr_set:
            rgr.index = l.index(rgr)

    
    def make_compatibility_groups(self, l: int=20, upstream_cleavage: int=12, downstream_cleavage: int=17) -> None:
        self.cgs = dict()
        lengths = dict()

        for step_iv, step_set in self.rgr_intervals.steps():
            if not step_set:
                continue
            for i in range(step_iv.length):
                gr_dict = dict()
                for rgr in step_set:
                    if not rgr in lengths:
                        lengths[rgr] = 0
                    if rgr.genomic_region.strand == '+':
                        read_gr = rgr.genomic_region.map((lengths[rgr]-upstream_cleavage, lengths[rgr]+downstream_cleavage))
                    else:
                        rgr_len = len(rgr.genomic_region)
                        read_gr = rgr.genomic_region.map((rgr_len-lengths[rgr]-upstream_cleavage, rgr_len-lengths[rgr]+downstream_cleavage))
                    if not read_gr in gr_dict:
                        gr_dict[read_gr] = set()
                    gr_dict[read_gr].add(rgr)
                for rgr_set in gr_dict.values():
                    CompatibilityGroup.add_length_to_cgs(self.cgs, [(rgr, lengths[rgr]) for rgr in rgr_set], 1)
                for rgr in step_set:
                    lengths[rgr] += 1
                    

    def make_equivalence_groups(self, runs: list[RiboSeqRun]) -> None:
        self.egs = dict()
        for run in runs:
            self.egs[run] = dict()
            for cg in self.cgs:
                for length in run.cleavage_model.non_zero_lengths:
                    for frame in range(3):
                        for oua in [True, False]:
                            rgr_frame = set()

                            fr = None
                            if run.cleavage_model.pmf(length, oua, fr) > 0:
                                for none in cg[0]:
                                    rgr_frame.add((none, None))

                            if self.iv.strand == '+':
                                fr = (frame+0)%3
                            else:
                                fr = (frame-0)%3
                            if run.cleavage_model.pmf(length, oua, fr) > 0:
                                for zero in cg[1]:
                                    rgr_frame.add((zero, fr))

                            if self.iv.strand == '+':
                                fr = (frame+1)%3
                            else:
                                fr = (frame-1)%3
                            if run.cleavage_model.pmf(length, oua, fr) > 0:
                                for one in cg[2]:
                                    rgr_frame.add((one, fr))

                            if self.iv.strand == '+':
                                fr = (frame+2)%3
                            else:
                                fr = (frame-2)%3
                            if run.cleavage_model.pmf(length, oua, fr) > 0:
                                for two in cg[3]:
                                    rgr_frame.add((two, fr))
                            
                            rgr_frame = frozenset(rgr_frame)

                            if not rgr_frame:
                                continue

                            eg_length = self.cgs[cg].length/3
                            if not (rgr_frame, length, oua) in self.egs[run]:
                                self.egs[run][(rgr_frame, length, oua)] = EquivalenceGroup()
                            
                            self.egs[run][(rgr_frame, length, oua)].length += eg_length
                            


    # TODO: optim
    def add_ribo_seq_alignments(self,
                                rsa: RiboSeqAlignment,
                                run: RiboSeqRun,
                                read_count: int,
                                overlap_likelihood_ratio_threshold: float=.99
                                ) -> None:
        
        #if rsa.matching_length < 15 or rsa.matching_length > 40:
        #    return
        length = len(rsa)
        oua = rsa.untemplated_addition
        rgr_frame = set()

        candidate_rgrs = set()
        for query_iv in rsa.genomic_region.intervals:
            for subject_iv, rgr_set in self.rgr_intervals[query_iv].steps():
                candidate_rgrs |= rgr_set

        if len(candidate_rgrs) == 1:
            rgr = candidate_rgrs.pop()
            try:
                iv_on_rgr = rgr.genomic_region.induce(rsa.genomic_region)
            except ValueError:
                return
            if rgr.type == 'ORF':
                frame = iv_on_rgr[0]%3
            else:
                frame = None
            rgr_frame.add((rgr, frame))

        else:
            for rgr in candidate_rgrs:
                try:
                    iv_on_rgr = rgr.genomic_region.induce(rsa.genomic_region)
                except ValueError:
                    continue
                if rgr.type == 'ORF':
                    frame = iv_on_rgr[0]%3
                else:
                    frame = None
                complete_likelihood = run.cleavage_model.pmf(
                    length,
                    oua,
                    frame,
                )
                if complete_likelihood == 0:
                    continue
                if iv_on_rgr[0] < 0 or iv_on_rgr[1] > len(rgr.genomic_region):
                    overlap_likelihood = run.cleavage_model.pmf(
                        length,
                        rsa.untemplated_addition,
                        frame,
                        region_start = -iv_on_rgr[0],
                        region_end= -iv_on_rgr[0] + len(rgr.genomic_region),
                    )
                    #try:
                    if overlap_likelihood/complete_likelihood >= overlap_likelihood_ratio_threshold:
                        rgr_frame.add((rgr, frame))
                    #except ZeroDivisionError:
                    #    return
                else:
                    rgr_frame.add((rgr, frame))
        
        if not rgr_frame:
            return
        
        rgr_frame = frozenset(rgr_frame)

        #if len(rgr_frame) == 1:
        #    pass

        try:
            self.egs[run][(rgr_frame, length, oua)].read_count += read_count
            run.read_count += read_count
        except KeyError:
            self.uncounted_reads += read_count

        try:
            self.read_counts[run] += read_count
        except KeyError:
            self.read_counts[run] = read_count
       
       

# find ORFs based on transcript sequence
# return list of intervals of all ORFs with min length
def find_orfs(
        seq: str,
        min_length: int=0, 
        start_codons: set=set(['ATG',]),# 'CTG', 'GTG', 'ACG']),
        stop_codons: set=set(['TAA', 'TAG', 'TGA']) ) -> list[tuple[int, int]]:
    orf_iv_on_transcript = []
    for i in range(3):
        starts = []
        for j in range(i, len(seq), 3):
            if seq[j:j+3] in start_codons:
                starts.append(j)
            if seq[j:j+3] in stop_codons:
                for start in starts:
                    if j - start >= min_length:
                        orf_iv_on_transcript.append((start,j+3))
                starts = []
    return orf_iv_on_transcript

