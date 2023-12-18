import HTSeq

from scipy.stats import poisson

from .genomic_features import ReadGeneratingRegion, Transcript
from .ribo_seq_alignment import RiboSeqAlignment
from .cleavage_model import CleavageModel
from .ribo_seq_run import RiboSeqRun


class CompatibilityGroup:
    def __init__(self, length: int=0) -> None:
        self.length = length
        self.read_count = 0


class EquivalenceGroup:
    def __init__(self) -> None:
        self.read_count = 0


class Locus:
    # permanent properties
    iv: HTSeq.GenomicInterval
    read_count: int
    read_counts: dict[RiboSeqRun:int]
    rgr_set = set[ReadGeneratingRegion]
    rgr_intervals: HTSeq.GenomicArrayOfSets
    transcripts: set[Transcript]
    status: str # inactive, active, complete
    overlap_likelihood_ratio_threshold: float = .99

    # temporary properties
    #cg: dict[frozenset[ReadGeneratingRegion]: CompatibilityGroup] # compatibility groups
    #eg: dict[(frozenset[(ReadGeneratingRegion, int)], int, bool): EquivalenceGroup] # association groups

    cgs: dict[RiboSeqRun:dict[frozenset[ReadGeneratingRegion]: CompatibilityGroup]]
    egs: dict[RiboSeqRun:dict[(frozenset[(ReadGeneratingRegion, int)], int, bool): EquivalenceGroup]]

    def __init__(self, iv: HTSeq.GenomicInterval, transcript_intervals: HTSeq.GenomicArrayOfSets) -> None:
        self.status = 'inactive'
        self.iv = iv
        self.read_count = 0
        self.read_counts = dict()
        self.rgr_set = set()
        self.rgr_intervals = HTSeq.GenomicArrayOfSets("auto", stranded=True, storage="step")

        self.transcripts = set()
        for iv, value in transcript_intervals[self.iv].steps():
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

    
    def make_compatibility_groups(self, runs: list[RiboSeqRun]) -> None:
        self.cgs = dict()
        for run in runs:
            self.cgs[run] = dict()
        for step_iv, step_set in self.rgr_intervals.steps():
            fr_set = frozenset(step_set)
            if not fr_set:
                continue # skip empty sets
            if not fr_set in self.cgs[runs[0]]:
                for run in runs:
                    cg = CompatibilityGroup()
                    self.cgs[run][fr_set] = cg
            for run in runs:
                self.cgs[run][fr_set].length += step_iv.length

    
    def activate(self, genome: dict[str: HTSeq._HTSeq.Sequence], runs: list[RiboSeqRun]) -> None:
        if self.status == 'inactive':
            self.status = 'active'
        else:
            raise ValueError('Locus is not inactive')
        self.eg = dict()
        self.egs = dict()
        for run in runs:
            self.egs[run] = dict()
        self.make_rgrs(genome)
        self.make_compatibility_groups(runs)


    def add_ribo_seq_alignment(self, rsa: RiboSeqAlignment, run: RiboSeqRun) -> None:
        if rsa.matching_length < 15 or rsa.matching_length > 40:
            return
        
        length = rsa.matching_length
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
                if iv_on_rgr[0] < 0 or iv_on_rgr[1] > len(rgr.genomic_region):
                    overlap_likelihood = run.cleavage_model.pmf(
                        length,
                        rsa.untemplated_addition,
                        frame,
                        region_start = -iv_on_rgr[0],
                        region_end= -iv_on_rgr[0] + len(rgr.genomic_region),
                    )
                    try:
                        if overlap_likelihood/complete_likelihood >= self.overlap_likelihood_ratio_threshold:
                            rgr_frame.add((rgr, frame))
                    except ZeroDivisionError:
                        return
                else:
                    rgr_frame.add((rgr, frame))
        
        if not rgr_frame:
            return
        
        rgr_frame = frozenset(rgr_frame)

        if not (rgr_frame, length, oua) in self.egs[run]:
            eg = EquivalenceGroup()
            self.egs[run][(rgr_frame, length, oua)] = eg
        self.egs[run][(rgr_frame, length, oua)].read_count += 1

        self.read_count += 1
        try:
            self.read_counts[run] += 1
        except KeyError:
            self.read_counts[run] = 1

    
    def fill_compatibility_groups(self, runs: list[RiboSeqRun]) -> None:
        for run in runs:
            for (eg_rgrs_frame, _, _), eg in self.egs[run].items():
                rgrs = frozenset({rgr_frame[0] for rgr_frame in eg_rgrs_frame})
                if not rgrs in self.cgs[run]:
                    cg = CompatibilityGroup(length=1)
                    self.cgs[run][rgrs] = cg
                self.cgs[run][rgrs].read_count += eg.read_count


    def run_orf_deconvolution(self,
                              runs: list[RiboSeqRun],
                              iterations: int=100,
                              activity_change_cutoff: float=.005,
                              )-> None:
        
        self.fill_compatibility_groups(runs)

        for rgr in self.rgr_set:
            rgr.activity = {}
            for run in runs:
                rgr.activity[run] = 1/len(self.rgr_set)
        
        for rgr in self.rgr_set:
            rgr.unscaled_activity = {}

        for rgr in self.rgr_set:
            rgr.average_activity = 1/len(self.rgr_set)

        self.active_rgrs = self.rgr_set.copy()

        self.rgr_activities = {(rgr, run):[] for rgr in self.active_rgrs for run in runs}
        self.rrc = {(rgr, run):[] for rgr in self.active_rgrs for run in runs}
        self.average_activities = {rgr:[rgr.average_activity] for rgr in self.active_rgrs}

        self.activity_changes = []

        for i in range(iterations):
            for rgr in self.active_rgrs:
                rgr.read_count = {}
                for run in runs:
                    rgr.read_count[run] = 0

            # E-step (assign reads to RGRs)
            for run in runs:
                con = False
                for (rgr_frame, length, oua), eg in self.egs[run].items():
                    count = eg.read_count
                    likelihoods = {}

                    # compute the likelihood for a read in this equivalence group to be generated by each overlapping orf
                    for rgr, frame in rgr_frame:
                        likelihoods[rgr] = run.cleavage_model.pmf(length, oua, frame) * rgr.average_activity

                    likelihood_sum = sum(likelihoods.values())
                    if likelihood_sum == 0:
                        con = True
                        break
                    p = {rgr: likelihoods[rgr]/likelihood_sum for rgr in likelihoods}
                    for rgr, frame in rgr_frame:
                        rgr.read_count[run] += count * p[rgr]

                    if con:
                        continue

            # M-step (update RGR activities)
            for run in runs:
                activity_sum = 0
                unscaled_activities = {}
                for rgr in self.active_rgrs:
                    unscaled_activities[rgr] = rgr.read_count[run] / len(rgr)
                    activity_sum += unscaled_activities[rgr]
                for rgr in self.active_rgrs:
                    if activity_sum != 0:
                        rgr.activity[run] = unscaled_activities[rgr] / activity_sum
                        rgr.unscaled_activity[run] = unscaled_activities[rgr]
                    else:
                        rgr.activity[run] = unscaled_activities[rgr]
            
            for rgr in self.active_rgrs:
                rgr.average_activity = sum([rgr.activity[run] for run in runs]) / len(runs)
                self.average_activities[rgr].append(rgr.average_activity)

            self.lls.append(self.incomplete_data_likelihood(runs))

            for rgr in self.rgr_set:
                for run in runs:
                    self.rgr_activities[(rgr, run)].append(rgr.activity[run])
            for rgr in self.active_rgrs:
                for run in runs:
                    self.rrc[(rgr, run)].append(rgr.read_count[run])

            # sum differences in region activities
            #if i>1:
            #    activity_change = 0
            #    for rgr in self.active_rgrs:
            #        activity_change += abs(self.average_activities[rgr][-1] - self.average_activities[rgr][-2])
            #    self.activity_changes.append(activity_change)
            #    # regularize if activity changes less than delta_cutoff, if this does not work break
            #    if activity_change < activity_change_cutoff:
            #        if not self.regularize(): # TODO implement this
            #            break

        #self.break_iteration = i

        if self.status == 'active':
            self.status = 'complete'
        else:
            raise ValueError('Locus is not active')
        
        #del self.egs
        #del self.cgs 


    def incomplete_data_likelihood(self, runs) -> float:
        ll = 0
        for run in runs:
            for cg_rgrs, cg in self.cgs[run].items():
                activity = sum([rgr.unscaled_activity[run] for rgr in cg_rgrs])
                expected_reads_in_cg = activity * cg.length
                actual_reads_in_cg = cg.read_count
                ll += poisson.logpmf(actual_reads_in_cg, expected_reads_in_cg)
        return ll
            

    def regularize(self, likelihood_cutoff: float=10, ) -> bool:
        success = False
        ll = self.incomplete_data_likelihood()
        regularized_rgrs = []
        for rgr in self.active_rgrs:
            rgr.original_activity = rgr.activity
        for removed_rgr in self.active_rgrs:
            # experimentally remove region, reassign reads and compute new likelihood
            for rgr in self.active_rgrs:
                rgr.read_count = 0

            # E-step
            con = False
            for (rgr_frame, length, oua), eg in self.eg.items():
                count = eg.read_count
                likelihoods = {}

                for rgr, frame in rgr_frame:
                    if rgr == removed_rgr:
                        likelihoods[rgr] = 0
                    else:
                        likelihoods[rgr] = self.cm.pmf(length, oua, frame) * rgr.activity

                likelihood_sum = sum(likelihoods.values())
                if likelihood_sum == 0:
                    con = True
                    break
                p = {rgr: likelihoods[rgr]/likelihood_sum for rgr in likelihoods}
                for rgr, frame in rgr_frame:
                    rgr.read_count += count * p[rgr]

            if con:
                continue

            # M-step
            for rgr in self.active_rgrs:
                rgr.activity = rgr.read_count / len(rgr)

            if self.incomplete_data_likelihood() + likelihood_cutoff > ll:
                success = True
                regularized_rgrs.append(removed_rgr)

            for rgr in self.active_rgrs:
                rgr.activity = rgr.original_activity

        for rgr in regularized_rgrs:
            self.deactivate_rgr(rgr)

        return success

        
    def deactivate_rgr(self, rem_rgr: ReadGeneratingRegion) -> None:
        self.active_rgrs = set([rgr for rgr in self.active_rgrs if rgr != rem_rgr])
        rem_rgr.activity = 0


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