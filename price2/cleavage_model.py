import numpy as np
from numba import njit
import HTSeq

from .reference_annotation import ReferenceAnnotation
from .ribo_seq_alignment import RiboSeqAlignment

class CleavageModel:

    def __init__(self, pl: np.array, pr:np.array, pu: float) -> None:
        self.pl = pl
        self.pr = pr
        self.pu = pu
        self.cds_lut = np.zeros((len(self.pl)+len(self.pr)+3, 3, 2), dtype=np.float64)
        for length in range(len(self.pl)+len(self.pr)+3):
            for frame in range(3):
                for oua in range(2):
                    self.cds_lut[length, frame, oua] = read_in_cds_likelihood(
                        pl = self.pl,
                        pr = self.pr,
                        pu = self.pu,
                        length = length,
                        frame = frame,
                        oua = oua,
                        region_start = 0,
                        region_end = 10**10,
                    )
        self.noise_lut = np.zeros((len(self.pl)+len(self.pr)+3, 2), dtype=np.float64)
        for length in range(len(self.pl)+len(self.pr)+3):
            for oua in range(2):
                self.noise_lut[length, oua] = read_in_noise_likelihood(
                    pl = self.pl,
                    pr = self.pr,
                    pu = self.pu,
                    length = length,
                    oua = oua,
                    region_start = 0,
                    region_end = 10**10,        
                )


    def pmf(
        self,
        length: int,
        oua: bool,
        frame: int=None,
        region_start: int=0,
        region_end: int=10**10,
        ) -> float:
        # region_start relative to read start
        # region_end relative to read start
        if length > len(self.pl) + len(self.pr) + 3 + int(oua):
            return 0

        if frame == None: # noise
            if region_start == 0 and region_end == 10**10:
                temp = self.noise_lut[length, int(oua)]
            else:
                temp = read_in_noise_likelihood(
                    self.pl, 
                    self.pr, 
                    self.pu, 
                    length, 
                    oua, 
                    region_start, 
                    region_end,
                    )
            return temp/3
        else: # CDS
            if region_start == 0 and region_end == 10**10:
                return self.cds_lut[length, frame, int(oua)]

            f0 = (-frame)%3
            f1 = (f0-1)%3

            if region_start and f0 != region_start%3:
                raise ValueError("region_start and frame are not compatible")

            if region_end < 10**10 and f0 != region_end%3:
                raise ValueError("region_end and frame are not compatible")

            return read_in_cds_likelihood(
            self.pl,
            self.pr,
            self.pu,
            length,
            frame,
            oua,
            region_start,
            region_end,
        )
        

@njit   
def read_in_cds_likelihood(
    pl: np.ndarray, 
    pr: np.ndarray, 
    pu: float, 
    length: int, 
    frame: int, 
    oua: bool, 
    region_start: int, 
    region_end: int
    ) -> float:
    
    f0 = (-frame)%3
    f1 = (f0-1)%3
    
    start_index = max(f1, region_start-1, length-len(pr)-3)
    if start_index%3 == f1%3:
        pass
    elif start_index%3 == (f1+1)%3:
        start_index += 2
    elif start_index%3 == (f1+2)%3:
        start_index += 1

    i = np.arange(
            start_index,
            min(len(pl), length-3, region_end-3),
            3)
    l = (pl[i]*pr[length-i-3-1]).sum()
    if oua:
        l = l * pu * 1/4
    else:
        l = l * pu * 3/4
        
        start_index = max(f0, region_start, length-len(pr)-3+1)
        if start_index%3 == f0%3:
            pass
        elif start_index%3 == (f0+1)%3:
            start_index += 2
        elif start_index%3 == (f0+2)%3:
            start_index += 1

            
        i = np.arange(
            start_index,
            min(len(pl),length-2, region_end-2),
            3)
        l += (pl[i]*pr[length-i-3]).sum() * (1-pu)

    return l

@njit
def read_in_noise_likelihood(
    pl: np.ndarray,
    pr: np.ndarray,
    pu: float,
    length: int,
    oua: bool,
    region_start: int,
    region_end: int
    ) -> float:

    i = np.arange(
        max(0, region_start-1, length-3-len(pr)), 
        min(len(pl), length-3, region_end-3)
        )
    l = (pl[i] * pr[length-i-3-1]).sum()

    if oua:
        l = l * pu * 1/4
    else:
        l = l * pu * 3/4

        i = np.arange(
            max(0, region_start, length-2-len(pr)), 
            min(len(pl), length-2, region_end-2)
            )
        l += (pl[i] * pr[length-i-3]).sum() * (1-pu)

    return l


class CleavageEstimator:
    
    def __init__(
            self,
            repeats: int=1_000, 
            maxiter: int=100,
            delta_cutoff: float=.001,
            seed: int=42
            ) -> None:

        self.table = np.zeros(shape=(100,3,2,1), dtype=np.int32)
        self.obs_min_len = 15
        self.obs_max_len = 40
        self.seed = seed
        self.repeats = repeats
        self.c = 0
        self.maxiter = maxiter
        self.delta_cutoff = delta_cutoff


    # TODO use this in pipelines instead of count_frame_starts
    def collect_data(self, 
                   reference_annotation: ReferenceAnnotation, 
                   alignments: HTSeq.BAM_Reader,
                   min_considered_length: int=15,
                   max_considered_length: int=40,
                   min_dist_to_start: int=30,
                   min_dist_to_end: int=30,
                   sufficient_counted_alns: int=100_000,
                   ) -> None:
        self.table = np.zeros(shape=(self.obs_max_len+10, 3, 2, 1), dtype=np.int32)
        self.dist_starts = np.zeros(shape=(200), dtype=np.int32)
        self.outside_cds = 0
        self.not_countable = 0
        self.counted_alns = 0
        for aln in alignments:
            aln = RiboSeqAlignment(aln)

            if not aln.unique:
                self.not_countable += 1
                continue
            #if aln.close_to_any_tis(reference_annotation):
            #    self.not_countable += 1
            #    continue
            if not min_considered_length <= aln.matching_length < max_considered_length:
                self.not_countable += 1
                continue
            transcript_candidates = reference_annotation.collect_coding_transcripts(aln.genomic_region)
            if len(transcript_candidates) == 0:
                self.outside_cds += 1
                continue
            frame = None
            dist_to_start = None

            # get frame
            for tr in transcript_candidates:
                try:
                    # why not do this on CDS right away?
                    #iv_on_tr = tr.exons.induce(aln.genomic_region)
                    iv_on_cds = tr.cds.induce(aln.genomic_region)
                except ValueError:
                    self.not_countable += 1
                    continue

                if iv_on_cds[0] > min_dist_to_start \
                and tr.coding_length - iv_on_cds[1] > min_dist_to_end:
                    new_frame = iv_on_cds[0]%3

                    if frame is None:
                        frame = new_frame
                    elif frame != new_frame:
                        self.not_countable += 1
                        break
                
            else:
                if not frame is None:
                    self.table[aln.matching_length, frame, int(aln.untemplated_addition), 0] += 1
                    self.counted_alns += 1
                    if self.counted_alns >= sufficient_counted_alns:
                        break


            # get dist_to_start
            for tr in transcript_candidates:
                try:
                    new_dist_to_start = tr.cds.induce(aln.genomic_region)[0]
                except ValueError:
                    new_dist_to_start = None
                
                if type(new_dist_to_start) == int:
                    if dist_to_start is None:
                        dist_to_start = new_dist_to_start
                    elif dist_to_start != new_dist_to_start:
                        break
            else:
                if (type(dist_to_start) == int) and (-100 < dist_to_start < 100):
                    self.dist_starts[dist_to_start+100] += 1


    def correct_table(self) -> None:
        temp_table = self.table.copy()
        (self.table[:,0,:,:], self.table[:,1,:,:], self.table[:,2,:,:]) = (temp_table[:,0,:,:],temp_table[:,2,:,:],temp_table[:,1,:,:])
  

    def run(self)->CleavageModel:

        self.best_ll, self.best_u, self.best_pl, self.best_pr = repeat(
            self.repeats,
            self.obs_max_len,
            self.obs_min_len,
            self.table,
            self.maxiter,
            self.c,
            self.delta_cutoff,
            self.seed,
        )
        shift = self.compute_shift()
        self.correct_max_pos(shift)
        return CleavageModel(
            self.best_pl,
            self.best_pr,
            self.best_u
            )


    def compute_shift(self, range_start:int=-25, range_end:int=-5) -> int:
        pl_rev = self.best_pl[::-1]
        pl_maxpos = pl_rev.argmax()

        overlayed = np.zeros(range_end-range_start)
        for i in range(range_start, range_end):
            overlayed[i-range_start] = sum(pl_rev * self.dist_starts[100+i-pl_maxpos:100+i-pl_maxpos+len(pl_rev)])
        
        overlayed_diff = overlayed[3:] - overlayed[:-3]
        return -(overlayed_diff.argmax() - pl_maxpos + range_start + len(pl_rev) + 2)
        

    def correct_max_pos(self, shift: int) -> None:
        pl = np.zeros(len(self.best_pl))
        for i in range(len(self.best_pl)):
            if i - shift >= 0 and i - shift < len(self.best_pl):
                pl[i] = self.best_pl[i-shift]

        pr = np.zeros(len(self.best_pr))
        for i in range(len(self.best_pr)):
            if i + shift >= 0 and i + shift < len(self.best_pl):
                pr[i] = self.best_pr[i+shift]

        pl = pl/pl.sum()
        pr = pr/pr.sum()
        
        self.best_pl = pl
        self.best_pr = pr


@njit
def compute_ll(
    table, 
    obs_min_len, 
    obs_max_len, 
    pl, 
    pr, 
    u, 
    c,
    ) -> float:
    ll = 0
    for length in range(obs_min_len,obs_max_len+1):
        for frame in range(3):
            frame1 = (frame-1)%3

            untemplated_addition = 1
            n = table[length,frame,untemplated_addition,c]

            if n > 0:
                i = np.arange(frame1,min(len(pl),length-3),3)
                p = (pl[i]*pr[length-i-3-1]).sum()
                ll += n*np.log(p)

            untemplated_addition = 0
            n = table[length,frame,untemplated_addition,c]

            if n > 0:
                i = np.arange(frame1,min(len(pl),length-3),3)
                p = (pl[i]*pr[length-i-3-1]).sum() * u
                i = np.arange(frame,min(len(pl),length-2),3)
                p += (pl[i]*pr[length-i-3]).sum() * (1-u)
                ll += n*np.log(p)
    return ll
    

@njit
def repeat(
    repeats,
    obs_max_len,
    obs_min_len,
    table,
    maxiter,
    c,
    delta_cutoff,
    seed=42,
):
    np.random.seed(seed)

    total = table[obs_min_len:obs_max_len+1,:,:,c].sum()
    
    best_ll = -np.inf

    for rep in range(max(repeats,1)):

        pl = np.random.rand(obs_max_len+1)
        pr = np.random.rand(obs_max_len+1)

        #maxpos = 12
        #pl[maxpos-1]*=4
        #pl[maxpos]*=10
        #pl[maxpos+1]*=4

        pl = pl/pl.sum()
        pr = pr/pr.sum()

        N = table[obs_min_len:obs_max_len+1,:,1,c].sum()
        u = N*4/3

        N += table[obs_min_len:obs_max_len+1,:,0,c].sum()
        u/=N

        better_ll = -np.inf

        for it in range(maxiter):
            eps = 1e-14

            ql0 = np.zeros(obs_max_len + 1)
            qr0 = np.zeros(obs_max_len + 1)
            ql1 = np.zeros(obs_max_len + 1)
            qr1 = np.zeros(obs_max_len + 1)
        
            qu = 0
        
            for length in range(obs_min_len, obs_max_len + 1):
                for frame in range(3):
                    untemplated_addition = 1
        
                    n = table[length, frame, untemplated_addition, c]
                    
                    frame1 = (frame - 1) % 3
                    left = np.arange(frame, length - 3, 3)
                    left1 = np.arange(frame1, length - 3, 3)
                    sum = eps
        
                    sum += (pl[left1] * pr[length - left1 - 3 - 1]).sum()
                    s = pl[left1] * pr[length - left1 - 3 - 1] / sum * n
                    ql1[left1 + 1] += s
                    qr1[length - left1 - 1 - 3] += s

        
                    qu += n

                    untemplated_addition = 0
        
                    n = table[length, frame, untemplated_addition, c]
        
                    sum0 = eps
                    sum1 = eps
        
                    prop = (u/(4-3*u))
        
                    sum1 += (pl[left1] * pr[length - left1 - 3 - 1]*prop).sum()
        
                    sum0 += (pl[left]*pr[length-left-3]*(1-prop)).sum()
                    sum = sum1 + sum0


                    s = pl[left1]*pr[length-left1-3-1]*prop/sum*n
                    ql1[left1]+=s
                    qr1[length-left1-1-3]+=s

        
                    s = pl[left]*pr[length-left-3]*(1-prop)/sum*n
                    ql0[left]+=s
                    qr0[length-left-3]+=s
        
                    qu+=sum1/sum*n

            old_pl = pl.copy()
            old_pr = pr.copy()
            for i in range(obs_max_len+1):
                pl[i] = ((ql1[i+1] if i+1<len(ql1) else 0) + ql0[i]) / total
                pr[i] = (qr1[i] + qr0[i]) / total

            N = table[obs_min_len:obs_max_len+1,:,:,c].sum()
            u=qu/N
            
            model_change = (np.absolute(old_pl - pl)).sum() + (np.absolute(old_pr - pr)).sum()
            if (model_change < delta_cutoff):
                better_ll = compute_ll(table, obs_min_len, obs_max_len, pl, pr, u, c)
                better_u = u
                break

        else:
            better_ll = compute_ll(table, obs_min_len, obs_max_len, pl, pr, u, c)
            better_u = u

        if better_ll > best_ll:
            best_ll = better_ll
            best_u = better_u
            best_pl = pl
            best_pr = pr

    return best_ll, best_u, best_pl, best_pr
