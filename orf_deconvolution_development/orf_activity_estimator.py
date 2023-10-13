from price2.cleavage_model import CleavageModel
import HTSeq
import pandas as pd
import numpy as np
from scipy.stats import poisson


class AssociationClass():
    def __init__(self)-> None:
        self.read_count = 0
        
class CompatibilityClass():
    def __init__(self) -> None:
        self.length = 0
        self.read_count = 0

class ORFActivityEstimator():

    def __init__(
            self, 
            cm: CleavageModel, 
            reads_path: str='',
            gtf_path: str='data/test.gtf',
            overlap_likelihood_ratio_threshold: float=.99,
            delta_cutoff: float=1,
            maxiter: int=100,
            ) -> None:
        self.cm = cm
        self.overlap_likelihood_ratio_threshold = overlap_likelihood_ratio_threshold
        self.delta_cutoff = delta_cutoff
        self.maxiter = maxiter
        self.region_activities = {}
        self.lls = []
        self.activity_changes = []

        self.regions_it = HTSeq.GenomicArrayOfSets('auto', stranded=True)
        self.regions = {}
        self.active_region_ids = set()
        for region in HTSeq.GFF_Reader(gtf_path):
            self.regions_it[region.iv] += region
            self.regions[region.attr['region_id']] = region
            self.active_region_ids.add(region.attr['region_id'])
            
        self.bam_file = HTSeq.BAM_Reader(reads_path)

        self.compatibility_classes = {}
        for step in self.regions_it.steps():
            step_regions = set()
            for step_region in step[1]:
                step_regions.add(step_region.name)
            step_regions = frozenset(step_regions)
            if step_regions not in self.compatibility_classes:
                self.compatibility_classes[step_regions] = CompatibilityClass()
            self.compatibility_classes[step_regions].length += step[0].length


    def fill_association_classes(self) -> None:

        # assign reads to association classes
        self.association_classes = {}
        for read_al in self.bam_file:
            # compute association class
            length = read_al.iv.length
            oua = read_al.optional_field('MD').startswith('0')
            region_frame = set()

            # get all overlapping regions
            if len(read_al.cigar) > 2:
                continue # TODO: consider spliced reads
            candidate_regions = set()
            for step in self.regions_it[read_al.iv].steps():
                for region in step[1]:
                    candidate_regions.add(region)
            
            if len(candidate_regions) == 1:
                region = candidate_regions.pop()                        
                if region.type == 'ORF':            
                    frame = (read_al.iv.start - region.iv.start)%3          
                else:            
                    frame = None            
                region_frame.add((region.name, frame))            

            else:
                for region in candidate_regions:
                    if region.type == 'ORF':
                        frame = (read_al.iv.start - region.iv.start)%3
                    else:
                        frame = None
                    complete_likelihood = self.cm.pmf(
                        length, 
                        oua,
                        frame, 
                        )
                    if region.iv.start > read_al.iv.start or region.iv.end < read_al.iv.end:
                        overlap_likelihood = self.cm.pmf(
                            read_al.iv.length, 
                            read_al.optional_field('MD').startswith('0'), 
                            frame, 
                            region_start = region.iv.start - read_al.iv.start,
                            region_end = region.iv.end - read_al.iv.start
                            )
                        if overlap_likelihood/complete_likelihood >= self.overlap_likelihood_ratio_threshold:
                            region_frame.add((region.name, frame))
                    else:
                        region_frame.add((region.name, frame))
            
            region_frame = frozenset(region_frame)
            step_regions = frozenset([rf[0] for rf in region_frame])
            try:
                if not (region_frame, length, oua) in self.association_classes:
                    self.association_classes[(region_frame, length, oua)] = AssociationClass()
                self.association_classes[(region_frame, length, oua)].read_count += 1 
                self.compatibility_classes[step_regions].read_count += 1
            except KeyError:
                # this exception can happen because reads on the borders of ORFs can be assigned to region combinations that are not in the compatibility classes
                pass
            
        self.total_read_count = sum([ac.read_count for ac in self.association_classes.values()])
       
        #df = pd.DataFrame.from_dict(self.association_classes, orient='index').rename(columns={0:'count'})

        self.rrc = {}
        for region_id in self.regions:
            self.regions[region_id].activity = 1/len(self.regions)
            self.region_activities[region_id] = [self.regions[region_id].activity]
            self.rrc[region_id] = []

        
    def run_orf_deconvolution(self, iterations: int=0) -> None:
        if iterations == 0:
            iterations = self.maxiter
        for i in range(iterations):
            for region_id in self.active_region_ids:
                self.regions[region_id].read_count = 0

            # E-step (assign reads to regions)
            for (region_frame, length, oua),ac in self.association_classes.items():
                count = ac.read_count
                likelihoods = {}

                #compute the likelihood for a read in this association class to be generated by each overlapping region
                for region_id, frame in region_frame:
                    likelihoods[region_id] = self.cm.pmf(length, oua, frame) * self.regions[region_id].activity

                # normalize likelihoods and assign read counts to regions
                likelihood_sum = sum(likelihoods.values())
                p = {k:likelihoods[k]/likelihood_sum for k in likelihoods}
                for region_id, frame in region_frame:
                    self.regions[region_id].read_count += count * p[region_id]

            # M-step (update region activities)
            for region_id in self.active_region_ids:
                self.regions[region_id].activity = self.regions[region_id].read_count/self.regions[region_id].iv.length
                self.regions[region_id].read_fraction = self.regions[region_id].read_count/self.total_read_count # do i need this?

            self.lls.append(self.incomplete_data_likelihood())

            for region_id in self.regions:
                self.region_activities[region_id].append(self.regions[region_id].activity)
            for region_id in self.active_region_ids:
                self.rrc[region_id].append(self.regions[region_id].read_count)

            # sum differences in region_activities
            if i>1:
                activity_change = 0
                for region_id in self.regions:
                    activity_change += abs(self.region_activities[region_id][-1] - self.region_activities[region_id][-2])
                self.activity_changes.append(activity_change)
                # break if activity changes less than delta_cutoff
                if activity_change < self.delta_cutoff:
                    #break
                    if not self.regularize(likelihood_cutoff=200):#3
                        print('break at iteration ', i)
                        break
                    

    # the incomplete data likelihood as defined in the lecture
    # product of likelihoods of all compatibility classes (expected read count vs actual read count assuming poisson distribution)
    def incomplete_data_likelihood(self) -> float:
        ll = 0
        for cc_regions, cc in self.compatibility_classes.items():
            activity = sum([self.regions[region_id].activity for region_id in cc_regions])
            expected_reads_in_cc = cc.length * activity
            actual_reads_in_cc = cc.read_count
            ll += poisson.logpmf(actual_reads_in_cc, expected_reads_in_cc)
        return ll

    # product of likelihoods of all association classes (expected read count vs actual read count assuming poisson distribution)
    #def incomplete_data_likelihood_3(self) -> float:
    #    ll = 0
    #    for (region_frames, length, oua), ac in self.association_classes.items():
    #        l = 0
    #        try:
    #            ac_length = self.compatibility_classes[frozenset(x[0] for x in region_frames)].length
    #            expected_reads = 0
    #            for region_id, frame in region_frames:
    #                expected_reads += self.regions[region_id].activity * self.cm.pmf(length, oua, frame) * ac_length
    #            actual_reads = ac.read_count
    #            ll += poisson.logpmf(actual_reads, expected_reads)
    #        except KeyError:
    #            pass
    #    return ll

        
    def deactivate_region(self, rem_reg_id: str, inplace: bool=True):
        # active regions
        active_region_ids = set([region_id for region_id in self.active_region_ids if region_id != rem_reg_id])

        ## compatibility classes
        #compatibility_classes = {}
        #for cc_regions, cc in self.compatibility_classes.items():
        #    if not rem_reg_id in cc_regions:
        #        compatibility_classes[cc_regions] = cc
        #    else:
        #        cc_regions = frozenset([region_id for region_id in cc_regions if region_id != rem_reg_id])
        #        if not cc_regions in compatibility_classes:
        #            compatibility_classes[cc_regions] = Compatibility_class()
        #        compatibility_classes[cc_regions].read_count += cc.read_count
        #        compatibility_classes[cc_regions].length += cc.length
#
        ## association classes
        #association_classes = {}
        #for (region_frame, length, oua), ac in self.association_classes.items():
        #    if not rem_reg_id in [x[0] for x in region_frame]:
        #        if not (region_frame, length, oua) in association_classes:
        #            association_classes[(region_frame, length, oua)] = ac
        #        else:
        #            association_classes[(region_frame, length, oua)].read_count += ac.read_count
        #    else:
        #        region_frame = frozenset([x for x in region_frame if x[0] != rem_reg_id])
        #        if not (region_frame, length, oua) in association_classes:
        #            association_classes[(region_frame, length, oua)] = Association_class()
        #        association_classes[(region_frame, length, oua)].read_count += ac.read_count


        if inplace:
            self.active_region_ids = active_region_ids
            #self.compatibility_classes = compatibility_classes
            #self.association_classes = association_classes
            self.regions[rem_reg_id].activity = 0


    #def regularize_1(self, activity_cutoff: float=0.1) -> bool:
    #    success = False
    #    for region_id, region in self.regions.items():
    #        if region_id in self.active_region_ids and region.type == 'ORF' and region.activity < activity_cutoff:
    #            print(f'remove region {region_id}')
    #            success = True
    #            self.deactivate_region(region_id, inplace=True)
    #    return success
        

    def regularize(self, likelihood_cutoff: float=5) -> bool:
        success = False
        ll = self.incomplete_data_likelihood()
        regularized_regions = []
        for region_id in self.active_region_ids:
            self.regions[region_id].original_activity = self.regions[region_id].activity
        for rem_region in self.active_region_ids:
            # experimentally remove region, reassign reads and compute new likelihood
            for region_id in self.active_region_ids:
                self.regions[region_id].read_count = 0
                
            # E-step
            con = False
            for (region_frame, length, oua), ac in self.association_classes.items():
                count = ac.read_count
                likelihoods = {}

                for region_id, frame in region_frame:
                    if region_id == rem_region:
                        likelihoods[region_id] = 0
                    else:
                        likelihoods[region_id] = self.cm.pmf(length, oua, frame) * self.regions[region_id].activity

                likelihood_sum = sum(likelihoods.values())
                if likelihood_sum == 0:
                    con = True
                    break
                p = {k:likelihoods[k]/likelihood_sum for k in likelihoods}
                for region_id, frame in region_frame:
                    self.regions[region_id].read_count += count * p[region_id]
            
            if con:
                continue

            # M-step
            for region_id in self.active_region_ids:
                self.regions[region_id].activity = self.regions[region_id].read_count / self.regions[region_id].iv.length

            #print(f'{rem_region}\t{self.incomplete_data_likelihood()}\t{ll}')
            if self.incomplete_data_likelihood() + likelihood_cutoff > ll:
                success = True
                regularized_regions.append(rem_region)

            for region_id in self.active_region_ids:
                self.regions[region_id].activity = self.regions[region_id].original_activity
            
        for region_id in regularized_regions:
            #print(f'remove region {region_id}')
            self.deactivate_region(region_id, inplace=True)

        return success



        


