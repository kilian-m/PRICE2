import HTSeq


class GenomicRegion:
    strand: str
    chrom: str
    intervals: list[HTSeq.GenomicInterval] # in order of the strand (like in GTF file)

    def __init__(self, intervals: list, chrom: str=None, strand: str=None) -> None:
        if chrom: 
            self.chrom = chrom
        else:
            self.chrom = intervals[0].chrom
        if strand:
            self.strand = strand
        else:
            self.strand = intervals[0].strand
        self.intervals = intervals


    def __eq__(self, other):
        if not isinstance(other, GenomicRegion):
            return False
        return (self.chrom == other.chrom\
                 and self.intervals == other.intervals\
                 and self.strand == other.strand)
    

    def __hash__(self):
        # hash changes when intervals are added!!!
        return hash((self.strand, self.chrom, tuple(self.intervals)))
    

    def __str__(self):
        s = f'{self.chrom}:{self.strand}'
        for i in self.intervals:
            s += f':[{i.start},{i.end})'
        return s
    
    
    def add_interval(self, interval: HTSeq.GenomicInterval) -> None:
        self.intervals.append(interval)



    # self is the reference region (e.g. the transcript)
    # other is the query region (e.g. the read)
    # return the interval of the query relative to the reference
    # if the query is not compatible with the reference, return False
    # not compatible means that the query spans intronic regions of the reference or skips exons of the reference
    # if the query overlaps the start or end of the reference the coordinates are returned
#    def induce(self, other: 'GenomicRegion') -> tuple[int, int]:
#        if self.chrom != other.chrom or self.strand != other.strand:
#            return False
#        ref_intervals = self.intervals
#        query_intervals = other.intervals
#
#        if self.strand == '-':
#            exons_end = ref_intervals[0].end
#            ref_intervals = [HTSeq.GenomicInterval(ri.chrom, exons_end-ri.end, exons_end-ri.start, ri.strand) for ri in ref_intervals]
#            query_intervals = [HTSeq.GenomicInterval(qi.chrom, exons_end-qi.end, exons_end-qi.start, qi.strand) for qi in query_intervals]
#
#        # !!! change this
#        #if ref_intervals[0].start > query_intervals[0].start\
#        #    or ref_intervals[-1].end < query_intervals[-1].end:
#        #    raise ValueError('Query region is not compatible with reference region.')
#
#        ref_index = 0
#        query_index = 0
#        current_length = 0
#
#        # increase ref_index until we reach query
#        while ref_intervals[ref_index].end < query_intervals[query_index].start:
#            current_length += ref_intervals[ref_index].length 
#            ref_index += 1
#            #TODO?
#            if ref_index == len(ref_intervals):
#                raise ValueError('Query region is not compatible with reference region.')
#
#        # case the query starts in the middle of an intron
#        if ref_intervals[ref_index].start > query_intervals[query_index].start\
#            and not ref_index == 0:
#            raise ValueError('Query region is not compatible with reference region.')
#        induced_start = current_length + query_intervals[query_index].start - ref_intervals[ref_index].start
#
#        # case query starts in exon and continues
#        if query_index < len(query_intervals)-1\
#            and ref_intervals[ref_index].end == query_intervals[query_index].end:
#            current_length += ref_intervals[ref_index].length
#            ref_index += 1
#            query_index += 1
#
#        # while reference and index span whole exons
#        while query_index < len(query_intervals)-1\
#            and ref_intervals[ref_index].end == query_intervals[query_index].end\
#            and ref_intervals[ref_index].start == query_intervals[query_index].start:
#            current_length += ref_intervals[ref_index].length
#            ref_index += 1
#            query_index += 1
#
#        # case query continues somewhere else than the reference
#        if ref_intervals[ref_index].start != query_intervals[query_index].start\
#            and query_index > 0:
#            raise ValueError('Query region is not compatible with reference region.')
#        
#        # case query ends somewhere else than the reference (and is not the last part)
#        if not ref_index == len(ref_intervals) - 1\
#            and ref_intervals[ref_index].end < query_intervals[query_index].end:
#            raise ValueError('Query region is not compatible with reference region.')
#            
#        induced_end = current_length + query_intervals[query_index].end - ref_intervals[ref_index].start
#        return (induced_start, induced_end)
    

#    def induce_old(self, other: 'GenomicRegion') -> tuple[int, int]:
#        if self.chrom != other.chrom or self.strand != other.strand:
#            raise ValueError('Query region is not compatible with reference region.')
#        ref_intervals = self.intervals
#        query_intervals = other.intervals
#
#        if self.strand == '-':
#            exons_end = ref_intervals[0].end
#            ref_intervals = [HTSeq.GenomicInterval(ri.chrom, exons_end-ri.end, exons_end-ri.start, ri.strand) for ri in ref_intervals]
#            query_intervals = [HTSeq.GenomicInterval(qi.chrom, exons_end-qi.end, exons_end-qi.start, qi.strand) for qi in query_intervals]
#
#        #c1
#        if query_intervals[0].start < query_intervals[-1].end < ref_intervals[0].start < ref_intervals[-1].end:
#            if len(query_intervals)>1:
#                raise ValueError('Query region is not compatible with reference region.')
#            else:
#                return (query_intervals[0].start - ref_intervals[0].start,
#                        query_intervals[0].end - ref_intervals[0].start)
#        
#        #c2
#        elif ref_intervals[0].start < ref_intervals[-1].end < query_intervals[0].start < query_intervals[-1].end:
#            if len(query_intervals)>1:
#                raise ValueError('Query region is not compatible with reference region.')
#            else:
#                ref_len = sum([ri.length for ri in ref_intervals])
#                return (query_intervals[0].start - ref_intervals[-1].end + ref_len,
#                        query_intervals[0].end - ref_intervals[-1].end + ref_len)
#
#        #c3
#        elif ref_intervals[0].start <= query_intervals[0].start <= query_intervals[-1].end <= ref_intervals[-1].end:
#            cut_ref_intervals = cut_intervals(ref_intervals,
#                    query_intervals[0].start, query_intervals[-1].end)
#            cut_query_intervals = cut_intervals(query_intervals,
#                    query_intervals[0].start, query_intervals[-1].end)
#            if cut_ref_intervals == cut_query_intervals:
#                return (interval_sum(ref_intervals, query_intervals[0].start),
#                        interval_sum(ref_intervals, query_intervals[-1].end))
#            else:
#                raise ValueError('Query region is not compatible with reference region.')
#            
#        #c4
#        elif query_intervals[0].start < ref_intervals[0].start < query_intervals[-1].end < ref_intervals[-1].end:
#            cut_ref_intervals = cut_intervals(ref_intervals,
#                    ref_intervals[0].start, query_intervals[-1].end)
#            cut_query_intervals = cut_intervals(query_intervals,
#                    ref_intervals[0].start, query_intervals[-1].end)
#            out_intervals = cut_intervals(query_intervals,
#                    query_intervals[0].start, ref_intervals[0].start)
#            if cut_ref_intervals == cut_query_intervals\
#                and len(out_intervals) == 1:
#                return (query_intervals[0].start - ref_intervals[0].start,
#                        interval_sum(ref_intervals, query_intervals[-1].end))
#            else:
#                raise ValueError('Query region is not compatible with reference region.')
#
#        #c5
#        elif ref_intervals[0].start < query_intervals[0].start < ref_intervals[-1].end < query_intervals[-1].end:
#            cut_ref_intervals = cut_intervals(ref_intervals,
#                    query_intervals[0].start, ref_intervals[-1].end)
#            cut_query_intervals = cut_intervals(query_intervals,
#                    query_intervals[0].start, ref_intervals[-1].end)
#            out_intervals = cut_intervals(query_intervals,
#                    ref_intervals[-1].end, query_intervals[-1].end)
#            if cut_ref_intervals == cut_query_intervals\
#                and len(out_intervals) == 1:
#                out_ref_intervals = cut_intervals(ref_intervals,
#                    ref_intervals[0].start, query_intervals[0].start)
#                s = sum([x[1]-x[0] for x in out_ref_intervals])
#                return (s, s + sum([x.end-x.start for x in query_intervals]))
#            else:
#                raise ValueError('Query region is not compatible with reference region.')
#
#        #c6
#        elif query_intervals[0].start < ref_intervals[0].start < ref_intervals[-1].end < query_intervals[-1].end:
#            cut_ref_intervals = cut_intervals(ref_intervals,
#                    ref_intervals[0].start, ref_intervals[-1].end)
#            cut_query_intervals = cut_intervals(query_intervals,
#                    ref_intervals[0].start, ref_intervals[-1].end)
#            out_intervals_start = cut_intervals(query_intervals,
#                    query_intervals[0].start, ref_intervals[0].start)
#            out_intervals_end = cut_intervals(query_intervals,
#                    ref_intervals[-1].end, query_intervals[-1].end)
#            if cut_ref_intervals == cut_query_intervals\
#                and len(out_intervals_start) == 1\
#                and len(out_intervals_end) == 1:
#                s = sum([x[1]-x[0] for x in out_intervals_start])
#                return (query_intervals[0].start - ref_intervals[0].start,
#                        s + query_intervals[-1].end - ref_intervals[-1].end)
#            else:
#                raise ValueError('Query region is not compatible with reference region.')
        

    def induce(self, other: 'GenomicRegion') -> tuple[int, int]:
        if self.chrom != other.chrom or self.strand != other.strand:
            raise ValueError('Query region is not compatible with reference region.')
        ref_intervals = self.intervals
        query_intervals = other.intervals

        if self.strand == '-':
            exons_end = ref_intervals[0].end
            ref_intervals = [HTSeq.GenomicInterval(ri.chrom, exons_end-ri.end, exons_end-ri.start, ri.strand) for ri in ref_intervals]
            query_intervals = [HTSeq.GenomicInterval(qi.chrom, exons_end-qi.end, exons_end-qi.start, qi.strand) for qi in query_intervals]

        # overlapping part
        s = max(ref_intervals[0].start, query_intervals[0].start)
        e = min(ref_intervals[-1].end, query_intervals[-1].end)
        cut_query_intervals = cut_intervals(query_intervals, s, e)
        cut_ref_intervals = cut_intervals(ref_intervals, s, e)
        if cut_query_intervals != cut_ref_intervals:
            raise ValueError('Query region is not compatible with reference region.')


        # upstream part
        upstream_query = cut_intervals(query_intervals, query_intervals[0].start, s)
        if len(upstream_query) > 1:
            raise ValueError('Query region is not compatible with reference region.')

        # downstream part
        downstream_query = cut_intervals(query_intervals, e, query_intervals[-1].end)
        if len(downstream_query) > 1:
            raise ValueError('Query region is not compatible with reference region.')
        
        upstream_ref = cut_intervals(ref_intervals, ref_intervals[0].start, s)

        if len(upstream_query) == 1:
            start = query_intervals[0].start - ref_intervals[0].start
        elif ref_intervals[-1].end <= query_intervals[0].start:
            start = sum(x.end-x.start for x in ref_intervals)\
                    + query_intervals[0].start - ref_intervals[-1].end
        else:
            start = sum([x[1]-x[0] for x in upstream_ref])
        end = start + sum([x.end-x.start for x in query_intervals])
        return (start, end)
            
        
        




    def map(self, transcript_interval: tuple[int, int]) -> 'GenomicRegion':
        transcript_start, transcript_end = transcript_interval
        if self.strand == '+':
            s = 0 # exon length from transcript start
            interval_index = 0
            region_starts = []
            region_ends = []
            while interval_index < len(self.intervals) - 1\
                and s + self.intervals[interval_index].length < transcript_start:
                s += self.intervals[interval_index].length
                interval_index += 1
            region_starts.append(self.intervals[interval_index].start + transcript_start - s)

            while interval_index < len(self.intervals) - 1\
                and s + self.intervals[interval_index].length < transcript_end:
                region_ends.append(self.intervals[interval_index].end)
                s += self.intervals[interval_index].length
                interval_index += 1
                region_starts.append(self.intervals[interval_index].start)
            region_ends.append(self.intervals[interval_index].start + transcript_end - s)

            ivs = zip(region_starts, region_ends)
            
        elif self.strand == '-':
            s = 0
            interval_index = 0
            region_length = transcript_end - transcript_start
            region_starts = []
            region_ends = []

            while interval_index < len(self.intervals) - 1\
                and s + self.intervals[interval_index].length < transcript_start:
                s += self.intervals[interval_index].length
                interval_index += 1
            region_start = self.intervals[interval_index].end - (transcript_start - s)
            region_starts.append(region_start)

            while interval_index < len(self.intervals) - 1\
                and s + self.intervals[interval_index].length < transcript_end:
                region_ends.append(self.intervals[interval_index].start)
                s += self.intervals[interval_index].length
                interval_index += 1
                region_starts.append(self.intervals[interval_index].end)
            region_ends.append(self.intervals[interval_index].end - (transcript_end - s))

            ivs = zip(region_ends, region_starts)

        region_intervals = [HTSeq.GenomicInterval(self.chrom, rs, re, self.strand) for rs, re in ivs if not rs == re]
        return GenomicRegion(intervals=region_intervals, strand=self.strand, chrom=self.chrom)


    def get_sequence(self, genome: dict[str:HTSeq._HTSeq.Sequence]) -> str:
        sequence = ''
        if self.strand == '+':
            for interval in self.intervals:
                sequence += str(genome[self.chrom][interval.start:interval.end])
        elif self.strand == '-':
            for interval in self.intervals:
                sequence += str(genome[self.chrom][interval.start:interval.end].get_reverse_complement())
        return sequence

def cut_intervals(interval_list, start, end)->list:
    l = []
    for iv in interval_list:
        if iv.end <= start:
            pass
        elif end <= iv.start:
            pass
        elif start <= iv.start <= iv.end <= end:
            l.append((iv.start, iv.end))
        elif iv.start < start < iv.end <= end:
            l.append((start, iv.end))
        elif start <= iv.start < end < iv.end:
            l.append((iv.start, end))
        elif iv.start < start < end < iv.end:
            l.append((start, end))
    return l


#def interval_sum(interval_list, end)->int:
#    s = 0
#    for iv in interval_list:
#        if iv.end < end:
#            s += iv.length
#        elif iv.start < end:
#            s += end - iv.start
#            break
#        else:
#            break
#    return s