import HTSeq
from HTSeq import GenomicInterval


class GenomicRegion:
    strand: str
    chrom: str
    intervals: list[GenomicInterval] # in order of the strand (like in GTF file)

    def __init__(self, intervals: list[GenomicInterval], chrom: str=None, strand: str=None) -> None:
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
    

    def __repr__(self):
        return str(self)


    def __len__(self):
        return sum([x.end-x.start for x in self.intervals])
    
    
    def add_interval(self, interval: HTSeq.GenomicInterval) -> None:
        self.intervals.append(interval)

        

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


