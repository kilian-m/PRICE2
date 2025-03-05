import HTSeq
import pandas as pd


from price2.genomic_region import GenomicRegion
from price2.genomic_features import Transcript, ReadGeneratingRegion
from price2.reference_annotation import ReferenceAnnotation


class RiboSeqAlignment:

    # alignment: HTSeq._HTSeq.SAM_Alignment # depracated
    # matching_length: int # depracated use matching_length() instead
    # unique: bool # depraceted use unique() instead
    genomic_region: GenomicRegion
    untemplated_addition: (
        bool  # observed untemplated addition is not part of the matching region
    )
    mapping_positions: int  # number of mapping positions in the genome

    def __init__(self, data):
        if isinstance(data, pd.DataFrame):
            df = data

            if df.iloc[0]["unique"]:
                self.mapping_positions = 1
            else:
                self.mapping_positions = 2  # TODO: change this

            self.untemplated_addition = df.iloc[0]["untemplated_addition"]
            intervals = []
            for _, row in df.iterrows():
                intervals.append(
                    HTSeq.GenomicInterval(
                        row["chrom"], row["start"], row["end"], row["strand"]
                    )
                )
            self.genomic_region = GenomicRegion(intervals)
            self.read_count = df.iloc[0]["count"]

        elif isinstance(data, HTSeq._HTSeq.SAM_Alignment):
            sa = data
            # self.alignment = sa
            # self.unique = self.alignment.optional_field('NH') == 1
            self.mapping_positions = sa.optional_field("NH")
            # self.untemplated_addition = self.has_untemplated_addition()

            if sa.iv.strand == "+":
                self.untemplated_addition = bool(
                    (sa.cigar[0].type == "S") and (sa.cigar[0].size == 1)
                )
            elif sa.iv.strand == "-":
                self.untemplated_addition = bool(
                    (sa.cigar[-1].type == "S") and (sa.cigar[-1].size == 1)
                )

            # self.matching_length = sum([x.size for x in self.alignment.cigar if x.type=='M'])

            intervals = []
            for cigar_op in sa.cigar:
                if cigar_op.type == "M":
                    intervals.append(cigar_op.ref_iv)

            if sa.iv.strand == "-":
                intervals = intervals[::-1]

            self.genomic_region = GenomicRegion(intervals)

        elif isinstance(data, dict):
            self.genomic_region = data["genomic_region"]
            self.untemplated_addition = data["untemplated_addition"]
            self.mapping_positions = data["mapping_positions"]

    def __repr__(self) -> str:
        return f"RiboSeqAlignment({self.genomic_region})"

    def __len__(self) -> int:
        return len(self.genomic_region)

    def __hash__(self) -> int:
        return hash(self.genomic_region) + self.untemplated_addition

    def __eq__(self, other: "RiboSeqAlignment") -> bool:
        if (
            self.genomic_region == other.genomic_region
            and self.untemplated_addition == other.untemplated_addition
            and self.mapping_positions == other.mapping_positions
        ):
            return True
        return False

    def unique(self) -> bool:
        return self.mapping_positions == 1

    def get_first_pos_in_genomic_coords(self) -> int:
        if self.genomic_region.strand == "+":
            return self.genomic_region.intervals[0].start
        else:
            return self.genomic_region.intervals[-1].start

    def get_last_pos_in_genomic_coords(self) -> int:
        if self.genomic_region.strand == "+":
            return self.genomic_region.intervals[-1].end
        else:
            return self.genomic_region.intervals[0].end

    # for cleavage model estimation
    def is_countable(
        self,
        reference_annotation: ReferenceAnnotation,
        max_considered_len: int = 50,
        min_considered_len: int = 15,
        min_dist_to_start: int = 30,
        min_dist_to_end: int = 30,
    ) -> bool:
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

    def get_frames(self, reference_annotation) -> tuple:
        transcript_candidates = reference_annotation.collect_coding_transcripts(
            self.genomic_region
        )
        if len(transcript_candidates) == 0:
            return None
        frames = [0, 0, 0]
        for transcript in transcript_candidates:
            frame = transcript.cds.induce(self.genomic_region)[0] % 3
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
