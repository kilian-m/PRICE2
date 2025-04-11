from price2.reference_annotation import ReferenceAnnotation
from price2.cleavage_model import CleavageModel, read_in_cds_likelihood
from price2.ribo_seq_alignment import RiboSeqAlignment

from enum import Enum

import HTSeq
import numpy as np


class CoveragePosition(Enum):
    start = 0
    middle = 1
    stop = 2


class CoverageModel:
    """
    class that models the higher ribosome footprint coverage at
    ORF start codons and codons one position upstream of stop codons
    """

    def __init__(
        self,
        ra: ReferenceAnnotation,
        sample_bam_path: str,
        cm: CleavageModel,
    ) -> None:

        p_site_2_start = np.zeros(250)

        for aln in HTSeq.BAM_Reader(sample_bam_path):
            aln = RiboSeqAlignment(aln)
            transcript_candidates = ra.collect_coding_transcripts(aln.genomic_region)
            if not transcript_candidates:
                continue
            s = set()
            for tr in transcript_candidates:
                try:
                    iv_on_cds = tr.cds.induce(aln.genomic_region)
                    transcript = tr
                except ValueError:
                    continue
                s.add(iv_on_cds)
            if len(s) != 1:
                continue

            iv_on_cds = s.pop()
            if not (-30 < iv_on_cds[0] < 200):
                continue

            if iv_on_cds[0] < len(transcript.cds) < iv_on_cds[1]:
                continue

            frame = iv_on_cds[0] % 3

            l = []
            for i in range((iv_on_cds[1] - iv_on_cds[0]) // 3):
                start = frame % 3 + 3 * i
                l.append(
                    read_in_cds_likelihood(
                        cm.pl,
                        cm.pr,
                        cm.pu,
                        len(aln),
                        frame,
                        aln.untemplated_addition,
                        start,
                        start + 3,
                    )
                )
            if max(l) < 0.01:
                continue
            l = np.array(l)
            l = l / sum(l)
            if max(l) < 0.8:
                continue
            p_site = iv_on_cds[0] + np.argmax(l) * 3
            # position 30 in the array is position 0 in the CDS
            p_site_2_start[p_site + 30] += 1

        self.start_factor = p_site_2_start[30] / p_site_2_start[33:230:3].mean()

        p_site_2_stop = np.zeros(250)
        for aln in HTSeq.BAM_Reader(sample_bam_path):
            aln = RiboSeqAlignment(aln)
            transcript_candidates = ra.collect_coding_transcripts(aln.genomic_region)
            if not transcript_candidates:
                continue
            s = set()
            for tr in transcript_candidates:
                try:
                    iv_on_cds = tr.cds.induce(aln.genomic_region)
                    transcript = tr
                except ValueError:
                    continue
                s.add(iv_on_cds)
            if len(s) != 1:
                continue

            iv_on_cds = s.pop()
            if not (-200 < (iv_on_cds[1] - len(transcript.cds)) < 30):
                continue

            if iv_on_cds[0] < 0 < iv_on_cds[1]:
                continue

            frame = iv_on_cds[0] % 3

            l = []
            for i in range((iv_on_cds[1] - iv_on_cds[0]) // 3):
                start = frame % 3 + 3 * i
                l.append(
                    read_in_cds_likelihood(
                        cm.pl,
                        cm.pr,
                        cm.pu,
                        len(aln),
                        frame,
                        aln.untemplated_addition,
                        start,
                        start + 3,
                    )
                )
            if max(l) < 0.01:
                continue
            l = np.array(l)
            l = l / sum(l)
            if max(l) < 0.8:
                continue
            p_site = iv_on_cds[0] + np.argmax(l) * 3
            p_site_2_stop[p_site - len(transcript.cds) + 220] += 1

        self.stop_factor = (
            p_site_2_stop[217] / p_site_2_stop[217 - 198 : 217 : 3].mean()
        )
