"""Generating the ORF candidates of a locus.

Every transcript of a locus contributes NOISE regions (its untranslated
parts, or the whole transcript when it has no annotated CDS) and one ORF
candidate per start/stop codon pair in its spliced sequence.  Candidates
with the same genomic footprint are merged, keeping the copy with the
longest flanking context, and each ORF is classified relative to the
annotated CDSs it overlaps.
"""

from __future__ import annotations

import HTSeq
import numpy as np
from pyfaidx import Fasta

from price2 import database
from price2.config import Config
from price2.genomic_features import ReadGeneratingRegion, Transcript
from price2.locus import Locus

# Transcript biotypes that count as long non-coding RNA.  Ensembl (>=97)
# and GENCODE (>=v31) use a single "lncRNA" biotype; earlier releases split
# the same class across the sub-biotypes listed below, so both vocabularies
# are accepted.  Note that Ensembl <=91 spells the antisense biotype
# "antisense_RNA" and 92..96 spell it "antisense".  "processed_transcript"
# and "retained_intron" are deliberately excluded: in the legacy vocabulary
# they are also used for non-coding transcripts of protein-coding genes.
LNCRNA_BIOTYPES: frozenset[str] = frozenset(
    {
        "lncRNA",
        "lincRNA",
        "antisense",
        "antisense_RNA",
        "sense_intronic",
        "sense_overlapping",
        "macro_lncRNA",
        "bidirectional_promoter_lncRNA",
        "3prime_overlapping_ncRNA",
    }
)

# Priority-ordered levels for ORF type assignment.  Within each level,
# having more than one matching label across compatible transcripts yields
# "other ORF"; the first level with exactly one match wins.
ORF_TYPES_LEVELS: list[set[str]] = [
    {"cORF"},
    {"N terminal extended cORF", "N terminal truncated cORF"},
    {
        "+1 uoORF",
        "+2 uoORF",
        "+1 doORF",
        "+2 doORF",
        "+1 iORF",
        "+2 iORF",
    },
    {"uORF", "dORF"},
    {"pcRNA-ORF", "lncRNA-ORF", "varRNA-ORF"},
]

def get_orf_type(
    orf: ReadGeneratingRegion,
    transcripts: set[Transcript],
) -> str:
    """Classify an ORF relative to the annotated CDSs of compatible transcripts.

    For each transcript in *transcripts* that contains the ORF's genomic
    footprint, the relationship between the ORF interval and the annotated
    CDS (in spliced transcript coordinates) is determined.  When an ORF is
    compatible with multiple transcripts the assignments are reconciled
    using :data:`ORF_TYPES_LEVELS`: the highest-priority level that has
    exactly one matching label is used; conflicting labels at the same
    level yield ``"other ORF"``.

    Parameters
    ----------
    orf : ReadGeneratingRegion
        An ORF-type RGR whose type should be classified.
    transcripts : set[Transcript]
        All transcripts belonging to the locus.

    Returns
    -------
    str
        ORF type label, e.g. ``'cORF'``, ``'uORF'``, ``'+0 iORF'``, or
        ``'other ORF'``.
    """
    assignments: dict[Transcript, str] = {}

    for tr in transcripts:
        try:
            orf_interval = tr.exons.map_to_local(orf.genomic_region)
        except ValueError:
            continue

        if tr.annotated_cds_iv is not None:
            cds_interval = tr.annotated_cds_iv
            orf_start, orf_end = orf_interval
            cds_start, cds_end = cds_interval

            if orf_interval == cds_interval:
                label = "cORF"
            elif orf_end == cds_end and orf_start < cds_start:
                label = "N terminal extended cORF"
            elif orf_end == cds_end and orf_start > cds_start:
                label = "N terminal truncated cORF"
            elif orf_end <= cds_start:
                label = "uORF"
            elif orf_start >= cds_end:
                label = "dORF"
            else:
                frame = (orf_start - cds_start) % 3
                if orf_start < cds_start < orf_end < cds_end:
                    label = f"+{frame} uoORF"
                elif cds_start < orf_start < cds_end < orf_end:
                    label = f"+{frame} doORF"
                elif cds_start < orf_start and orf_end < cds_end:
                    label = f"+{frame} iORF"
                else:
                    label = "other ORF"

        elif tr.biotype == "protein_coding":
            label = "pcRNA-ORF"
        elif tr.biotype in LNCRNA_BIOTYPES:
            label = "lncRNA-ORF"
        else:
            label = "varRNA-ORF"

        assignments[tr] = label

    if not assignments:
        return "other ORF"

    label_set = set(assignments.values())
    for level in ORF_TYPES_LEVELS:
        matches = label_set & level
        if len(matches) > 1:
            return "other ORF"
        if len(matches) == 1:
            return matches.pop()

    return "other ORF"


def find_orfs(
    seq: str,
    start_codons: list[str] | tuple[str, ...] = ("ATG",),
    stop_codons: list[str] | tuple[str, ...] = ("TAA", "TAG", "TGA"),
    min_length: int = 0,
) -> list[tuple[int, int]]:
    """Find all ORFs in a transcript sequence.

    Scans all three reading frames for start/stop codon pairs and
    returns 0-based, half-open intervals **including** the stop codon.

    Parameters
    ----------
    seq : str
        Spliced transcript nucleotide sequence.
    start_codons : list[str] or tuple[str, ...]
        Codons accepted as translation initiation sites.
    stop_codons : list[str] or tuple[str, ...]
        Codons accepted as translation termination sites.
    min_length : int
        Minimum ORF length (nt, excluding stop codon) to report.

    Returns
    -------
    list[tuple[int, int]]
        ``(start, end)`` intervals in transcript coordinates.  *end*
        includes the 3-nt stop codon.
    """
    start_codons_set = set(start_codons)
    stop_codons_set = set(stop_codons)
    orf_iv_on_transcript: list[tuple[int, int]] = []
    for i in range(3):
        starts: list[int] = []
        for j in range(i, len(seq), 3):
            codon = seq[j : j + 3]
            if codon in start_codons_set:
                starts.append(j)
            if codon in stop_codons_set:
                for start in starts:
                    if j - start >= min_length:
                        orf_iv_on_transcript.append((start, j + 3))
                starts = []
    return orf_iv_on_transcript


def make_rgrs(
    loc,
    genome: Fasta,
    config: Config,
    min_length_to_end: int = 30,
) -> None:
    """Generate ORF and noise ReadGeneratingRegions for this locus.

    For each transcript, noise regions are created upstream and
    downstream of the annotated CDS (if present) or spanning the
    full transcript otherwise.  ORF candidates are found by scanning
    the spliced transcript sequence for start/stop codon pairs.
    Duplicate RGRs (identical genomic footprint) are deduplicated,
    keeping the copy with the longest flanking context.

    Parameters
    ----------
    genome : pyfaidx.Fasta
        Indexed FASTA handle keyed by chromosome name.
    config : Config
        Configuration providing ``start_codons`` and ``stop_codons``.
    min_length_to_end : int
        Minimum combined length of ORF plus flanking transcript
        distance (in nucleotides) for an ORF to be retained.
    """
    loc.rgr_set: set[ReadGeneratingRegion] = set()
    orf_dict: dict[ReadGeneratingRegion, ReadGeneratingRegion] = {}
    noise_dict: dict[ReadGeneratingRegion, ReadGeneratingRegion] = {}

    for transcript in loc.transcripts:
        if (transcript.annotated_cds_iv is not None) and (
            (cds_start := transcript.annotated_cds_iv[0]) > 5
        ):
            # cds_start = transcript.exons.map_to_local(transcript.cds)[0]
            noise1 = ReadGeneratingRegion(
                "NOISE",
                transcript,
                f"{transcript.id}_a",
                (0, cds_start),
            )

            noise2 = ReadGeneratingRegion(
                "NOISE",
                transcript,
                f"{transcript.id}_b",
                (cds_start, len(transcript.exons)),
            )

            for noise in [noise1, noise2]:
                if noise not in noise_dict:
                    noise_dict[noise] = noise
                else:
                    existing = noise_dict[noise]
                    existing_span = (
                        existing.dist_to_transcript_end
                        + existing.dist_to_transcript_start
                    )
                    new_span = (
                        noise.dist_to_transcript_end
                        + noise.dist_to_transcript_start
                    )
                    if new_span > existing_span:
                        noise_dict[noise] = noise

        else:
            noise = ReadGeneratingRegion(
                "NOISE",
                transcript,
                transcript.id,
                (0, len(transcript.exons)),
            )
            if noise not in noise_dict:
                noise_dict[noise] = noise
            else:
                existing = noise_dict[noise]
                existing_span = (
                    existing.dist_to_transcript_end
                    + existing.dist_to_transcript_start
                )
                new_span = (
                    noise.dist_to_transcript_end
                    + noise.dist_to_transcript_start
                )
                if new_span > existing_span:
                    noise_dict[noise] = noise

        seq = transcript.exons.get_sequence(genome)
        c = 0
        for orf_iv_on_transcript in find_orfs(
            seq, config.start_codons, config.stop_codons
        ):
            rgr_iv_on_transcript = (
                orf_iv_on_transcript[0],
                orf_iv_on_transcript[1] - 3,
            )  # remove stop codon
            c += 1
            orf = ReadGeneratingRegion(
                "ORF",
                transcript,
                f"{transcript.id}_{c:04d}",
                rgr_iv_on_transcript,
            )
            if len(orf) + orf.dist_to_transcript_end < min_length_to_end:
                continue
            if len(orf) + orf.dist_to_transcript_start < min_length_to_end:
                continue
            if orf not in orf_dict:
                orf_dict[orf] = orf
            else:
                existing = orf_dict[orf]
                existing_span = (
                    existing.dist_to_transcript_end
                    + existing.dist_to_transcript_start
                )
                new_span = orf.dist_to_transcript_end + orf.dist_to_transcript_start
                if new_span > existing_span:
                    orf_dict[orf] = orf

    for noise in noise_dict.values():
        noise.transcript.rgr_set.add(noise)
    loc.rgr_set |= set(noise_dict.values())
    for orf in orf_dict.values():
        orf.orf_type = get_orf_type(orf, loc.transcripts)
        orf.transcript.add_orf(orf)
    loc.rgr_set |= set(orf_dict.values())

    # ``rgr.index`` addresses the design-matrix column blocks and the rows
    # of ``result``, so every RGR carries one from the moment the set is
    # built.  ``remove_rgrs`` re-densifies them after a removal.
    for c, rgr in enumerate(loc.rgr_set):
        rgr.index = c

    loc.gene_ids_complete = {
        rgr.transcript.gene_id for rgr in loc.rgr_set
    }


def build_rgrs(
    locus: Locus,
    db_path: str,
    genome: Fasta,
    config: Config,
    min_explained_reads: float,
) -> bool:
    """Filter a locus's transcripts by read support and build its RGRs.

    Greedily selects transcripts that jointly explain the most observed
    reads above ``min_explained_reads``, prunes the locus's transcript
    set and ``transcript_intervals`` accordingly, and calls
    :meth:`~price2.locus.Locus.make_rgrs` when transcripts remain.

    Parameters
    ----------
    locus : Locus
        Pre-RGR locus skeleton, mutated in place.
    db_path : str
        Path to ``price.db`` (read-only access for
        ``transcript_read_counts``).
    genome : pyfaidx.Fasta
        Worker-local indexed FASTA handle.
    config : Config
        Parsed PRICE configuration.
    min_explained_reads : float
        Threshold (count, not per-run) used to discard transcripts with
        insufficient read support.

    Returns
    -------
    bool
        ``True`` when the locus retained at least one transcript and
        RGRs were built; ``False`` for empty loci that downstream code
        should skip.
    """
    with database.connect(db_path) as db:
        rows = db.execute(
            "SELECT transcript_read_counts_blob FROM transcript_read_counts "
            "WHERE locus_id = ?",
            (locus.id,),
        ).fetchall()

    transcript_read_counts: dict = {}
    for (blob,) in rows:
        for k, v in database.decompress_blob(blob).items():
            transcript_read_counts[k] = transcript_read_counts.get(k, 0) + v

    tr_ids = [t.id for t in locus.transcripts]
    explaining_transcripts_reads_list = []

    if tr_ids and transcript_read_counts:
        n_tr = len(tr_ids)
        n_rs = len(transcript_read_counts)
        tr_to_col = {tr_id: i for i, tr_id in enumerate(tr_ids)}

        M = np.zeros((n_rs, n_tr), dtype=bool)
        counts = np.zeros(n_rs, dtype=np.float64)

        for row_i, (read_set, count) in enumerate(transcript_read_counts.items()):
            counts[row_i] = count
            for member in read_set:
                if member in tr_to_col:
                    M[row_i, tr_to_col[member]] = True

        while counts.sum() > 0:
            weighted = M.T @ counts
            best_col = int(np.argmax(weighted))
            best_score = weighted[best_col]
            if best_score == 0:
                break
            explaining_transcripts_reads_list.append(
                (tr_ids[best_col], best_score)
            )
            counts[M[:, best_col]] = 0.0

    transcripts_dict = {tr.id: tr for tr in locus.transcripts}

    locus.transcripts_number = len(locus.transcripts)
    locus.transcripts = [
        transcripts_dict[tr_id]
        for tr_id, count in explaining_transcripts_reads_list
        if count > min_explained_reads
    ]

    new_tr_intervals = HTSeq.GenomicArray(
        list(locus.transcript_intervals.chrom_vectors.keys()), typecode="O"
    )
    for step_iv, step_set in locus.transcript_intervals.steps():
        new_step_set = set()
        for tr in step_set:
            if tr in locus.transcripts:
                new_step_set.add(tr)
        new_tr_intervals[step_iv] = new_step_set
    locus.transcript_intervals = new_tr_intervals

    if not locus.transcripts:
        return False

    make_rgrs(locus, genome, config)
    return True