"""Equivalence group construction for Ribo-seq deconvolution.

Enumerates, per read length, the runs of read start positions that share a
genomic footprint across a locus's transcripts (:func:`read_start_runs`) and
maps them, through the cleavage model, to the sets of compatible ORFs
(ReadGeneratingRegions).  These equivalence groups are the core data
structure consumed by the deconvolution optimiser.

The equivalence-group key
-------------------------
Every group is keyed by ``(cells, read_length, oua)``: the set of
``(RGR, frame, coverage position)`` combinations a read of that length, with
or without a 5' untemplated addition, is compatible with.  A combination is
packed into one integer *cell* (see :func:`pack_cell`)::

    cell = rgr.index * CELL_CODES + frame_code * 3 + coverage_position

with ``frame_code`` 0-2 for the reading frame of an ORF and 3 for a NOISE
region, so ``cells`` is a ``frozenset[int]``.  :func:`make_equivalence_groups`
maps every key to the group's length (in codons); the same cells then make
up the rows of :class:`price2.read_routing.ReadRouting`, which is where the
groups live once the reads are routed.  Because the cells carry
``rgr.index``, every key has to be remapped when RGRs are removed and the
survivors are re-indexed (:meth:`~price2.read_routing.ReadRouting.without_rgrs`).
"""

from __future__ import annotations

import logging
from collections import defaultdict

from price2.cleavage_model import NO_FRAME
from price2.coverage_position import CoveragePosition
from price2.genomic_features import Transcript

logger = logging.getLogger(__name__)


#: Distinct ``frame_code * 3 + coverage_position`` values per RGR: four frame
#: codes (0, 1, 2 and :data:`NO_FRAME` for NOISE, the frame axis of the
#: cleavage model's look-up table) times three coverage positions.
CELL_CODES = 12


def pack_cell(
    rgr_index: int, frame: int | None, covpos: CoveragePosition
) -> int:
    """Pack an ``(RGR, frame, coverage position)`` combination into one int.

    Parameters
    ----------
    rgr_index : int
        ``rgr.index`` of the region.
    frame : int or None
        Reading frame relative to the ORF start, or ``None`` for NOISE.
    covpos : CoveragePosition
        Which part of the coverage profile the combination belongs to.
    """
    frame_code = NO_FRAME if frame is None else frame
    return rgr_index * CELL_CODES + frame_code * 3 + covpos.value


def unpack_cell(cell: int) -> tuple[int, int | None, CoveragePosition]:
    """Inverse of :func:`pack_cell`: ``(rgr_index, frame, covpos)``.

    Hot paths use ``divmod(cell, CELL_CODES)`` directly (or ``numpy.divmod``
    on an array of cells); this is for the callers that need the objects.
    """
    rgr_index, code = divmod(cell, CELL_CODES)
    frame_code, covpos = divmod(code, 3)
    return rgr_index, (None if frame_code == NO_FRAME else frame_code), CoveragePosition(covpos)


def _codon_span(start: int, end: int, phase: int) -> tuple[int, int]:
    """Codon indices ``[sc, ec)`` of the positions ``≡ phase (mod 3)`` in ``[start, end)``.

    Position ``3 * c + phase`` is codon ``c`` of that phase, so the span is
    ``ceil((start - phase) / 3)`` to ``ceil((end - phase) / 3)``.
    """
    return -(-(start - phase) // 3), -(-(end - phase) // 3)


class EquivalenceGroupIntervals:
    """Transcript positions where reads are compatible with a set of RGRs.

    Stores three lists — one per reading-frame phase (0, 1, 2) — of
    ``(start_codon, end_codon, cell)`` intervals, *cell* being the packed
    ``(rgr, frame, CoveragePosition)`` of :func:`pack_cell`: a read starting
    at position ``3 * codon + phase`` of the transcript is compatible with
    the cell.  Using interval lists instead of per-position dicts avoids
    O(interval_length) inner loops: ``add_rgr`` becomes O(1) per call, and
    the sweep-line in ``get_egs_dict`` is O(n_intervals * log(n_intervals)).
    """

    def __init__(self) -> None:
        self.intervals: tuple[list, list, list] = ([], [], [])

    def add_rgr(
        self,
        rgr,
        start: int,
        end: int,
        phase: int | None,
        frame: int | None = None,
        covpos: CoveragePosition = CoveragePosition.middle,
    ) -> None:
        """Register a ReadGeneratingRegion over a transcript position range.

        Parameters
        ----------
        rgr :
            The ReadGeneratingRegion to register (its ``index`` goes into
            the cell).
        start : int
            First transcript position (nucleotide coordinate) of the interval.
        end : int
            One-past-the-last transcript position of the interval.
        phase : int or None
            ``(read_start - transcript_start) % 3`` for coding RGRs;
            ignored (all phases used) for NOISE RGRs.
        frame : int or None
            Reading frame relative to ORF start (0, 1, or 2).
        covpos : CoveragePosition
            Which part of the coverage profile this interval corresponds to.
        """
        phases = (0, 1, 2) if not rgr.is_orf else (phase,)
        cell = pack_cell(rgr.index, frame, covpos)
        for ph in phases:
            sc, ec = _codon_span(start, end, ph)
            if sc < ec:
                self.intervals[ph].append((sc, ec, cell))

    def add_slice(
        self, source: EquivalenceGroupIntervals, start: int, length: int
    ) -> None:
        """Append *source*'s intervals over ``[start, start + length)``, re-based to 0.

        A read starting at position ``start + p`` of *source*'s transcript
        is the read starting at ``p`` here, so the phase of a source
        interval shifts by ``start`` and its codons by the span's first
        codon.

        Parameters
        ----------
        source : EquivalenceGroupIntervals
            The intervals of one transcript.
        start : int
            First transcript position (nucleotide coordinate) of the window.
        length : int
            Length of the window in nucleotides.
        """
        end = start + length
        shift = start % 3
        for phase in range(3):
            sc, ec = _codon_span(start, end, phase)
            if sc >= ec:
                continue
            dst = self.intervals[(phase - shift) % 3]
            for interval_sc, interval_ec, cell in source.intervals[phase]:
                lo = max(interval_sc, sc)
                hi = min(interval_ec, ec)
                if lo < hi:
                    dst.append((lo - sc, hi - sc, cell))

    def get_egs_dict(
        self,
        read_length: int,
        oua: bool,
        key_cache: dict | None = None,
    ) -> dict:
        """Convert internal intervals to a dict of EquivalenceGroups.

        Parameters
        ----------
        read_length : int
            Read length associated with these equivalence groups.
        oua : bool
            Whether the reads carry a 5' untemplated addition.
        key_cache : dict or None
            Optional shared cache used to intern the
            ``(cells, read_length, oua)`` key tuples so that equivalent keys
            produced for different runs reference the same Python objects.
            Pass the same dict across all calls to share keys.

        Returns
        -------
        dict
            Maps ``(cells, read_length, oua)`` keys (see the module
            docstring) to the group's length in codons.
        """
        egs: dict = defaultdict(int)
        for phase in range(3):
            interval_list = self.intervals[phase]
            if not interval_list:
                continue

            # Build events: (pos, type, cell)
            # type=0 for interval-end (deactivate), type=1 for interval-start
            # (activate).  Sorting puts ends before starts at the same position,
            # preserving half-open [sc, ec) semantics.
            events: list = []
            for sc, ec, cell in interval_list:
                events.append((sc, 1, cell))
                events.append((ec, 0, cell))
            events.sort(key=lambda x: (x[0], x[1]))

            # refcount dict: cell -> number of currently-open intervals.
            # Needed because ``add_slice`` can emit duplicate (sc, ec, cell)
            # entries when the same RGR appears on multiple transcripts; plain
            # set semantics would discard the cell too early on the first end
            # event.  A cell is "active" as long as refcount > 0.
            active: dict = {}
            prev_pos: int | None = None

            for pos, typ, cell in events:
                if prev_pos is not None and pos != prev_pos and active:
                    key = (frozenset(active), read_length, oua)
                    if key_cache is not None:
                        key = key_cache.setdefault(key, key)
                    egs[key] += pos - prev_pos
                if typ == 0:  # end event
                    cnt = active.get(cell, 1) - 1
                    if cnt <= 0:
                        active.pop(cell, None)
                    else:
                        active[cell] = cnt
                else:  # start event
                    active[cell] = active.get(cell, 0) + 1
                prev_pos = pos

        return egs


def _exons_in_transcription_order(
    transcript: Transcript, minus: bool
) -> list[tuple[int, int]]:
    """The transcript's exons as genomic ``(start, end)`` in transcription order.

    Abutting exons (one ending where the next begins) are merged: a read's
    footprint does not see such a boundary, so neither may the runs.
    """
    merged: list[tuple[int, int]] = []
    for iv in transcript.exons.intervals:
        if merged and merged[-1][1] == iv.start:
            merged[-1] = (merged[-1][0], iv.end)
        else:
            merged.append((iv.start, iv.end))
    return merged[::-1] if minus else merged


def read_start_runs(
    transcripts: list[Transcript], read_length: int, strand: str
) -> list[tuple[dict[Transcript, int], int]]:
    """Runs of read start positions sharing a footprint and its transcripts.

    A read of length ``L`` starting at position ``p`` of transcript ``t``
    (``0 <= p <= len(t) - L``) has a genomic footprint: the ``L`` bases of
    ``t`` from ``p`` on, following ``t``'s splicing.  Starts on different
    transcripts are the same read when their footprints coincide; the read
    is then compatible with every one of those transcripts.  This function
    returns the maximal runs of consecutive start positions whose set of
    compatible transcripts is constant, each as ``(positions, length)``:
    the transcript coordinate of the run's first start on every compatible
    transcript, and the number of starts in the run.  Runs partition the
    valid starts of every transcript, so summing anything over their
    positions sums it over all possible reads.

    Two starts share a footprint exactly when their 5' bases coincide
    genomically and the reads cross the same junctions, so the starts are
    grouped by the *chain* of intron boundaries a read from them crosses
    (empty for a read within one exon).  Within a chain a transcript's
    starts form one range of 5'-end coordinates, bounded by its first exon
    of the chain on the 5' side and its last exon on the 3' side; a sweep
    over those range boundaries yields the runs.

    Parameters
    ----------
    transcripts : list[Transcript]
        The locus's transcripts; a run lists its transcripts in this order.
    read_length : int
        Read length ``L``.
    strand : str
        The locus strand (``"+"`` or ``"-"``).
    """
    minus = strand == "-"
    rank = {transcript: i for i, transcript in enumerate(transcripts)}
    # chain -> [(g_lo, g_hi, transcript, exon_5p_end, exon_offset)]: the range
    # of 5'-end genomic coordinates (inclusive) of the transcript's starts in
    # that chain, plus what maps a coordinate back to a transcript position.
    ranges: dict[tuple, list] = {}
    for transcript in transcripts:
        exons = _exons_in_transcription_order(transcript, minus)
        offset = 0
        for i, (start_i, end_i) in enumerate(exons):
            len_i = end_i - start_i
            # ``u`` = bases from the 5' end of the read to the 3' end of exon
            # ``i`` (the read's first exon), 1 <= u <= len_i.  A read within
            # the exon needs u >= L; one crossing to exon ``j`` needs
            # 1 <= L - u - D < = len_j, D being the exons in between.
            chain_ranges: list[tuple[tuple, int, int]] = []
            if len_i >= read_length:
                chain_ranges.append(((), read_length, len_i))
            chain: list[int] = [start_i if minus else end_i]
            between = 0
            for start_j, end_j in exons[i + 1:]:
                if read_length - between - 1 < 1:
                    break
                chain.append(end_j if minus else start_j)
                u_lo = max(1, read_length - between - (end_j - start_j))
                u_hi = min(len_i, read_length - between - 1)
                if u_lo <= u_hi:
                    chain_ranges.append((tuple(chain), u_lo, u_hi))
                between += end_j - start_j
                chain.append(start_j if minus else end_j)
            for key, u_lo, u_hi in chain_ranges:
                if minus:
                    g_lo, g_hi = start_i - 1 + u_lo, start_i - 1 + u_hi
                else:
                    g_lo, g_hi = end_i - u_hi, end_i - u_lo
                ranges.setdefault(key, []).append(
                    (g_lo, g_hi, transcript, end_i if minus else start_i, offset)
                )
            offset += len_i

    runs: list[tuple[dict[Transcript, int], int]] = []
    for chain_ranges in ranges.values():
        events: list[tuple[int, int, int]] = []
        for k, (g_lo, g_hi, _, _, _) in enumerate(chain_ranges):
            events.append((g_lo, 1, k))
            events.append((g_hi + 1, 0, k))
        events.sort(key=lambda e: (e[0], e[1]))
        active: set[int] = set()
        prev = None
        for pos, kind, k in events:
            if prev is not None and pos != prev and active:
                # The run's first start in transcription order.
                first = prev if not minus else pos - 1
                positions = {}
                for k_active in sorted(active, key=lambda k: rank[chain_ranges[k][2]]):
                    _, _, transcript, exon_5p, offset = chain_ranges[k_active]
                    positions[transcript] = offset + (
                        exon_5p - 1 - first if minus else first - exon_5p
                    )
                runs.append((positions, pos - prev))
            if kind:
                active.add(k)
            else:
                active.discard(k)
            prev = pos
    return runs


def _equivalence_groups_dict(
    runs: list[tuple[dict[Transcript, int], int]],
    egis: dict[Transcript, EquivalenceGroupIntervals],
    read_length: int,
    oua: bool,
    key_cache: dict,
) -> dict:
    """Aggregate equivalence groups over all read-start runs.

    For each run (see :func:`read_start_runs`) collects the partial
    equivalence groups contributed by its transcripts' coverage intervals
    over the run's positions.  Groups with matching keys are merged by
    summing their lengths.

    Parameters
    ----------
    runs : list
        The read-start runs of the locus for *read_length*.
    egis : dict[Transcript, EquivalenceGroupIntervals]
        Per-transcript equivalence-group intervals for the current read length
        and oua flag.
    read_length : int
        Read length for which these groups are computed.
    oua : bool
        Whether the reads carry a 5' untemplated addition.
    key_cache : dict
        Shared cache interning the equivalence-group keys across runs and
        read lengths.

    Returns
    -------
    dict
        Maps equivalence-group keys to their lengths.
    """
    egs_dict: dict = {}
    for positions, length in runs:
        window = EquivalenceGroupIntervals()
        for transcript, start in positions.items():
            window.add_slice(egis[transcript], start, length)
        run_egs = window.get_egs_dict(read_length, oua, key_cache=key_cache)
        for k, run_length in run_egs.items():
            egs_dict[k] = egs_dict.get(k, 0) + run_length
    return egs_dict


def _cleavage_dist_signature(cleavage_model, read_length: int, oua: bool) -> tuple:
    """Signature of the cleavage distances that determine equivalence groups.

    :func:`_equivalence_intervals` reads the cleavage model only through
    :meth:`~price2.cleavage_model.CleavageModel.dist_to_orf_bounds` for
    frames ``(None, 0, 1, 2)``.  Two runs that share this signature for a
    given ``(read_length, oua)`` therefore produce identical equivalence
    groups, so the :func:`_equivalence_groups_dict` result can be reused
    between them.

    Returns
    -------
    tuple
        The ``(start, end)`` bounds, or ``None``, for frames ``(None, 0, 1, 2)``.
    """
    return tuple(
        cleavage_model.dist_to_orf_bounds(read_length, oua, frame)
        for frame in (None, 0, 1, 2)
    )


def make_equivalence_groups(loc, runs: list) -> dict:
    """Compute all equivalence groups for a locus across all runs.

    For every run and every read length present in its cleavage model,
    enumerates the read-start runs of the locus and collects the equivalence
    groups over them.

    A shared key cache interns the ``(cells, read_length, oua)`` tuples so
    that identical keys produced for different runs reference the same
    Python objects.  In typical multi-run loci this reduces ``loc.egs`` memory by
    one to two orders of magnitude.

    Two cross-run caches avoid redundant work, which matters most when many
    runs are present:

    * :func:`read_start_runs` depends only on the transcripts and the read
      length, not the run, so the runs are built once per read length.
    * :func:`_equivalence_groups_dict` depends on the run only through the
      cleavage-distance signature (see :func:`_cleavage_dist_signature`), so
      its contribution is computed once per ``(read_length, oua, signature)``
      and reused across runs that share that signature.

    Parameters
    ----------
    loc :
        A locus object with ``transcripts`` (list of Transcript) and ``iv``.
    runs : list
        List of RiboSeqRun objects, each providing a cleavage model.

    Returns
    -------
    dict
        Maps each run to a dict of equivalence-group key → length in codons.
    """
    key_cache: dict = {}
    start_runs: dict = {}  # read_length -> read-start runs
    contributions: dict = {}  # (read_length, oua, signature) -> {key: length}

    def contribution(cleavage_model, read_length: int, oua: bool) -> dict:
        signature = _cleavage_dist_signature(cleavage_model, read_length, oua)
        found = contributions.get((read_length, oua, signature))
        if found is None:
            if read_length not in start_runs:
                start_runs[read_length] = read_start_runs(
                    loc.transcripts, read_length, loc.iv.strand
                )
            egis = {
                tr: _equivalence_intervals(tr, cleavage_model, read_length, oua)
                for tr in loc.transcripts
            }
            found = _equivalence_groups_dict(
                start_runs[read_length], egis, read_length, oua, key_cache
            )
            contributions[(read_length, oua, signature)] = found
        return found

    egs: dict = {}
    for run in runs:
        run_egs: dict = {}
        egs[run] = run_egs
        for read_length in run.cleavage_model.non_zero_lengths:
            for oua in (True, False):
                found = contribution(run.cleavage_model, read_length, oua)
                for k, length in found.items():
                    run_egs[k] = run_egs.get(k, 0) + length
    return egs


def _equivalence_intervals(
    transcript: Transcript,
    cleavage_model,
    read_length: int,
    oua: bool,
) -> EquivalenceGroupIntervals:
    """Build equivalence-group intervals for one transcript.

    Uses the cleavage model to determine, for each ReadGeneratingRegion on
    *transcript*, which transcript positions can produce a read of
    *read_length* and in which frame/coverage-position category.

    Parameters
    ----------
    transcript : Transcript
        The transcript for which to compute intervals.
    cleavage_model :
        Cleavage model providing
        :meth:`~price2.cleavage_model.CleavageModel.dist_to_orf_bounds`.
    read_length : int
        Read length to model.
    oua : bool
        Whether the reads carry a 5' untemplated addition.

    Returns
    -------
    EquivalenceGroupIntervals
        Populated intervals for assignment of reads to RGRs on this transcript.
    """
    egi = EquivalenceGroupIntervals()
    # Bind the enum members once: ``CoveragePosition.start`` etc. is a
    # DynamicClassAttribute lookup, and this loop accesses them millions of
    # times across a run.  Local names are the identical singleton objects.
    cov_start = CoveragePosition.start
    cov_middle = CoveragePosition.middle
    cov_stop = CoveragePosition.stop
    # One past the last position a read of this length can start at.
    last_start = len(transcript.exons) - read_length + 1
    for rgr in transcript.rgr_set:
        lo, hi = rgr.iv_on_transcript
        if not rgr.is_orf:
            bounds = cleavage_model.dist_to_orf_bounds(read_length, oua, None)
            if bounds is None:
                continue
            rsos, rsoe = bounds
            egi.add_rgr(rgr, max(0, lo + rsos), max(3, min(last_start, hi + rsoe)), None)
            continue
        for frame in (0, 1, 2):  # frame relative to ORF start
            bounds = cleavage_model.dist_to_orf_bounds(read_length, oua, frame)
            if bounds is None:
                continue
            # Read start to ORF start / to ORF end.
            rsos, rsoe = bounds
            phase = (lo + rsos) % 3  # phase relative to transcript start
            # The start codon, the body and the stop codon of the ORF, as
            # the read-start ranges that cover them.
            for covpos, start, end in (
                (cov_start, lo + rsos, max(0, lo + 3 + rsoe)),
                (cov_middle, lo + 3 + rsos, min(last_start, hi - 3 + rsoe)),
                (cov_stop, hi - 3 + rsos, min(last_start, hi + rsoe)),
            ):
                egi.add_rgr(rgr, max(0, start), end, phase, frame, covpos)
    return egi
