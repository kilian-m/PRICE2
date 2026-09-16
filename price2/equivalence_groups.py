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
region.  :func:`make_equivalence_groups` numbers the distinct keys of a
locus and gives every run its groups with their lengths (in codons), as
arrays (:class:`EquivalenceGroups`); the same cells then make up the rows of
:class:`price2.read_routing.ReadRouting`, which is where the groups live once
the reads are routed.  Because the cells carry
``rgr.index``, every key has to be remapped when RGRs are removed and the
survivors are re-indexed (:meth:`~price2.read_routing.ReadRouting.without_rgrs`).
"""

from __future__ import annotations

import logging
from collections import defaultdict

import numpy as np
from numba import njit, types
from numba.typed import Dict

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


def _equivalence_groups_dict_reference(
    runs: list[tuple[dict[Transcript, int], int]],
    egis: dict[Transcript, EquivalenceGroupIntervals],
    read_length: int,
    oua: bool,
    key_cache: dict,
) -> dict:
    """Aggregate equivalence groups over all read-start runs, one run at a time.

    The reference for :func:`_equivalence_groups_dict`, which computes the
    same groups in one compiled sweep.  For each run (see
    :func:`read_start_runs`) collects the partial equivalence groups
    contributed by its transcripts' coverage intervals over the run's
    positions.  Groups with matching keys are merged by summing their
    lengths.

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


# --------------------------------------------------------------------------- #
# The compiled sweep
# --------------------------------------------------------------------------- #
#
# The reference above clips every transcript's intervals into every read-start
# run's window and sweeps each window in Python — a few thousand intervals
# times a few dozen runs per (read length, oua) contribution, tens of
# millions of Python steps per read-dense locus.  The path below builds each
# transcript's intervals as arrays and sweeps all runs of a contribution in
# one compiled pass.  The active set of a sweep is tracked by reference
# counts over the cells and identified by two additive 64-bit hashes plus its
# size; a set is snapshotted the first time its identity appears, so the
# groups come back as distinct cell arrays with their summed lengths.

#: Cell identity of a group.  Two independent multiplicative hashes of the
#: cells, summed over the distinct cells of a set, plus the set's size.
_HASH_A = np.uint64(0x9E3779B97F4A7C15)
_HASH_B = np.uint64(0xC2B2AE3D27D4EB4F)
_IDENTITY_TYPE = types.UniTuple(types.uint64, 3)
_INDEX_TYPE = types.int64


@njit(cache=True)
def _mix(x, k):
    """A 64-bit hash of *x* (splitmix-style)."""
    z = (x + np.uint64(1)) * k
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


@njit(cache=True)
def _sweep_runs(
    run_length, mem_ptr, mem_tr, mem_start,
    iv_ptr, iv_sc, iv_ec, iv_cell, n_cells,
):
    """Sweep every read-start run of one contribution.

    Parameters
    ----------
    run_length : int64[n_runs]
        Positions per run.
    mem_ptr, mem_tr, mem_start : int64
        CSR over the runs of their (transcript, local start) members.
    iv_ptr, iv_sc, iv_ec, iv_cell : int64
        CSR over ``transcript * 3 + phase`` of the transcripts' intervals
        ``[sc, ec)`` in codons, with their cell.
    n_cells : int
        One more than the largest cell.

    Returns
    -------
    n_groups, group_ptr, group_cells, group_length
        The distinct groups: their cells (CSR, sorted) and summed lengths.
    """
    n_runs = run_length.shape[0]
    # Event buffer, grown as needed: key = (phase, pos, type), plus the cell.
    cap = 1024
    ev_key = np.empty(cap, dtype=np.int64)
    ev_cell = np.empty(cap, dtype=np.int64)
    count = np.zeros(n_cells, dtype=np.int64)
    active = np.empty(n_cells, dtype=np.int64)
    where = np.full(n_cells, -1, dtype=np.int64)
    seen = Dict.empty(key_type=_IDENTITY_TYPE, value_type=_INDEX_TYPE)
    out_cap = 4096
    out_cells = np.empty(out_cap, dtype=np.int64)
    grp_cap = 1024
    out_ptr = np.zeros(grp_cap + 1, dtype=np.int64)
    out_len = np.zeros(grp_cap, dtype=np.int64)
    n_groups = 0
    n_out = 0
    for r in range(n_runs):
        length = run_length[r]
        n_ev = 0
        for m in range(mem_ptr[r], mem_ptr[r + 1]):
            start = mem_start[m]
            end = start + length
            shift = start % 3
            tr = mem_tr[m]
            for phase in range(3):
                sc = -((start - phase) // -3)
                ec = -((end - phase) // -3)
                if sc >= ec:
                    continue
                dst = (phase - shift) % 3
                base = (dst << 40)
                for k in range(iv_ptr[tr * 3 + phase], iv_ptr[tr * 3 + phase + 1]):
                    lo = iv_sc[k]
                    if lo < sc:
                        lo = sc
                    hi = iv_ec[k]
                    if hi > ec:
                        hi = ec
                    if lo < hi:
                        if n_ev + 2 > cap:
                            cap *= 2
                            new_key = np.empty(cap, dtype=np.int64)
                            new_cell = np.empty(cap, dtype=np.int64)
                            new_key[:n_ev] = ev_key[:n_ev]
                            new_cell[:n_ev] = ev_cell[:n_ev]
                            ev_key = new_key
                            ev_cell = new_cell
                        ev_key[n_ev] = base + ((lo - sc) << 1) + 1
                        ev_cell[n_ev] = iv_cell[k]
                        ev_key[n_ev + 1] = base + ((hi - sc) << 1)
                        ev_cell[n_ev + 1] = iv_cell[k]
                        n_ev += 2
        if n_ev == 0:
            continue
        order = np.argsort(ev_key[:n_ev], kind="mergesort")
        n_active = 0
        h1 = np.uint64(0)
        h2 = np.uint64(0)
        prev_key = np.int64(-1)
        for e in range(n_ev):
            key = ev_key[order[e]]
            cell = ev_cell[order[e]]
            phase_pos = key >> 1
            if prev_key >= 0 and phase_pos != (prev_key >> 1) and n_active > 0:
                # Emit [prev_pos, pos) unless the phase changed (the last
                # event of a phase is an end that empties the set, so it
                # cannot).
                seg_len = phase_pos - (prev_key >> 1)
                ident = (h1, h2, np.uint64(n_active))
                if ident in seen:
                    g = seen[ident]
                else:
                    g = n_groups
                    seen[ident] = g
                    n_groups += 1
                    if n_groups > grp_cap:
                        grp_cap *= 2
                        new_ptr = np.zeros(grp_cap + 1, dtype=np.int64)
                        new_len = np.zeros(grp_cap, dtype=np.int64)
                        new_ptr[:n_groups] = out_ptr[:n_groups]
                        new_len[:n_groups - 1] = out_len[:n_groups - 1]
                        out_ptr = new_ptr
                        out_len = new_len
                    if n_out + n_active > out_cap:
                        while n_out + n_active > out_cap:
                            out_cap *= 2
                        new_cells = np.empty(out_cap, dtype=np.int64)
                        new_cells[:n_out] = out_cells[:n_out]
                        out_cells = new_cells
                    snapshot = np.sort(active[:n_active])
                    out_cells[n_out:n_out + n_active] = snapshot
                    n_out += n_active
                    out_ptr[n_groups] = n_out
                out_len[g] += seg_len
            if key & 1:
                if count[cell] == 0:
                    where[cell] = n_active
                    active[n_active] = cell
                    n_active += 1
                    h1 += _mix(np.uint64(cell), _HASH_A)
                    h2 += _mix(np.uint64(cell), _HASH_B)
                count[cell] += 1
            else:
                count[cell] -= 1
                if count[cell] == 0:
                    n_active -= 1
                    last = active[n_active]
                    pos = where[cell]
                    active[pos] = last
                    where[last] = pos
                    where[cell] = -1
                    h1 -= _mix(np.uint64(cell), _HASH_A)
                    h2 -= _mix(np.uint64(cell), _HASH_B)
            prev_key = key
    return n_groups, out_ptr[:n_groups + 1], out_cells[:n_out], out_len[:n_groups]


def _codon_spans(start, end, phase):
    """Vectorised :func:`_codon_span`."""
    return -((start - phase) // -3), -((end - phase) // -3)


def _transcript_intervals(
    transcript: Transcript, cleavage_model, read_length: int, oua: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The intervals of :func:`_equivalence_intervals`, as arrays.

    Returns ``(phase, sc, ec, cell)`` with every RGR's intervals of every
    phase; the order is immaterial.
    """
    rgrs = sorted(transcript.rgr_set, key=lambda rgr: rgr.index)
    last_start = len(transcript.exons) - read_length + 1
    phases: list[np.ndarray] = []
    scs: list[np.ndarray] = []
    ecs: list[np.ndarray] = []
    cells: list[np.ndarray] = []

    def add(phase, start, end, cell):
        sc, ec = _codon_spans(start, end, phase)
        keep = sc < ec
        phases.append(phase[keep])
        scs.append(sc[keep])
        ecs.append(ec[keep])
        cells.append(cell[keep])

    orfs = [rgr for rgr in rgrs if rgr.is_orf]
    if orfs:
        lo = np.array([rgr.iv_on_transcript[0] for rgr in orfs], dtype=np.int64)
        hi = np.array([rgr.iv_on_transcript[1] for rgr in orfs], dtype=np.int64)
        index = np.array([rgr.index for rgr in orfs], dtype=np.int64)
        for frame in (0, 1, 2):
            bounds = cleavage_model.dist_to_orf_bounds(read_length, oua, frame)
            if bounds is None:
                continue
            rsos, rsoe = bounds
            phase = (lo + rsos) % 3
            base = index * CELL_CODES + frame * 3
            for covpos, start, end in (
                (CoveragePosition.start.value, lo + rsos, np.maximum(0, lo + 3 + rsoe)),
                (CoveragePosition.middle.value, lo + 3 + rsos, np.minimum(last_start, hi - 3 + rsoe)),
                (CoveragePosition.stop.value, hi - 3 + rsos, np.minimum(last_start, hi + rsoe)),
            ):
                add(phase, np.maximum(0, start), end, base + covpos)
    noise = [rgr for rgr in rgrs if not rgr.is_orf]
    if noise:
        bounds = cleavage_model.dist_to_orf_bounds(read_length, oua, None)
        if bounds is not None:
            rsos, rsoe = bounds
            lo = np.array([rgr.iv_on_transcript[0] for rgr in noise], dtype=np.int64)
            hi = np.array([rgr.iv_on_transcript[1] for rgr in noise], dtype=np.int64)
            index = np.array([rgr.index for rgr in noise], dtype=np.int64)
            start = np.maximum(0, lo + rsos)
            end = np.maximum(3, np.minimum(last_start, hi + rsoe))
            cell = index * CELL_CODES + NO_FRAME * 3 + CoveragePosition.middle.value
            for ph in (0, 1, 2):
                add(np.full(lo.shape, ph, dtype=np.int64), start, end, cell)
    if not phases:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty, empty, empty
    return (
        np.concatenate(phases), np.concatenate(scs), np.concatenate(ecs),
        np.concatenate(cells),
    )


def _equivalence_groups_arrays(
    runs: list[tuple[dict[Transcript, int], int]],
    transcripts: list[Transcript],
    cleavage_model,
    read_length: int,
    oua: bool,
) -> _Contribution:
    """Aggregate equivalence groups over all read-start runs, in one compiled sweep.

    Computes what :func:`_equivalence_groups_dict_reference` computes from
    :func:`_equivalence_intervals`: every run's window of every transcript,
    swept for the stretches over which the set of compatible cells is
    constant, summed by set — as arrays (:class:`_Contribution`).

    Parameters
    ----------
    runs : list
        The read-start runs of the locus for *read_length*.
    transcripts : list[Transcript]
        The locus's transcripts (the runs' members are among them).
    cleavage_model :
        The run's cleavage model, read through ``dist_to_orf_bounds``.
    read_length, oua : int, bool
        The read shape the groups are computed for.
    """
    rank = {tr: i for i, tr in enumerate(transcripts)}
    n_tr = len(transcripts)
    # The transcripts' intervals, CSR over ``transcript * 3 + phase``.
    parts = [_transcript_intervals(tr, cleavage_model, read_length, oua) for tr in transcripts]
    slot = np.concatenate(
        [np.repeat(t, len(part[0])) * 3 + part[0] for t, part in enumerate(parts)]
    ) if parts else np.zeros(0, dtype=np.int64)
    order = np.argsort(slot, kind="stable")
    iv_sc = np.concatenate([part[1] for part in parts])[order] if parts else np.zeros(0, dtype=np.int64)
    iv_ec = np.concatenate([part[2] for part in parts])[order] if parts else np.zeros(0, dtype=np.int64)
    iv_cell = np.concatenate([part[3] for part in parts])[order] if parts else np.zeros(0, dtype=np.int64)
    iv_ptr = np.zeros(n_tr * 3 + 1, dtype=np.int64)
    np.cumsum(np.bincount(slot, minlength=n_tr * 3), out=iv_ptr[1:])
    if iv_cell.size == 0 or not runs:
        return _Contribution(
            read_length, oua, np.zeros(1, dtype=np.int64),
            np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64),
        )
    n_cells = int(iv_cell.max()) + 1

    run_length = np.array([length for _, length in runs], dtype=np.int64)
    mem_counts = np.array([len(positions) for positions, _ in runs], dtype=np.int64)
    mem_ptr = np.zeros(len(runs) + 1, dtype=np.int64)
    np.cumsum(mem_counts, out=mem_ptr[1:])
    mem_tr = np.fromiter(
        (rank[tr] for positions, _ in runs for tr in positions),
        dtype=np.int64, count=int(mem_counts.sum()),
    )
    mem_start = np.fromiter(
        (start for positions, _ in runs for start in positions.values()),
        dtype=np.int64, count=int(mem_counts.sum()),
    )

    n_groups, group_ptr, group_cells, group_length = _sweep_runs(
        run_length, mem_ptr, mem_tr, mem_start, iv_ptr, iv_sc, iv_ec, iv_cell, n_cells
    )
    return _Contribution(read_length, oua, group_ptr, group_cells, group_length)


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


class EquivalenceGroups:
    """The equivalence groups of a locus, as arrays.

    A *key* is a distinct ``(cells, read_length, oua)`` (see the module
    docstring), numbered once for the locus; every run's groups are the
    keys it produced with their lengths in codons.  This is what
    :meth:`price2.read_routing.ReadRouting.build` turns into the rows of the
    design matrix, without a Python object per group.

    Attributes
    ----------
    key_ptr, key_cells : numpy.ndarray
        The keys' cells, CSR over the keys, sorted within a key.
    key_rl, key_oua : numpy.ndarray
        Read length (``int32``) and untemplated-addition flag (``uint8``)
        of every key.
    run_key, run_length : dict[str, numpy.ndarray]
        Per run id, the keys of its groups and their lengths.
    """

    __slots__ = ("key_ptr", "key_cells", "key_rl", "key_oua", "run_key", "run_length")

    def __init__(self, key_ptr, key_cells, key_rl, key_oua, run_key, run_length):
        self.key_ptr = key_ptr
        self.key_cells = key_cells
        self.key_rl = key_rl
        self.key_oua = key_oua
        self.run_key = run_key
        self.run_length = run_length

    @property
    def n_keys(self) -> int:
        return self.key_rl.shape[0]

    @property
    def key_nnz(self) -> np.ndarray:
        return np.diff(self.key_ptr)

    @property
    def n_rows(self) -> int:
        """Groups over all runs: the rows of the design matrix."""
        return sum(ids.size for ids in self.run_key.values())

    def as_dicts(self, runs: list) -> dict:
        """``{run: {(frozenset(cells), read_length, oua): length}}``, for tests."""
        cells = self.key_cells.tolist()
        ptr = self.key_ptr.tolist()
        keys = [
            (frozenset(cells[ptr[k]:ptr[k + 1]]), int(self.key_rl[k]), bool(self.key_oua[k]))
            for k in range(self.n_keys)
        ]
        return {
            run: {
                keys[k]: int(length)
                for k, length in zip(self.run_key[run.id].tolist(), self.run_length[run.id].tolist())
            }
            for run in runs
        }


class _Contribution:
    """The groups of one ``(read length, oua, cleavage signature)``: arrays."""

    __slots__ = ("read_length", "oua", "ptr", "cells", "length")

    def __init__(self, read_length: int, oua: bool, ptr, cells, length) -> None:
        self.read_length = read_length
        self.oua = oua
        self.ptr = ptr
        self.cells = cells
        self.length = length

    @classmethod
    def from_dict(cls, read_length: int, oua: bool, groups: dict) -> _Contribution:
        keys = [key for key in groups if key[0]]
        nnz = np.array([len(key[0]) for key in keys], dtype=np.int64)
        ptr = np.zeros(len(keys) + 1, dtype=np.int64)
        np.cumsum(nnz, out=ptr[1:])
        cells = np.array(
            [c for key in keys for c in sorted(key[0])], dtype=np.int64
        )
        length = np.array([groups[key] for key in keys], dtype=np.int64)
        return cls(read_length, oua, ptr, cells, length)

    @property
    def n_groups(self) -> int:
        return self.length.shape[0]


def _assemble(
    contributions: list[_Contribution], run_ids: list[str], run_parts: dict[str, list[int]]
) -> EquivalenceGroups:
    """Number the distinct keys of every contribution and lay the runs out over them.

    Keys are matched by their hash (see :func:`price2.read_routing.key_hashes`)
    and every group verified against its key's first occurrence; should two
    different keys hash alike, an exact pass takes over.
    """
    from price2.read_routing import _cells_equal, key_hashes

    if not contributions:
        empty = np.zeros(0, dtype=np.int64)
        return EquivalenceGroups(
            np.zeros(1, dtype=np.int64), empty, np.zeros(0, dtype=np.int32),
            np.zeros(0, dtype=np.uint8),
            {run_id: empty for run_id in run_ids},
            {run_id: empty for run_id in run_ids},
        )
    # Every group of every contribution, concatenated.
    nnz = np.concatenate([np.diff(c.ptr) for c in contributions])
    cells = np.concatenate([c.cells for c in contributions])
    rl = np.concatenate(
        [np.full(c.n_groups, c.read_length, dtype=np.int32) for c in contributions]
    )
    oua = np.concatenate(
        [np.full(c.n_groups, int(c.oua), dtype=np.uint8) for c in contributions]
    )
    offset = np.cumsum([0] + [c.n_groups for c in contributions])
    hashes = key_hashes(rl, oua, nnz, cells)
    _, first, inverse = np.unique(hashes, return_index=True, return_inverse=True)
    inverse = inverse.reshape(-1)
    off = np.cumsum(nnz) - nnz
    rep = first[inverse]
    ok = (rl == rl[rep]) & (oua == oua[rep]) & (nnz == nnz[rep])
    ok[ok] = _cells_equal(nnz, cells, off, np.flatnonzero(ok), nnz, cells, off, rep[ok])
    if not ok.all():  # pragma: no cover - only on a hash collision
        table: dict = {}
        key_of = np.empty(nnz.size, dtype=np.int64)
        firsts = []
        for g in range(nnz.size):
            key = (int(rl[g]), int(oua[g]), cells[off[g]:off[g] + nnz[g]].tobytes())
            k = table.get(key)
            if k is None:
                k = table[key] = len(firsts)
                firsts.append(g)
            key_of[g] = k
        first = np.array(firsts, dtype=np.int64)
    else:
        key_of = inverse
    n_keys = first.size
    key_nnz = nnz[first]
    key_ptr = np.zeros(n_keys + 1, dtype=np.int64)
    np.cumsum(key_nnz, out=key_ptr[1:])
    gather = np.repeat(off[first], key_nnz) + (
        np.arange(int(key_nnz.sum())) - np.repeat(key_ptr[:-1], key_nnz)
    )
    groups = EquivalenceGroups(
        key_ptr, cells[gather], rl[first], oua[first], {}, {}
    )
    for run_id in run_ids:
        parts = run_parts[run_id]
        if parts:
            groups.run_key[run_id] = np.concatenate(
                [key_of[offset[i]:offset[i + 1]] for i in parts]
            )
            groups.run_length[run_id] = np.concatenate(
                [contributions[i].length for i in parts]
            )
        else:
            groups.run_key[run_id] = np.zeros(0, dtype=np.int64)
            groups.run_length[run_id] = np.zeros(0, dtype=np.int64)
    return groups


def make_equivalence_groups(loc, runs: list, reference: bool = False) -> EquivalenceGroups:
    """Compute all equivalence groups for a locus across all runs.

    For every run and every read length present in its cleavage model,
    enumerates the read-start runs of the locus and collects the equivalence
    groups over them.

    Two cross-run caches avoid redundant work, which matters most when many
    runs are present:

    * :func:`read_start_runs` depends only on the transcripts and the read
      length, not the run, so the runs are built once per read length.
    * A contribution depends on the run only through the cleavage-distance
      signature (see :func:`_cleavage_dist_signature`), so it is computed
      once per ``(read_length, oua, signature)`` and shared by the runs
      with that signature.

    Parameters
    ----------
    loc :
        A locus object with ``transcripts`` (list of Transcript) and ``iv``.
    runs : list
        List of RiboSeqRun objects, each providing a cleavage model.
    reference : bool, optional
        Use the one-run-at-a-time Python sweep
        (:func:`_equivalence_groups_dict_reference`) instead of the compiled
        one; for tests.

    Returns
    -------
    EquivalenceGroups
        The distinct keys and, per run, its groups with their lengths.
    """
    start_runs: dict = {}  # read_length -> read-start runs
    contributions: list[_Contribution] = []
    index: dict = {}  # (read_length, oua, signature) -> position in contributions
    key_cache: dict = {}

    def contribution(cleavage_model, read_length: int, oua: bool) -> int:
        signature = _cleavage_dist_signature(cleavage_model, read_length, oua)
        found = index.get((read_length, oua, signature))
        if found is None:
            if read_length not in start_runs:
                start_runs[read_length] = read_start_runs(
                    loc.transcripts, read_length, loc.iv.strand
                )
            if reference:
                egis = {
                    tr: _equivalence_intervals(tr, cleavage_model, read_length, oua)
                    for tr in loc.transcripts
                }
                groups = _equivalence_groups_dict_reference(
                    start_runs[read_length], egis, read_length, oua, key_cache
                )
                part = _Contribution.from_dict(read_length, oua, groups)
            else:
                part = _equivalence_groups_arrays(
                    start_runs[read_length], loc.transcripts, cleavage_model,
                    read_length, oua,
                )
            found = index[(read_length, oua, signature)] = len(contributions)
            contributions.append(part)
        return found

    run_parts: dict = {}
    for run in runs:
        parts = run_parts[run.id] = []
        for read_length in run.cleavage_model.non_zero_lengths:
            for oua in (True, False):
                i = contribution(run.cleavage_model, int(read_length), oua)
                if contributions[i].n_groups:
                    parts.append(i)
    return _assemble(contributions, [run.id for run in runs], run_parts)


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
