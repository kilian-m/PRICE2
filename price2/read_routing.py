"""From reads to the design matrix of a locus.

Loads a locus's collapsed reads, decides which regions each read is
compatible with and in which frame and coverage position, assigns reads to
their equivalence groups, and turns the groups into the sparse design matrix
the solver sees.  :class:`EgRoutingCache` freezes the weight-independent part
of that work so the multimapping EM can repeat it as a few array operations.

All functions take the :class:`~price2.locus.Locus` they operate on as their
first argument and update its state in place.
"""

from __future__ import annotations

import bisect
from typing import TYPE_CHECKING

import HTSeq
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from price2 import database, multimap
from price2.coverage_position import CoveragePosition
from price2.equivalence_groups import CELL_CODES, NO_FRAME
from price2.genomic_region import GenomicRegion
from price2.ribo_seq_alignment import RiboSeqAlignment
from price2.ribo_seq_run import RiboSeqRun

if TYPE_CHECKING:
    from price2.locus import Locus

# The ``frame_code * 3 + coverage_position`` part of a packed cell (see
# ``equivalence_groups.pack_cell``): a NOISE region has no frame and only a
# middle position; an ORF's codes are ``frame * 3 + _START / _MIDDLE / _STOP``.
_START = CoveragePosition.start.value
_MIDDLE = CoveragePosition.middle.value
_STOP = CoveragePosition.stop.value
_NOISE_MIDDLE = NO_FRAME * 3 + _MIDDLE


class EgRoutingCache:
    """Weight-independent per-locus state reused across EM iterations.

    Between multimapping-EM iterations only the fractional read weights
    change: the RGR set, the equivalence groups and hence the design matrix
    ``X`` are fixed (the light M-step runs with ``prune=False``).  This
    caches everything a light M-step would otherwise re-derive from the
    reads — the read → design-matrix-row routing and the geometry of ``X`` —
    so an iteration reduces to a weighted ``bincount`` plus a vectorised
    rebuild of ``X.data``.

    Invalidated (set to ``None`` on the locus) whenever the RGR set changes,
    since that re-indexes RGRs and collapses equivalence groups.

    It holds only arrays, ints and strings — no RGR, transcript or
    equivalence-group objects — so it can be pickled on its own.  A light
    M-step therefore loads just this blob (a few numpy ``memcpy``s) instead
    of unpickling the locus's whole object graph, which is ~96% of the cost
    of restoring a prepared locus.

    Attributes
    ----------
    n_rows : int
        Number of design-matrix rows (non-empty equivalence groups).
    n_reads : dict[str, int]
        Reads per run at build time; guards against a changed read order.
    eg_row : dict[str, numpy.ndarray]
        Per run, the row each read feeds: ``>=0`` a row index, ``-1`` the
        read's key is absent from ``egs`` (uncounted), ``-2`` the read is
        compatible with no RGR.
    counts0 : dict[str, numpy.ndarray]
        Per run, each read's raw (unweighted) count.
    mm_idx, mm_gk, mm_base : dict[str, numpy.ndarray]
        Per run, the read positions that carry a multimapping slot, their
        group keys, and their baseline cross-locus mass.
    slot_gk, slot_rl, slot_oua, slot_nnz : dict[str, numpy.ndarray]
        Per run, one entry per multimapping slot: group key, read length,
        untemplated-addition flag, and number of compatible RGR cells.
    slot_rgr, slot_code : dict[str, numpy.ndarray]
        Per run, the flattened slot cells split into RGR index and
        ``frame_code * 3 + coverage_position`` (``divmod(cell, CELL_CODES)``).
    cell_rgr, cell_code : numpy.ndarray
        One entry per design-matrix cell: the RGR index, and
        ``frame_code * 3 + coverage_position`` packed into a byte.
    row_nnz, row_len, row_rl, row_oua, row_run : numpy.ndarray
        Per row: cell count, EG length, read length, untemplated-addition
        flag, and run index.
    num_rgrs : int
        RGR count at build time (design-matrix column blocks).
    rgr_ids : tuple[str, ...]
        RGR identifiers, indexed by ``rgr.index``.
    rgr_lengths : numpy.ndarray
        RGR lengths, indexed by ``rgr.index``.
    """

    __slots__ = (
        "n_rows", "n_reads", "eg_row", "counts0", "mm_idx", "mm_gk",
        "mm_base", "slot_gk", "slot_rl", "slot_oua", "slot_nnz",
        "slot_rgr", "slot_code", "cell_rgr", "cell_code", "row_nnz",
        "row_len", "row_rl", "row_oua", "row_run", "num_rgrs",
        "rgr_ids", "rgr_lengths",
    )

    def __init__(self, **fields: object) -> None:
        for name, value in fields.items():
            setattr(self, name, value)


def load_reads(
    loc, db_path: str, drop_multimappers: bool = False
) -> None:
    """Load reads for this locus from the SQLite database.

    Populates :attr:`rsas_dict` (mapping run id to a list of
    :class:`RiboSeqAlignment` objects) and :attr:`run_read_count`.

    Parameters
    ----------
    db_path : str
        Path to the ``price.db`` SQLite database.
    drop_multimappers : bool, optional
        Discard stored reads that align to more than one genomic locus
        (``NH > 1``) instead of counting them at full weight here.  Set
        when ``multimap_em`` is disabled, which leaves no mechanism for
        spreading such a read over the loci it aligns to.  A database
        collected with ``multimap_em`` disabled holds no multimapping
        reads to begin with, so this is a no-op there; it matters when a
        database collected with the EM enabled is deconvolved without it.
        Note that ``transcript_read_counts`` -- used only for the
        transcript-support filter -- still includes them in that case.
    """
    with database.connect(db_path) as db:
        rows = db.execute(
            "SELECT run_id, reads_blob FROM reads WHERE locus_id = ?",
            (loc.id,),
        ).fetchall()
    loc.run_read_count = {}
    loc.rsas_dict = {}
    chrom = loc.iv.chrom
    strand = loc.iv.strand
    # Memoize GenomicRegion objects by their interval-coordinate signature.
    # Reads are exact-deduplicated within a run at collection time, but the
    # same coordinates recur across runs (typically 40-80% of reads); sharing
    # one immutable GenomicRegion across runs avoids rebuilding its intervals
    # and hash.  Scoped per locus, so it is freed when the locus is done.
    region_cache: dict[tuple, GenomicRegion] = {}
    for run_id, blob in rows:
        reads_df = database.decompress_blob(blob)

        # Vectorized: pull columns into numpy arrays and locate per-read
        # boundaries from is_first_iv, avoiding groupby / iterrows / per-row
        # Series construction (the dominant cost of read loading).
        is_first = reads_df["is_first_iv"].to_numpy()
        starts = reads_df["start"].to_numpy()
        ends = reads_df["end"].to_numpy()
        uas = reads_df["untemplated_addition"].to_numpy()
        uniques = reads_df["unique"].to_numpy()
        counts = reads_df["count"].to_numpy()

        boundaries = np.flatnonzero(is_first)
        read_ends = np.append(boundaries[1:], len(is_first))
        if drop_multimappers:
            keep = uniques[boundaries].astype(bool)
            boundaries = boundaries[keep]
            read_ends = read_ends[keep]

        rsas_run = []
        for b, e in zip(boundaries.tolist(), read_ends.tolist()):
            if e - b == 1:
                sig = (int(starts[b]), int(ends[b]))
            else:
                sig = tuple(
                    (int(starts[j]), int(ends[j])) for j in range(b, e)
                )
            gr = region_cache.get(sig)
            if gr is None:
                intervals = [
                    HTSeq.GenomicInterval(
                        chrom, int(starts[j]), int(ends[j]), strand
                    )
                    for j in range(b, e)
                ]
                gr = GenomicRegion(
                    intervals=intervals, chrom=chrom, strand=strand
                )
                region_cache[sig] = gr
            rsas_run.append(
                RiboSeqAlignment(
                    gr,
                    untemplated_addition=bool(uas[b]),
                    unique=bool(uniques[b]),
                    read_count=int(counts[b]),
                )
            )
        loc.rsas_dict[run_id] = rsas_run

        loc.run_read_count[run_id] = int(
            counts[boundaries].astype(np.int64).sum()
        )


def rgr_compatibility(
    loc: Locus,
    rsa: RiboSeqAlignment,
    run: RiboSeqRun,
    overlap_likelihood_ratio_threshold: float = 0.2,
) -> frozenset[int] | None:
    """Determine which RGRs a read alignment is compatible with.

    For each RGR overlapping the read, compute the reading frame and
    coverage-profile position (start / middle / stop).  Partial
    overlaps are kept only when the cleavage-model probability ratio
    exceeds *overlap_likelihood_ratio_threshold*.  Each combination is
    returned as a packed cell (``equivalence_groups.pack_cell``), so the
    result is the first element of the read's equivalence-group key.

    Parameters
    ----------
    rsa : RiboSeqAlignment
        A single Ribo-seq read alignment.
    run : RiboSeqRun
        The Ribo-seq run that produced *rsa*.
    overlap_likelihood_ratio_threshold : float
        Minimum ratio of partial-overlap cleavage probability to
        full-overlap probability for a partial overlap to be accepted.

    Returns
    -------
    frozenset[int] or None
        The packed ``(rgr, frame, coverage_position)`` cells, or ``None``
        when no compatible RGR is found.
    """

    overlap_transcripts = set(loc.transcripts)

    cells = set()

    bp_starts, bp_ends, bp_sets = loc.transcript_breakpoint_index
    for query_iv in rsa.genomic_region.intervals:
        i = bisect.bisect_right(bp_ends, query_iv.start)
        while i < len(bp_starts) and bp_starts[i] < query_iv.end:
            overlap_transcripts &= bp_sets[i]
            i += 1

    # Hoist per-read / per-run invariants out of the transcript x rgr
    # loops.  ``len(rsa) == len(rsa.genomic_region)`` (a cached value) and
    # the untemplated-addition flag are the same for every candidate RGR,
    # so compute them once.  The full-overlap cleavage likelihood is an
    # exact lookup-table entry (verified bit-identical to
    # ``CleavageModel.pmf`` across the whole domain), so index the LUT
    # directly instead of paying a pmf() call frame per RGR — only the
    # partial-overlap (region-bounded) likelihoods still call pmf.
    pmf = run.cleavage_model.pmf
    cds_lut = run.cleavage_model.cds_lut
    noise_lut = run.cleavage_model.noise_lut
    lut_len = cds_lut.shape[0]
    read_length = len(rsa)
    oua = rsa.untemplated_addition
    oua_i = int(oua)
    thr = overlap_likelihood_ratio_threshold
    in_lut = read_length < lut_len
    noise_cl = noise_lut[read_length, oua_i] if in_lut else 0.0

    for tr in overlap_transcripts:
        try:
            rsa_iv_on_tr = tr.exons.map_to_local(rsa.genomic_region)
        except ValueError:
            continue
        rsa_lo, rsa_hi = rsa_iv_on_tr
        for rgr in tr.rgr_set:
            rgr_lo, rgr_hi = rgr.iv_on_transcript
            # full overlap with orf
            if rgr.type == "NOISE":
                frame = None
                if (rgr_lo <= rsa_lo) and (rgr_hi >= rsa_hi):
                    if noise_cl > 0:
                        cells.add(rgr.index * CELL_CODES + _NOISE_MIDDLE)
                elif (rsa_lo <= rgr_lo <= rsa_hi) or (
                    rsa_lo <= rgr_hi <= rsa_hi
                ):
                    region_start = rgr_lo - rsa_lo
                    region_end = rgr_hi - rsa_lo
                    if (
                        ol := pmf(
                            read_length,
                            oua,
                            frame,
                            region_start=region_start,
                            region_end=region_end,
                        )
                    ) > 0:
                        cl = noise_cl
                        if cl == 0:
                            continue
                        # cl > 0 here, so test ol > thr*cl instead of
                        # ol/cl > thr: same result, no scalar-divide overflow
                        # when cl is a tiny denormal.
                        if ol > thr * cl:
                            cells.add(rgr.index * CELL_CODES + _NOISE_MIDDLE)
            elif rgr.type == "ORF":
                orf = rgr
                if (rgr_lo <= rsa_lo) and (rgr_hi >= rsa_hi):
                    frame = (rsa_lo - rgr_lo) % 3
                    if (cds_lut[read_length, frame, oua_i] if in_lut else 0.0) > 0:
                        cells.add(orf.index * CELL_CODES + frame * 3 + _MIDDLE)
                # part overlap with orf
                elif (rsa_lo <= rgr_lo <= rsa_hi) or (
                    rsa_lo <= rgr_hi <= rsa_hi
                ):
                    frame = (rsa_lo - rgr_lo) % 3
                    orf_cells = orf.index * CELL_CODES + frame * 3
                    cl = cds_lut[read_length, frame, oua_i] if in_lut else 0.0
                    cl_ok = not cl == 0
                    # consider overlap likelihood
                    # compute at which position in the read the orf starts
                    region_start = rgr_lo + 3 - rsa_lo
                    # compute at which position in the read the orf ends
                    region_end = rgr_hi - 3 - rsa_lo
                    if (
                        ol := pmf(
                            read_length,
                            oua,
                            frame,
                            region_start=region_start,
                            region_end=region_end,
                        )
                    ) > 0:
                        # ol > thr*cl avoids the ol/cl scalar-divide overflow
                        # when cl is a tiny denormal (cl_ok guarantees cl > 0);
                        # identical result since thr > 0.
                        if cl_ok and (ol > thr * cl):
                            cells.add(orf_cells + _MIDDLE)
                    # consider coverage profile - start
                    # compute where the orf starts relative to the read
                    start_position = (
                        rgr_lo - rsa_lo,
                        rgr_lo + 3 - rsa_lo,
                    )
                    if (
                        ol := pmf(
                            read_length,
                            oua,
                            frame,
                            region_start=start_position[0],
                            region_end=start_position[1],
                        )
                    ) > 0:
                        if cl_ok and (
                            # ol * run.coverage_model.start_factor / cl
                            # (as ol > thr*cl: avoids scalar-divide overflow
                            #  for denormal cl; cl_ok guarantees cl > 0)
                            ol
                            > thr * cl
                        ):
                            cells.add(orf_cells + _START)

                    # consider coverage profile - stop
                    # compute where the orf ends relative to the read
                    stop_position = (
                        rgr_hi - 3 - rsa_lo,
                        rgr_hi - rsa_lo,
                    )
                    if (
                        ol := pmf(
                            read_length,
                            oua,
                            frame,
                            region_start=stop_position[0],
                            region_end=stop_position[1],
                        )
                    ) > 0:
                        if cl_ok and (
                            # ol * run.coverage_model.stop_factor / cl
                            # (as ol > thr*cl: avoids scalar-divide overflow
                            #  for denormal cl; cl_ok guarantees cl > 0)
                            ol
                            > thr * cl
                        ):
                            cells.add(orf_cells + _STOP)

    if not cells:
        return None
    return frozenset(cells)


def count_well_fitting_reads(loc: Locus, runs: list[RiboSeqRun]) -> None:
    """Count well-fitting reads per RGR and run.

    A read is *well-fitting* when its length and untemplated-addition
    status match a high-probability entry in the run's cleavage
    model.  Results are stored in :attr:`wfr_df` (a DataFrame
    indexed by RGR id with one column per run).

    Parameters
    ----------
    runs : list[RiboSeqRun]
        Ribo-seq runs to process.
    """
    well_fitting_rcs = {}
    for run in runs:
        well_fitting_rcs[run.id] = {}
        for rgr in loc.rgrs:
            if rgr.type == "ORF":
                well_fitting_rcs[run.id][rgr.id] = 0
    # ORF id by ``rgr.index`` (``None`` for NOISE), to resolve the cells.
    orf_id_of = [rgr.id if rgr.type == "ORF" else None for rgr in loc.rgrs]
    for run in runs:
        well_fitting_indices = run.cleavage_model.get_high_prob_indices()
        well_fitting_length_oua = {(l, oua) for l, f, oua in well_fitting_indices}
        for rsa in loc.rsas_dict[run.id]:
            if (
                len(rsa),
                int(rsa.untemplated_addition),
            ) not in well_fitting_length_oua:
                continue
            cells = rgr_compatibility(loc, rsa, run)
            if not cells:
                continue

            # One RGR can appear under several coverage positions for the
            # same read (a read spanning a short ORF overlaps both its
            # start- and stop-codon regions), so deduplicate before
            # counting: a read contributes its count once per RGR.
            orf_ids = {orf_id_of[cell // CELL_CODES] for cell in cells}
            orf_ids.discard(None)
            for rgr_id in orf_ids:
                well_fitting_rcs[run.id][rgr_id] += rsa.read_count

    loc.wfr_df = (
        pd.DataFrame.from_dict(well_fitting_rcs).replace(np.nan, 0).astype(np.int32)
    )


def assign_reads_to_egs(
    loc: Locus,
    runs: list[RiboSeqRun],
    mm_data: dict | None = None,
    build_cache: bool = False,
) -> None:
    """Assign reads to their equivalence groups.

    Each read is matched to its ``(cells, read_length, oua)`` key (see
    :mod:`price2.equivalence_groups`) and added to the corresponding
    :class:`EquivalenceGroup`.
    Reads whose key is absent (due to earlier filtering) are counted
    in :attr:`uncounted_reads`.

    When ``mm_data`` is supplied (multimapping EM mode), each
    cross-locus multimapping read contributes a *fractional* count at
    this locus instead of its full count.  For a slot with collapsed
    count ``c``, baseline cross-locus mass ``base`` and current
    fractional weight ``weight`` the effective contribution is
    ``c - base + weight`` (single-slot multimappers keep full weight;
    cross-locus reads are down-weighted so their total mass across all
    their loci sums to one).  The routing needed by the E-step is
    cached in :attr:`mm_slots`.

    Parameters
    ----------
    runs : list[RiboSeqRun]
        Ribo-seq runs to process.
    mm_data : dict, optional
        ``{run_id: {group_key: (base, weight)}}`` for this locus's
        multimapping slots, or ``None`` for classic full-weight
        counting.
    build_cache : bool, optional
        Record the weight-independent read routing and design-matrix
        geometry in :attr:`eg_cache` while assigning.  Later EM
        iterations then take :meth:`_assign_reads_from_cache`, which
        skips the per-read :func:`rgr_compatibility` recomputation.
    """
    cache = loc.eg_cache
    if cache is not None and not build_cache:
        _assign_reads_from_cache(loc, runs, mm_data, cache)
        return

    # Row layout of the design matrix, in the order ``egs_to_sparse``
    # emits rows.  Built up-front so each read can record the row it
    # feeds instead of re-deriving its equivalence-group key later.
    if build_cache:
        row_of_key: dict = {}
        n_rows = 0
        for run in runs:
            d: dict = {}
            for key in loc.egs[run]:
                if not key[0]:
                    continue
                d[key] = n_rows
                n_rows += 1
            row_of_key[run.id] = d
        eg_row: dict = {}
        counts0: dict = {}
        mm_idx: dict = {}
        mm_gk: dict = {}
        mm_base: dict = {}

    # {run_id: {group_key: (cells, read_length, oua)}} —
    # the compatibility routing the E-step needs to recompute λ.
    loc.mm_slots = {run.id: {} for run in runs}
    for run in runs:
        run_id = run.id
        run_mm = mm_data.get(run_id) if mm_data else None

        if build_cache:
            n_reads = len(loc.rsas_dict[run_id])
            rows_arr = np.full(n_reads, -2, dtype=np.int32)
            counts_arr = np.zeros(n_reads, dtype=np.float64)
            run_rows = row_of_key[run_id]
            mi: list = []
            mg: list = []
            mb: list = []

        for i, rsa in enumerate(loc.rsas_dict[run_id]):
            cells = rgr_compatibility(loc, rsa, run)
            read_count = rsa.read_count

            if build_cache:
                counts_arr[i] = rsa.read_count
                if cells:
                    rows_arr[i] = run_rows.get(
                        (
                            cells,
                            len(rsa),
                            rsa.untemplated_addition,
                        ),
                        -1,
                    )

            if run_mm is not None and not rsa.unique:
                gk = multimap.alignment_group_key(rsa)
                slot = run_mm.get(gk)
                if slot is not None:
                    base, weight = slot
                    # ``base`` is summed over the spilled alignments at
                    # indexing time, independently of the collapsed slot
                    # count ``read_count``; floor the non-cross-locus
                    # remainder at zero so that any disagreement between
                    # the two can never drive the Poisson response
                    # negative.
                    read_count = max(0.0, read_count - base) + weight
                    if build_cache:
                        mi.append(i)
                        mg.append(gk)
                        mb.append(base)
                    if cells:
                        loc.mm_slots[run_id][gk] = (
                            cells,
                            len(rsa),
                            rsa.untemplated_addition,
                        )

            if not cells:
                continue

            try:
                loc.egs[run][
                    (cells, len(rsa), rsa.untemplated_addition)
                ].read_count += read_count
                run.read_count += read_count
            except KeyError:
                loc.uncounted_reads += read_count

            try:
                loc.read_counts[run] += read_count
            except KeyError:
                loc.read_counts[run] = read_count

        if build_cache:
            eg_row[run_id] = rows_arr
            counts0[run_id] = counts_arr
            mm_idx[run_id] = np.array(mi, dtype=np.int32)
            mm_gk[run_id] = np.array(mg, dtype=np.int64)
            mm_base[run_id] = np.array(mb, dtype=np.float64)

    loc.counted_reads = {}
    for run in runs:
        loc.counted_reads[run.id] = 0
        for v in loc.egs[run].values():
            loc.counted_reads[run.id] += v.read_count

    if build_cache:
        loc.eg_cache = _make_eg_cache(loc, 
            runs, n_rows, eg_row, counts0, mm_idx, mm_gk, mm_base
        )


def _make_eg_cache(
    loc: Locus,
    runs: list[RiboSeqRun],
    n_rows: int,
    eg_row: dict,
    counts0: dict,
    mm_idx: dict,
    mm_gk: dict,
    mm_base: dict,
) -> EgRoutingCache:
    """Freeze the design-matrix geometry into a compact cell encoding.

    Storing ``X`` itself would add ~1.1 MB per locus; instead the packed
    cells of the equivalence-group keys are stored split into
    ``(rgr index, frame_code*3 + coverage position)`` plus per-row
    ``(length, read length, oua, run)``, from which ``X.data`` is recomputed
    with a handful of vectorised look-ups.

    The multimapping-slot routing is encoded the same way.
    """
    cell_rgr: list = []
    cell_code: list = []
    row_nnz: list = []
    row_len: list = []
    row_rl: list = []
    row_oua: list = []
    row_run: list = []
    for run_index, run in enumerate(runs):
        for (cells, read_length, oua), eg in loc.egs[run].items():
            if not cells:
                continue
            row_nnz.append(len(cells))
            row_len.append(eg.length)
            row_rl.append(read_length)
            row_oua.append(int(oua))
            row_run.append(run_index)
            for cell in cells:
                rgr_index, code = divmod(cell, CELL_CODES)
                cell_rgr.append(rgr_index)
                cell_code.append(code)

    slot_gk: dict = {}
    slot_rl: dict = {}
    slot_oua: dict = {}
    slot_nnz: dict = {}
    slot_rgr: dict = {}
    slot_code: dict = {}
    for run in runs:
        gks, rls, ouas, nnzs, srgr, scode = [], [], [], [], [], []
        for gk, (rfc, read_length, oua) in loc.mm_slots[run.id].items():
            gks.append(gk)
            rls.append(read_length)
            ouas.append(int(oua))
            nnzs.append(len(rfc))
            for cell in rfc:
                rgr_index, code = divmod(cell, CELL_CODES)
                srgr.append(rgr_index)
                scode.append(code)
        slot_gk[run.id] = np.array(gks, dtype=np.int64)
        slot_rl[run.id] = np.array(rls, dtype=np.int32)
        slot_oua[run.id] = np.array(ouas, dtype=np.uint8)
        slot_nnz[run.id] = np.array(nnzs, dtype=np.int32)
        slot_rgr[run.id] = np.array(srgr, dtype=np.int32)
        slot_code[run.id] = np.array(scode, dtype=np.uint8)

    num_rgrs = len(loc.rgrs)
    rgr_lengths = np.fromiter(
        (len(rgr) for rgr in loc.rgrs), dtype=np.int64, count=num_rgrs
    )

    return EgRoutingCache(
        n_rows=n_rows,
        n_reads={r.id: len(loc.rsas_dict[r.id]) for r in runs},
        eg_row=eg_row,
        counts0=counts0,
        mm_idx=mm_idx,
        mm_gk=mm_gk,
        mm_base=mm_base,
        slot_gk=slot_gk,
        slot_rl=slot_rl,
        slot_oua=slot_oua,
        slot_nnz=slot_nnz,
        slot_rgr=slot_rgr,
        slot_code=slot_code,
        num_rgrs=num_rgrs,
        rgr_ids=tuple(rgr.id for rgr in loc.rgrs),
        rgr_lengths=rgr_lengths,
        cell_rgr=np.array(cell_rgr, dtype=np.int32),
        cell_code=np.array(cell_code, dtype=np.uint8),
        row_nnz=np.array(row_nnz, dtype=np.int32),
        row_len=np.array(row_len, dtype=np.int64),
        row_rl=np.array(row_rl, dtype=np.int32),
        row_oua=np.array(row_oua, dtype=np.uint8),
        row_run=np.array(row_run, dtype=np.uint8),
    )


def _assign_reads_from_cache(
    loc: Locus,
    runs: list[RiboSeqRun],
    mm_data: dict | None,
    cache: EgRoutingCache,
) -> None:
    """Re-derive the response ``y`` from cached routing (no per-read work).

    Only the fractional weights change between EM iterations, so the
    per-read equivalence-group routing recorded in *cache* stays valid and
    the response reduces to a weighted ``bincount`` over rows.
    """
    y = np.zeros(cache.n_rows, dtype=np.float64)
    loc.read_counts = {}
    loc.counted_reads = {}
    loc.uncounted_reads = 0.0

    rsas = loc.rsas_dict
    for run in runs:
        run_id = run.id
        # The reads are only needed for this length sanity check here; an
        # intermediate light M-step skips loading them (the response is
        # rebuilt from ``cache.counts0``), so the check only runs when the
        # reads were actually loaded.
        if (
            rsas is not None
            and run_id in rsas
            and len(rsas[run_id]) != cache.n_reads[run_id]
        ):
            raise RuntimeError(
                f"locus {loc.id}: cached routing has "
                f"{cache.n_reads[run_id]} reads for run {run_id} but "
                f"{len(rsas[run_id])} were loaded"
            )
        counts = cache.counts0[run_id].copy()
        run_mm = mm_data.get(run_id) if mm_data else None
        idx = cache.mm_idx[run_id]
        if run_mm is not None and idx.size:
            weights = np.fromiter(
                (run_mm[gk][1] for gk in cache.mm_gk[run_id]),
                dtype=np.float64,
                count=idx.size,
            )
            counts[idx] = (
                np.maximum(0.0, counts[idx] - cache.mm_base[run_id]) + weights
            )

        rows = cache.eg_row[run_id]
        counted = rows >= 0
        y += np.bincount(
            rows[counted], weights=counts[counted], minlength=cache.n_rows
        )
        counted_sum = float(counts[counted].sum())
        uncounted = rows == -1
        loc.uncounted_reads += float(counts[uncounted].sum())
        loc.counted_reads[run_id] = counted_sum
        run.read_count += counted_sum
        # ``read_counts`` gains an entry only when the run has at least one
        # read compatible with some RGR, matching the uncached path.
        compatible = counted | uncounted
        if compatible.any():
            loc.read_counts[run] = float(counts[compatible].sum())

    # Downstream consumers (the final pass's pruning, the likelihood-ratio
    # test, ``estimate_activities``) read counts off the EG objects.  A
    # light M-step loads only the cache, so there are no EG objects and
    # nothing downstream that reads them.
    if loc.egs:
        i = 0
        for run in runs:
            for key, eg in loc.egs[run].items():
                if not key[0]:
                    continue
                eg.read_count = y[i]
                i += 1
    loc._eg_y = y


def design_matrix_from_cache(
    cache: EgRoutingCache,
    cm_lut: np.ndarray,
    coverage_params: np.ndarray,
    num_runs: int,
) -> csr_matrix:
    """Rebuild ``X`` from the cached cell encoding, fully vectorised."""
    nnz_per_row = cache.row_nnz
    run_c = np.repeat(cache.row_run, nnz_per_row)
    rl_c = np.repeat(cache.row_rl, nnz_per_row)
    oua_c = np.repeat(cache.row_oua, nnz_per_row)
    len_c = np.repeat(cache.row_len, nnz_per_row)
    code = cache.cell_code.astype(np.int64)
    frame_c, cov_c = code // 3, code % 3

    data = (
        len_c
        * cm_lut[run_c, rl_c, frame_c, oua_c]
        * coverage_params[run_c, cov_c]
    )
    rows_idx = np.repeat(np.arange(cache.n_rows, dtype=np.int64), nnz_per_row)
    cols_idx = cache.cell_rgr.astype(np.int64) * num_runs + run_c
    # COO construction (as in ``egs_to_sparse``) so that a row touching the
    # same RGR at several coverage positions sums those cells.
    return csr_matrix(
        (data, (rows_idx, cols_idx)),
        shape=(cache.n_rows, cache.num_rgrs * num_runs),
        dtype=np.float64,
    )


def egs_to_sparse(
    locus_egs: dict,
    runs: list[RiboSeqRun],
    cm_lut: np.ndarray,
    coverage_params: np.ndarray,
    num_rgrs: int,
    num_runs: int,
) -> tuple[csr_matrix, np.ndarray]:
    """Convert locus equivalence groups to a sparse CSR design matrix.

    Builds the design matrix ``X`` and response vector ``y`` for the
    identity-link Poisson GLM directly from the locus's native EG
    dictionary, avoiding Numba typed-List construction entirely.

    Each row corresponds to one ``(EG, run)`` pair.  Column
    ``rgr_index * num_runs + run_index`` receives the value::

        length * cm_lut[run, read_length, frame, oua] * coverage_params[run, cov_pos]

    Parameters
    ----------
    locus_egs : dict
        ``Locus.egs`` — mapping from :class:`RiboSeqRun` to a dict of
        ``(cells, read_length, oua) -> EquivalenceGroup`` (see
        :mod:`price2.equivalence_groups` for the key).
    runs : list[RiboSeqRun]
        Ordered list of runs (determines run indices).
    cm_lut : np.ndarray, shape ``(num_runs, max_read_len, 4, 2)``
        Cleavage-model look-up table.
    coverage_params : np.ndarray, shape ``(num_runs, 3)``
        Coverage-model factors.
    num_rgrs : int
        Number of RGRs (= number of column groups).
    num_runs : int
        Number of Ribo-seq runs.

    Returns
    -------
    X : csr_matrix, shape ``(n_EGs_total, num_rgrs * num_runs)``
        Sparse design matrix.
    y : np.ndarray, shape ``(n_EGs_total,)``
        Observed read counts.
    """
    # Pass 1: count rows (non-empty EGs) and total non-zeros so we can
    # pre-size numpy arrays.  Building Python int/float lists with one
    # entry per CSR cell costs ~80 B/cell on CPython and dominates peak
    # RSS at this stage; numpy buffers are 12 B/cell instead.
    n_rows = 0
    nnz = 0
    for run in runs:
        for (cells, _, _), _ in locus_egs[run].items():
            sz = len(cells)
            if sz == 0:
                continue
            n_rows += 1
            nnz += sz

    rows_idx = np.empty(nnz, dtype=np.int64)
    cols_idx = np.empty(nnz, dtype=np.int64)
    data = np.empty(nnz, dtype=np.float64)
    y = np.empty(n_rows, dtype=np.float64)

    # Pass 2: populate.
    row = 0
    cell = 0
    for run_index, run in enumerate(runs):
        for (cells, read_length, oua), eg in locus_egs[run].items():
            if not cells:
                continue
            y[row] = eg.read_count
            length = eg.length
            oua_int = int(oua)
            for packed in cells:
                rgr_index, code = divmod(packed, CELL_CODES)
                frame_code, cov_pos = divmod(code, 3)
                rows_idx[cell] = row
                cols_idx[cell] = rgr_index * num_runs + run_index
                data[cell] = (
                    length
                    * cm_lut[run_index, read_length, frame_code, oua_int]
                    * coverage_params[run_index, cov_pos]
                )
                cell += 1
            row += 1

    n_cols = num_rgrs * num_runs
    X = csr_matrix(
        (data, (rows_idx, cols_idx)), shape=(n_rows, n_cols), dtype=np.float64
    )
    return X, y


def multimap_lambdas(loc: Locus, runs: list[RiboSeqRun]) -> list:
    """Compute the per-slot origin rate ``λ`` for multimapping reads.

    For each recorded multimapping slot, ``λ`` is the read's design-
    matrix row *without* the geometric ``length`` factor dotted with
    the current activities — i.e. ``Σ cleavage · coverage · activity``
    over the read's compatible ORFs, which is the per-read expected
    rate the E-step normalises across a read's loci
    (``λ = δ_EG / length_EG``).

    Parameters
    ----------
    runs : list[RiboSeqRun]
        Ribo-seq runs, in the order used to build :attr:`result`.

    Returns
    -------
    list of (run_id, group_key, lam)
        One entry per multimapping slot recorded at this locus.
    """
    cache = loc.eg_cache
    if cache is None and not any(loc.mm_slots.get(run.id) for run in runs):
        return []

    num_runs = len(runs)
    cm_lut = np.zeros(
        (num_runs, runs[0].cleavage_model.cds_lut.shape[0], 4, 2)
    )
    for i, run in enumerate(runs):
        cm_lut[i, :, 3, :] = run.cleavage_model.noise_lut
        cm_lut[i, :, :3, :] = run.cleavage_model.cds_lut

    coverage_params = np.zeros((num_runs, 3))
    for i, run in enumerate(runs):
        coverage_params[i, 0] = run.coverage_model.start_factor
        coverage_params[i, 1] = 1
        coverage_params[i, 2] = run.coverage_model.stop_factor

    if cache is not None:
        return _multimap_lambdas_from_cache(loc, 
            runs, cache, cm_lut, coverage_params
        )

    out = []
    for run_index, run in enumerate(runs):
        for gk, (rfc, read_length, oua) in loc.mm_slots[run.id].items():
            oua_int = int(oua)
            lam = 0.0
            for cell in rfc:
                rgr_index, code = divmod(cell, CELL_CODES)
                frame_code, cov_pos = divmod(code, 3)
                lam += (
                    cm_lut[run_index, read_length, frame_code, oua_int]
                    * coverage_params[run_index, cov_pos]
                    * loc.result[rgr_index, run_index]
                )
            out.append((run.id, gk, float(lam)))
    return out


def _multimap_lambdas_from_cache(
    loc: Locus,
    runs: list[RiboSeqRun],
    cache: EgRoutingCache,
    cm_lut: np.ndarray,
    coverage_params: np.ndarray,
) -> list:
    """Vectorised :meth:`compute_multimap_lambdas` over the cached slots."""
    out: list = []
    for run_index, run in enumerate(runs):
        gks = cache.slot_gk[run.id]
        if gks.size == 0:
            continue
        nnz = cache.slot_nnz[run.id]
        read_length = np.repeat(cache.slot_rl[run.id], nnz)
        oua = np.repeat(cache.slot_oua[run.id], nnz)
        code = cache.slot_code[run.id].astype(np.int64)
        contribution = (
            cm_lut[run_index, read_length, code // 3, oua]
            * coverage_params[run_index, code % 3]
            * loc.result[cache.slot_rgr[run.id], run_index]
        )
        slot_of_cell = np.repeat(np.arange(gks.size), nnz)
        lam = np.bincount(
            slot_of_cell, weights=contribution, minlength=gks.size
        )
        out.extend(
            (run.id, int(gk), float(value)) for gk, value in zip(gks, lam)
        )
    return out
