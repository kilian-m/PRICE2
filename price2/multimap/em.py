"""The global E-step, run once between two M-step fan-outs."""

from __future__ import annotations

import sqlite3 as sql

import numpy as np

from price2 import database
from price2.multimap.linkage import Linkage, _linkage


def _slot_vector(
    cur: sql.Cursor, table: str, column: str, iteration: int, link: Linkage
) -> np.ndarray:
    """Gather one iteration's per-locus blobs into a dense per-slot vector.

    *table* holds one bare ``float64`` buffer per locus in canonical slot
    order; *link* says where each locus's slots sit.  Loci without a row
    stay zero.
    """
    vector = np.zeros(link.n_slots, dtype=np.float64)
    cur.execute(
        f"SELECT locus_id, {column} FROM {table} WHERE iteration = ?",
        (iteration,),
    )
    for locus_id, blob in cur.fetchall():
        vector[link.locus_slice(link.locus_index[locus_id])] = np.frombuffer(
            blob, dtype=np.float64
        )
    return vector


def e_step(db_path: str, iteration: int) -> float:
    """Recompute fractional slot weights from the just-finished M-step.

    For every multimap group, normalises its members' current origin
    rates ``λ`` across the group's slots and distributes the group's read
    count accordingly, accumulating a new weight per slot.  Writes those
    weights as ``group_weights`` for ``iteration + 1`` and returns a
    convergence metric versus ``iteration``'s weights.

    Parameters
    ----------
    db_path : str
        Path to ``price.db``.
    iteration : int
        Iteration whose ``group_lambdas`` drive the update; new weights
        are written for ``iteration + 1``.

    Returns
    -------
    float
        The L1 fraction of total read mass reassigned this iteration
        (``Σ|w_new − w_old| / Σ w_new``) — 0.0 when there are no multimap
        groups.

    Notes
    -----
    Everything is addressed by integer slot id against the cached static
    linkage (see :func:`_linkage`), so the whole update is two ``bincount``
    reductions over the membership rows: one to normalise λ within each
    multimap group, one to accumulate each slot's weight across the groups
    that share it.  λ and the weights travel as bare ``float64`` buffers in
    the canonical per-locus slot order, so no key matching is needed.
    """
    link = _linkage(db_path)
    n_slots = link.n_slots
    if n_slots == 0:
        return 0.0
    member_mmg = link.member_mmg
    member_slot = link.member_slot
    mmg_count = link.mmg_count
    n_groups = mmg_count.size

    with database.connect(db_path) as db:
        cur = db.cursor()
        lam_slot = _slot_vector(cur, "group_lambdas", "lam_blob", iteration, link)
        old_slot = _slot_vector(cur, "group_weights", "weight_blob", iteration, link)

    # Responsibility per membership row: λ / Σλ within its group, or a uniform
    # 1/n split when the group's λ sums to zero (its read fits no ORF anywhere,
    # so it is neither lost nor arbitrarily concentrated).  Written as
    # ``count * (λ/Σλ + b)`` — normalise first, scale by the read count second.
    #
    # The order matters.  Precomputing ``count / Σλ`` as its own factor is one
    # gather cheaper, but a group's λ can sum to a *subnormal* positive value,
    # and that quotient then overflows to ``inf`` even though the weight it
    # scales is bounded by ``count``.  The rows where λ is 0 pick up ``0 * inf``
    # = ``NaN`` from it.  Dividing at the row keeps every intermediate inside
    # the bound the maths already guarantees: λ/Σλ ≤ 1 for non-negative λ.
    lam_cell = lam_slot[member_slot]
    lam_sum = np.bincount(member_mmg, weights=lam_cell, minlength=n_groups)
    n_cells = np.bincount(member_mmg, minlength=n_groups)
    positive = lam_sum > 0.0
    denom = np.where(positive, lam_sum, 1.0)
    offset = np.where(positive, 0.0, 1.0 / np.maximum(n_cells, 1))
    weight_cell = (
        lam_cell / denom[member_mmg] + offset[member_mmg]
    ) * mmg_count[member_mmg]

    # New weight per slot = Σ contributions across the MMGs sharing it.
    new_slot = np.bincount(member_slot, weights=weight_cell, minlength=n_slots)
    del lam_cell, weight_cell

    # Convergence = Σ|w_new − w_old| / Σ w_new — a total-variation measure
    # robust to the slot count.  From iteration 1 on the total mass is
    # conserved (each read's weight sums to one); iteration 0's large value
    # reflects the one-off removal of the classic full-weight double count.
    total_mass = float(new_slot.sum())
    moved = float(np.abs(new_slot - old_slot).sum())

    # A non-finite weight must stop the EM, and it takes an explicit check to
    # make it do so: ``NaN > 0`` is False, so a single NaN anywhere in the
    # slots would fall through to the ``else`` below and report a convergence
    # metric of exactly 0.0 — indistinguishable from a perfectly converged
    # run, and enough to end the loop after one iteration.
    if not (np.isfinite(total_mass) and np.isfinite(moved)):
        n_bad = int((~np.isfinite(new_slot)).sum())
        raise FloatingPointError(
            f"the E-step produced {n_bad} non-finite slot weight(s) of "
            f"{n_slots} at iteration {iteration}; refusing to derive a "
            f"convergence metric from them. The new weights were not written."
        )
    rel = moved / total_mass if total_mass > 0 else 0.0

    # Write new weights, then prune spent state.  Only weights[it+1] (the next
    # M-step's response) and locus_activities[it] (its warm start) are still
    # needed.
    it_next = iteration + 1
    weight_rows = [
        (it_next, locus_id, new_slot[link.locus_slice(i)].tobytes())
        for locus_id, i in link.locus_index.items()
    ]
    with database.connect(db_path, wal_writer=True, commit=True) as db:
        cur = db.cursor()
        cur.execute("DELETE FROM group_weights WHERE iteration = ?", (it_next,))
        cur.executemany("INSERT INTO group_weights VALUES (?, ?, ?)", weight_rows)
        cur.execute("DELETE FROM group_weights WHERE iteration <= ?", (iteration,))
        cur.execute("DELETE FROM group_lambdas WHERE iteration <= ?", (iteration,))
        cur.execute(
            "DELETE FROM locus_activities WHERE iteration < ?", (iteration,)
        )
    return rel
