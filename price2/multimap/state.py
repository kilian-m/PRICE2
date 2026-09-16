"""The EM state in ``price.db``: its lifecycle, and what a worker reads and writes per locus.

``multimap_slot_base``
    Per-slot baseline weight = number of MMG reads passing through it
    (the full-count reference the worker subtracts), as canonical-order
    arrays per locus (:func:`price2.multimap.linkage.slot_base_blob`).
``group_weights``
    Per-iteration fractional slot weights, produced by the E-step and
    read by the M-step workers.
``group_lambdas``
    Per-iteration per-slot origin rates ``λ``, produced by the M-step
    workers and consumed by the E-step.
``locus_activities``
    Per-iteration per-locus activity matrices (keyed by ``rgr.id``) used
    to warm-start the next M-step.

``group_weights`` and ``group_lambdas`` are bare ``float64`` buffers over a
locus's slots in canonical order (see :mod:`price2.multimap.linkage`); only
the baseline keeps its keys, since the workers need them.
"""

from __future__ import annotations

import sqlite3 as sql

import numpy as np

from price2 import database
from price2.multimap.linkage import LocusSlots, _run_index, run_index_from


def reset_em_state(db_path: str) -> None:
    """Clear per-iteration EM state and re-seed iteration-0 weights.

    Called at the start of every EM run (warm or cold).  The *linkage*
    (``multimap_linkage.npz`` and ``multimap_slot_base``) depends only on
    the collected alignments and is preserved; the *iteration* tables (``group_lambdas``,
    ``locus_activities``, ``group_weights``) are wiped so a warm re-run
    cannot consume a previous run's stale λ/weights/activities for a slot
    that is not re-emitted this run.  Iteration-0 ``group_weights`` are
    re-seeded from the slot baseline (weight = base → classic full
    counts) so the first M-step reproduces classic behaviour.

    Parameters
    ----------
    db_path : str
        Path to ``price.db`` (must already carry the slot baselines).
    """
    with database.connect(db_path, commit=True) as db:
        cur = db.cursor()
        cur.execute("DELETE FROM group_lambdas")
        cur.execute("DELETE FROM locus_activities")
        cur.execute("DELETE FROM group_weights")
        cur.execute("DELETE FROM prepared_loci")
        # Databases collected before ``prepared_loci_cache`` existed lack it.
        database.create_prepared_cache_table(cur)
        cur.execute("DELETE FROM prepared_loci_cache")
        cur.executemany(
            "INSERT INTO group_weights VALUES (0, ?, ?)",
            _baseline_weight_rows(cur),
        )


def _baseline_weight_rows(cur: sql.Cursor) -> list:
    """``(locus_id, weight_blob)`` seeding iteration-0 weights from the baseline.

    Iteration-0 weights equal the baseline (full counts — classic behaviour),
    written as a dense ``float64`` buffer in canonical slot order.  Reads
    through the caller's cursor: ``build_multimap_index`` calls this inside its
    write transaction, where a second connection could not see the rows it just
    inserted.
    """
    run_index = run_index_from(cur)
    return [
        (locus_id, LocusSlots.from_blob(blob, run_index).base.tobytes())
        for locus_id, blob in cur.execute(
            "SELECT locus_id, base_blob FROM multimap_slot_base"
        ).fetchall()
    ]


def slot_locus_ids(db_path: str) -> set:
    """Return the set of locus ids that carry at least one multimap slot.

    Loci absent from this set have no multimapping reads to reassign, so
    their response ``y`` and activities do not change across EM
    iterations; the light M-step fan-out can skip them entirely and only
    compute them once, in the final full pass.

    Parameters
    ----------
    db_path : str
        Path to ``price.db``.

    Returns
    -------
    set of str
        Locus ids present in ``multimap_slot_base``.
    """
    with database.connect(db_path) as db:
        return {
            loc_id
            for loc_id, in db.execute(
                "SELECT DISTINCT locus_id FROM multimap_slot_base"
            )
        }


def em_resume_point(db_path: str) -> tuple[int, set] | None:
    """Locate the EM iteration an interrupted run should continue from.

    The EM checkpoints itself: :func:`e_step` writes the next iteration's
    ``group_weights`` and then deletes everything it has consumed, so the
    database is left holding exactly the inputs of one M-step — its
    fractional weights, and the previous iteration's ``locus_activities``
    as a warm start.  The iteration those weights belong to is therefore
    the one to run next, and the loci that already wrote their activities
    for it are the ones that M-step had finished before the interruption.

    Parameters
    ----------
    db_path : str
        Path to ``price.db``.

    Returns
    -------
    tuple[int, set] or None
        ``(iteration, finished_locus_ids)``, or ``None`` when no weights
        are stored — nothing to resume, the EM starts from scratch.
    """
    with database.connect(db_path) as db:
        cur = db.cursor()
        if not database.table_exists(cur, "group_weights"):
            return None  # never ran an EM here
        row = cur.execute("SELECT MAX(iteration) FROM group_weights").fetchone()
        if row is None or row[0] is None:
            return None
        iteration = int(row[0])
        finished = {
            locus_id
            for locus_id, in cur.execute(
                "SELECT locus_id FROM locus_activities WHERE iteration = ?",
                (iteration,),
            )
        }
        return iteration, finished


def load_locus_mm_data(
    db_path: str, locus_id: str, iteration: int
) -> LocusSlots | None:
    """Load a locus's multimapping slots with their weights at an iteration.

    Parameters
    ----------
    db_path : str
        Path to ``price.db``.
    locus_id : str
        Locus whose multimapping slots are requested.
    iteration : int
        EM iteration whose ``group_weights`` should be used.

    Returns
    -------
    LocusSlots or None
        ``None`` when the locus has no multimapping slot.  A locus whose
        weight row is missing for the iteration (should not happen) gets
        its baseline, i.e. full weights.
    """
    with database.connect(db_path) as db:
        base_row = db.execute(
            "SELECT base_blob FROM multimap_slot_base WHERE locus_id = ?",
            (locus_id,),
        ).fetchone()
        if base_row is None:
            return None
        w_row = db.execute(
            "SELECT weight_blob FROM group_weights "
            "WHERE locus_id = ? AND iteration = ?",
            (locus_id, iteration),
        ).fetchone()
    return LocusSlots.from_blob(
        base_row[0], _run_index(db_path), w_row[0] if w_row is not None else None
    )


def load_warm_activities(
    db_path: str, locus_id: str, iteration: int
) -> dict | None:
    """Load the activity matrix persisted for ``iteration`` at a locus.

    Parameters
    ----------
    db_path : str
        Path to ``price.db``.
    locus_id : str
        Locus identifier.
    iteration : int
        Iteration whose activities are requested (the previous M-step).

    Returns
    -------
    dict or None
        ``{rgr_id: numpy.ndarray of shape (num_runs,)}`` or ``None`` when
        no activities were stored for that iteration (e.g. iteration 0).
    """
    with database.connect(db_path) as db:
        row = db.execute(
            "SELECT activities_blob FROM locus_activities "
            "WHERE locus_id = ? AND iteration = ?",
            (locus_id, iteration),
        ).fetchone()
    if row is None:
        return None
    return database.decompress_blob(row[0])


def write_locus_em_output(
    db_path: str,
    locus_id: str,
    iteration: int,
    activities: dict,
    lambdas: list,
    slots: LocusSlots | None,
) -> None:
    """Persist a light M-step's activities and per-slot ``λ`` values.

    Parameters
    ----------
    db_path : str
        Path to ``price.db``.
    locus_id : str
        Locus identifier.
    iteration : int
        EM iteration that produced these values.
    activities : dict
        ``{rgr_id: numpy.ndarray}`` activity matrix (keyed by stable
        ``rgr.id`` so it survives index re-densification).
    lambdas : list of (run_id, group_key, lam)
        Per-slot origin rates for this locus's multimapping slots.  Slots whose
        reads were filtered out are absent and score ``λ = 0``.
    slots : LocusSlots or None
        The locus's slots, which fix the order λ is written in; ``None`` for
        a locus without multimapping slots.

    Raises
    ------
    FloatingPointError
        When a λ is not finite: the E-step could not normalise it, and
        catching it here names the locus.
    """
    lam_vector = None
    if slots is not None:
        # Dense over the locus's slots, in canonical order, so the E-step can
        # drop it straight into its per-slot vector.
        lam_vector = slots.lambda_vector(lambdas)
        if not np.isfinite(lam_vector).all():
            raise FloatingPointError(
                f"locus {locus_id}: {int((~np.isfinite(lam_vector)).sum())} "
                f"non-finite λ at iteration {iteration}"
            )
    with database.connect(db_path, wal_writer=True, commit=True) as db:
        cur = db.cursor()
        cur.execute(
            "INSERT OR REPLACE INTO locus_activities VALUES (?, ?, ?)",
            (iteration, locus_id, database.compress_blob(activities)),
        )
        if lam_vector is not None:
            cur.execute(
                "INSERT OR REPLACE INTO group_lambdas VALUES (?, ?, ?)",
                (iteration, locus_id, lam_vector.tobytes()),
            )
