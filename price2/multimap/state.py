"""The EM state in ``price.db``: its lifecycle, and what a worker reads and writes per locus.

``multimap_slot_base``
    Per-slot baseline weight = number of MMG reads passing through it
    (the full-count reference the worker subtracts).
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
from collections import defaultdict

import numpy as np

from price2 import database
from price2.multimap.linkage import _run_index, _slot_keys


def reset_em_state(db_path: str) -> None:
    """Clear per-iteration EM state and re-seed iteration-0 weights.

    Called at the start of every EM run (warm or cold).  The *linkage*
    tables (``multimap_groups``, ``multimap_group_slots``,
    ``multimap_slot_base``) depend only on the collected alignments and
    are preserved; the *iteration* tables (``group_lambdas``,
    ``locus_activities``, ``group_weights``) are wiped so a warm re-run
    cannot consume a previous run's stale λ/weights/activities for a slot
    that is not re-emitted this run.  Iteration-0 ``group_weights`` are
    re-seeded from the slot baseline (weight = base → classic full
    counts) so the first M-step reproduces classic behaviour.

    Parameters
    ----------
    db_path : str
        Path to ``price.db`` (must already contain the linkage tables).
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
    run_ids = sorted(r for (r,) in cur.execute("SELECT run_id FROM runs").fetchall())
    run_index = {run_id: i for i, run_id in enumerate(run_ids)}
    rows = []
    for locus_id, blob in cur.execute(
        "SELECT locus_id, base_blob FROM multimap_slot_base"
    ).fetchall():
        base_map = database.unpickle_blob(blob)
        keys = _slot_keys(base_map, run_index)
        weights = np.fromiter(
            (base_map[k] for k in keys), dtype=np.float64, count=len(keys)
        )
        rows.append((locus_id, weights.tobytes()))
    return rows


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
) -> dict:
    """Load per-slot ``(base, weight)`` for a locus at a given iteration.

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
    dict
        ``{run_id: {group_key: (base, weight)}}``.  A slot present in the
        baseline but missing a weight row (should not happen) falls back
        to its baseline (full weight).
    """
    with database.connect(db_path) as db:
        base_row = db.execute(
            "SELECT base_blob FROM multimap_slot_base WHERE locus_id = ?",
            (locus_id,),
        ).fetchone()
        if base_row is None:
            return {}
        w_row = db.execute(
            "SELECT weight_blob FROM group_weights "
            "WHERE locus_id = ? AND iteration = ?",
            (locus_id, iteration),
        ).fetchone()
    base_map = database.unpickle_blob(base_row[0])  # {(run_id, group_key): base}
    # ``weight_blob`` is a bare float64 buffer over the locus's slots in
    # canonical order (see ``_slot_keys``); a missing row falls back to the
    # baseline, i.e. full weight.
    weights = (
        np.frombuffer(w_row[0], dtype=np.float64) if w_row is not None else None
    )
    keys = _slot_keys(base_map, _run_index(db_path))

    out: dict = defaultdict(dict)
    for i, key in enumerate(keys):
        run_id, gk = key
        base = base_map[key]
        out[run_id][gk] = (base, float(weights[i]) if weights is not None else base)
    return dict(out)


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
    mm_data: dict | None = None,
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
    mm_data : dict, optional
        ``{run_id: {group_key: (base, weight)}}`` for this locus, which fixes
        the canonical slot order λ is written in.  Required when *lambdas* is
        non-empty.
    """
    with database.connect(db_path, wal_writer=True, commit=True) as db:
        cur = db.cursor()
        cur.execute(
            "INSERT OR REPLACE INTO locus_activities VALUES (?, ?, ?)",
            (iteration, locus_id, database.compress_blob(activities)),
        )
        if lambdas:
            if mm_data is None:
                raise ValueError("mm_data is required to order a locus's lambdas")
            # Dense over the locus's slots, in canonical order, so the E-step can
            # drop it straight into its per-slot vector.
            run_index = _run_index(db_path)
            keys = sorted(
                (
                    (run_index[run_id], gk)
                    for run_id, slots in mm_data.items()
                    for gk in slots
                )
            )
            position = {key: i for i, key in enumerate(keys)}
            lam_vector = np.zeros(len(keys), dtype=np.float64)
            for run_id, gk, lam in lambdas:
                lam_vector[position[(run_index[run_id], gk)]] = lam
            cur.execute(
                "INSERT OR REPLACE INTO group_lambdas VALUES (?, ?, ?)",
                (iteration, locus_id, lam_vector.tobytes()),
            )
