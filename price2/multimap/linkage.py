"""The canonical slot order and the static linkage arrays of the E-step."""

from __future__ import annotations

import logging
import os

import numpy as np

from price2 import database

logger = logging.getLogger(__name__)


#
# A *slot* is a ``(run, locus, group_key)`` triple.  Every per-locus vector the
# EM exchanges — the baseline, the fractional weights, λ — is stored in one
# canonical order: the locus's slots sorted by ``(run index, group_key)``.  That
# lets the weights and λ travel as bare ``float64`` buffers rather than pickled
# dicts with tuple keys, and lets the E-step address every slot by integer.
#
# The linkage itself (which slots belong to which multimap group, and each
# group's read count) depends only on the collected alignments, so it is built
# once into ``multimap_linkage.npz`` beside the database and reloaded thereafter.

_RUN_INDEX_CACHE: dict = {}
_LINKAGE_CACHE: dict = {}


def _run_index(db_path: str) -> dict:
    """Return ``{run_id: run index}``, memoised per process."""
    cached = _RUN_INDEX_CACHE.get(db_path)
    if cached is None:
        with database.connect(db_path) as db:
            run_ids = sorted(
                r for (r,) in db.execute("SELECT run_id FROM runs")
            )
        cached = {run_id: i for i, run_id in enumerate(run_ids)}
        _RUN_INDEX_CACHE[db_path] = cached
    return cached


def _slot_keys(base_map: dict, run_index: dict) -> list:
    """Canonical order of a locus's slots: sorted by ``(run index, group_key)``."""
    return sorted(base_map, key=lambda k: (run_index[k[0]], k[1]))


def linkage_path(db_path: str) -> str:
    """Path of the cached static-linkage arrays for *db_path*."""
    return os.path.join(
        os.path.dirname(os.path.abspath(db_path)), "multimap_linkage.npz"
    )


def _invalidate_linkage(db_path: str) -> None:
    """Drop cached linkage arrays that describe a superseded index."""
    stale = linkage_path(db_path)
    if os.path.exists(stale):
        os.remove(stale)
    _LINKAGE_CACHE.pop(db_path, None)


def _build_linkage(db_path: str) -> dict:
    """Materialise the static linkage as integer arrays (slow path, run once).

    Reading ``multimap_group_slots`` (tens of millions of rows, with a TEXT
    ``locus_id``) and re-deriving the slot identities dominated every E-step.
    Here it happens once; afterwards the E-step is two ``bincount``s.
    """
    run_index = _run_index(db_path)
    with database.connect(db_path) as db:
        cur = db.cursor()

        locus_ids = sorted(
            lid for (lid,) in cur.execute("SELECT locus_id FROM multimap_slot_base")
        )
        locus_index = {lid: i for i, lid in enumerate(locus_ids)}

        n_groups = cur.execute("SELECT COUNT(*) FROM multimap_groups").fetchone()[0]
        mmg_run = np.zeros(n_groups, dtype=np.int32)
        mmg_count = np.zeros(n_groups, dtype=np.float64)
        cur.execute("SELECT mmg_id, run_id, count FROM multimap_groups")
        while chunk := cur.fetchmany(1 << 20):
            for mmg_id, run_id, count in chunk:
                mmg_run[mmg_id] = run_index[run_id]
                mmg_count[mmg_id] = count

        n_members = cur.execute(
            "SELECT COUNT(*) FROM multimap_group_slots"
        ).fetchone()[0]
        member_mmg = np.empty(n_members, dtype=np.int32)
        member_locus = np.empty(n_members, dtype=np.int32)
        member_gk = np.empty(n_members, dtype=np.int64)
        cur.execute("SELECT mmg_id, locus_id, group_key FROM multimap_group_slots")
        i = 0
        while chunk := cur.fetchmany(1 << 20):
            for mmg_id, locus_id, group_key in chunk:
                member_mmg[i] = mmg_id
                member_locus[i] = locus_index[locus_id]
                member_gk[i] = group_key
                i += 1

    # Identify slots by sorting membership rows into the canonical order; a
    # slot's run is its group's run.
    if n_members == 0:
        member_slot = np.empty(0, dtype=np.int32)
        n_slots = 0
        locus_off = np.zeros(len(locus_ids) + 1, dtype=np.int64)
    else:
        member_run = mmg_run[member_mmg]
        order = np.lexsort((member_gk, member_run, member_locus))
        sorted_locus = member_locus[order]
        sorted_run = member_run[order]
        sorted_gk = member_gk[order]
        starts = np.empty(n_members, dtype=bool)
        starts[0] = True
        np.not_equal(sorted_locus[1:], sorted_locus[:-1], out=starts[1:])
        starts[1:] |= sorted_run[1:] != sorted_run[:-1]
        starts[1:] |= sorted_gk[1:] != sorted_gk[:-1]
        slot_of_sorted = np.cumsum(starts) - 1
        n_slots = int(slot_of_sorted[-1]) + 1
        if n_slots > np.iinfo(np.int32).max:
            raise OverflowError(f"{n_slots} slots exceed the int32 slot index")
        member_slot = np.empty(n_members, dtype=np.int32)
        member_slot[order] = slot_of_sorted

        slot_locus = sorted_locus[starts]
        locus_off = np.searchsorted(
            slot_locus, np.arange(len(locus_ids) + 1, dtype=np.int32)
        ).astype(np.int64)

    link = {
        "member_mmg": member_mmg,
        "member_slot": member_slot,
        "mmg_count": mmg_count,
        "locus_off": locus_off,
        "locus_ids": np.array(locus_ids),
        "n_slots": np.array(n_slots),
    }
    np.savez(linkage_path(db_path), **link)
    logger.info(
        "multimap linkage cached: %d slots, %d groups, %d memberships",
        n_slots,
        n_groups,
        n_members,
    )
    return link


def _linkage(db_path: str) -> dict:
    """Return the static linkage arrays, building and caching them on first use."""
    cached = _LINKAGE_CACHE.get(db_path)
    if cached is not None:
        return cached
    path = linkage_path(db_path)
    if os.path.exists(path):
        with np.load(path, allow_pickle=False) as data:
            cached = {k: data[k] for k in data.files}
    else:
        cached = _build_linkage(db_path)
    cached["locus_index"] = {
        lid: i for i, lid in enumerate(cached["locus_ids"].tolist())
    }
    _LINKAGE_CACHE[db_path] = cached
    return cached
