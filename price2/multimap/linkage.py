"""The canonical slot order and the static linkage arrays of the E-step."""

from __future__ import annotations

import logging
import os
import sqlite3 as sql
from dataclasses import dataclass

import numpy as np

from price2 import database

logger = logging.getLogger(__name__)


#
# A *slot* is a ``(run, locus, group_key)`` triple.  Every per-locus vector the
# EM exchanges — the baseline, the fractional weights, λ — is stored in one
# canonical order, that of :class:`LocusSlots`: the locus's slots sorted by
# ``(run index, group_key)``.  That lets the weights and λ travel as bare
# ``float64`` buffers rather than pickled dicts with tuple keys, and lets the
# E-step address every slot by integer.
#
# The linkage itself (which slots belong to which multimap group, and each
# group's read count) depends only on the collected alignments, so
# :func:`~price2.multimap.index.build_multimap_index` writes it once, as
# ``multimap_linkage.npz`` beside the database, and every E-step reloads it.

_RUN_INDEX_CACHE: dict = {}
_LINKAGE_CACHE: dict[str, Linkage] = {}


@dataclass(frozen=True)
class LocusSlots:
    """A locus's multimapping slots in canonical order.

    The order — by run index, then group key — is the one every per-locus
    buffer of the EM uses (``multimap_slot_base`` seeds it, ``group_weights``
    and ``group_lambdas`` follow it, and the linkage arrays sort their
    membership rows by the same key), so a worker reads its weights and
    writes its λ through this object without any key matching.

    Attributes
    ----------
    run_ids : tuple[str, ...]
        The run of every slot.
    group_keys : tuple[int, ...]
        The group key of every slot.
    base : numpy.ndarray
        The baseline weight of every slot (the full-count reference).
    weights : numpy.ndarray
        The current fractional weight of every slot.
    """

    run_ids: tuple[str, ...]
    group_keys: tuple[int, ...]
    base: np.ndarray
    weights: np.ndarray

    @classmethod
    def from_base_map(
        cls, base_map: dict, run_index: dict, weight_blob: bytes | None = None
    ) -> LocusSlots:
        """Order a locus's ``{(run_id, group_key): base}`` map.

        Parameters
        ----------
        base_map : dict
            The locus's ``multimap_slot_base`` entry.
        run_index : dict
            ``{run_id: run index}`` (see :func:`_run_index`).
        weight_blob : bytes, optional
            The locus's ``group_weights`` buffer; without it the weights
            equal the baseline (full counts).
        """
        keys = sorted(base_map, key=lambda k: (run_index[k[0]], k[1]))
        base = np.fromiter(
            (base_map[k] for k in keys), dtype=np.float64, count=len(keys)
        )
        if weight_blob is None:
            weights = base
        else:
            weights = np.frombuffer(weight_blob, dtype=np.float64)
            if weights.size != base.size:
                raise ValueError(
                    f"{weights.size} weights stored for {base.size} slots"
                )
        return cls(
            tuple(k[0] for k in keys), tuple(k[1] for k in keys), base, weights
        )

    @classmethod
    def from_blob(
        cls, blob: bytes, run_index: dict, weight_blob: bytes | None = None
    ) -> LocusSlots:
        """Decode a ``multimap_slot_base`` entry.

        Two layouts exist: the ``{(run_id, group_key): base}`` map of
        earlier releases (:meth:`from_base_map`) and the canonical-order
        arrays ``(run index, group key, base)`` the index writes now
        (:func:`slot_base_blob`).
        """
        stored = database.unpickle_blob(blob)
        if isinstance(stored, dict):
            return cls.from_base_map(stored, run_index, weight_blob)
        run_idx, group_keys, base = stored
        id_of = {i: run_id for run_id, i in run_index.items()}
        if weight_blob is None:
            weights = base
        else:
            weights = np.frombuffer(weight_blob, dtype=np.float64)
            if weights.size != base.size:
                raise ValueError(
                    f"{weights.size} weights stored for {base.size} slots"
                )
        return cls(
            tuple(id_of[i] for i in run_idx.tolist()),
            tuple(group_keys.tolist()),
            base,
            weights,
        )

    def base_map(self) -> dict[tuple[str, int], float]:
        """``{(run_id, group_key): base}``, the layout of the earlier blobs."""
        return {
            (run_id, gk): base
            for run_id, gk, base in zip(self.run_ids, self.group_keys, self.base.tolist())
        }

    def __len__(self) -> int:
        return len(self.run_ids)

    def by_run(self) -> dict[str, dict[int, tuple[float, float]]]:
        """``{run_id: {group_key: (base, weight)}}``, as the read routing consumes it."""
        out: dict[str, dict[int, tuple[float, float]]] = {}
        for run_id, gk, base, weight in zip(
            self.run_ids, self.group_keys, self.base.tolist(), self.weights.tolist()
        ):
            out.setdefault(run_id, {})[gk] = (base, weight)
        return out

    def lambda_vector(self, lambdas: list[tuple[str, int, float]]) -> np.ndarray:
        """Lay ``(run_id, group_key, λ)`` triples out over the slots; absent slots score 0."""
        position = {
            key: i for i, key in enumerate(zip(self.run_ids, self.group_keys))
        }
        vector = np.zeros(len(self), dtype=np.float64)
        for run_id, gk, lam in lambdas:
            vector[position[(run_id, gk)]] = lam
        return vector


@dataclass(frozen=True)
class Linkage:
    """The static linkage arrays the E-step addresses slots by.

    Attributes
    ----------
    member_mmg, member_slot : numpy.ndarray
        One entry per (multimap group, slot) membership: the group and the
        slot it links.
    mmg_count : numpy.ndarray
        Read count of every multimap group.
    locus_off : numpy.ndarray
        ``locus_off[i]:locus_off[i + 1]`` are the slots of locus ``i``.
    locus_ids : numpy.ndarray
        Locus id of every locus index.
    n_slots : int
        Total number of slots.
    locus_index : dict
        ``{locus_id: locus index}``.
    """

    member_mmg: np.ndarray
    member_slot: np.ndarray
    mmg_count: np.ndarray
    locus_off: np.ndarray
    locus_ids: np.ndarray
    n_slots: int
    locus_index: dict

    @classmethod
    def from_memberships(
        cls,
        member_mmg: np.ndarray,
        member_locus: np.ndarray,
        member_run: np.ndarray,
        member_gk: np.ndarray,
        mmg_count: np.ndarray,
        locus_ids: list[str],
    ) -> Linkage:
        """Number the slots of the membership rows in canonical order.

        A slot is identified by sorting the rows into the order of
        :class:`LocusSlots` (locus index, then run index, then group key), so
        a slot's position within its locus is the position the locus's base
        map gives it.

        Parameters
        ----------
        member_mmg, member_locus, member_run, member_gk : numpy.ndarray
            One entry per (multimap group, slot) membership: the group, and
            the slot's locus index (into *locus_ids*), run index and group
            key.
        mmg_count : numpy.ndarray
            Read count of every multimap group.
        locus_ids : list of str
            The loci that carry slots, sorted; ``member_locus`` indexes it.
        """
        n_members = member_mmg.size
        if n_members == 0:
            member_slot = np.empty(0, dtype=np.int32)
            n_slots = 0
            locus_off = np.zeros(len(locus_ids) + 1, dtype=np.int64)
        else:
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

        return cls.from_arrays(
            {
                "member_mmg": np.asarray(member_mmg, dtype=np.int32),
                "member_slot": member_slot,
                "mmg_count": np.asarray(mmg_count, dtype=np.float64),
                "locus_off": locus_off,
                "locus_ids": np.array(locus_ids),
                "n_slots": np.array(n_slots),
            }
        )

    @classmethod
    def from_arrays(cls, arrays: dict) -> Linkage:
        locus_ids = np.asarray(arrays["locus_ids"])
        return cls(
            member_mmg=arrays["member_mmg"],
            member_slot=arrays["member_slot"],
            mmg_count=arrays["mmg_count"],
            locus_off=arrays["locus_off"],
            locus_ids=locus_ids,
            n_slots=int(arrays["n_slots"]),
            locus_index={lid: i for i, lid in enumerate(locus_ids.tolist())},
        )

    @classmethod
    def load(cls, path: str) -> Linkage:
        with np.load(path, allow_pickle=False) as data:
            return cls.from_arrays({k: data[k] for k in data.files})

    def save(self, path: str) -> None:
        np.savez(
            path,
            member_mmg=self.member_mmg,
            member_slot=self.member_slot,
            mmg_count=self.mmg_count,
            locus_off=self.locus_off,
            locus_ids=self.locus_ids,
            n_slots=np.array(self.n_slots),
        )

    def locus_slice(self, locus_index: int) -> slice:
        """The slots of one locus within a per-slot vector."""
        return slice(
            int(self.locus_off[locus_index]), int(self.locus_off[locus_index + 1])
        )


def slot_base_blob(
    run_idx: np.ndarray, group_keys: np.ndarray, base: np.ndarray
) -> bytes:
    """Encode one locus's slot baselines, given in canonical order."""
    return database.pickle_blob(
        (
            np.ascontiguousarray(run_idx, dtype=np.int32),
            np.ascontiguousarray(group_keys, dtype=np.int64),
            np.ascontiguousarray(base, dtype=np.float64),
        )
    )


def run_index_from(cur: sql.Cursor) -> dict:
    """Return ``{run_id: run index}`` — the runs of the database, sorted by id."""
    run_ids = sorted(r for (r,) in cur.execute("SELECT run_id FROM runs").fetchall())
    return {run_id: i for i, run_id in enumerate(run_ids)}


def _run_index(db_path: str) -> dict:
    """Return ``{run_id: run index}``, memoised per process."""
    cached = _RUN_INDEX_CACHE.get(db_path)
    if cached is None:
        with database.connect(db_path) as db:
            cached = run_index_from(db.cursor())
        _RUN_INDEX_CACHE[db_path] = cached
    return cached


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


def load_linkage(db_path: str) -> Linkage:
    """Return the static linkage of *db_path*'s index, memoised per process.

    Raises
    ------
    FileNotFoundError
        When ``multimap_linkage.npz`` is missing beside the database: the
        index was built by a PRICE2 that kept the linkage in SQLite tables
        instead, and only a cold re-collection can rebuild it.
    """
    cached = _LINKAGE_CACHE.get(db_path)
    if cached is not None:
        return cached
    path = linkage_path(db_path)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} is missing: the multimapping linkage index of {db_path} "
            "was built by an older PRICE2. Re-collect the run with "
            "warm_start=false to rebuild it."
        )
    cached = Linkage.load(path)
    _LINKAGE_CACHE[db_path] = cached
    return cached
