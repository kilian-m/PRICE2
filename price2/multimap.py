"""EM-based fractional assignment of multimapping reads.

Classic PRICE2 solves one locus at a time and counts every read at its
full weight in *every* locus it overlaps, so a read that aligns to
several loci is multi-counted.  This module adds an
Expectation-Maximisation outer loop around the existing per-locus
deconvolution:

* **M-step** — the existing per-locus group-LASSO Poisson deconvolution
  (unchanged in shape), run in the existing fan-out, but with a
  *fractional* response ``y_i = Σ_r f_{r,ℓ}·read_count`` for multimapping
  reads.
* **E-step** — a global reduce that, given the current per-locus
  activities, re-assigns each multimapping read fractionally across the
  loci it aligns to::

      f_{r,ℓ} = λ_{r,ℓ} / Σ_{ℓ'∈L(r)} λ_{r,ℓ'}

  where ``λ_{r,ℓ}`` is the per-read origin rate at locus ``ℓ`` under the
  current model — exactly the read's design-matrix row (cleavage ×
  coverage × activity) summed over its compatible ORFs, i.e.
  ``δ_EG / length_EG``.

The only cross-locus coupling is the E-step normalisation, which is a
per-read lookup; the M-step never leaves the per-locus fan-out, so there
is no joint optimisation and connected components never need merging.

Cross-locus linkage
-------------------
Reads that share the same *set* of alignment slots behave identically in
the E-step, so they are collapsed into **multimap groups** (MMGs).  A
slot is a ``(locus_id, group_key)`` pair, where ``group_key`` is a stable
hash of the read's spliced coordinates and 5' untemplated-addition state
at that locus — computable both at collection time and inside a worker
from a :class:`~price2.ribo_seq_alignment.RiboSeqAlignment`.  Only reads
with **≥2** in-locus slots need EM treatment; intergenic alignments never
enter a locus fetch and are therefore ignored for free ("loci-only").

Persistence
-----------
All state lives in ``price.db`` next to the existing tables:

``multimap_alignments``
    ``(run_id, qname_hash, locus_id, group_key)`` — one row per in-locus
    multimapping alignment, written during read collection.
``multimap_group_slots`` / ``multimap_groups``
    MMG membership and per-MMG read counts, derived after collection.
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
``prepared_loci`` / ``prepared_loci_cache``
    The weight-independent per-locus state an EM iteration reuses: the
    prepared :class:`~price2.locus.Locus`, and — separately, because it
    holds only arrays — its :class:`~price2.locus.EgRoutingCache`.

``group_weights`` and ``group_lambdas`` are bare ``float64`` buffers over a
locus's slots in canonical order (sorted by run index, then ``group_key``);
only the baseline keeps its keys, since the workers need them.  Alongside the
database, ``multimap_linkage.npz`` caches the static MMG membership as integer
arrays; it is rebuilt automatically if absent.
"""

from __future__ import annotations

import hashlib
import logging
import os
import sqlite3 as sql
import struct
import zlib
from collections import defaultdict
from pickle import dumps, loads

import numpy as np

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Stable hashing (process-independent; Python's str hash is salted)            #
# --------------------------------------------------------------------------- #

def qname_hash(query_name: str) -> int:
    """Return a stable 63-bit hash of a read's query name.

    Parameters
    ----------
    query_name : str
        BAM ``QNAME``; identical for all alignments of one physical read.

    Returns
    -------
    int
        A non-negative 63-bit integer suitable for an SQLite ``INTEGER``
        column (SQLite integers are signed 64-bit).
    """
    digest = hashlib.blake2b(query_name.encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little") >> 1


def group_key(ivs_tuple: tuple, untemplated_addition: bool) -> int:
    """Return a stable 63-bit hash identifying a read's slot at a locus.

    Two alignments with the same spliced coordinates and untemplated-
    addition state share a ``group_key``.  The encoding is deterministic
    across processes so the value computed at collection time matches the
    value recomputed from an in-memory alignment inside a worker.

    Parameters
    ----------
    ivs_tuple : tuple of (int, int)
        The alignment's exonic intervals as ``(start, end)`` pairs, in
        chromosome order.
    untemplated_addition : bool
        Whether a 5' untemplated addition was detected.

    Returns
    -------
    int
        A non-negative 63-bit integer.
    """
    h = hashlib.blake2b(digest_size=8)
    h.update(b"\x01" if untemplated_addition else b"\x00")
    for start, end in ivs_tuple:
        h.update(struct.pack("<qq", int(start), int(end)))
    return int.from_bytes(h.digest(), "little") >> 1


def alignment_group_key(rsa) -> int:
    """Compute the :func:`group_key` of an in-memory alignment.

    Parameters
    ----------
    rsa : RiboSeqAlignment
        A loaded alignment whose ``genomic_region`` gives its intervals.

    Returns
    -------
    int
        The slot hash matching the value stored at collection time.
    """
    ivs = tuple(
        (iv.start, iv.end) for iv in rsa.genomic_region.intervals
    )
    return group_key(ivs, rsa.untemplated_addition)


# --------------------------------------------------------------------------- #
# Schema                                                                       #
# --------------------------------------------------------------------------- #

def create_alignment_table(cur: sql.Cursor) -> None:
    """Create the raw multimapping-alignment table (drop any stale copy)."""
    cur.execute("DROP TABLE IF EXISTS multimap_alignments")
    cur.execute(
        """CREATE TABLE multimap_alignments (
               run_id     TEXT    NOT NULL,
               qname_hash INTEGER NOT NULL,
               locus_id   TEXT    NOT NULL,
               group_key  INTEGER NOT NULL
           )"""
    )


def create_em_tables(cur: sql.Cursor) -> None:
    """Create the derived EM state tables (dropping any stale copies)."""
    for table in (
        "multimap_group_slots",
        "multimap_groups",
        "multimap_slot_base",
        "group_weights",
        "group_lambdas",
        "locus_activities",
        "prepared_loci",
        "prepared_loci_cache",
    ):
        cur.execute(f"DROP TABLE IF EXISTS {table}")

    cur.execute(
        """CREATE TABLE multimap_groups (
               mmg_id  INTEGER PRIMARY KEY,
               run_id  TEXT    NOT NULL,
               count   INTEGER NOT NULL
           )"""
    )
    cur.execute(
        """CREATE TABLE multimap_group_slots (
               mmg_id    INTEGER NOT NULL,
               locus_id  TEXT    NOT NULL,
               group_key INTEGER NOT NULL
           )"""
    )
    cur.execute(
        "CREATE INDEX idx_mgs_mmg ON multimap_group_slots(mmg_id)"
    )
    # Per-slot state is stored as ONE blob per slot-locus rather than one row
    # per slot: at genome scale that is ~40K rows instead of ~42M, so every
    # E-step read/write/prune touches ~1000x fewer rows.  The baseline keeps
    # its keys (a pickled ``{(run_id, group_key): value}`` dict) because the
    # workers need them; the weights and λ are bare ``float64`` buffers over
    # the locus's slots in canonical order (see ``_slot_keys``).  The
    # multimap_* linkage tables remain per-slot, and are cached as integer
    # arrays in ``multimap_linkage.npz`` (see ``_linkage``).
    cur.execute(
        """CREATE TABLE multimap_slot_base (
               locus_id  TEXT PRIMARY KEY,
               base_blob BLOB NOT NULL
           )"""
    )
    cur.execute(
        """CREATE TABLE group_weights (
               iteration   INTEGER NOT NULL,
               locus_id    TEXT    NOT NULL,
               weight_blob BLOB    NOT NULL,
               PRIMARY KEY (iteration, locus_id)
           )"""
    )
    cur.execute(
        """CREATE TABLE group_lambdas (
               iteration INTEGER NOT NULL,
               locus_id  TEXT    NOT NULL,
               lam_blob  BLOB    NOT NULL,
               PRIMARY KEY (iteration, locus_id)
           )"""
    )
    cur.execute(
        """CREATE TABLE locus_activities (
               iteration      INTEGER NOT NULL,
               locus_id       TEXT    NOT NULL,
               activities_blob BLOB   NOT NULL,
               PRIMARY KEY (iteration, locus_id)
           )"""
    )
    cur.execute(
        """CREATE TABLE prepared_loci (
               locus_id  TEXT PRIMARY KEY,
               prep_blob BLOB NOT NULL
           )"""
    )
    # The per-locus ``EgRoutingCache`` (see ``Config.eg_cache``), stored apart
    # from the locus because it holds only arrays: a light M-step loads this
    # blob alone and never unpickles the locus's object graph.
    cur.execute(
        """CREATE TABLE prepared_loci_cache (
               locus_id   TEXT PRIMARY KEY,
               cache_blob BLOB NOT NULL
           )"""
    )


def enable_wal(db_path: str) -> None:
    """Switch ``price.db`` to WAL mode for concurrent worker writes.

    The EM M-step makes every worker a writer (activities + λ) from up to
    ``config.processes`` processes at once.  WAL lets many readers and one
    writer proceed without blocking, and combined with ``busy_timeout``
    serialises the brief commit windows safely.

    ``journal_mode`` is a persistent database property; ``synchronous`` is
    per-connection, so the writer connections set it themselves via
    :func:`_prepare_writer`.

    Parameters
    ----------
    db_path : str
        Path to the SQLite database.
    """
    db = sql.connect(db_path, timeout=120)
    try:
        db.execute("PRAGMA journal_mode = WAL")
    finally:
        db.close()


def _prepare_writer(db_path: str) -> sql.Connection:
    """Open a writer connection tuned for WAL concurrency.

    Sets ``busy_timeout`` (wait rather than fail under contention) and
    ``synchronous = NORMAL`` (safe under WAL, avoids an fsync per commit).
    Both pragmas are per-connection, so every writer must set them.
    """
    db = sql.connect(db_path, timeout=120)
    db.execute("PRAGMA busy_timeout = 120000")
    db.execute("PRAGMA synchronous = NORMAL")
    return db


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
    db = sql.connect(db_path, timeout=120)
    cur = db.cursor()
    cur.execute("PRAGMA busy_timeout = 120000")
    cur.execute("DELETE FROM group_lambdas")
    cur.execute("DELETE FROM locus_activities")
    cur.execute("DELETE FROM group_weights")
    cur.execute("DELETE FROM prepared_loci")
    # Databases collected before ``prepared_loci_cache`` existed lack the table.
    cur.execute(
        "CREATE TABLE IF NOT EXISTS prepared_loci_cache ("
        "locus_id TEXT PRIMARY KEY, cache_blob BLOB NOT NULL)"
    )
    cur.execute("DELETE FROM prepared_loci_cache")
    cur.executemany(
        "INSERT INTO group_weights VALUES (0, ?, ?)",
        _baseline_weight_rows(cur),
    )
    db.commit()
    db.close()


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
        base_map = loads(blob)
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
    db = sql.connect(db_path, timeout=120)
    cur = db.cursor()
    cur.execute("SELECT DISTINCT locus_id FROM multimap_slot_base")
    ids = {loc_id for loc_id, in cur.fetchall()}
    db.close()
    return ids


# --------------------------------------------------------------------------- #
# Index construction (run once, after read collection)                         #
# --------------------------------------------------------------------------- #

def build_multimap_index(db_path: str) -> int:
    """Collapse recorded alignments into multimap groups and seed weights.

    Reads ``multimap_alignments``, keeps only reads that touch **≥2**
    distinct in-locus slots, collapses reads that share an identical slot
    set into one multimap group (MMG) with a member count, and writes the
    derived tables (``multimap_groups``, ``multimap_group_slots``,
    ``multimap_slot_base``) plus the iteration-0 ``group_weights`` seed
    (``weight = base`` → full counts, i.e. classic behaviour before any
    reassignment).

    Parameters
    ----------
    db_path : str
        Path to ``price.db``.

    Returns
    -------
    int
        Number of multimap groups created.
    """
    db = sql.connect(db_path, timeout=120)
    cur = db.cursor()
    cur.execute("PRAGMA busy_timeout = 120000")
    create_em_tables(cur)
    db.commit()

    # Group a read's in-locus alignment slots.  Ordered by (run, qname)
    # so each read's rows arrive consecutively → constant-memory streaming.
    cur.execute(
        """SELECT run_id, qname_hash, locus_id, group_key
               FROM multimap_alignments
               ORDER BY run_id, qname_hash"""
    )

    # signature (frozenset of slots) per run -> read count, so identical
    # cross-locus patterns collapse into a single MMG.
    sig_counts: dict[str, dict[frozenset, int]] = defaultdict(
        lambda: defaultdict(int)
    )

    cur_run = None
    cur_qname = None
    cur_slots: set = set()

    def _flush(run_id, slots):
        # Keep only reads with >=2 distinct in-locus slots.
        if run_id is not None and len(slots) >= 2:
            sig_counts[run_id][frozenset(slots)] += 1

    for run_id, qh, locus_id, gk in cur:
        if run_id != cur_run or qh != cur_qname:
            _flush(cur_run, cur_slots)
            cur_run, cur_qname = run_id, qh
            cur_slots = set()
        cur_slots.add((locus_id, gk))
    _flush(cur_run, cur_slots)

    # Materialise MMGs, membership and per-slot baselines.
    group_rows: list = []
    slot_rows: list = []
    base: dict = defaultdict(float)
    mmg_id = 0
    for run_id, sigs in sig_counts.items():
        for slots, count in sigs.items():
            group_rows.append((mmg_id, run_id, count))
            for locus_id, gk in slots:
                slot_rows.append((mmg_id, locus_id, gk))
                base[(run_id, locus_id, gk)] += count
            mmg_id += 1

    # Collapse per-slot baselines into one pickled dict per slot-locus.
    base_by_locus: dict = defaultdict(dict)
    for (run_id, locus_id, gk), b in base.items():
        base_by_locus[locus_id][(run_id, gk)] = b
    base_rows = [(locus_id, dumps(d)) for locus_id, d in base_by_locus.items()]

    cur.executemany(
        "INSERT INTO multimap_groups VALUES (?, ?, ?)", group_rows
    )
    cur.executemany(
        "INSERT INTO multimap_group_slots VALUES (?, ?, ?)", slot_rows
    )
    cur.executemany(
        "INSERT INTO multimap_slot_base VALUES (?, ?)", base_rows
    )
    cur.executemany(
        "INSERT INTO group_weights VALUES (0, ?, ?)",
        _baseline_weight_rows(cur),
    )
    # The cached linkage arrays describe the index we just replaced.
    stale = linkage_path(db_path)
    if os.path.exists(stale):
        os.remove(stale)
    _LINKAGE_CACHE.pop(db_path, None)
    db.commit()
    db.close()
    logger.info(
        "multimap index: %d groups over %d slots", mmg_id, len(base)
    )
    return mmg_id


def has_multimap_index(db_path: str) -> bool:
    """Return ``True`` if the EM linkage tables exist and are populated."""
    db = sql.connect(db_path, timeout=120)
    try:
        cur = db.cursor()
        cur.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name='multimap_groups'"
        )
        if cur.fetchone() is None:
            return False
        cur.execute("SELECT 1 FROM multimap_groups LIMIT 1")
        return cur.fetchone() is not None
    finally:
        db.close()


# --------------------------------------------------------------------------- #
# Canonical slot ordering and the static linkage arrays                        #
# --------------------------------------------------------------------------- #
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
        db = sql.connect(db_path, timeout=120)
        try:
            db.execute("PRAGMA busy_timeout = 120000")
            run_ids = sorted(r for (r,) in db.execute("SELECT run_id FROM runs"))
        finally:
            db.close()
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


def _build_linkage(db_path: str) -> dict:
    """Materialise the static linkage as integer arrays (slow path, run once).

    Reading ``multimap_group_slots`` (tens of millions of rows, with a TEXT
    ``locus_id``) and re-deriving the slot identities dominated every E-step.
    Here it happens once; afterwards the E-step is two ``bincount``s.
    """
    run_index = _run_index(db_path)
    db = sql.connect(db_path, timeout=120)
    cur = db.cursor()
    cur.execute("PRAGMA busy_timeout = 120000")

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

    n_members = cur.execute("SELECT COUNT(*) FROM multimap_group_slots").fetchone()[0]
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
    db.close()

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


# --------------------------------------------------------------------------- #
# Worker-side helpers (per locus, inside a fan-out process)                     #
# --------------------------------------------------------------------------- #

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
    db = sql.connect(db_path, timeout=120)
    cur = db.cursor()
    cur.execute("PRAGMA busy_timeout = 120000")

    cur.execute(
        "SELECT base_blob FROM multimap_slot_base WHERE locus_id = ?",
        (locus_id,),
    )
    base_row = cur.fetchone()
    if base_row is None:
        db.close()
        return {}
    base_map = loads(base_row[0])  # {(run_id, group_key): base}

    cur.execute(
        "SELECT weight_blob FROM group_weights "
        "WHERE locus_id = ? AND iteration = ?",
        (locus_id, iteration),
    )
    w_row = cur.fetchone()
    db.close()
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
    db = sql.connect(db_path, timeout=120)
    cur = db.cursor()
    cur.execute("PRAGMA busy_timeout = 120000")
    cur.execute(
        "SELECT activities_blob FROM locus_activities "
        "WHERE locus_id = ? AND iteration = ?",
        (locus_id, iteration),
    )
    row = cur.fetchone()
    db.close()
    if row is None:
        return None
    return loads(zlib.decompress(row[0]))


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
    db = _prepare_writer(db_path)
    cur = db.cursor()
    cur.execute(
        "INSERT OR REPLACE INTO locus_activities VALUES (?, ?, ?)",
        (iteration, locus_id, zlib.compress(dumps(activities))),
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
    db.commit()
    db.close()


def save_locus_cache(db_path: str, locus_id: str, loc) -> None:
    """Persist a locus's :class:`~price2.locus.EgRoutingCache` on its own.

    The cache references no RGR, transcript or equivalence-group objects, so
    a light M-step can restore it — plus the locus id and interval, all it
    otherwise needs — without unpickling the locus itself.

    Parameters
    ----------
    db_path : str
        Path to ``price.db``.
    locus_id : str
        Locus identifier.
    loc : Locus
        A locus whose ``eg_cache`` has been built.
    """
    payload = {"id": loc.id, "iv": loc.iv, "cache": loc.eg_cache}
    blob = zlib.compress(dumps(payload, protocol=5))
    db = _prepare_writer(db_path)
    db.execute(
        "INSERT OR REPLACE INTO prepared_loci_cache VALUES (?, ?)",
        (locus_id, blob),
    )
    db.commit()
    db.close()


def load_locus_cache(db_path: str, locus_id: str):
    """Return the cached routing payload for *locus_id*, or ``None``."""
    db = sql.connect(db_path, timeout=120)
    cur = db.cursor()
    cur.execute("PRAGMA busy_timeout = 120000")
    try:
        cur.execute(
            "SELECT cache_blob FROM prepared_loci_cache WHERE locus_id = ?",
            (locus_id,),
        )
        row = cur.fetchone()
    except sql.OperationalError:  # table absent (pre-cache database)
        row = None
    finally:
        db.close()
    if row is None:
        return None
    return loads(zlib.decompress(row[0]))


def load_light_locus(db_path: str, locus_id: str):
    """Return a minimal :class:`~price2.locus.Locus` for a light M-step.

    The returned locus carries only ``id``, ``iv`` and ``eg_cache`` — enough
    for ``get_reads_from_db``, ``set_warm_start``, ``assign_reads_to_egs``,
    ``deconvolve(prune=False)``, ``compute_multimap_lambdas`` and
    ``activities_by_id``.  It has no ``rgr_set``, ``egs`` or ``transcripts``,
    which is the whole point: restoring those dominates the cost of loading a
    prepared locus.

    Parameters
    ----------
    db_path : str
        Path to ``price.db``.
    locus_id : str
        Locus identifier.

    Returns
    -------
    Locus or None
        ``None`` when no cache was stored for this locus.
    """
    from price2.locus import Locus  # local: locus imports this module

    payload = load_locus_cache(db_path, locus_id)
    if payload is None:
        return None
    loc = Locus.__new__(Locus)
    loc.id = payload["id"]
    loc.iv = payload["iv"]
    loc.eg_cache = payload["cache"]
    loc._eg_y = None
    loc.read_counts = {}
    loc.uncounted_reads = 0
    return loc


def save_prepared_locus(db_path: str, locus_id: str, loc) -> None:
    """Cache a locus's weight-independent prepared state for later EM passes.

    The RGR candidate set, coverage/deconvolution-filter results and
    equivalence-group *geometry* depend only on raw (unweighted) reads, so
    they are identical in every EM iteration.  Persisting them after the
    first light M-step lets subsequent iterations skip ORF generation, the
    two filter passes and the EG DAG build — the dominant per-locus cost.
    The bulky ``rsas_dict`` (reads) is excluded from the blob and reloaded
    from the ``reads`` table on a cache hit.

    Parameters
    ----------
    db_path : str
        Path to ``price.db``.
    locus_id : str
        Locus identifier.
    loc : Locus
        A locus prepared up to and including ``make_equivalence_groups``
        (accumulators still zero, no ``result`` yet).
    """
    saved_rsas = getattr(loc, "rsas_dict", None)
    saved_rrc = getattr(loc, "run_read_count", None)
    saved_cache = getattr(loc, "eg_cache", None)
    loc.rsas_dict = {}
    loc.run_read_count = {}
    # The routing cache lives in its own table (``save_locus_cache``).
    loc.eg_cache = None
    # Per-iteration state must not ride along: a cached locus is reloaded for
    # every later iteration, and its equivalence groups accumulate read counts
    # with ``+=``.
    saved_eg_counts = [
        (eg, eg.read_count)
        for egs in getattr(loc, "egs", {}).values()
        for eg in egs.values()
    ]
    saved_transient = {
        name: getattr(loc, name)
        for name in ("uncounted_reads", "read_counts", "counted_reads", "_eg_y")
        if hasattr(loc, name)
    }
    for eg, _ in saved_eg_counts:
        eg.read_count = 0
    loc.uncounted_reads = 0
    loc.read_counts = {}
    loc.counted_reads = {}
    loc._eg_y = None
    try:
        blob = zlib.compress(dumps(loc))
    finally:
        loc.rsas_dict = saved_rsas
        loc.run_read_count = saved_rrc
        loc.eg_cache = saved_cache
        for eg, count in saved_eg_counts:
            eg.read_count = count
        for name, value in saved_transient.items():
            setattr(loc, name, value)
    db = _prepare_writer(db_path)
    db.execute(
        "INSERT OR REPLACE INTO prepared_loci VALUES (?, ?)",
        (locus_id, blob),
    )
    db.commit()
    db.close()


def load_prepared_locus(db_path: str, locus_id: str, with_cache: bool = True):
    """Return a cached prepared :class:`Locus`, or ``None`` if absent.

    The returned locus has an empty ``rsas_dict``; the caller must call
    ``get_reads_from_db`` to repopulate reads before assigning them.

    Parameters
    ----------
    db_path : str
        Path to ``price.db``.
    locus_id : str
        Locus identifier.

    with_cache : bool, optional
        Re-attach the locus's ``eg_cache`` from ``prepared_loci_cache``
        (default).  The final full pass needs both.

    Returns
    -------
    Locus or None
    """
    db = sql.connect(db_path, timeout=120)
    cur = db.cursor()
    cur.execute("PRAGMA busy_timeout = 120000")
    try:
        cur.execute(
            "SELECT prep_blob FROM prepared_loci WHERE locus_id = ?",
            (locus_id,),
        )
        row = cur.fetchone()
    except sql.OperationalError:
        # Table absent (e.g. no EM run in progress).
        row = None
    finally:
        db.close()
    if row is None:
        return None
    loc = loads(zlib.decompress(row[0]))
    if with_cache:
        payload = load_locus_cache(db_path, locus_id)
        if payload is not None:
            loc.eg_cache = payload["cache"]
    return loc


# --------------------------------------------------------------------------- #
# Global E-step (run once between M-step fan-outs)                             #
# --------------------------------------------------------------------------- #

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
    n_slots = int(link["n_slots"])
    if n_slots == 0:
        return 0.0
    member_mmg = link["member_mmg"]
    member_slot = link["member_slot"]
    mmg_count = link["mmg_count"]
    locus_off = link["locus_off"]
    locus_index = link["locus_index"]
    n_groups = mmg_count.size

    db = _prepare_writer(db_path)
    cur = db.cursor()

    lam_slot = np.zeros(n_slots, dtype=np.float64)
    cur.execute(
        "SELECT locus_id, lam_blob FROM group_lambdas WHERE iteration = ?",
        (iteration,),
    )
    for locus_id, blob in cur.fetchall():
        i = locus_index[locus_id]
        lam_slot[locus_off[i]:locus_off[i + 1]] = np.frombuffer(
            blob, dtype=np.float64
        )

    # Responsibility per membership row: λ / Σλ within its group, or a uniform
    # 1/n split when the group's λ sums to zero (its read fits no ORF anywhere,
    # so it is neither lost nor arbitrarily concentrated).  Folded into an
    # affine form ``count * (a·λ + b)`` so each row needs two gathers, not a
    # branch.
    lam_cell = lam_slot[member_slot]
    lam_sum = np.bincount(member_mmg, weights=lam_cell, minlength=n_groups)
    n_cells = np.bincount(member_mmg, minlength=n_groups)
    positive = lam_sum > 0.0
    scale = np.where(positive, mmg_count / np.where(positive, lam_sum, 1.0), 0.0)
    offset = np.where(positive, 0.0, mmg_count / np.maximum(n_cells, 1))
    weight_cell = lam_cell * scale[member_mmg] + offset[member_mmg]

    # New weight per slot = Σ contributions across the MMGs sharing it.
    new_slot = np.bincount(member_slot, weights=weight_cell, minlength=n_slots)
    del lam_cell, weight_cell

    # Previous weights, in the same slot order, for the convergence metric.
    old_slot = np.zeros(n_slots, dtype=np.float64)
    cur.execute(
        "SELECT locus_id, weight_blob FROM group_weights WHERE iteration = ?",
        (iteration,),
    )
    for locus_id, blob in cur.fetchall():
        i = locus_index[locus_id]
        old_slot[locus_off[i]:locus_off[i + 1]] = np.frombuffer(
            blob, dtype=np.float64
        )

    # Convergence = Σ|w_new − w_old| / Σ w_new — a total-variation measure
    # robust to the slot count.  From iteration 1 on the total mass is
    # conserved (each read's weight sums to one); iteration 0's large value
    # reflects the one-off removal of the classic full-weight double count.
    total_mass = float(new_slot.sum())
    moved = float(np.abs(new_slot - old_slot).sum())
    rel = moved / total_mass if total_mass > 0 else 0.0

    # Write new weights, then prune spent state.  Only weights[it+1] (the next
    # M-step's response) and locus_activities[it] (its warm start) are still
    # needed.
    it_next = iteration + 1
    weight_rows = [
        (it_next, locus_id, new_slot[locus_off[i]:locus_off[i + 1]].tobytes())
        for locus_id, i in locus_index.items()
    ]
    cur.execute("DELETE FROM group_weights WHERE iteration = ?", (it_next,))
    cur.executemany(
        "INSERT INTO group_weights VALUES (?, ?, ?)", weight_rows
    )
    cur.execute("DELETE FROM group_weights WHERE iteration <= ?", (iteration,))
    cur.execute("DELETE FROM group_lambdas WHERE iteration <= ?", (iteration,))
    cur.execute("DELETE FROM locus_activities WHERE iteration < ?", (iteration,))
    db.commit()
    db.close()
    return rel
