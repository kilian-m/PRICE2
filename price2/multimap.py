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
"""

from __future__ import annotations

import hashlib
import logging
import sqlite3 as sql
import struct
import zlib
from collections import defaultdict
from pickle import dumps, loads

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
    # Per-slot state (baseline, weights, λ) is stored as ONE blob per
    # slot-locus — a pickled ``{(run_id, group_key): value}`` dict — rather
    # than one row per slot.  At genome scale that is ~40K rows instead of
    # ~42M, so every E-step read/write/prune touches ~1000x fewer rows
    # (the per-slot compute stays vectorised in memory).  The multimap_*
    # linkage tables remain per-slot because the E-step needs them expanded.
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
    cur.execute(
        "INSERT INTO group_weights "
        "SELECT 0, locus_id, base_blob FROM multimap_slot_base"
    )
    db.commit()
    db.close()


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
    # iteration-0 weights == baseline == full counts (classic behaviour).
    cur.executemany(
        "INSERT INTO group_weights VALUES (0, ?, ?)", base_rows
    )
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
    weight_map = loads(w_row[0]) if w_row is not None else {}

    out: dict = defaultdict(dict)
    for (run_id, gk), b in base_map.items():
        out[run_id][gk] = (b, weight_map.get((run_id, gk), b))
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
        Per-slot origin rates for this locus's multimapping slots.
    """
    db = _prepare_writer(db_path)
    cur = db.cursor()
    cur.execute(
        "INSERT OR REPLACE INTO locus_activities VALUES (?, ?, ?)",
        (iteration, locus_id, zlib.compress(dumps(activities))),
    )
    if lambdas:
        lam_map = {(run_id, gk): lam for run_id, gk, lam in lambdas}
        cur.execute(
            "INSERT OR REPLACE INTO group_lambdas VALUES (?, ?, ?)",
            (iteration, locus_id, dumps(lam_map)),
        )
    db.commit()
    db.close()



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
    loc.rsas_dict = {}
    loc.run_read_count = {}
    try:
        blob = zlib.compress(dumps(loc))
    finally:
        loc.rsas_dict = saved_rsas
        loc.run_read_count = saved_rrc
    db = _prepare_writer(db_path)
    db.execute(
        "INSERT OR REPLACE INTO prepared_loci VALUES (?, ?)",
        (locus_id, blob),
    )
    db.commit()
    db.close()


def load_prepared_locus(db_path: str, locus_id: str):
    """Return a cached prepared :class:`Locus`, or ``None`` if absent.

    The returned locus has an empty ``rsas_dict``; the caller must call
    ``get_reads_from_db`` to repopulate reads before assigning them.

    Parameters
    ----------
    db_path : str
        Path to ``price.db``.
    locus_id : str
        Locus identifier.

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
    return loads(zlib.decompress(row[0]))


# --------------------------------------------------------------------------- #
# Global E-step (run once between M-step fan-outs)                             #
# --------------------------------------------------------------------------- #

def e_step(db_path: str, iteration: int) -> float:
    """Recompute fractional slot weights from the just-finished M-step.

    For every multimap group, normalises its members' current origin
    rates ``λ`` across the group's slots and distributes the group's read
    count accordingly, accumulating a new weight per slot.  Writes those
    weights as ``group_weights`` for ``iteration + 1`` and returns a
    convergence metric (max relative change versus ``iteration``'s
    weights).

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
    Vectorised with pandas/numpy: the per-group normalisation and per-slot
    accumulation are ``groupby`` reductions rather than a Python loop over
    every multimap group, which matters at genome scale (tens of millions
    of groups/slots).  The result is identical to the scalar formulation
    up to floating-point summation order.
    """
    import numpy as np
    import pandas as pd

    db = _prepare_writer(db_path)
    cur = db.cursor()
    it_next = iteration + 1

    # λ is stored one blob per locus; expand to per-slot records so it can
    # be joined against the (per-slot) MMG membership.
    lam_records: list = []
    cur.execute(
        "SELECT locus_id, lam_blob FROM group_lambdas WHERE iteration = ?",
        (iteration,),
    )
    for locus_id, blob in cur.fetchall():
        for (run_id, gk), lam_val in loads(blob).items():
            lam_records.append((run_id, locus_id, gk, lam_val))

    slots = pd.DataFrame(
        cur.execute(
            "SELECT mmg_id, locus_id, group_key FROM multimap_group_slots"
        ).fetchall(),
        columns=["mmg_id", "locus_id", "group_key"],
    )
    if slots.empty:
        db.close()
        return 0.0
    groups = pd.DataFrame(
        cur.execute(
            "SELECT mmg_id, run_id, count FROM multimap_groups"
        ).fetchall(),
        columns=["mmg_id", "run_id", "count"],
    )
    lam = pd.DataFrame(
        lam_records, columns=["run_id", "locus_id", "group_key", "lam"]
    )

    # Attach each slot's run_id + MMG read count (via mmg_id), then its λ
    # (via the slot key); a slot with no λ this iteration defaults to 0.
    df = slots.merge(groups, on="mmg_id", how="left")
    df = df.merge(lam, on=["run_id", "locus_id", "group_key"], how="left")
    df["lam"] = df["lam"].fillna(0.0)

    # Responsibility per slot: λ / Σλ within its group, or a uniform 1/n
    # split when the group's λ sums to zero (its read fits no ORF anywhere,
    # so it is neither lost nor arbitrarily concentrated).
    grp = df.groupby("mmg_id", sort=False)["lam"]
    lam_sum = grp.transform("sum").to_numpy()
    n_slots = grp.transform("size").to_numpy()
    lam_arr = df["lam"].to_numpy()
    pos = lam_sum > 0.0
    frac = np.where(pos, lam_arr / np.where(pos, lam_sum, 1.0), 1.0 / n_slots)
    df["weight"] = df["count"].to_numpy() * frac

    # New weight per slot = Σ contributions across the MMGs sharing it.
    new_w = df.groupby(
        ["run_id", "locus_id", "group_key"], sort=False, as_index=False
    )["weight"].sum()

    # Previous weights (one blob per locus) for the convergence metric.
    old_by_locus: dict = {}
    cur.execute(
        "SELECT locus_id, weight_blob FROM group_weights WHERE iteration = ?",
        (iteration,),
    )
    for locus_id, blob in cur.fetchall():
        old_by_locus[locus_id] = loads(blob)

    # Collapse the new per-slot weights into one blob per locus, computing
    # the L1 mass reassigned vs the previous weights in the same pass.
    # Convergence = Σ|w_new − w_old| / Σ w_new — a total-variation measure
    # robust to the slot count.  From iteration 1 on the total mass is
    # conserved (each read's weight sums to one); iteration 0's large value
    # reflects the one-off removal of the classic full-weight double count.
    weight_rows: list = []
    moved = 0.0
    total_mass = 0.0
    for locus_id, sub in new_w.groupby("locus_id", sort=False):
        d = {
            (r, int(g)): float(w)
            for r, g, w in zip(
                sub["run_id"], sub["group_key"], sub["weight"]
            )
        }
        weight_rows.append((it_next, locus_id, dumps(d)))
        old_d = old_by_locus.get(locus_id, {})
        for k, w in d.items():
            moved += abs(w - old_d.get(k, 0.0))
            total_mass += w
        for k, w in old_d.items():
            if k not in d:
                moved += abs(w)
    # Loci in the old weights but absent from the new (slot set is fixed, so
    # none in practice) — count their full mass as moved.
    new_loci = set(new_w["locus_id"].unique())
    for locus_id, old_d in old_by_locus.items():
        if locus_id not in new_loci:
            moved += sum(abs(w) for w in old_d.values())
    rel = moved / total_mass if total_mass > 0 else 0.0

    # Write new weights (one blob per locus), then prune spent state.  Only
    # weights[it+1] (the next M-step's response) and locus_activities[it]
    # (its warm start) are still needed; with one blob per locus these
    # writes and deletes touch ~40K rows, not tens of millions.
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
