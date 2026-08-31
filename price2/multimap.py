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
Raw in-locus multimapping alignments are *not* stored in ``price.db``.  A
human genome run emits ~7.3e8 of them, and both the insert and the
``ORDER BY``-driven read-back dominated data collection.  They are instead
spilled to plain ``.npy`` arrays under ``<w_dir>/mm_spill/<run_id>/`` (one
file triple per locus chunk, written in parallel by the collection
workers) and consumed once by :func:`build_multimap_index`, which deletes
the directory afterwards.  Loci are referenced by their index into
``mm_spill/loci.npy`` rather than by their ``loc_*`` string.

The derived EM state lives in ``price.db`` next to the existing tables:

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
import shutil
import sqlite3 as sql
import struct
import zlib
from collections import defaultdict
from multiprocessing import Pool
from pickle import dumps, loads

import numpy as np

logger = logging.getLogger(__name__)

#: Directory (under the working directory) holding the spilled raw
#: multimapping alignments between collection and index construction.
SPILL_DIRNAME = "mm_spill"


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

def create_em_tables(cur: sql.Cursor) -> None:
    """Create the derived EM state tables (dropping any stale copies)."""
    for table in (
        "multimap_alignments",  # legacy: superseded by the mm_spill files
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
    # No index on mmg_id: the sole reader, ``_build_linkage``, full-scans the
    # table once, and building a 4.5e7-row index cost more than the scan it
    # never accelerated.
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
    db = sql.connect(db_path, timeout=120)
    try:
        cur = db.cursor()
        cur.execute("PRAGMA busy_timeout = 120000")
        try:
            row = cur.execute(
                "SELECT MAX(iteration) FROM group_weights"
            ).fetchone()
        except sql.OperationalError:  # tables absent (never ran an EM here)
            return None
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
    finally:
        db.close()


# --------------------------------------------------------------------------- #
# Alignment spill files (written during collection, consumed by the index)     #
# --------------------------------------------------------------------------- #

def spill_dir(db_path: str) -> str:
    """Return the multimapping spill directory next to ``price.db``."""
    return os.path.join(os.path.dirname(db_path) or ".", SPILL_DIRNAME)


def init_spill(db_path: str, locus_ids: list[str]) -> str:
    """Prepare the spill directory and record the locus-index mapping.

    Spilled alignments name their locus by position in *locus_ids*, so a
    resumed collection may only keep an existing spill when the ordering is
    unchanged; otherwise the directory is rebuilt from scratch.  Spills of
    runs that were already collected are preserved, since the index build
    needs every run's alignments.

    Parameters
    ----------
    db_path : str
        Path to ``price.db``.
    locus_ids : list of str
        Locus ids in the order the collection workers index them.

    Returns
    -------
    str
        Path to the spill directory.
    """
    root = spill_dir(db_path)
    loci_path = os.path.join(root, "loci.npy")
    wanted = np.asarray(locus_ids, dtype=np.str_)

    if os.path.exists(loci_path):
        stored = np.load(loci_path)
        if stored.shape == wanted.shape and (stored == wanted).all():
            return root
        logger.warning(
            "existing multimap spill in %s was built for a different locus "
            "set; discarding it", root,
        )
    shutil.rmtree(root, ignore_errors=True)
    os.makedirs(root)
    np.save(loci_path, wanted)
    return root


def reset_run_spill(root: str, run_id: str) -> None:
    """Drop any spill left by a previous, incomplete pass over ``run_id``."""
    shutil.rmtree(os.path.join(root, run_id), ignore_errors=True)


#: Rows a worker buffers before spilling them to disk.  Only bounds peak
#: memory (~20 bytes/row, so ~40 MB per worker); the common case is that a
#: worker never reaches it and spills exactly once, when the run finishes.
SPILL_FLUSH_ROWS = 2_000_000

#: ``run_spill_dir -> {"q": [...], "l": [...], "g": [...], "n": int, "seq": int}``
#: Process-local: each collection worker accumulates only its own alignments.
_SPILL_BUFFERS: dict[str, dict] = {}
_SPILL_FINALIZER = None

_SPILL_COLUMNS = (("q", np.uint64), ("l", np.uint32), ("g", np.uint64))


def write_spill(
    run_spill_dir: str,
    qname_hashes: list[int],
    locus_indices: list[int],
    group_keys: list[int],
) -> None:
    """Buffer one locus chunk's multimapping alignments for the spill.

    Nothing reaches disk here in the normal case.  Alignments accumulate in
    a process-local buffer and are written by :func:`flush_spill`, either
    when the buffer passes ``SPILL_FLUSH_ROWS`` or when the worker exits.

    Writing per *chunk* instead was the obvious thing and does not scale:
    a chunk is at most four loci, so a human run is ~12 000 chunks, and at
    three files each that is ~36 000 files **per run** — 1.6 M for a 45-BAM
    set, which exhausts a typical filesystem inode quota long before it runs
    out of space.  Buffering per worker writes three files per worker
    instead, ~100x fewer, for the same bytes.

    Parameters
    ----------
    run_spill_dir : str
        ``<spill_dir>/<run_id>``; created when the buffer is flushed.
    qname_hashes, locus_indices, group_keys : list of int
        One entry per in-locus multimapping alignment.
    """
    if not qname_hashes:
        return

    global _SPILL_FINALIZER
    if _SPILL_FINALIZER is None:
        # NOT atexit: multiprocessing children end in os._exit(), which skips
        # atexit handlers entirely.  util.Finalize is what _exit_function runs,
        # and it is only reached if the pool is closed and joined rather than
        # terminated (see DataCollector._collect_run).
        from multiprocessing.util import Finalize

        _SPILL_FINALIZER = Finalize(None, flush_spill, exitpriority=16)

    buf = _SPILL_BUFFERS.get(run_spill_dir)
    if buf is None:
        buf = {"q": [], "l": [], "g": [], "n": 0, "seq": 0}
        _SPILL_BUFFERS[run_spill_dir] = buf

    for (key, dtype), values in zip(
        _SPILL_COLUMNS, (qname_hashes, locus_indices, group_keys)
    ):
        buf[key].append(np.asarray(values, dtype=dtype))
    buf["n"] += len(qname_hashes)

    if buf["n"] >= SPILL_FLUSH_ROWS:
        flush_spill(run_spill_dir)


def flush_spill(run_spill_dir: str | None = None) -> int:
    """Write buffered alignments out as one ``.npy`` triple per call.

    Called on the size threshold, from the worker's exit finalizer, and
    directly by the in-process fallback path.  Files are named by process id
    and flush sequence, which keeps them unique across the pool;
    :func:`_load_run_spill` only globs and concatenates, so the naming and
    the number of files carry no meaning.

    Parameters
    ----------
    run_spill_dir : str, optional
        Flush only this run's buffer; ``None`` (the default, and what the
        exit finalizer uses) flushes every buffer this process holds.

    Returns
    -------
    int
        Number of alignments written.
    """
    targets = list(_SPILL_BUFFERS) if run_spill_dir is None else [run_spill_dir]
    written = 0
    for target in targets:
        buf = _SPILL_BUFFERS.get(target)
        if buf is None or buf["n"] == 0:
            continue
        os.makedirs(target, exist_ok=True)
        stem = f"{os.getpid():07d}-{buf['seq']:04d}"
        for key, dtype in _SPILL_COLUMNS:
            parts = buf[key]
            arr = (
                np.concatenate(parts) if parts else np.empty(0, dtype=dtype)
            )
            final = os.path.join(target, f"{stem}.{key}.npy")
            # Ends in .npy already, so np.save will not append another suffix;
            # the rename keeps a crash from leaving a half-written array.
            tmp = f"{final}.tmp.npy"
            np.save(tmp, arr)
            os.replace(tmp, final)
            parts.clear()
        written += buf["n"]
        buf["n"] = 0
        buf["seq"] += 1
    return written


def discard_spill(db_path: str) -> None:
    """Delete the spill directory (called once the index is built)."""
    shutil.rmtree(spill_dir(db_path), ignore_errors=True)


# --------------------------------------------------------------------------- #
# Index construction (run once, after read collection)                         #
# --------------------------------------------------------------------------- #

def _npy_row_count(path: str) -> int:
    """Return an ``.npy`` file's row count by reading only its header."""
    with open(path, "rb") as fh:
        version = np.lib.format.read_magic(fh)
        if version == (1, 0):
            shape, _, _ = np.lib.format.read_array_header_1_0(fh)
        else:
            shape, _, _ = np.lib.format.read_array_header_2_0(fh)
    return shape[0]


def _load_run_spill(run_spill_dir: str) -> tuple:
    """Concatenate a run's spilled chunk arrays into three flat columns.

    Sizes are taken from the ``.npy`` headers so the destination can be
    allocated once and each chunk read straight into it — a run holds
    thousands of chunk files, so neither mapping them all at once (file
    descriptors) nor concatenating a list of arrays (peak memory) is
    acceptable.
    """
    chunks = sorted(
        f[:-6] for f in os.listdir(run_spill_dir) if f.endswith(".q.npy")
    )
    if not chunks:
        return (
            np.empty(0, dtype=np.uint64),
            np.empty(0, dtype=np.uint32),
            np.empty(0, dtype=np.uint64),
        )

    paths = [
        tuple(os.path.join(run_spill_dir, f"{c}.{s}.npy") for s in "qlg")
        for c in chunks
    ]
    total = sum(_npy_row_count(p[0]) for p in paths)
    qh = np.empty(total, dtype=np.uint64)
    li = np.empty(total, dtype=np.uint32)
    gk = np.empty(total, dtype=np.uint64)

    off = 0
    for q_path, l_path, g_path in paths:
        part = np.load(q_path)
        n = part.shape[0]
        qh[off:off + n] = part
        li[off:off + n] = np.load(l_path)
        gk[off:off + n] = np.load(g_path)
        off += n
    return qh, li, gk


def _index_run(args: tuple) -> tuple:
    """Collapse one run's spilled alignments into multimap groups.

    Fully vectorised: the per-read slot grouping and the collapse of
    identical slot *sets* are sorts and boundary scans over flat arrays,
    replacing the row-by-row Python loop over the (~7e8-row) alignment
    table that dominated the previous implementation.

    Parameters
    ----------
    args : tuple
        ``(run_spill_dir,)``.

    Returns
    -------
    tuple
        ``(counts, slot_k, slot_li, slot_gk)`` where ``counts[i]`` is the
        number of reads in MMG ``i``, ``slot_k[i]`` its slot count, and
        ``slot_li`` / ``slot_gk`` the concatenated per-MMG slots (a CSR
        layout with row lengths ``slot_k``).
    """
    (run_spill_dir,) = args
    empty = (
        np.empty(0, np.int64), np.empty(0, np.int64),
        np.empty(0, np.uint32), np.empty(0, np.uint64),
    )

    qh, li, gk = _load_run_spill(run_spill_dir)
    if qh.size == 0:
        return empty

    # 1. Drop reads that can only have a single in-locus slot.  Rows of one
    #    read are distinct (locus, group_key) pairs — one row per alignment —
    #    so a read with a single row has a single slot.  Reads with >=2 rows
    #    are a superset of those with >=2 slots; the exact filter is applied
    #    after the sort below.  This prunes ~80% of the rows before the
    #    (much more expensive) three-key sort.
    _, inv, counts = np.unique(qh, return_inverse=True, return_counts=True)
    del qh
    inv = inv.reshape(-1)  # numpy>=2 returns the input's shape
    keep = (counts >= 2)[inv]  # index the small array, not an int64 gather
    del counts
    read_id = inv[keep]
    del inv
    li = li[keep]
    gk = gk[keep]
    del keep
    if read_id.size == 0:
        return empty

    # 2. Sort rows by (read, locus, group_key): each read's slots become
    #    contiguous and canonically ordered, so a signature is a slice.
    order = np.lexsort((gk, li, read_id))
    read_id = read_id[order]
    li = li[order]
    gk = gk[order]
    del order

    # 3. Deduplicate identical slots within a read, then recount.
    n = read_id.size
    fresh = np.empty(n, dtype=bool)
    fresh[0] = True
    np.logical_or(
        read_id[1:] != read_id[:-1],
        np.logical_or(li[1:] != li[:-1], gk[1:] != gk[:-1]),
        out=fresh[1:],
    )
    read_id = read_id[fresh]
    li = li[fresh]
    gk = gk[fresh]
    del fresh

    n = read_id.size
    first = np.empty(n, dtype=bool)
    first[0] = True
    np.not_equal(read_id[1:], read_id[:-1], out=first[1:])
    del read_id
    starts = np.flatnonzero(first)
    del first
    k = np.diff(np.append(starts, n))

    # A read whose duplicate rows collapsed to one slot drops out here.
    ok = k >= 2
    if not ok.all():
        row_sel = np.repeat(ok, k)
        li = li[row_sel]
        gk = gk[row_sel]
        k = k[ok]
        starts = np.concatenate(([0], np.cumsum(k)[:-1]))
        del row_sel
    del ok
    if k.size == 0:
        return empty

    # 4. Collapse reads that share an identical slot set into one MMG.
    #    Reads are bucketed by slot count so each bucket is a dense
    #    (n_reads, k) matrix on which `np.unique(axis=0)` is exact.
    g_counts: list = []
    g_k: list = []
    g_li: list = []
    g_gk: list = []
    for kk in np.unique(k):
        sel = np.flatnonzero(k == kk)
        base_idx = starts[sel][:, None] + np.arange(kk)
        mat = np.empty((sel.size, 2 * kk), dtype=np.uint64)
        mat[:, 0::2] = li[base_idx]
        mat[:, 1::2] = gk[base_idx]
        uniq, cnt = np.unique(mat, axis=0, return_counts=True)
        g_counts.append(cnt.astype(np.int64))
        g_k.append(np.full(uniq.shape[0], kk, dtype=np.int64))
        g_li.append(uniq[:, 0::2].reshape(-1).astype(np.uint32))
        g_gk.append(uniq[:, 1::2].reshape(-1))

    return (
        np.concatenate(g_counts),
        np.concatenate(g_k),
        np.concatenate(g_li),
        np.concatenate(g_gk),
    )


def build_multimap_index(db_path: str, processes: int = 1) -> int:
    """Collapse spilled alignments into multimap groups and seed weights.

    Reads the per-run spill files written during collection, keeps only
    reads that touch **≥2** distinct in-locus slots, collapses reads that
    share an identical slot set into one multimap group (MMG) with a
    member count, and writes the derived tables (``multimap_groups``,
    ``multimap_group_slots``, ``multimap_slot_base``) plus the iteration-0
    ``group_weights`` seed (``weight = base`` → full counts, i.e. classic
    behaviour before any reassignment).  The spill directory is deleted on
    success.

    Parameters
    ----------
    db_path : str
        Path to ``price.db``.
    processes : int, optional
        Maximum number of runs to index concurrently.  Each concurrent run
        holds its own alignment columns in memory (~20 bytes per in-locus
        multimapping alignment, plus a like-sized sort buffer), so this is
        the knob that bounds peak RSS.

    Returns
    -------
    int
        Number of multimap groups created.
    """
    root = spill_dir(db_path)
    run_ids = sorted(
        d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))
    ) if os.path.isdir(root) else []

    if not run_ids:
        db = sql.connect(db_path, timeout=120)
        create_em_tables(db.cursor())
        db.commit()
        db.close()
        _invalidate_linkage(db_path)
        logger.warning(
            "no multimapping alignments were spilled to %s; the EM linkage "
            "index is empty", root,
        )
        return 0

    locus_ids = np.load(os.path.join(root, "loci.npy")).tolist()
    tasks = [(os.path.join(root, r),) for r in run_ids]
    n_proc = max(1, min(processes, len(tasks)))

    # Fork before opening the database: SQLite connections must not be carried
    # across fork().  ``imap`` then lets the parent insert one run's rows while
    # the remaining runs are still being collapsed.
    pool = Pool(n_proc) if n_proc > 1 else None

    db = sql.connect(db_path, timeout=120)
    cur = db.cursor()
    cur.execute("PRAGMA busy_timeout = 120000")
    create_em_tables(cur)
    db.commit()

    # Merge the per-run results into the global MMG id space and accumulate
    # per-slot baselines.  Everything below is O(#slots), not O(#alignments).
    base_by_locus: dict = defaultdict(dict)
    mmg_id = 0
    n_slots = 0

    try:
        results = (
            pool.imap(_index_run, tasks) if pool
            else (_index_run(t) for t in tasks)
        )
        for run_id, (counts, slot_k, slot_li, slot_gk) in zip(run_ids, results):
            if counts.size == 0:
                continue
            mmg_ids = np.arange(mmg_id, mmg_id + counts.size, dtype=np.int64)
            mmg_id += counts.size

            cur.executemany(
                "INSERT INTO multimap_groups VALUES (?, ?, ?)",
                zip(mmg_ids.tolist(), [run_id] * counts.size, counts.tolist()),
            )
            cur.executemany(
                "INSERT INTO multimap_group_slots VALUES (?, ?, ?)",
                zip(
                    np.repeat(mmg_ids, slot_k).tolist(),
                    # __getitem__ hands back the interned locus string rather
                    # than minting one per slot row (there are ~4.5e7 of them).
                    map(locus_ids.__getitem__, slot_li.tolist()),
                    slot_gk.tolist(),
                ),
            )

            # base[slot] = Σ read counts of the MMGs passing through it.
            per_slot_count = np.repeat(counts, slot_k)
            order = np.lexsort((slot_gk, slot_li))
            s_li = slot_li[order]
            s_gk = slot_gk[order]
            new = np.empty(s_li.size, dtype=bool)
            new[0] = True
            np.logical_or(
                s_li[1:] != s_li[:-1], s_gk[1:] != s_gk[:-1], out=new[1:]
            )
            slot_of = np.cumsum(new) - 1
            totals = np.bincount(slot_of, weights=per_slot_count[order])
            u_li = s_li[new]
            u_gk = s_gk[new]
            n_slots += u_li.size

            # One pickled {(run_id, group_key): base} dict per locus.
            bounds = np.flatnonzero(
                np.concatenate(([True], u_li[1:] != u_li[:-1]))
            )
            for b, e in zip(bounds.tolist(), bounds[1:].tolist() + [u_li.size]):
                d = base_by_locus[locus_ids[u_li[b]]]
                for g, t in zip(u_gk[b:e].tolist(), totals[b:e].tolist()):
                    d[(run_id, g)] = t
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    base_rows = [(locus_id, dumps(d)) for locus_id, d in base_by_locus.items()]
    cur.executemany("INSERT INTO multimap_slot_base VALUES (?, ?)", base_rows)
    # Released before the re-read below: at genome scale the baseline is ~4e7
    # slots, and `_baseline_weight_rows` loads every blob back again.
    del base_rows, base_by_locus
    # iteration-0 weights == baseline == full counts (classic behaviour), as a
    # dense buffer in canonical slot order; reads back the rows just inserted,
    # so it must share this cursor's transaction.
    cur.executemany(
        "INSERT INTO group_weights VALUES (0, ?, ?)",
        _baseline_weight_rows(cur),
    )
    db.commit()
    db.close()

    _invalidate_linkage(db_path)
    discard_spill(db_path)
    logger.info("multimap index: %d groups over %d slots", mmg_id, n_slots)
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

    # A non-finite weight must stop the EM, and it takes an explicit check to
    # make it do so: ``NaN > 0`` is False, so a single NaN anywhere in the
    # slots would fall through to the ``else`` below and report a convergence
    # metric of exactly 0.0 — indistinguishable from a perfectly converged
    # run, and enough to end the loop after one iteration.
    if not (np.isfinite(total_mass) and np.isfinite(moved)):
        n_bad = int((~np.isfinite(new_slot)).sum())
        db.close()
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
