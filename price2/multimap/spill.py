"""The alignment spill: raw multimapping alignments between collection and indexing.

Raw in-locus multimapping alignments are *not* stored in ``price.db``.  A
human genome run emits ~7.3e8 of them, and both the insert and the
``ORDER BY``-driven read-back dominated data collection.  They are instead
spilled to plain ``.npy`` arrays under ``<w_dir>/mm_spill/<run_id>/``, one
file triple per collection worker flush, and consumed once by
:func:`price2.multimap.index.build_multimap_index`, which deletes the
directory afterwards.  Loci are referenced by their index into
``mm_spill/loci.npy`` rather than by their ``loc_*`` string.
"""

from __future__ import annotations

import logging
import os
import shutil

import numpy as np

logger = logging.getLogger(__name__)

#: Directory (under the working directory) holding the spilled raw
#: multimapping alignments between collection and index construction.
SPILL_DIRNAME = "mm_spill"


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


#: A run's spill, collapsed into its multimap groups right after the run was
#: mapped (:func:`price2.multimap.index.index_run_spill`), replaces the raw
#: ``<run_id>/`` directory by this file.
GROUPS_SUFFIX = ".groups.npz"


def run_spill_dir(root: str, run_id: str) -> str:
    """The raw spill directory of one run."""
    return os.path.join(root, run_id)


def groups_path(root: str, run_id: str) -> str:
    """The collapsed groups file of one run (see :data:`GROUPS_SUFFIX`)."""
    return os.path.join(root, run_id + GROUPS_SUFFIX)


def run_spill_present(root: str, run_id: str) -> bool:
    """Whether a run's alignments are in the spill, raw or already collapsed."""
    return os.path.isdir(run_spill_dir(root, run_id)) or os.path.exists(
        groups_path(root, run_id)
    )


def spilled_run_ids(root: str) -> list[str]:
    """The runs with a raw spill directory or a groups file under *root*, sorted."""
    if not os.path.isdir(root):
        return []
    found = set()
    for name in os.listdir(root):
        if os.path.isdir(os.path.join(root, name)):
            found.add(name)
        elif name.endswith(GROUPS_SUFFIX):
            found.add(name[: -len(GROUPS_SUFFIX)])
    return sorted(found)


def spill_bytes(root: str, run_id: str) -> int:
    """The size of a run's raw spill files on disk."""
    directory = run_spill_dir(root, run_id)
    if not os.path.isdir(directory):
        return 0
    return sum(
        os.path.getsize(os.path.join(directory, name)) for name in os.listdir(directory)
    )


def reset_run_spill(root: str, run_id: str) -> None:
    """Drop any spill left by a previous, incomplete pass over ``run_id``."""
    shutil.rmtree(run_spill_dir(root, run_id), ignore_errors=True)
    try:
        os.remove(groups_path(root, run_id))
    except FileNotFoundError:
        pass


#: Rows a worker buffers before spilling them to disk.  Only bounds peak
#: memory (~20 bytes/row, so ~40 MB per worker); the common case is that a
#: worker never reaches it and spills exactly once, when the run finishes.
SPILL_FLUSH_ROWS = 2_000_000

_SPILL_COLUMNS = (("q", np.uint64), ("l", np.uint32), ("g", np.uint64))


class _SpillBuffer:
    """One collection worker's alignments of one run not yet written out."""

    __slots__ = ("columns", "n", "seq")

    def __init__(self) -> None:
        #: Per column, the arrays appended so far.
        self.columns: dict[str, list[np.ndarray]] = {
            key: [] for key, _ in _SPILL_COLUMNS
        }
        #: Rows buffered.
        self.n = 0
        #: Flushes done, numbering the files.
        self.seq = 0


#: ``run_spill_dir -> buffer``.  Process-local: each collection worker
#: accumulates only its own alignments.
_SPILL_BUFFERS: dict[str, _SpillBuffer] = {}
_SPILL_FINALIZER = None


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
    instead, ~100x fewer, for the same bytes.  The collector also flushes
    every worker once a run's chunks are all mapped
    (:func:`price2.data_collector._mapping_task`), so a run's spill is
    complete before its reads are committed.

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
        # terminated (see DataCollector._map_reads).
        from multiprocessing.util import Finalize

        _SPILL_FINALIZER = Finalize(None, flush_spill, exitpriority=16)

    buf = _SPILL_BUFFERS.get(run_spill_dir)
    if buf is None:
        buf = _SPILL_BUFFERS[run_spill_dir] = _SpillBuffer()

    for (key, dtype), values in zip(
        _SPILL_COLUMNS, (qname_hashes, locus_indices, group_keys)
    ):
        buf.columns[key].append(np.asarray(values, dtype=dtype))
    buf.n += len(qname_hashes)

    if buf.n >= SPILL_FLUSH_ROWS:
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
        if buf is None or buf.n == 0:
            continue
        os.makedirs(target, exist_ok=True)
        stem = f"{os.getpid():07d}-{buf.seq:04d}"
        for key, dtype in _SPILL_COLUMNS:
            parts = buf.columns[key]
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
        written += buf.n
        buf.n = 0
        buf.seq += 1
    return written


def discard_spill(db_path: str) -> None:
    """Delete the spill directory (called once the index is built)."""
    shutil.rmtree(spill_dir(db_path), ignore_errors=True)
