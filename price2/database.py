"""SQLite access to ``price.db``.

Everything PRICE2 persists between stages lives in one SQLite database in
the working directory.  This module owns what every other module used to
spell out by hand:

* how a connection is opened — :func:`connect` waits on a busy database
  instead of failing, commits on a clean exit when asked to, and always
  closes;
* what the tables look like — the ``create_*`` functions hold every DDL
  statement, so the schema can be read in one place;
* how Python objects become blobs — the ``*_blob`` codecs, one pair per
  storage format.  The formats are part of the on-disk contract with
  existing databases and must not change.

Worker processes must open their own connections: a SQLite handle must not
be carried across ``fork()``.
"""

from __future__ import annotations

import sqlite3 as sql
import zlib
from collections.abc import Iterator
from contextlib import contextmanager
from pickle import dumps, loads

#: How long a connection waits for a locked database before giving up, in
#: seconds.  Many workers commit small transactions concurrently during the
#: EM, so waiting is the norm, not the exception.
DEFAULT_TIMEOUT = 120.0

STATE_TABLE = "run_state"
PROGRESS_TABLE = "progress"

# Tables holding the multimapping-EM state; dropped and re-created together
# by :func:`create_em_tables`.  ``multimap_alignments``, ``multimap_groups``
# and ``multimap_group_slots`` are legacy tables of earlier releases that are
# only ever dropped (the linkage lives in ``multimap_linkage.npz`` now).
_EM_TABLES = (
    "multimap_alignments",
    "multimap_group_slots",
    "multimap_groups",
    "multimap_slot_base",
    "group_weights",
    "group_lambdas",
    "locus_activities",
    "prepared_loci",
    "prepared_loci_cache",
)


# --------------------------------------------------------------------------- #
# Connections
# --------------------------------------------------------------------------- #


@contextmanager
def connect(
    db_path: str,
    *,
    timeout: float = DEFAULT_TIMEOUT,
    commit: bool = False,
    wal_writer: bool = False,
) -> Iterator[sql.Connection]:
    """Open ``price.db`` for the duration of a ``with`` block.

    Parameters
    ----------
    db_path : str
        Path to the database; created if it does not exist.
    timeout : float, optional
        Seconds to wait for a locked database before raising
        ``sqlite3.OperationalError``.
    commit : bool, optional
        Commit when the block exits without an exception.  An exception
        closes the connection uncommitted, which rolls the transaction
        back.
    wal_writer : bool, optional
        Tune the connection for the many concurrent writers of the EM
        M-step: ``synchronous = NORMAL`` is safe under WAL (see
        :func:`enable_wal`) and avoids an ``fsync`` per commit.

    Yields
    ------
    sqlite3.Connection
    """
    db = sql.connect(db_path, timeout=timeout)
    try:
        if wal_writer:
            db.execute("PRAGMA synchronous = NORMAL")
        yield db
        if commit:
            db.commit()
    finally:
        db.close()


def enable_wal(db_path: str) -> None:
    """Switch the database to write-ahead logging.

    The EM M-step makes every worker a writer, from up to
    ``config.processes`` processes at once.  WAL lets many readers and one
    writer proceed without blocking, and combined with the busy timeout
    serialises the brief commit windows safely.  ``journal_mode`` is a
    persistent property of the database file, so this is done once per run.

    Parameters
    ----------
    db_path : str
        Path to the database.
    """
    with connect(db_path) as db:
        db.execute("PRAGMA journal_mode = WAL")


def table_exists(cur: sql.Cursor, name: str) -> bool:
    """Return whether table *name* exists in the connected database."""
    row = cur.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (name,),
    ).fetchone()
    return row is not None


# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #


def create_collection_tables(cur: sql.Cursor) -> None:
    """Create the tables the data collection fills, if they do not exist.

    ``runs`` and ``loci`` hold one pickled object per row; ``reads`` and
    ``transcript_read_counts`` hold one compressed blob per (locus, run).
    """
    cur.execute(
        """CREATE TABLE IF NOT EXISTS runs (
               run_id   TEXT PRIMARY KEY,
               run_blob BLOB
           )"""
    )
    cur.execute(
        """CREATE TABLE IF NOT EXISTS loci (
               locus_id TEXT PRIMARY KEY,
               loc_blob BLOB
           )"""
    )
    cur.execute(
        """CREATE TABLE IF NOT EXISTS reads (
               locus_id   TEXT NOT NULL,
               run_id     TEXT NOT NULL,
               reads_blob BLOB NOT NULL
           )"""
    )
    cur.execute(
        """CREATE TABLE IF NOT EXISTS transcript_read_counts (
               locus_id                    TEXT NOT NULL,
               run_id                      TEXT NOT NULL,
               transcript_read_counts_blob BLOB NOT NULL
           )"""
    )


def create_read_indexes(cur: sql.Cursor) -> None:
    """Index the per-locus read tables once the collection has filled them."""
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_reads_locus_id ON reads(locus_id)"
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_trc_locus_id "
        "ON transcript_read_counts(locus_id)"
    )


def create_state_table(cur: sql.Cursor) -> None:
    """Create the key/value table :mod:`price2.run_state` records into."""
    cur.execute(
        f"CREATE TABLE IF NOT EXISTS {STATE_TABLE} "
        "(key TEXT PRIMARY KEY, value TEXT NOT NULL)"
    )


def create_progress_table(cur: sql.Cursor) -> None:
    """Create the table of finished ``(stage, key)`` pairs (:mod:`price2.run_state`)."""
    cur.execute(
        f"CREATE TABLE IF NOT EXISTS {PROGRESS_TABLE} ("
        "stage TEXT NOT NULL, key TEXT NOT NULL, value TEXT NOT NULL DEFAULT '', "
        "PRIMARY KEY (stage, key))"
    )


#: Layout version of the ``prepared_loci`` / ``prepared_loci_cache`` blobs
#: (a pickled ``Locus`` and its ``ReadRouting``).  Part of the deconvolution fingerprint
#: (:func:`price2.run_state.deconvolution_fingerprint`), so a run resumed by
#: a PRICE2 that pickles the locus differently starts its EM over instead of
#: unpickling blobs it cannot use.  Bump it whenever the pickled state
#: changes shape.
PREPARED_LOCI_FORMAT = "3"


def create_em_tables(cur: sql.Cursor) -> None:
    """Create the multimapping-EM state tables, dropping any stale copies.

    See :mod:`price2.multimap` for what each table holds and why the
    per-slot vectors are stored as one blob per locus.
    """
    for table in _EM_TABLES:
        cur.execute(f"DROP TABLE IF EXISTS {table}")

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
               iteration       INTEGER NOT NULL,
               locus_id        TEXT    NOT NULL,
               activities_blob BLOB    NOT NULL,
               PRIMARY KEY (iteration, locus_id)
           )"""
    )
    cur.execute(
        """CREATE TABLE prepared_loci (
               locus_id  TEXT PRIMARY KEY,
               prep_blob BLOB NOT NULL
           )"""
    )
    create_prepared_cache_table(cur)


def create_prepared_cache_table(cur: sql.Cursor) -> None:
    """Create ``prepared_loci_cache`` if it does not exist.

    Kept separate from :func:`create_em_tables` because databases collected
    before the table existed must gain it on their first warm start.
    """
    cur.execute(
        """CREATE TABLE IF NOT EXISTS prepared_loci_cache (
               locus_id   TEXT PRIMARY KEY,
               cache_blob BLOB NOT NULL
           )"""
    )


# --------------------------------------------------------------------------- #
# Blob codecs
# --------------------------------------------------------------------------- #


def pickle_blob(obj: object) -> bytes:
    """Serialise *obj* as a plain pickle (``runs``, ``loci``, slot bases)."""
    return dumps(obj)


def unpickle_blob(blob: bytes) -> object:
    """Inverse of :func:`pickle_blob`."""
    return loads(blob)


def compress_blob(obj: object, protocol: int | None = None) -> bytes:
    """Serialise *obj* as a zlib-compressed pickle.

    Used for the large per-locus payloads: reads, transcript read counts,
    prepared loci and their routing caches, and EM activities.
    """
    return zlib.compress(dumps(obj, protocol=protocol))


def decompress_blob(blob: bytes) -> object:
    """Inverse of :func:`compress_blob`."""
    return loads(zlib.decompress(blob))
