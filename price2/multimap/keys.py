"""Stable hashes naming reads and slots (Python's ``str`` hash is salted).

A *slot* is a ``(locus, group_key)`` pair: :func:`group_key` identifies a
read's placement at a locus by its spliced coordinates and untemplated-
addition state, so the value computed at collection time matches the one a
worker recomputes from an in-memory alignment; :func:`qname_hash` identifies
the physical read across its alignments.
"""

from __future__ import annotations

import hashlib
import struct


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
