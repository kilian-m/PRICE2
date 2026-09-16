"""The per-locus state an EM iteration reuses: the prepared locus and its routing.

``prepared_loci`` holds the prepared :class:`~price2.locus.Locus` (see
:meth:`~price2.locus.Locus.prepared_copy`) and ``prepared_loci_cache`` —
separately, because it holds only arrays — its
:class:`~price2.read_routing.ReadRouting` together with the design matrix
built from it, which a light M-step loads alone.
"""

from __future__ import annotations

from scipy.sparse import csr_matrix

from price2 import database


def save_locus_routing(
    db_path: str, locus_id: str, loc, design_matrix: csr_matrix | None = None
) -> None:
    """Persist a locus's :class:`~price2.read_routing.ReadRouting` on its own.

    The routing references no RGR, transcript or read objects, so a light
    M-step can restore it — plus the locus id and interval, all it
    otherwise needs — without unpickling the locus itself.

    Parameters
    ----------
    db_path : str
        Path to ``price.db``.
    locus_id : str
        Locus identifier.
    loc : Locus
        A locus whose ``routing`` has been built.
    design_matrix : csr_matrix, optional
        The design matrix built from that routing.  It is a pure function of
        the routing and the runs' models, both fixed for the run, so storing
        it saves every later pass its construction.
    """
    payload = {"id": loc.id, "iv": loc.iv, "routing": loc.routing}
    if design_matrix is not None:
        X = design_matrix
        payload["design"] = (X.data, X.indices, X.indptr, X.shape)
    blob = database.compress_blob(payload, protocol=5)
    with database.connect(db_path, wal_writer=True, commit=True) as db:
        db.execute(
            "INSERT OR REPLACE INTO prepared_loci_cache VALUES (?, ?)",
            (locus_id, blob),
        )


def load_locus_routing(db_path: str, locus_id: str):
    """Return the stored routing payload for *locus_id*, or ``None``."""
    with database.connect(db_path) as db:
        cur = db.cursor()
        row = None
        if database.table_exists(cur, "prepared_loci_cache"):
            cur.execute(
                "SELECT cache_blob FROM prepared_loci_cache WHERE locus_id = ?",
                (locus_id,),
            )
            row = cur.fetchone()
    if row is None:
        return None
    return database.decompress_blob(row[0])


def _stored_design_matrix(payload: dict) -> csr_matrix | None:
    stored = payload.get("design")
    if stored is None:
        return None
    data, indices, indptr, shape = stored
    return csr_matrix((data, indices, indptr), shape=shape)


def load_light_locus(db_path: str, locus_id: str):
    """Return a minimal :class:`~price2.locus.Locus` for a light M-step.

    The returned locus carries only ``id``, ``iv`` and ``routing`` (with
    the stored design matrix) — enough for ``set_warm_start``,
    ``assign_reads_to_egs``, ``deconvolve(prune=False)``,
    ``compute_multimap_lambdas`` and ``activities_by_id``.  It has no ``rgrs``
    or ``transcripts``, which is the whole point: restoring those dominates
    the cost of loading a prepared locus.

    Parameters
    ----------
    db_path : str
        Path to ``price.db``.
    locus_id : str
        Locus identifier.

    Returns
    -------
    Locus or None
        ``None`` when no routing was stored for this locus.
    """
    from price2.locus import Locus  # local: locus imports this module

    payload = load_locus_routing(db_path, locus_id)
    if payload is None:
        return None
    return Locus.light(
        payload["id"],
        payload["iv"],
        payload["routing"],
        _stored_design_matrix(payload),
    )


def save_prepared_locus(db_path: str, locus_id: str, loc) -> None:
    """Cache a locus's weight-independent prepared state for later EM passes.

    The RGR candidate set and the coverage/deconvolution-filter results
    depend only on raw (unweighted) reads, so they are identical in every EM
    iteration.  Persisting them after the first light M-step lets subsequent
    iterations skip ORF generation and the two filter passes — the dominant
    per-locus cost.  The reads are reloaded from the ``reads`` table on a
    hit, and the routing (which also holds the equivalence-group geometry)
    lives in its own table (:func:`save_locus_routing`); see
    :meth:`~price2.locus.Locus.prepared_copy`.

    Parameters
    ----------
    db_path : str
        Path to ``price.db``.
    locus_id : str
        Locus identifier.
    loc : Locus
        A prepared locus.
    """
    blob = database.compress_blob(loc.prepared_copy())
    with database.connect(db_path, wal_writer=True, commit=True) as db:
        db.execute(
            "INSERT OR REPLACE INTO prepared_loci VALUES (?, ?)",
            (locus_id, blob),
        )


def load_prepared_locus(db_path: str, locus_id: str):
    """Return a stored prepared :class:`Locus` with its routing, or ``None``.

    The returned locus has an empty ``rsas_dict``; the caller must call
    ``get_reads_from_db`` to repopulate reads before assigning them.
    ``None`` when either the locus or its routing is absent (both are
    written by the first light M-step).

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
    with database.connect(db_path) as db:
        cur = db.cursor()
        row = None
        if database.table_exists(cur, "prepared_loci"):
            cur.execute(
                "SELECT prep_blob FROM prepared_loci WHERE locus_id = ?",
                (locus_id,),
            )
            row = cur.fetchone()
    if row is None:
        return None
    payload = load_locus_routing(db_path, locus_id)
    if payload is None:
        return None
    loc = database.decompress_blob(row[0])
    loc.routing = payload["routing"]
    loc.set_design_matrix(_stored_design_matrix(payload))
    return loc
