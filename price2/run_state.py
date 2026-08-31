"""Run-state bookkeeping that makes a PRICE2 run resumable.

A run is expensive enough that an interruption — a wall-clock limit, a node
failure, a manual ``Ctrl-C`` — should not throw away the work already done.
Every stage already persists its results (``price.db`` for collection,
``processed_loci.txt`` plus the appended output files for deconvolution, the
per-iteration EM tables for the multimapping EM), so resuming is mostly a
matter of knowing *what may still be trusted*.

That is what this module records.  Beside the collected data in ``price.db``
it keeps a small ``run_state`` key/value table holding two fingerprints of
the configuration:

``collection_fingerprint``
    Over the options that decide what ends up *in* ``price.db``: the
    annotation, the genome, the BAMs, the read-end mode and whether
    multimapping reads were kept.  A change here invalidates the database
    itself, which PRICE2 refuses to overwrite silently — see
    :func:`plan_resume`.
``deconvolution_fingerprint``
    Over every other option that can change a result (filters, penalties,
    solver settings, the EM parameters, the export selection) plus the
    PRICE2 version.  A change here leaves the collected data usable but
    invalidates the deconvolution: its outputs, its per-locus progress and
    the EM checkpoint are dropped and it starts over.

Path options are fingerprinted by their *basename*, so relocating an
analysis directory (staging it on a compute node's local disk, say) does not
invalidate anything, while pointing the run at a different annotation does.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
import hashlib
import logging
import os
import re
import sqlite3 as sql

from price2.config import Config

logger = logging.getLogger(__name__)

#: Options that only say *where* things are or *how fast* to go.  They cannot
#: change a result, so they are excluded from both fingerprints.
_IGNORED_FIELDS: frozenset[str] = frozenset(
    {
        "base_dir",
        "o_dir",
        "w_dir",
        "l_file",
        "log_level",
        "processes",
        "timeout",
        "memory_limit_gb",
        "worker_max_tasks",
        "save_memory",
        "warm_start",
        "mu_gpu",
        "mu_gpu_min_rows",
        "mu_broker",
        "mu_broker_procs",
        "mu_broker_streams",
    }
)

#: Options that decide the content of ``price.db``.  Everything the data
#: collection reads: the inputs themselves, the read-end mode, the run
#: quality gate, and ``multimap_em`` (which decides whether multimapping
#: reads are stored at all and whether the linkage index is built).
_COLLECTION_FIELDS: tuple[str, ...] = (
    "gtf_path",
    "fasta_path",
    "bam_dir",
    "bam_ids",
    "align_ends_type",
    "high_quality_runs_only",
    "multimap_em",
)

#: Options fingerprinted by basename rather than by full path.
_PATH_FIELDS: frozenset[str] = frozenset({"gtf_path", "fasta_path", "bam_dir"})

_TABLE = "run_state"


@dataclass
class ResumePlan:
    """What a warm start may reuse from a previous invocation.

    Attributes
    ----------
    skip_collection : bool
        The database holds a complete data collection; skip straight to
        deconvolution.  When ``False`` the collection stages run, each
        skipping the runs and loci it already stored.
    reuse_deconvolution : bool
        Per-locus progress (``processed_loci.txt``), the already written
        output files and the multimapping-EM checkpoint are valid and are
        picked up where they stopped.  When ``False`` the deconvolution
        starts over with a cleared output directory.
    reason : str
        Human-readable explanation, logged by the caller.
    """

    skip_collection: bool
    reuse_deconvolution: bool
    reason: str


class IncompatibleRunStateError(RuntimeError):
    """Raised when the existing database was collected under other options."""


# --------------------------------------------------------------------------- #
# Fingerprints                                                                  #
# --------------------------------------------------------------------------- #


def _normalise(name: str, value: object) -> str:
    """Return the fingerprint representation of one option value."""
    if name in _PATH_FIELDS and isinstance(value, str) and value:
        return os.path.basename(os.path.normpath(value))
    if isinstance(value, (list, set)):
        return repr(sorted(str(v) for v in value))
    return repr(value)


def _digest(items: list[tuple[str, str]]) -> str:
    """Return a short stable hash over ``(name, value)`` pairs."""
    h = hashlib.sha256()
    for name, value in sorted(items):
        h.update(name.encode())
        h.update(b"=")
        h.update(value.encode())
        h.update(b"\n")
    return h.hexdigest()[:32]


def _price2_version() -> str:
    """Return the installed PRICE2 version, or ``"unknown"``."""
    try:
        from importlib.metadata import PackageNotFoundError, version

        return version("price2")
    except (ImportError, PackageNotFoundError):
        return "unknown"


def collection_fingerprint(config: Config) -> str:
    """Hash the options that decide the content of ``price.db``.

    Parameters
    ----------
    config : Config
        Fully populated configuration object.

    Returns
    -------
    str
        32-character hex digest.
    """
    return _digest(
        [
            (name, _normalise(name, getattr(config, name)))
            for name in _COLLECTION_FIELDS
        ]
    )


def deconvolution_fingerprint(config: Config) -> str:
    """Hash every option that can change a deconvolution result.

    Includes the collection options (a different database implies a
    different deconvolution) and the PRICE2 version, since the cached
    per-locus state of an EM run is pickled and therefore version-bound.

    Parameters
    ----------
    config : Config
        Fully populated configuration object.

    Returns
    -------
    str
        32-character hex digest.
    """
    items = [
        (f.name, _normalise(f.name, getattr(config, f.name)))
        for f in fields(config)
        if f.name not in _IGNORED_FIELDS
    ]
    items.append(("price2_version", _price2_version()))
    return _digest(items)


# --------------------------------------------------------------------------- #
# The state table                                                               #
# --------------------------------------------------------------------------- #


def read_state(db_path: str) -> dict[str, str]:
    """Return the stored run state, empty when there is none.

    Parameters
    ----------
    db_path : str
        Path to ``price.db``.

    Returns
    -------
    dict[str, str]
        Key/value pairs; empty for a missing database, a database written
        by a PRICE2 older than this table, or an unset state.
    """
    if not os.path.exists(db_path):
        return {}
    db = sql.connect(db_path, timeout=120)
    try:
        cur = db.cursor()
        cur.execute("PRAGMA busy_timeout = 120000")
        try:
            rows = cur.execute(f"SELECT key, value FROM {_TABLE}").fetchall()
        except sql.OperationalError:  # table absent
            return {}
        return {key: value for key, value in rows}
    finally:
        db.close()


def write_state(db_path: str, **entries: str) -> None:
    """Create the state table if needed and upsert *entries*.

    Parameters
    ----------
    db_path : str
        Path to ``price.db``.  Created if it does not exist.
    **entries : str
        Key/value pairs to store.
    """
    db = sql.connect(db_path, timeout=120)
    try:
        cur = db.cursor()
        cur.execute("PRAGMA busy_timeout = 120000")
        cur.execute(
            f"CREATE TABLE IF NOT EXISTS {_TABLE} "
            "(key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        cur.executemany(
            f"INSERT OR REPLACE INTO {_TABLE} VALUES (?, ?)",
            [(key, str(value)) for key, value in entries.items()],
        )
        db.commit()
    finally:
        db.close()


# --------------------------------------------------------------------------- #
# The resume decision                                                           #
# --------------------------------------------------------------------------- #


def plan_resume(config: Config, db_path: str) -> ResumePlan:
    """Decide what an existing working directory may contribute to this run.

    Parameters
    ----------
    config : Config
        Fully populated configuration object.
    db_path : str
        Path to ``price.db`` in ``config.w_dir``.

    Returns
    -------
    ResumePlan
        What to skip and what to redo.

    Raises
    ------
    IncompatibleRunStateError
        When the database was collected under different collection options.
        Resuming would mix incompatible data, and silently discarding a
        collection that can take days is worse still, so the run stops and
        leaves the choice to the user.
    """
    state = read_state(db_path)
    stored_collection = state.get("collection_fingerprint")
    collection = collection_fingerprint(config)

    if stored_collection is not None and stored_collection != collection:
        raise IncompatibleRunStateError(
            f"{db_path} was collected with different options "
            f"({', '.join(_COLLECTION_FIELDS)}), so its reads and loci do "
            "not match this configuration. Point the run at a different "
            "base_dir, or set warm_start=false to discard the working "
            "directory and collect the data again."
        )

    if stored_collection is None:
        # A database from a PRICE2 that predates this table, or one whose
        # collection was interrupted before the fingerprint was written.
        # Trust it — the alternative is discarding a valid collection — but
        # adopt it so that any later change is caught.
        logger.warning(
            "%s carries no run state; assuming it matches this "
            "configuration and adopting it.",
            db_path,
        )

    stored_version = state.get("price2_version")
    version = _price2_version()
    if stored_version is not None and stored_version != version:
        logger.warning(
            "%s was written by PRICE2 %s, this is %s. The collected data is "
            "reused; the deconvolution starts over.",
            db_path,
            stored_version,
            version,
        )

    skip_collection = state.get("collection_complete") == "1"
    stored_deconvolution = state.get("deconvolution_fingerprint")
    deconvolution = deconvolution_fingerprint(config)
    reuse_deconvolution = (
        skip_collection and stored_deconvolution == deconvolution
    )

    if not skip_collection:
        reason = "resuming the data collection"
    elif reuse_deconvolution:
        reason = "resuming the ORF deconvolution"
    else:
        reason = (
            "reusing the collected data; the deconvolution options changed, "
            "so the deconvolution starts over"
        )

    return ResumePlan(
        skip_collection=skip_collection,
        reuse_deconvolution=reuse_deconvolution,
        reason=reason,
    )


def record_configuration(config: Config, db_path: str) -> None:
    """Store this run's fingerprints and version in ``price.db``.

    Parameters
    ----------
    config : Config
        Fully populated configuration object.
    db_path : str
        Path to ``price.db``.
    """
    write_state(
        db_path,
        collection_fingerprint=collection_fingerprint(config),
        deconvolution_fingerprint=deconvolution_fingerprint(config),
        price2_version=_price2_version(),
    )


# --------------------------------------------------------------------------- #
# Repairing appended output files                                               #
# --------------------------------------------------------------------------- #
#
# A worker appends a locus's rows to the shared output files and only then
# records the locus in ``processed_loci.txt``.  Nothing makes those two writes
# atomic, so an interrupted run leaves two kinds of damage behind: a half
# written final line, and complete rows belonging to loci that never got
# marked done — which a resume re-runs, appending their rows a second time.
# Both are repaired here, before the resumed run writes anything: partial
# lines are truncated, and every row whose locus is not recorded as finished
# is dropped.
#
# Rows carry their locus in ``locus_id``/``loc_id`` (the TSVs, by column) or
# in the ``loc_id``/``locus_id`` GTF attribute.  BED12 records carry no locus,
# so they are filtered by the ORF ids that survive in the sibling TSV; where
# that TSV does not exist the BED cannot be repaired exactly and the caller is
# told to start the deconvolution over instead.

#: Extensions of the line-oriented files the workers append to.
_APPENDED_SUFFIXES: tuple[str, ...] = (".tsv", ".bed", ".gtf", ".txt")

#: Column names identifying the locus a TSV row belongs to.
_LOCUS_COLUMNS: tuple[str, ...] = ("locus_id", "loc_id")

#: Column index of the locus id in the headerless intermediate TSV written by
#: ``ReadGeneratingRegion.to_tsv_line`` (id, gene_id, loc_id, location, type).
_HEADERLESS_LOCUS_COLUMN = 2

_GTF_LOCUS = re.compile(r'\b(?:locus_id|loc_id) "([^"]+)"')

_TAIL_CHUNK = 1 << 20


def _truncate_to_last_newline(path: str) -> bool:
    """Drop a trailing partial line from *path*.

    Parameters
    ----------
    path : str
        File to repair.

    Returns
    -------
    bool
        ``True`` when bytes were removed.
    """
    size = os.path.getsize(path)
    if size == 0:
        return False
    with open(path, "r+b") as fh:
        fh.seek(-1, os.SEEK_END)
        if fh.read(1) == b"\n":
            return False
        end = size
        while end > 0:
            start = max(0, end - _TAIL_CHUNK)
            fh.seek(start)
            chunk = fh.read(end - start)
            idx = chunk.rfind(b"\n")
            if idx != -1:
                fh.truncate(start + idx + 1)
                return True
            end = start
        # No newline anywhere: the whole file is one partial line.
        fh.truncate(0)
    return True


def repair_partial_lines(*paths: str) -> None:
    """Trim trailing partial lines from appended outputs before a resume.

    A worker appends a locus's rows under a lock and marks the locus done
    only afterwards, so an interrupted run can leave a half-written final
    line.  The locus itself is re-run on resume — but its truncated line
    would survive and corrupt the file, so drop it here.

    Parameters
    ----------
    *paths : str
        Files to repair, and directories to walk for
        ``.tsv``/``.bed``/``.gtf``/``.txt`` files.
    """
    targets: list[str] = []
    for path in paths:
        if os.path.isdir(path):
            for root, _, names in os.walk(path):
                targets += [
                    os.path.join(root, name)
                    for name in names
                    if name.endswith(_APPENDED_SUFFIXES)
                ]
        elif os.path.isfile(path):
            targets.append(path)

    for target in targets:
        try:
            if _truncate_to_last_newline(target):
                logger.warning(
                    "dropped a partial trailing line from %s", target
                )
        except OSError as exc:
            logger.warning("could not repair %s: %s", target, exc)


def _rewrite(path: str, keep) -> int:
    """Rewrite *path* keeping only the lines *keep* accepts.

    Parameters
    ----------
    path : str
        File to filter in place.
    keep : callable
        ``keep(line) -> bool``.

    Returns
    -------
    int
        Number of dropped lines.
    """
    tmp = path + ".resume-tmp"
    dropped = 0
    with open(path) as src, open(tmp, "w") as dst:
        for line in src:
            if keep(line):
                dst.write(line)
            else:
                dropped += 1
    if dropped:
        os.replace(tmp, path)
    else:
        os.remove(tmp)
    return dropped


def _filter_tsv(path: str, done: set[str]) -> set[str]:
    """Drop rows of unfinished loci from a TSV, returning the ids kept.

    Parameters
    ----------
    path : str
        TSV written by :meth:`price2.locus.Locus.to_tsv` or the
        performance-measurement log.
    done : set of str
        Locus ids recorded as finished.

    Returns
    -------
    set of str
        First-column values of the surviving rows — the ORF/region ids a
        sibling BED is filtered by.
    """
    with open(path) as fh:
        first = fh.readline()
    if not first:
        return set()

    header = first.rstrip("\n").split("\t")
    column = next(
        (header.index(name) for name in _LOCUS_COLUMNS if name in header),
        None,
    )
    has_header = column is not None
    if column is None:
        column = _HEADERLESS_LOCUS_COLUMN

    kept: set[str] = set()

    def keep(line: str, _state={"first": True}) -> bool:
        if _state["first"]:
            _state["first"] = False
            if has_header:
                return True
        fields_ = line.rstrip("\n").split("\t")
        if len(fields_) <= column or fields_[column] not in done:
            return False
        kept.add(fields_[0])
        return True

    _rewrite(path, keep)
    return kept


def _filter_gtf(path: str, done: set[str]) -> None:
    """Drop records of unfinished loci from a GTF file."""

    def keep(line: str) -> bool:
        match = _GTF_LOCUS.search(line)
        return match is not None and match.group(1) in done

    _rewrite(path, keep)


def _filter_bed(path: str, kept_ids: set[str]) -> None:
    """Drop records whose ORF id did not survive in the sibling TSV."""

    def keep(line: str) -> bool:
        fields_ = line.split("\t")
        return len(fields_) > 3 and fields_[3].split(":")[0] in kept_ids

    _rewrite(path, keep)


def repair_outputs(o_dir: str, processed_loci_path: str) -> bool:
    """Make the output directory consistent with the finished-locus list.

    Truncates partial trailing lines and drops every row written by a locus
    that is not recorded in ``processed_loci.txt``: the resumed run re-runs
    those loci, and their rows would otherwise appear twice.

    Parameters
    ----------
    o_dir : str
        The run's output directory.
    processed_loci_path : str
        Path to ``processed_loci.txt`` in the working directory.

    Returns
    -------
    bool
        ``True`` when the outputs are consistent.  ``False`` when a BED file
        could not be filtered because its sibling TSV is absent — the caller
        must then discard the output directory and deconvolve every locus
        again, since duplicate BED records cannot be ruled out.
    """
    repair_partial_lines(o_dir, processed_loci_path)

    done: set[str] = set()
    if os.path.exists(processed_loci_path):
        with open(processed_loci_path) as fh:
            done = {line.strip() for line in fh if line.strip()}

    tsv_files: list[str] = []
    gtf_files: list[str] = []
    bed_files: list[str] = []
    txt_files: list[str] = []
    for root, _, names in os.walk(o_dir):
        for name in names:
            path = os.path.join(root, name)
            if name.endswith(".tsv"):
                tsv_files.append(path)
            elif name.endswith(".gtf"):
                gtf_files.append(path)
            elif name.endswith(".bed"):
                bed_files.append(path)
            elif name == "failed_loci.txt":
                txt_files.append(path)

    kept_ids: dict[str, set[str]] = {}
    for path in tsv_files:
        kept_ids[path] = _filter_tsv(path, done)
    for path in gtf_files:
        _filter_gtf(path, done)
    for path in txt_files:
        _rewrite(path, lambda line: line.strip() in done)

    consistent = True
    for path in bed_files:
        sibling = path[: -len(".bed")] + ".tsv"
        if sibling in kept_ids:
            _filter_bed(path, kept_ids[sibling])
        else:
            logger.warning(
                "%s carries no locus column and %s does not exist, so rows "
                "of unfinished loci cannot be dropped from it.",
                path,
                sibling,
            )
            consistent = False

    return consistent
