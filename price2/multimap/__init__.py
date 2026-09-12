"""EM-based fractional assignment of multimapping reads.

Classic PRICE2 solves one locus at a time and counts every read at its
full weight in *every* locus it overlaps, so a read that aligns to
several loci is multi-counted.  This package adds an
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

Reads that share the same *set* of alignment slots behave identically in
the E-step, so they are collapsed into **multimap groups** (MMGs).  A slot
is a ``(locus_id, group_key)`` pair (:mod:`~price2.multimap.keys`).  Only
reads with **>= 2** in-locus slots need EM treatment; intergenic alignments
never enter a locus fetch and are therefore ignored for free.

The package follows the data through the run:

:mod:`~price2.multimap.keys`
    The stable hashes naming reads and slots.
:mod:`~price2.multimap.spill`
    The alignments spilled to disk during collection.
:mod:`~price2.multimap.index`
    Collapsing the spill into the linkage and the slot baselines, once.
:mod:`~price2.multimap.linkage`
    The canonical slot order and the static linkage arrays
    (``multimap_linkage.npz``) the E-step addresses slots by.
:mod:`~price2.multimap.state`
    The per-iteration EM state in ``price.db`` and what a worker reads and
    writes per locus.
:mod:`~price2.multimap.prepared`
    The prepared locus and its routing, reused by every iteration.
:mod:`~price2.multimap.em`
    The E-step itself.

The names below are the package's public API.
"""

from price2.multimap.em import e_step
from price2.multimap.index import build_multimap_index, has_multimap_index
from price2.multimap.keys import alignment_group_key, group_key, qname_hash
from price2.multimap.linkage import Linkage, LocusSlots, linkage_path, load_linkage
from price2.multimap.prepared import (
    load_light_locus,
    load_locus_routing,
    load_prepared_locus,
    save_locus_routing,
    save_prepared_locus,
)
from price2.multimap.spill import (
    SPILL_DIRNAME,
    SPILL_FLUSH_ROWS,
    discard_spill,
    flush_spill,
    init_spill,
    reset_run_spill,
    spill_dir,
    write_spill,
)
from price2.multimap.state import (
    em_resume_point,
    load_locus_mm_data,
    load_warm_activities,
    reset_em_state,
    slot_locus_ids,
    write_locus_em_output,
)

__all__ = [
    "Linkage",
    "LocusSlots",
    "SPILL_DIRNAME",
    "SPILL_FLUSH_ROWS",
    "alignment_group_key",
    "build_multimap_index",
    "discard_spill",
    "e_step",
    "em_resume_point",
    "flush_spill",
    "group_key",
    "has_multimap_index",
    "init_spill",
    "linkage_path",
    "load_linkage",
    "load_light_locus",
    "load_locus_mm_data",
    "load_locus_routing",
    "load_prepared_locus",
    "load_warm_activities",
    "qname_hash",
    "reset_em_state",
    "reset_run_spill",
    "save_locus_routing",
    "save_prepared_locus",
    "slot_locus_ids",
    "spill_dir",
    "write_locus_em_output",
    "write_spill",
]
