"""Vendored subset of the ``gridforge`` package (the KCL-only synthetic-grid
generator) — the *shared* implementations GridCrucible reuses instead of
re-implementing.

Provenance
----------
Copied verbatim from GridForge 0.1.0, its initial public release
(https://github.com/Snodgrass-Research-Group/GridForge-Public). Only the transitive closure actually imported by
GridCrucible is vendored here — six modules:

* :mod:`._gridforge.model`                  — ``GridModel`` container
* :mod:`._gridforge.solver`                 — machine-aware Gurobi env sizing
* :mod:`._gridforge.dispatch.economic_dispatch` — ``EconomicDispatcher`` (UCED/ED)
* :mod:`._gridforge.network.conductors`     — ``ConductorCatalog``
* :mod:`._gridforge.network.impedance`      — ``compute_rxb``, ``GMDTables``, ``ybus``
* :mod:`._gridforge.network.topology`       — ``create_ol_matrices`` and helpers

Why vendored (not a dependency): GridCrucible ships as an independent repo, so
it carries its own copy of the shared GridForge code rather than importing the
sibling package. Keep these files in sync with GridForge when the upstream
shared surface changes; do not edit them here except to track upstream. The
GridCrucible-owned adapters that wire these into the pipeline live in
:mod:`gridcrucible.gridforge_bridge`.
"""
