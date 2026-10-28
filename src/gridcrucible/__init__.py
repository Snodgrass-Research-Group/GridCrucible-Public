"""GridCrucible: KCL-KVL compliant synthetic transmission-grid generator.

Python conversion of the dissertation QNCA pipeline (main_SRTV.m +
MultiLevelProgram.m). Three nested loops refine a fully connected candidate
graph into a realistic network: OHM (inner — QP honoring KVL with L1
sparsity), TUNE (middle — thermal conductor/transformer updates), TRIM
(outer — metric-informed topology reduction).

Start here: docs/ARCHITECTURE.md · docs/CALL_TREE.md.

Shares a handful of helpers with GridForge (the KCL-only pipeline: economic
dispatch, R/X/B impedance, conductor catalog, candidate/topology math). Rather
than depend on that sibling package, GridCrucible carries its own pinned copy
of just those modules under :mod:`gridcrucible._gridforge`, so this repository
installs and runs on its own. The adapters that wire them into the pipeline
live in :mod:`gridcrucible.gridforge_bridge`.
"""

from .config import CrucibleConfig
from .core import GridCrucible
from .model import ACPFResult, CrucibleModel, ScenarioSet

__version__ = "0.1.0"
__all__ = ["GridCrucible", "CrucibleConfig", "CrucibleModel", "ScenarioSet",
           "ACPFResult"]
