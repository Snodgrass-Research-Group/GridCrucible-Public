"""gridforge-bridge tests: the adapters wiring real
gridforge implementations into GridCrucible's injection points.

The full-scale validation lives in the real-SRTV check (2026-07-09):
4,625 substations / 719 generators / 2,491,428 candidates — EXACT match
with the reference-run GMD index, draws reused verbatim, gridforge RXB
across all paths in ~2 s.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gridcrucible.config import CrucibleConfig

gp = pytest.importorskip("gurobipy")
pytest.importorskip("gridcrucible._gridforge")   # vendored shared GridForge subset
DATA_DIR = CrucibleConfig().io.data_dir

pytestmark = pytest.mark.skipif(
    not (DATA_DIR / "gmd_tables.parquet").exists(),
    reason="run the migration first")


def test_uced_adapter_balances():
    from gridcrucible.gridforge_bridge import uced

    fleet = pd.DataFrame({
        "BusNum": [0, 2, 5], "MinMW": [10.0, 5, 0], "MaxMW": [200.0, 150, 100],
        "MinMVar": -50.0, "MaxMVar": 60.0,
        "FuelType": ["Gas", "Coal", "Hydro"], "UnitType": ["CT", "ST", "HY"],
        "A": [100.0, 120, 50], "B": [25.0, 20, 5], "C": [0.02, 0.015, 0.001],
    })
    x, out = uced(fleet, 1.0, np.array([250.0]), reserve=0.1)
    assert x.sum() == pytest.approx(250.0, rel=1e-6)
    assert len(x) == len(out)
    assert "BusNum" in out.columns              # mapped back to MATLAB schema


def test_kvl_variant_dir_index_consistency(tmp_path):
    """The bridged pooled table must preserve our master cond_index order —
    conductor_idx values index the SAME table everywhere."""
    from gridcrucible.gridforge_bridge import _kvl_variant_dir
    from gridcrucible.network.conductors import load_conductor_catalog

    cfg = CrucibleConfig()
    pooled = pd.read_parquet(_kvl_variant_dir(cfg) / "pooled.parquet")
    master = load_conductor_catalog(cfg.io.data_dir)
    assert len(pooled) == len(master) == 148
    assert (pooled["cond_index"].to_numpy()
            == master.index.to_numpy()).all()
    assert (pooled["mva"].to_numpy() == master["mva"].to_numpy()).all()


def test_gmd_tables_shape():
    from gridcrucible.gridforge_bridge import make_gmd_tables

    cfg = CrucibleConfig()
    g = make_gmd_tables(cfg, region="SRTV")
    assert g.gmd_table.shape == (1000, 6)
    ft = g.gmd_ft(np.array([1, 182, 1000]), np.array([69.0, 69.0, 69.0]))
    assert ft.tolist() == [4.0, 7.0, 15.0]      # GMDDataSRTV.m breakpoints
