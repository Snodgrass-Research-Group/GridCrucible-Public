"""power/dcopf.py — MatpowerDCOPF.m port (docs/algorithms/ElectricalComparison.md).

Property tests on the tiny 6-bus model: DC energy balance, stat-column
contract vs the reference parquets, the rating relax/tighten ladder, and
the MATLAB per-unit sniff.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pandapower")

from gridcrucible.power.dcopf import _PU_SNIFF, run_dcopf_stats

#: Column names the reference parquets (pf_dcopfstats) share.
_MATLAB_COLS = {"Network", "Loads", "LoadMW", "LimitRelaxPercent", "GenMW",
                "MaxAngleDiff", "MeanAngleDiff", "Losses", "LossPercent",
                "PercentLoadedMean", "PercentLoadedStd"}


@pytest.fixture
def snet(tiny_model) -> np.ndarray:
    """The tier in PER-UNIT, mirroring the real ElectricalComparison chain:
    the MATLAB pu sniff (column sums < 35e3 → ×SBase) restores the MW
    values inside run_dcopf_stats. Feeding raw MW from a system this small
    would be inflated ×100 by the sniff — MATLAB shares that quirk."""
    return tiny_model.scenarios.tiers["full"] / 100.0   # (6, 2) complex, pu


class TestRunDCOPFStats:
    def test_feasible_base_case(self, tiny_model, snet):
        df = run_dcopf_stats(tiny_model, snet, table="candidates")
        assert len(df) == snet.shape[1]
        assert _MATLAB_COLS <= set(df.columns)
        assert df["success"].all()
        # feasible without relaxation on the roomy tiny model
        assert (df["LimitRelaxPercent"] == 0).all()
        # DC OPF energy balance: dispatch == load (lossless model)
        assert df["GenMW"].to_numpy() == pytest.approx(
            df["LoadMW"].to_numpy(), rel=1e-4)
        assert (df["MaxAngleDiff"] >= df["MeanAngleDiff"]).all()
        assert (df["PercentLoadedMean"].between(0, 100)).all()

    def test_network_loads_labels(self, tiny_model, snet):
        df = run_dcopf_stats(tiny_model, snet, network="KCL/KVL",
                             loads="DC1", table="candidates")
        assert (df["Network"] == "KCL/KVL").all()
        assert (df["Loads"] == "DC1").all()

    def test_relaxation_ladder_on_tight_ratings(self, tiny_model, snet):
        """Choke the ratings so the base case is DCOPF-infeasible: the port
        must relax (LimitRelaxPercent > 0) and still report success."""
        tight = tiny_model
        tight.candidates["capacity_mva"] = 20.0          # << flows required
        df = run_dcopf_stats(tight, snet, table="candidates",
                             dc_overload_percent_max=np.full(
                                 snet.shape[1], 400.0))
        assert df["success"].all()
        assert (df["LimitRelaxPercent"] > 0).all()
        # tightening never ends ABOVE the first relax step
        assert (df["LimitRelaxPercent"] <= 400.0 * 3).all()

    def test_pu_sniff_matches_mw_input(self, tiny_model, snet):
        """Columns summing below the 35e3 threshold are treated as pu and
        scaled by SBase — pu and MW inputs must give identical stats."""
        df_mw = run_dcopf_stats(tiny_model, snet, table="candidates")
        assert np.abs(snet).sum(axis=0).max() < _PU_SNIFF  # both paths sniff
        df_pu = run_dcopf_stats(tiny_model, snet / 100.0, table="candidates")
        # sniff multiplies BOTH by 100 -> pu input ends up 100x smaller load
        assert df_pu["LoadMW"].to_numpy() == pytest.approx(
            df_mw["LoadMW"].to_numpy() / 100.0)

    def test_gen_out_of_service_by_scenario(self, tiny_model):
        """A scenario with no positive injection at bus 3 must switch that
        generator out: all dispatch lands on the slack (bus 0)."""
        n = tiny_model.n_substations
        col = np.zeros((n, 1), complex)                  # pu (sniff ×100)
        col[1, 0] = -0.8                                 # one 80 MW load
        col[0, 0] = 0.8                                  # slack covers it
        df = run_dcopf_stats(tiny_model, col, table="candidates")
        assert bool(df.loc[0, "success"])
        assert df.loc[0, "GenMW"] == pytest.approx(80.0, rel=1e-3)
