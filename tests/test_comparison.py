"""metrics/comparison.py
(docs/algorithms/RealismMetricsActual.md, ElectricalComparison.md).

Real-data checks run on the WECC case (smaller than EI); the EI path is
identical code with different parquets.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gridcrucible.config import CrucibleConfig
from gridcrucible.metrics.comparison import (
    HIGH_MVA, LOW_MVA, VOLTAGE_CLASSES, _branch_stats, _length_stats,
    comparison_stats, electrical_comparison, hist_areas,
    nearest_voltage_class, realism_metrics_actual, weibull_fit)

DATA_DIR = CrucibleConfig().io.data_dir

needs_data = pytest.mark.skipif(
    not (DATA_DIR / "wecc_branch.parquet").exists(),
    reason="run the migration first")


class TestHelpers:
    def test_weibull_fit_recovers_parameters(self):
        rng = np.random.default_rng(7)
        a, b = 0.05, 1.4
        x = np.arange(2.0, 60.0, 2.0)
        y = a * b * x ** (b - 1) * np.exp(-a * x ** b)
        fit = weibull_fit(x, y + rng.normal(0, 1e-5, x.size))
        assert fit is not None
        assert fit["a"] == pytest.approx(a, rel=0.05)
        assert fit["b"] == pytest.approx(b, rel=0.05)
        assert fit["r2"] > 0.99

    def test_nearest_voltage_class(self):
        got = nearest_voltage_class(np.array([66, 115, 138, 345, 525, 882]))
        assert got.tolist() == [69, 138, 138, 345, 500, 765]

    def test_length_stats_shape_and_999_row(self):
        kv = np.array([69, 69, 230, 345, 500, 765])
        length = pd.Series([10.0, 20, 30, 40, 50, 60])
        df = _length_stats(length, kv)
        assert df["Volt"].tolist() == VOLTAGE_CLASSES.tolist()
        assert df.loc[df["Volt"] == 69, "Total"].item() == 30.0
        # 999 row: sum of the 230..765 class totals
        assert df.loc[df["Volt"] == 999, "Total"].item() == 30 + 40 + 50 + 60
        assert df.loc[df["Volt"] == 138, "Mean"].item() == 0.0  # NaN -> 0

    def test_branch_stats_gva_miles(self):
        kv = np.array([230, 230, 500])
        length = pd.Series([100.0, 100.0, 10.0])
        limit = pd.Series([1000.0, 1000.0, 2000.0])
        df = _branch_stats(length, limit, kv)
        assert df.loc[df["Volt"] == 230, "GVAMiles"].item() == pytest.approx(200.0)
        assert df.loc[df["Volt"] == 999, "GVAMiles"].item() == pytest.approx(220.0)
        assert df.loc[df["Volt"] == 999, "Count"].item() == 3

    def test_hist_areas_by_label(self):
        x = np.arange(2.0, 40.0, 2.0)
        c = {"hist": {"bin_mid": x, "norm": 0.5 * np.exp(-0.1 * x)},
             "exp_fit": {"a": 0.5, "b": -0.1, "r2": 0.99, "area": 4.2},
             "power_fit": None}
        df = hist_areas({"KCL/KVL": c, "EI All Branches": None})
        assert df["Label_CB"].tolist() == ["KCL/KVL", "EI All Branches"]
        assert df.loc[0, "Area_Exp"] == 4.2
        assert np.isnan(df.loc[1, "Exp_RSq"])


@pytest.fixture(scope="module")
def stats():
    return realism_metrics_actual("WECC", DATA_DIR, use_cache=False)


@needs_data
class TestRealismMetricsActualWECC:

    def test_flavors_and_bbr(self, stats):
        for flavor in ("branch", "bus"):
            assert 1.0 < stats[flavor]["bbr"] < 2.0     # real grids ~1.2-1.5

    def test_degree_fits_present(self, stats):
        fits = stats["bus"]["degree_fits"]
        assert {"node_degree", "sub_degree_lines", "sub_degree",
                "bus_degree_lines", "bus_degree",
                "bus_degree_in_service"} <= set(fits)
        nd = fits["node_degree"]
        assert nd["fit"] is not None
        assert nd["fit"]["slope"] < 0                   # degree dist decays

    def test_cycles(self, stats):
        c = stats["branch"]["cycles"]
        assert c is not None
        # fundamental cycle bases on large sparse graphs run long (WECC
        # branch case: mean ~54) — the <80 truncation before fitting exists
        # for exactly this reason
        assert 3 <= c["mean_length"] < 200
        assert (np.asarray(c["hist"]["bin_mid"]) < 80).any()
        assert c["exp_fit"] is None or c["exp_fit"]["b"] < 0

    def test_form715_tables(self, stats):
        length = pd.DataFrame(stats["form715"]["length"])
        branch = pd.DataFrame(stats["form715"]["branch"])
        assert length["Volt"].tolist() == VOLTAGE_CLASSES.tolist()
        assert (branch["GVAMiles"] >= 0).all()
        # WECC branch table has no usable XF rows -> no separate XF hists
        # required (EI has them); line hists exist for the classes present
        hists = stats["form715"]["mva_hist"]
        assert any(h is not None for h in hists.values())

    def test_bus_kv_counts(self, stats):
        counts = stats["bus"]["bus_kv_counts"]
        assert sum(counts.values()) > 0

    def test_cache_roundtrip(self, tmp_path):
        import shutil
        for f in ("wecc_substations", "wecc_buses", "wecc_branch", "form1"):
            shutil.copy(DATA_DIR / f"{f}.parquet", tmp_path / f"{f}.parquet")
        (tmp_path / "reference_cases").mkdir()
        for f in ("WECCBranchSanitized", "WECCLargeIslandSanitized"):
            shutil.copy(DATA_DIR / "reference_cases" / f"{f}.npz",
                        tmp_path / "reference_cases" / f"{f}.npz")
        first = realism_metrics_actual("WECC", tmp_path)
        assert (tmp_path / "comparison_cache" / "wecc_actual.json").exists()
        second = realism_metrics_actual("WECC", tmp_path)   # from cache
        assert second["branch"]["bbr"] == pytest.approx(first["branch"]["bbr"])


@needs_data
class TestKclKvlBusMap:
    def test_full_permutation(self):
        from gridcrucible.metrics.comparison import kcl_kvl_bus_map
        m = kcl_kvl_bus_map(DATA_DIR)
        assert list(m.columns) == ["KCLBus", "KVLBus", "BusName"]
        n = len(m)
        assert n == 75432
        # a true permutation of 1..n on BOTH sides (Num == row+1 upstream)
        assert sorted(m["KCLBus"]) == list(range(1, n + 1))
        assert sorted(m["KVLBus"]) == list(range(1, n + 1))
        # the mapping is non-trivial (~9.5% of buses renumbered)
        moved = (m["KCLBus"] != m["KVLBus"]).mean()
        assert 0.05 < moved < 0.20


class TestComparisonStats:
    def test_weibull_added_to_cycles(self, tiny_model, config):
        out = comparison_stats(tiny_model, config, table="candidates")
        assert out["cycles"] is not None
        assert "weibull_fit" in out["cycles"]


@needs_data
class TestElectricalComparison:
    def test_grid_on_tiny_model(self, tiny_model):
        snet = tiny_model.scenarios.tiers["full"]        # MW, col sums > 100
        dc, opf = electrical_comparison(
            {"KCL/KVL": tiny_model}, {"KCL/KVL": snet / 100.0},
            table="candidates")
        # pu columns sum < 100 -> zero-scenario drop removes everything
        assert dc.empty and opf.empty
        dc, opf = electrical_comparison(
            {"KCL/KVL": tiny_model}, {"KCL/KVL": snet},
            table="candidates")
        assert len(opf) == snet.shape[1]
        assert (opf["Network"] == "KCL/KVL").all()
        assert (opf["Loads"] == "KCL/KVL").all()
        assert opf["success"].all()
