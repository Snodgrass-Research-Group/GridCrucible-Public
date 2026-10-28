"""UCCD complementary dispatch + generator fleet builder.

The fleet test regresses against Gen_reference.csv — the 719-unit SRTV fleet
exported from the author's real MATLAB run.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gridcrucible.config import CrucibleConfig
from gridcrucible.data.dispatch import uccd
from gridcrucible.data.generators import (create_generator_data_national,
                                          regional_fleet)

gp = pytest.importorskip("gurobipy")
DATA_DIR = CrucibleConfig().io.data_dir


def _fleet(n=12, seed=1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "MinMW": 10.0, "MaxMW": rng.uniform(80, 300, n).round(),
        "A": 100.0, "B": rng.uniform(15, 45, n).round(2),
        "C": rng.uniform(0.005, 0.05, n).round(4),
        "FuelType": ["gas"] * (n - 2) + ["Hydro", "Nuclear"],
    })


class TestUCCD:
    def test_balances_load(self):
        fleet = _fleet()
        load = np.array([400.0, 300.0])
        x, out = uccd(fleet, 1.0, load, reserve=0.1)
        assert x.sum() == pytest.approx(load.sum(), rel=1e-6)
        assert len(x) == len(out)
        assert (x > 0.1).all()                      # cleanup removed tiny units

    def test_complementary_vs_cheap_dispatch(self):
        """CD must load expensive units more than the plain cheap QP does."""
        from gridcrucible.data.dispatch import _solve_with_fallback
        fleet = _fleet(n=20, seed=7)
        load = np.array([1200.0])
        x_cd, out = uccd(fleet, 1.0, load, reserve=0.1)
        x_cheap = _solve_with_fallback(fleet, float(load.sum()))
        # marginal-cost-weighted output: CD ≥ cheap dispatch
        def weighted_cost(x, f):
            b = f["B"].to_numpy(float)
            return float((b * x).sum() / x.sum())
        assert weighted_cost(x_cd, out) >= weighted_cost(x_cheap, fleet) - 1e-9

    def test_hydro_scaling(self):
        fleet = _fleet()
        x0, out0 = uccd(fleet, 1.0, np.array([500.0]), 0.1)
        x1, out1 = uccd(fleet, 0.2, np.array([500.0]), 0.1)   # dry year
        hydro_max0 = out0.loc[out0["FuelType"] == "Hydro", "MaxMW"]
        hydro_max1 = out1.loc[out1["FuelType"] == "Hydro", "MaxMW"]
        if len(hydro_max0) and len(hydro_max1):
            assert hydro_max1.iloc[0] == pytest.approx(0.2 * hydro_max0.iloc[0])
        assert x1.sum() == pytest.approx(500.0, rel=1e-6)

    def test_bounds_respected(self):
        fleet = _fleet(n=15, seed=3)
        x, out = uccd(fleet, 1.0, np.array([900.0]), 0.1)
        assert (x <= out["MaxMW"].to_numpy() + 1e-6).all()

    def test_original_index_preserved(self):
        fleet = _fleet()
        _, out = uccd(fleet, 1.0, np.array([500.0]), 0.1)
        assert set(out.index) <= set(fleet.index)   # rows traceable to input


@pytest.fixture(scope="module")
def gen_nat():
    if not (DATA_DIR / "generator_import.parquet").exists():
        pytest.skip("run the migration first")
    return create_generator_data_national(DATA_DIR)


class TestGeneratorFleet:

    def test_group_explosion(self, gen_nat):
        raw = pd.read_parquet(DATA_DIR / "generator_import.parquet")
        assert len(gen_nat) == int(raw.iloc[:, 1].sum())   # Σ NumGen
        assert (gen_nat["Index"] == np.arange(1, len(gen_nat) + 1)).all()
        assert (gen_nat["MinMW"] < gen_nat["MaxMW"]).all()

    def test_fuel_rules(self, gen_nat):
        nuc = gen_nat[gen_nat["FuelType"] == "Nuclear"]
        if len(nuc):
            assert (nuc["MinMW"] == pytest.approx(0.99 * nuc["MaxMW"])).all() \
                or (nuc["MinMW"] >= 0.98 * nuc["MaxMW"]).all()
        hyd = gen_nat[gen_nat["FuelType"] == "Hydro"]
        if len(hyd):
            assert (hyd["MinMW"] == 0).all()

    def test_regression_vs_reference_run(self, gen_nat):
        """The SRTV regional subset must reproduce the author's real fleet."""
        ref_csv = DATA_DIR / "matlab_exports" / "Gen_reference.csv"
        if not ref_csv.exists():
            pytest.skip("Gen_reference.csv not exported")
        ref = pd.read_csv(ref_csv)
        ext_col = "BusNumExt" if "BusNumExt" in ref.columns else None
        if ext_col is None:
            pytest.skip("reference lacks external bus numbers")
        srtv = regional_fleet(gen_nat, ref[ext_col].unique())
        assert len(srtv) == len(ref)                # 719 units
        assert srtv["MaxMW"].sum() == pytest.approx(ref["MaxMW"].sum(), rel=1e-6)
        assert srtv["MinMW"].sum() == pytest.approx(ref["MinMW"].sum(), rel=1e-3)
        # per-bus unit counts match exactly
        ours = srtv.groupby("BusNum").size().sort_index()
        theirs = ref.groupby(ext_col).size().sort_index()
        assert ours.equals(theirs)
