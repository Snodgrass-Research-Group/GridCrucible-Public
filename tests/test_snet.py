"""tests: build_snet (CreateSNet port) on a synthetic 30-bus system.

Profiles are 24 hours so the ±60 h windows clamp (warning expected) and the
dispatch blocks stay small enough for fast Gurobi solves.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gridcrucible.data.scenarios import (_commitment_subsample,
                                         _diff_num_cd, _diff_num_ed,
                                         build_snet)
from gridcrucible.model import CrucibleModel, ScenarioSet

gp = pytest.importorskip("gurobipy")

N_BUS = 30
GEN_BUSES = [0, 3, 7, 12, 20]


def _uced_stub(fleet, hydro_scaling, load_mw, reserve, gurobi_env=None):
    """Proportional-to-capacity dispatch (stands in for gridforge UCED)."""
    total = float(np.asarray(load_mw).sum())
    cap = fleet["MaxMW"].to_numpy(float)
    return total * cap / cap.sum(), fleet


@pytest.fixture
def snet_setup(config):
    rng = np.random.default_rng(11)
    fleet = pd.DataFrame({
        "BusNum": GEN_BUSES,
        "MinMW": [50.0, 10, 0, 20, 5],
        "MaxMW": [400.0, 300, 200, 350, 150],
        "MinMVar": -50.0, "MaxMVar": 80.0,
        "A": 100.0, "B": [20.0, 25, 30, 22, 35],
        "C": [0.01, 0.02, 0.03, 0.015, 0.04],
        "FuelType": ["Coal", "Gas", "Hydro", "Gas", "Gas"],
    })
    subs = pd.DataFrame({
        "sub_id": range(N_BUS), "external_num": np.arange(1, N_BUS + 1),
        "load_mw": 0.0, "load_mvar": 0.0,
    })
    gen_arr = np.array(GEN_BUSES)
    load_arr = np.setdiff1d(np.arange(N_BUS), gen_arr)
    scen_meta = ScenarioSet(slack_bus=0, gen_buses=gen_arr[1:],
                            load_buses=load_arr,
                            v_setpoints=np.ones(N_BUS))
    model = CrucibleModel(substations=subs, generators=fleet,
                          scenarios=scen_meta, meta={"s_base": 100.0})
    bus_input = pd.DataFrame({
        "Region": ["SRTV"] * N_BUS,
        "Type": ["Gen" if i in GEN_BUSES else "Load" for i in range(N_BUS)],
    })
    # 24-hour profiles for the 25 load buses, no bus-number column
    base = 10.0 + 5.0 * rng.random((len(load_arr), 1))
    shape = 0.7 + 0.3 * np.sin(np.linspace(0, 2 * np.pi, 24))[None, :]
    p = base * shape
    p[3, :] = 0.05                                # tiny load -> floor case
    fn = lambda region, data_dir: (p, 0.3 * p)
    config.case.regions = ["SRTV"]
    return model, bus_input, fn


class TestDiffNumTables:
    @pytest.mark.parametrize("n,regional,expect", [
        (4600, False, 0), (12000, False, 1), (12000, True, 2),
        (18000, False, 2), (25000, False, 3)])
    def test_ed(self, n, regional, expect):
        assert _diff_num_ed(n, regional) == expect

    def test_cd_low_ratio_wins(self):
        assert _diff_num_cd(4600, False, 1.05, ["SRTV"], 0) == 0
        assert _diff_num_cd(4600, False, 1.5, ["RFCW"], 0) == 0
        assert _diff_num_cd(4600, False, 1.5, ["SRTV"], 0) == 4


class TestCommitmentSubsample:
    def test_dedup_keeps_max(self):
        gen_rows = np.array([0, 1])
        # scenarios 0-2 commit both plants; scenario 3 commits only plant 1
        # (0.05 MW is BELOW the 0.1 commitment threshold)
        block = np.array([[100, 120, 90, 0.05],
                          [50, 60, 45, 80.0],
                          [-150, -180, -135, -80.05]])
        out = _commitment_subsample(block.astype(complex), gen_rows, 0)
        # scenarios 0,1,2 share commitment {0,1}; scenario 3 = {1} only
        assert out.shape[1] == 2
        assert out[0, 0].real == 120                # max-total kept
        assert out[1, 1].real == 80.0

    def test_higher_diff_merges_more(self):
        rng = np.random.default_rng(2)
        block = (rng.random((6, 12)) * 100).astype(complex)
        gen_rows = np.arange(3)
        n0 = _commitment_subsample(block, gen_rows, 0).shape[1]
        n2 = _commitment_subsample(block, gen_rows, 2).shape[1]
        assert n2 <= n0


class TestBuildSNet:
    @pytest.mark.filterwarnings("ignore:scenario window")
    def test_tiers_and_balance(self, config, snet_setup):
        model, bus_input, fn = snet_setup
        ss = build_snet(model, config, bus_input, profile_fn=fn,
                        uced_fn=_uced_stub)
        assert set(ss.tiers) == {"full", "complete", "advanced",
                                 "intermediate", "basic"}
        full = ss.tiers["full"]
        assert full.shape[0] == N_BUS
        # 24h profile -> clamped windows -> 2 x (24+24) = 96 cols (ED + CD)
        assert full.shape[1] == 96
        # every scenario balances: dispatch == load  =>  column sums ~ 0
        assert np.abs(full.real.sum(axis=0)).max() < 1e-6
        # tier ordering: more aggressive dedup -> fewer or equal scenarios
        sizes = {t: ss.tiers[t].shape[1] for t in ss.tiers}
        assert sizes["basic"] <= sizes["intermediate"] \
            <= sizes["advanced"] <= sizes["complete"] <= sizes["full"]
        # < 2000 buses: basic aliases intermediate (MATLAB size rule)
        assert np.array_equal(ss.tiers["basic"], ss.tiers["intermediate"])

    @pytest.mark.filterwarnings("ignore:scenario window")
    def test_peak_loads_written_and_floor(self, config, snet_setup):
        model, bus_input, fn = snet_setup
        build_snet(model, config, bus_input, profile_fn=fn, uced_fn=_uced_stub)
        subs = model.substations
        assert subs["load_mw"].sum() > 0            # peak scenario written
        assert subs["gen_max_mw"].iloc[0] == 400.0  # per-bus aggregates
        # the 0.05 MW load bus got floored to 0.2 + 0.12j
        floored = subs[np.isclose(subs["load_mw"], 0.2)]
        assert len(floored) >= 1
        assert np.isclose(floored["load_mvar"].iloc[0], 0.12)

    def test_uced_missing_message(self, config, snet_setup):
        model, bus_input, fn = snet_setup
        with pytest.raises(ImportError, match="gridforge-owned"):
            build_snet(model, config, bus_input, profile_fn=fn, uced_fn=None)

    def test_regional_mode_not_ported(self, config, snet_setup):
        model, bus_input, fn = snet_setup
        config.dispatch.mode = "Regional"
        with pytest.raises(NotImplementedError):
            build_snet(model, config, bus_input, profile_fn=fn,
                       uced_fn=_uced_stub)
