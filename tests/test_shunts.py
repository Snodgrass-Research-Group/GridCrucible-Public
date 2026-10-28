"""tests: shunt placement (winnowing) + bus-level SNet + XF sizing."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gridcrucible.config import CrucibleConfig
from gridcrucible.model import ScenarioSet
from gridcrucible.network.shunts import (correct_power_factors,
                                         place_reactive_compensation)

gp = pytest.importorskip("gurobipy")
DATA_DIR = CrucibleConfig().io.data_dir


class TestPowerFactorCorrection:
    def test_bands(self):
        # PF 0.8 load -> retarget 0.97; PF 0.99 load untouched-ish
        p = np.array([100.0, 100.0])
        q = np.array([75.0, 14.2])          # PF 0.8, PF ~0.99
        out = correct_power_factors(p + 1j * q)
        pf_out = np.abs(out.real) / np.abs(out)
        assert pf_out[0] == pytest.approx(0.97, abs=1e-9)
        assert pf_out[1] > 0.98
        assert np.array_equal(out.real, p)   # P untouched


class TestWinnowing:
    def _ac_model(self, tiny_model):
        cand = tiny_model.candidates
        cand["r_pu"] = 0.1 * cand["reactance_x"]
        cand["x_pu"] = cand["reactance_x"]
        cand["b_pu"] = 0.002
        tiny_model.buses = tiny_model.substations.copy()
        tiny_model.buses["voltage_kv"] = 138.0
        tiny_model.buses["sub_id"] = tiny_model.buses["sub_id"]
        # bus-level complete tier: balanced injections, 3 scenarios
        p = -tiny_model.substations["load_mw"].to_numpy(float)
        total = -p.sum()
        p[0] += 0.45 * total
        p[3] += 0.55 * total
        q = 0.3 * p
        tier = np.column_stack([(p + 1j * q) * f for f in (1.0, 0.8, 0.9)])
        tiny_model.scenarios = ScenarioSet(
            tiers={"complete": tier}, slack_bus=0,
            gen_buses=np.array([3]),
            load_buses=np.array([1, 2, 4, 5]),
            v_setpoints=np.ones(6))
        tiny_model.meta["ed_block_width"] = 3
        return tiny_model

    def test_places_and_converges(self, config, tiny_model):
        model = self._ac_model(tiny_model)
        res = place_reactive_compensation(model, config, per_switch=0.3)
        assert res.converged
        buses = model.buses
        assert "sync_cond" in buses.columns
        # condensers only at non-generator buses
        assert not buses.loc[[0, 3], "sync_cond"].any()
        # winnowing kept fewer than everything
        assert buses["sync_cond"].sum() <= 4
        # slack reasserted at the max-capacity bus (bus 3: 250 MW)
        assert buses.at[3, "bus_type"] == 3
        assert model.acpf is res

    def test_requires_complete_tier(self, config, tiny_model):
        model = self._ac_model(tiny_model)
        model.scenarios.tiers = {}
        with pytest.raises(ValueError, match="Complete tier"):
            place_reactive_compensation(model, config)


@pytest.mark.skipif(not (DATA_DIR / "conductor_catalog.parquet").exists(),
                    reason="run the migration first")
class TestBusLevelSNetAndXFSizing:
    def test_full_stage6_flow(self, config, split_model):
        """create_buses -> bus-level build_snet -> size_transformers_by_dcpf
        on the split fixture (the CreateBuses.m step-8 chain)."""
        from gridcrucible.data.scenarios import build_snet
        from gridcrucible.loops.tune import TUNELoop
        from gridcrucible.network.buses import (create_buses,
                                                size_transformers_by_dcpf)
        from gridcrucible.network.conductors import load_conductor_catalog

        model = split_model
        xfdata = pd.read_parquet(DATA_DIR / "xfdata.parquet")
        xfxr = pd.read_parquet(DATA_DIR / "xfxr.parquet")
        create_buses(model, config, xfdata, xfxr)
        n_buses = len(model.buses)

        bus_input = pd.DataFrame({
            "Region": ["TEST"] * 4,
            "Type": ["Load", "Gen", "Load", "Gen"],
        })
        prof = np.full((2, 24), 30.0)            # 2 load subs, 24 hours
        fn = lambda region, data_dir: (prof, 0.3 * prof)
        uced = lambda fleet, h, load, r, gurobi_env=None: (
            float(np.sum(load)) * fleet["MaxMW"].to_numpy(float)
            / fleet["MaxMW"].sum(), fleet)
        config.case.regions = ["TEST"]
        config.dispatch.use_cd = False
        ss = build_snet(model, config, bus_input, profile_fn=fn,
                        uced_fn=uced, bus_frame="buses")
        model.scenarios = ss
        assert ss.tiers["full"].shape[0] == n_buses
        # gen-zero-padding places profile rows AFTER the gen rows (MATLAB's
        # "gen buses first" convention): sub 2's min-voltage bus gets load
        assert abs(ss.tiers["full"][2].real).max() > 0
        assert ss.tiers["full"].real.sum(axis=0) == pytest.approx(
            np.zeros(ss.tiers["full"].shape[1]), abs=1e-6)  # balanced

        catalog = load_conductor_catalog(DATA_DIR)
        tune = TUNELoop(config, catalog, xfdata, xfxr,
                        impedance_fn=lambda df, cat: None,
                        rng=np.random.default_rng(3))
        cand = model.candidates
        assert (cand.loc[cand["transmission_xf"], "capacity_mva"]
                == 2000).all()
        size_transformers_by_dcpf(model, config, tune)
        sized = model.candidates
        assert not (sized.loc[sized["transmission_xf"], "capacity_mva"]
                    == 2000).any()               # placeholders re-rated
        assert (sized["max_dcpf"] >= 0).all()
