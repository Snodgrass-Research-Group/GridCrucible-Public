"""tests: pandapower bridge + DC/AC power-flow adapters."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gridcrucible.model import CrucibleModel
from gridcrucible.power import pandapower_io as ppio
from gridcrucible.power import power_flow as pf


def _balanced_injections(model: CrucibleModel) -> np.ndarray:
    """P vector: loads at their MW, generation split at buses 0 and 3."""
    p = -model.substations["load_mw"].to_numpy(float)
    total = -p.sum()
    p[0] += 0.45 * total
    p[3] += 0.55 * total
    return p


class TestBMatrix:
    def test_structure(self, tiny_model):
        A = tiny_model.incidence()
        x = tiny_model.candidates["reactance_x"].to_numpy(float)
        B = pf.b_matrix(A, x)
        assert B.shape == (6, 6)
        assert abs(B - B.T).max() < 1e-12          # symmetric
        assert np.abs(B.sum(axis=1)).max() < 1e-9  # row sums zero (no shunts)

    def test_contingency_zeroes_line(self, tiny_model):
        A = tiny_model.incidence()
        x = tiny_model.candidates["reactance_x"].to_numpy(float)
        B_out = pf.b_matrix(A, x, contingency=6)   # chord 0-3
        x_dropped = x.copy()
        x_dropped[6] = np.inf
        B_ref = pf.b_matrix(A, x_dropped)
        assert abs(B_out - B_ref).max() < 1e-12


class TestDCDirect:
    def test_path_graph_hand_solution(self):
        """0-1-2 path, inject +100 at 0, -100 at 2: both lines carry 100 MW."""
        subs = pd.DataFrame({"sub_id": [0, 1, 2]})
        cand = pd.DataFrame({"from_bus": [0, 1], "to_bus": [1, 2],
                             "reactance_x": [0.1, 0.2]})
        m = CrucibleModel(substations=subs, candidates=cand,
                          meta={"s_base": 100.0})
        res = pf.run_dc_pf(m, np.array([100.0, 0.0, -100.0]), engine="direct")
        assert res.flows_mw == pytest.approx([100.0, 100.0], abs=1e-9)
        assert res.theta_rad[0] == 0.0
        # theta drop = x * p_pu
        assert res.theta_rad[1] == pytest.approx(-0.1 * 1.0)

    def test_kcl_holds(self, tiny_model):
        p = _balanced_injections(tiny_model)
        res = pf.run_dc_pf(tiny_model, p, engine="direct")
        A = tiny_model.incidence()
        assert A @ res.flows_mw == pytest.approx(p, abs=1e-8)

    def test_antisymmetry(self, tiny_model):
        p = _balanced_injections(tiny_model)
        f1 = pf.run_dc_pf(tiny_model, p, engine="direct").flows_mw
        f2 = pf.run_dc_pf(tiny_model, -p, engine="direct").flows_mw
        assert f1 == pytest.approx(-f2, abs=1e-9)

    def test_contingency_moves_flow(self, tiny_model):
        p = _balanced_injections(tiny_model)
        base = pf.run_dc_pf(tiny_model, p, engine="direct")
        out = pf.run_dc_pf(tiny_model, p, engine="direct", contingency=6)
        assert out.flows_mw[6] == pytest.approx(0.0, abs=1e-9)
        # flow redistributes onto the remaining lines, KCL still holds
        assert not np.allclose(out.flows_mw, base.flows_mw)
        A = tiny_model.incidence()
        assert A @ out.flows_mw == pytest.approx(p, abs=1e-8)

    def test_disconnected_raises(self, tiny_model):
        cand = tiny_model.candidates
        keep = ~((cand["from_bus"] == 5) | (cand["to_bus"] == 5))
        tiny_model.candidates = cand[keep].reset_index(drop=True)
        with pytest.raises(pf.SingularNetworkError):
            pf.run_dc_pf(tiny_model, _balanced_injections(tiny_model),
                         engine="direct")


class TestPandapowerEngine:
    def test_engines_agree(self, tiny_model):
        p = _balanced_injections(tiny_model)
        direct = pf.run_dc_pf(tiny_model, p, engine="direct")
        via_pp = pf.run_dc_pf(tiny_model, p, engine="pandapower")
        assert via_pp.flows_mw == pytest.approx(direct.flows_mw, abs=1e-6)
        assert via_pp.theta_rad == pytest.approx(direct.theta_rad, abs=1e-9)

    def test_to_pandapower_structure(self, tiny_model):
        net = ppio.to_pandapower(tiny_model, table="candidates",
                                 injections_mw=_balanced_injections(tiny_model),
                                 slack_bus=0)
        assert len(net.bus) == 6
        assert sorted(net.line["crucible_row"]) == list(range(9))
        assert len(net.trafo) == 0            # tap==0, same kV everywhere
        assert net.ext_grid["bus"].iloc[0] == 0
        # slack bus's positive injection rides on the ext_grid, so sgens hold
        # only the OTHER positive injections; loads hold all negative ones
        p = _balanced_injections(tiny_model)
        assert net.sgen["p_mw"].sum() == pytest.approx(p[3])
        assert net.load["p_mw"].sum() == pytest.approx(-p[p < 0].sum())

    def test_flows_row_order(self, tiny_model):
        """Shuffle candidate rows: flows must follow crucible_row, not
        pandapower table position."""
        p = _balanced_injections(tiny_model)
        base = pf.run_dc_pf(tiny_model, p, engine="pandapower").flows_mw
        perm = np.random.default_rng(7).permutation(9)
        shuffled = tiny_model.copy()
        shuffled.candidates = tiny_model.candidates.iloc[perm].reset_index(drop=True)
        got = pf.run_dc_pf(shuffled, p, engine="pandapower").flows_mw
        assert got == pytest.approx(base[perm], abs=1e-6)


class TestDCStats:
    def test_frame_and_formulas(self, tiny_model):
        tiny_model.candidates["r_pu"] = 0.01
        p = _balanced_injections(tiny_model)
        flows, stats = pf.run_dc_pf_full(tiny_model, p[:, None],
                                         table="candidates")
        assert flows.shape == (9, 1)
        row = stats.iloc[0]
        assert row["GenMW"] == pytest.approx(p[p > 0].sum())
        assert row["LoadMW"] == pytest.approx(-p[p < 0].sum())
        expected_loss = 100.0 * (((flows[:, 0] / 100.0) ** 2) * 0.01).sum()
        assert row["Losses"] == pytest.approx(expected_loss)
        assert row["LinesOverloaded"] + (stats.iloc[0]["PercentLinesOverloaded"] == 0) >= 0
        # capacities are 230 MVA and flows are small -> nothing overloaded
        assert row["LinesOverloaded"] == 0
        assert row["OverloadPercentMax"] == 0.0

    def test_overload_detection(self, tiny_model):
        tiny_model.candidates["capacity_mva"] = 10.0  # force overloads
        p = _balanced_injections(tiny_model)
        _, stats = pf.run_dc_pf_full(tiny_model, p[:, None], table="candidates")
        assert stats.iloc[0]["LinesOverloaded"] > 0
        assert stats.iloc[0]["OverloadPercentMax"] > 0


class TestACPF:
    def _ac_ready(self, tiny_model) -> CrucibleModel:
        br = tiny_model.candidates.copy()
        br["r_pu"] = 0.1 * br["reactance_x"]
        br["x_pu"] = br["reactance_x"]
        br["b_pu"] = 0.002
        tiny_model.branches = br
        return tiny_model

    def test_converges_and_harvests(self, tiny_model):
        model = self._ac_ready(tiny_model)
        res = pf.run_ac_pf(model)
        assert res.converged
        assert res.v_mag.shape == (6,)
        assert 0.9 < res.v_mag.min() <= res.v_mag.max() < 1.1
        assert res.v_angle[0] == pytest.approx(0.0, abs=1e-9)  # slack ref
        assert np.isfinite(res.flows_mva).all() and (res.flows_mva >= 0).all()
        assert res.stats["total_loss_mw"].iloc[0] > 0  # r > 0 -> real losses

    def test_candidates_only_rejected(self, tiny_model):
        with pytest.raises(ValueError, match="r_pu"):
            pf.run_ac_pf(tiny_model)


class TestReferenceCase:
    def test_load_and_dc_solve(self):
        from gridcrucible.config import CrucibleConfig
        data_dir = CrucibleConfig().io.data_dir
        if not (data_dir / "reference_cases" / "WECCLargeIslandSanitized.json").exists():
            pytest.skip("reference cases not migrated on this machine")
        import pandapower as pp
        net = ppio.load_reference_case("WECCLargeIslandSanitized", data_dir)
        pp.rundcpp(net)
        assert net.converged and len(net.bus) > 20000
