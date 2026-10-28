"""tests: OHM Loop, DC formulation (QP + Newton + γ sweep).

Gurobi note: the pip-installed size-limited license comfortably covers these
(2m=18 variables). On the author's machine the academic license is used.
"""

from __future__ import annotations

import numpy as np
import pytest

from gridcrucible.loops.ohm import OHMLoop, OHMResult
from gridcrucible.power.power_flow import dc_pf_arrays

gp = pytest.importorskip("gurobipy")


@pytest.fixture
def ohm(config) -> OHMLoop:
    return OHMLoop(config)


def _arrays(model):
    A = model.incidence("candidates")
    cand = model.candidates
    return A, cand, np.full(len(cand), 500.0)  # generous MaxRateA (MVA)


def _injections_pu(model) -> np.ndarray:
    p = -model.substations["load_mw"].to_numpy(float)
    total = -p.sum()
    p[0] += 0.45 * total
    p[3] += 0.55 * total
    return p / 100.0


class TestQPCore:
    def test_gamma_zero_equals_dc_power_flow(self, ohm, tiny_model):
        """γ→0: min ½Xp² s.t. Ap=P IS the DC power flow (the KVL trick)."""
        A, cand, _ = _arrays(tiny_model)
        snet = _injections_pu(tiny_model)
        qp = ohm._get_qp(cand["reactance_x"].to_numpy(float),
                         cand["l1_cost"].to_numpy(float), A)
        flows = qp.solve(snet, gamma=0.0)
        ref = dc_pf_arrays(A, cand["reactance_x"].to_numpy(float), snet)
        assert flows == pytest.approx(ref.flows_mw, abs=1e-6)

    def test_kcl_exact_at_all_gammas(self, ohm, tiny_model):
        A, cand, _ = _arrays(tiny_model)
        snet = _injections_pu(tiny_model)
        qp = ohm._get_qp(cand["reactance_x"].to_numpy(float),
                         cand["l1_cost"].to_numpy(float), A)
        for gamma in (0.0, 1e-4, 1e-2, 1.0):
            flows = qp.solve(snet, gamma)
            assert A @ flows == pytest.approx(snet, abs=1e-6)

    def test_large_gamma_sparsifies(self, ohm, tiny_model):
        """L1 pressure drives more paths to ~zero as γ grows; kept path
        (l1_cost=0) is never penalized."""
        A, cand, _ = _arrays(tiny_model)
        snet = _injections_pu(tiny_model)
        X = cand["reactance_x"].to_numpy(float)
        c = cand["l1_cost"].to_numpy(float)
        qp = ohm._get_qp(X, c, A)
        n_zero = [int((np.abs(qp.solve(snet, g)) < 1e-5).sum())
                  for g in (0.0, 0.05, 5.0)]
        assert n_zero[0] <= n_zero[1] <= n_zero[2]
        assert n_zero[2] > n_zero[0]  # sparsification actually happened

    def test_rhs_update_across_scenarios(self, ohm, tiny_model):
        """Workspace reuse: second scenario through the same model gives the
        same answer as a fresh solve."""
        A, cand, _ = _arrays(tiny_model)
        X = cand["reactance_x"].to_numpy(float)
        c = cand["l1_cost"].to_numpy(float)
        s1 = _injections_pu(tiny_model)
        s2 = 0.6 * s1
        qp = ohm._get_qp(X, c, A)
        qp.solve(s1, 1e-3)
        reused = qp.solve(s2, 1e-3)
        fresh = OHMLoop(ohm.config)._get_qp(X, c, A).solve(s2, 1e-3)
        assert reused == pytest.approx(fresh, abs=1e-7)


class TestNewton:
    def test_agrees_with_qp(self, config, tiny_model):
        """MATLAB's NewtonOptiAgree check: smoothed-Newton flows track the QP
        within the smoothing error at moderate γ."""
        config.ohm.solver = "newton"
        ohm_n = OHMLoop(config)
        A, cand, _ = _arrays(tiny_model)
        X = cand["reactance_x"].to_numpy(float)
        c = cand["l1_cost"].to_numpy(float)
        snet = _injections_pu(tiny_model)
        seed = dc_pf_arrays(A, X, snet)
        p, theta, eps, alfa, ok = ohm_n._newton_step(
            X, c, A, snet, 1e-3, seed.flows_mw.copy(),
            seed.theta_rad[1:].copy(), 0.01, 1.0)
        assert ok
        qp_flows = OHMLoop(config)._get_qp(X, c, A).solve(snet, 1e-3)
        # ε=0.01 smoothing: agreement to ~ε on the flow scale
        assert p == pytest.approx(qp_flows, abs=0.02)
        assert A @ p == pytest.approx(snet, abs=5e-3)  # KCL to Newton tol


class TestSweep:
    def test_terminates_with_zero_update(self, config, tiny_model):
        """A reachable terminate_frac ends the sweep with update_gamma=0."""
        config.ohm.gamma_grid = (-3.0, 1.0, 6)
        ohm = OHMLoop(config)
        ohm.terminate_frac = 0.2  # 2 of 9 paths at ~0 is reachable
        A, cand, cap = _arrays(tiny_model)
        res = ohm.solve_scenario(cand, A, _injections_pu(tiny_model),
                                 cap / 100.0)
        assert isinstance(res, OHMResult)
        assert res.update_gamma == 0
        assert res.flows_abs.shape == (9,)
        assert (res.flows_abs >= 0).all()
        assert res.dcpf_flows.shape == (9,)

    def test_unreachable_target_requests_higher_gamma(self, config, tiny_model):
        """Tiny γ grid + demanding target: sweep fails, update_gamma=+1."""
        config.ohm.gamma_grid = (-9.0, -8.0, 3)
        ohm = OHMLoop(config)
        ohm.terminate_frac = 0.9
        A, cand, cap = _arrays(tiny_model)
        res = ohm.solve_scenario(cand, A, _injections_pu(tiny_model),
                                 cap / 100.0)
        assert res.update_gamma == +1

    def test_stall_handler_resets_grid(self, config, tiny_model):
        config.ohm.gamma_grid = (2.0, 2.5, 2)  # γ ≥ 100 immediately
        ohm = OHMLoop(config)
        ohm.terminate_frac = 0.99              # unreachable
        A, cand, cap = _arrays(tiny_model)
        ohm.solve_scenario(cand, A, _injections_pu(tiny_model), cap / 100.0)
        assert ohm.gammas[0] == pytest.approx(1e-4)
        assert ohm.gammas[-1] == pytest.approx(1e-3)
        assert 0.05 <= ohm.terminate_frac < 0.99


class TestGammaState:
    def test_shift_directions(self, ohm):
        lo, hi = np.log10(ohm.gammas[0]), np.log10(ohm.gammas[-1])
        ohm.shift_gammas(+1)
        assert np.log10(ohm.gammas[0]) == pytest.approx(lo + 0.5)
        assert len(ohm.gammas) == 7
        ohm.shift_gammas(0)
        assert len(ohm.gammas) == 8  # widened both ends

    def test_solve_all_feedback_settles(self, config, tiny_model):
        config.ohm.gamma_grid = (-5.0, -4.0, 3)  # low: forces +1 feedback rounds
        ohm = OHMLoop(config)
        ohm.terminate_frac = 0.2
        snet_mw = np.column_stack([_injections_pu(tiny_model) * 100.0,
                                   60.0 * _injections_pu(tiny_model)])
        results = ohm.solve_all(tiny_model, snet_mw, np.full(9, 500.0))
        assert len(results) == 2
        assert all(r.update_gamma == 0 for r in results)
        assert ohm.gammas[0] > 1e-5  # grid actually shifted up


class TestTuneScaling:
    def test_scaling_bounds(self, config, tiny_model):
        ohm = OHMLoop(config)
        ohm.terminate_frac = 0.2
        snet_mw = _injections_pu(tiny_model)[:, None] * 100.0
        s = ohm.tune_snet_scaling(tiny_model, snet_mw, np.full(9, 500.0))
        assert s == 1.0 or 1.0 < s <= 1000.0
        # gammas restored after the widened tuning call
        assert len(ohm.gammas) == config.ohm.gamma_grid[2]

    def test_huge_ratio_resets_to_one(self, config, tiny_model):
        """MATLAB quirk: ratio > 1000 -> scaling = 1 (not clamped to 1000)."""
        ohm = OHMLoop(config)
        ohm.terminate_frac = 0.2
        snet_mw = _injections_pu(tiny_model)[:, None] * 0.01  # microscopic flows
        s = ohm.tune_snet_scaling(tiny_model, snet_mw, np.full(9, 500.0))
        assert s == 1.0
