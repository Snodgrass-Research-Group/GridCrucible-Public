"""CrucibleModel property tests."""

from __future__ import annotations

import numpy as np
import pytest

from gridcrucible.model import SCENARIO_TIERS, ScenarioSet

BBR = {"high": 20.0, "mid": 12.0, "midlow": 6.0, "low": 3.0}


class TestIncidence:
    def test_shape_and_signs(self, tiny_model):
        A = tiny_model.incidence("candidates")
        assert A.shape == (6, 9)
        dense = A.toarray()
        for j, (f, t) in enumerate(zip(tiny_model.candidates["from_bus"],
                                       tiny_model.candidates["to_bus"])):
            assert dense[f, j] == 1.0 and dense[t, j] == -1.0
        # each column: exactly one +1, one -1
        assert (dense.sum(axis=0) == 0).all()
        assert (np.abs(dense).sum(axis=0) == 2).all()

    def test_kcl_row_space(self, tiny_model):
        """A @ p sums flows out of each bus — balanced flows satisfy KCL."""
        A = tiny_model.incidence()
        p = np.zeros(9)
        p[0] = 5.0   # 0->1
        p[1] = 5.0   # 1->2
        inj = A @ p
        assert inj[0] == 5.0 and inj[2] == -5.0 and inj[1] == 0.0


class TestNodeDegree:
    def test_matches_diag_AAT(self, tiny_model):
        A = tiny_model.incidence()
        expected = (A @ A.T).diagonal()
        assert np.array_equal(tiny_model.node_degree(), expected)

    def test_ring_plus_chords(self, tiny_model):
        # ring gives every bus degree 2; one chord each adds 1 -> all 3
        assert (tiny_model.node_degree() == 3).all()


class TestValidate:
    def test_clean_model_passes(self, tiny_model):
        assert tiny_model.validate() == []

    def test_self_loop_caught(self, tiny_model):
        tiny_model.candidates.loc[0, "to_bus"] = tiny_model.candidates.loc[0, "from_bus"]
        assert any("self-loop" in v for v in tiny_model.validate())

    def test_disconnection_caught(self, tiny_model):
        # cut bus 5 off completely (drop edges 4-5, 5-0, 2-5)
        cand = tiny_model.candidates
        keep_rows = ~((cand["from_bus"] == 5) | (cand["to_bus"] == 5))
        tiny_model.candidates = cand[keep_rows].reset_index(drop=True)
        assert any("components" in v for v in tiny_model.validate())

    def test_kept_path_with_l1_cost_caught(self, tiny_model):
        tiny_model.candidates.loc[8, "l1_cost"] = 2.0  # row 8 has keep=True
        assert any("l1_cost" in v for v in tiny_model.validate())

    def test_scenario_shape_caught(self, tiny_model):
        tiny_model.scenarios.tiers["full"] = np.zeros((4, 2), complex)
        assert any("scenarios[full]" in v for v in tiny_model.validate())


class TestDensityTier:
    @pytest.mark.parametrize("density,tier", [
        (25.0, "basic"), (20.0, "intermediate"), (15.0, "intermediate"),
        (12.0, "advanced"), (7.0, "advanced"),
        (6.0, "complete"), (4.5, "complete"), (3.0, "complete"),
        (2.9, "full"), (1.0, "full"),
    ])
    def test_matlab_branch_order(self, density, tier):
        assert ScenarioSet().tier_for_density(density, BBR) == tier

    def test_all_tiers_reachable(self):
        tiers = {ScenarioSet().tier_for_density(d, BBR)
                 for d in (25, 15, 8, 4, 1)}
        assert tiers == set(SCENARIO_TIERS)

    def test_model_density(self, tiny_model):
        assert tiny_model.density == pytest.approx(9 / 6)
