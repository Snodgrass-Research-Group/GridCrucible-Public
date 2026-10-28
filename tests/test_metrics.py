"""Graph-metric leaves + realism_metrics."""

from __future__ import annotations

import numpy as np
import pytest

from gridcrucible.metrics import graph_metrics as gm
from gridcrucible.metrics.realism import realism_metrics


class TestLeaves:
    def test_dedup_and_simple_degree(self):
        # triangle with a parallel edge 0-1
        f = np.array([0, 1, 2, 1])
        t = np.array([1, 2, 0, 0])
        deg = gm.node_degree_simple(f, t, 3)
        assert deg.tolist() == [2, 2, 2]          # parallel edge removed

    def test_eigencentrality_star(self):
        """Star K1,4: hub carries the largest centrality; λ = √4 = 2."""
        f = np.array([0, 0, 0, 0])
        t = np.array([1, 2, 3, 4])
        vec, val = gm.eigencentrality(f, t, 5)
        assert abs(val) == pytest.approx(2.0, abs=1e-9)
        assert np.abs(vec).argmax() == 0

    def test_fiedler_partitions_barbell(self):
        """Two triangles + bridge: v₂ signs split the two halves."""
        f = np.array([0, 1, 2, 3, 4, 5, 2])
        t = np.array([1, 2, 0, 4, 5, 3, 3])
        lam2, v2 = gm.fiedler_vector(f, t, 6)
        assert lam2 > 0
        assert set(np.sign(v2[:3]).tolist()) != set(np.sign(v2[3:]).tolist()) \
            or (np.sign(v2[:3]) != np.sign(v2[3:])).all()

    def test_normalized_laplacian_spectrum_bounds(self, tiny_model):
        fb = tiny_model.candidates["from_bus"].to_numpy()
        tb = tiny_model.candidates["to_bus"].to_numpy()
        eigs = gm.normalized_laplacian_eigs(fb, tb, 6)
        assert len(eigs) == 6
        assert eigs.min() == pytest.approx(0.0, abs=1e-9)   # connected graph
        assert eigs.max() <= 2.0 + 1e-9                     # normalized bound

    def test_singular_values_match_eigvals(self, tiny_model):
        """B is symmetric PSD: singular values == |eigenvalues|."""
        cand = tiny_model.candidates
        fb, tb = cand["from_bus"].to_numpy(), cand["to_bus"].to_numpy()
        x = cand["reactance_x"].to_numpy(float)
        sv = gm.b_matrix_singular_values(fb, tb, x, 6)
        from gridcrucible.power.power_flow import b_matrix
        B = b_matrix(tiny_model.incidence(), x).toarray()
        ev = np.abs(np.linalg.eigvalsh(B))
        assert np.sort(sv) == pytest.approx(np.sort(ev), abs=1e-8)

    def test_tier_offset_invariance(self, tiny_model):
        """Grounding choice must not matter: recompute with a permuted bus
        order and compare TIER values."""
        cand = tiny_model.candidates
        fb, tb = cand["from_bus"].to_numpy(), cand["to_bus"].to_numpy()
        x = cand["reactance_x"].to_numpy(float)
        gen = np.array([0, 3])
        _, tier = gm.tier_efficiency(fb, tb, x, gen, 6)
        assert tier.shape == (9,)
        assert (tier >= 0).all()
        # relabel buses (0<->5) and map gen list: TIER per branch invariant
        relabel = np.array([5, 1, 2, 3, 4, 0])
        _, tier2 = gm.tier_efficiency(relabel[fb], relabel[tb], x,
                                      relabel[gen], 6)
        assert tier2 == pytest.approx(tier, abs=1e-9)

    def test_cycle_basis_ring_chords(self, tiny_model):
        cand = tiny_model.candidates
        lengths = gm.cycle_basis_lengths(cand["from_bus"].to_numpy(),
                                         cand["to_bus"].to_numpy())
        assert len(lengths) == 9 - 6 + 1          # m - n + 1 independent cycles
        assert (lengths >= 3).all()

    def test_exp_fit_recovers_coefficients(self):
        x = np.arange(2, 40, 2, dtype=float)
        y = 0.8 * np.exp(-0.15 * x)
        fit = gm.exp_fit(x, y)
        assert fit["a"] == pytest.approx(0.8, rel=1e-6)
        assert fit["b"] == pytest.approx(-0.15, rel=1e-6)
        assert fit["r2"] == pytest.approx(1.0, abs=1e-9)

    def test_power_fit_recovers_coefficients(self):
        x = np.arange(2, 40, 2, dtype=float)
        y = 2.0 * x ** -1.3 + 0.01
        fit = gm.power_fit(x, y)
        assert fit["b"] == pytest.approx(-1.3, rel=1e-4)
        assert fit["r2"] == pytest.approx(1.0, abs=1e-8)

    def test_degenerate_fits_return_none(self):
        assert gm.exp_fit(np.array([1.0]), np.array([2.0])) is None
        assert gm.log_linear_fit(np.array([1, 2]), np.array([0.5, 0.5])) is None


class TestRealismMetrics:
    def test_full_dict(self, config, tiny_model):
        tiny_model.candidates["tap"] = 0.0
        out = realism_metrics(tiny_model, config)
        assert out["cycles"] is not None
        assert out["cycles"]["mean_length"] >= 3
        assert set(out["degree_fits"]) == {
            "node_degree", "sub_degree_lines", "sub_degree",
            "bus_degree_lines", "bus_degree"}
        # tiny_model is 3-regular: dist[0] == 0 -> MATLAB would crash,
        # Python returns fit=None but keeps the distribution
        nd = out["degree_fits"]["node_degree"]
        assert nd["fit"] is None
        assert nd["dist"][2] == pytest.approx(1.0)   # all buses degree 3
        assert out["mean_node_degree"] == pytest.approx(3.0)
        assert tiny_model.meta["realism"] is out

    def test_mva_hist_per_level(self, config, tiny_model):
        cand = tiny_model.candidates
        cand["capacity_mva"] = np.linspace(100, 400, len(cand))
        out = realism_metrics(tiny_model, config)
        assert 138 in out["mva_hist"]
        h = out["mva_hist"][138]
        assert h["counts"].sum() == len(cand)
        assert h["norm"].sum() == pytest.approx(1.0)

    def test_sogol_block(self, config, tiny_model):
        out = realism_metrics(tiny_model, config, include_sogol=True)
        s = out["sogol"]
        assert s["dominant_eigenvalue"] > 0
        assert s["tier"].shape == (9,)
        assert len(s["normalized_laplacian_eigs"]) == 6
        assert s["node_degree_simple"].tolist() == [3] * 6
