"""hardware.py — runtime core/memory detection + tuned Gurobi envs."""

from __future__ import annotations

import pytest

from gridcrucible import hardware


class TestDetection:
    def test_cpu_count_positive(self):
        assert hardware.cpu_count() >= 1

    def test_total_memory_positive(self):
        # best-effort: on any supported dev/CI box this should resolve
        assert hardware.total_memory_gb() > 0.5

    def test_params_divide_across_workers(self):
        one = hardware.gurobi_params(1)
        four = hardware.gurobi_params(4)
        assert one["Threads"] >= four["Threads"] >= 1
        if "MemLimit" in one:
            assert one["MemLimit"] == pytest.approx(4 * four["MemLimit"]) \
                or four["MemLimit"] == 1.0            # floor engaged
            assert one["MemLimit"] <= hardware.total_memory_gb()


class TestEnv:
    def test_env_carries_tuned_params(self):
        gp = pytest.importorskip("gurobipy")
        env = hardware.make_gurobi_env(n_workers=2)
        if env is None:
            pytest.skip("no gurobi license available")
        m = gp.Model(env=env)
        assert m.Params.Threads == hardware.gurobi_params(2)["Threads"]
        if "MemLimit" in hardware.gurobi_params(2):
            assert m.Params.MemLimit == pytest.approx(
                hardware.gurobi_params(2)["MemLimit"])
