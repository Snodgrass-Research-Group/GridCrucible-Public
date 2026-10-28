"""tests: CheckpointRegistry + the GridCrucible orchestrator.

The smoke test runs the REAL pipeline end-to-end (stages 0-5) on a synthetic
20-bus region with injected stand-ins for the gridforge-owned pieces
(CreateOLMatrices, UCED) and the load-profile bridge — proving that OHM,
TRIM, and TUNE cooperate through the orchestrator until TRIM converges.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gridcrucible.config import CrucibleConfig
from gridcrucible.core import GridCrucible
from gridcrucible.io.checkpoints import CheckpointRegistry

gp = pytest.importorskip("gurobipy")

REAL_DATA = CrucibleConfig().io.data_dir
N_BUS = 20
GEN_ROWS = [0, 7, 14]          # rows in the fake BusInputData (Type == Gen)


# ---------------------------------------------------------------------------
# fixtures: fake data_dir + injected gridforge stand-ins
# ---------------------------------------------------------------------------

def _fake_data_dir(tmp: Path) -> Path:
    d = tmp / "data"
    d.mkdir()
    (d / "manifest.json").write_text('{"fake": true}')
    rng = np.random.default_rng(21)
    pd.DataFrame({
        "BusName": [f"BUS{i}" for i in range(N_BUS)],
        "latitude": 35.0 + 0.4 * rng.random(N_BUS),
        "longitude": -87.0 - 0.4 * rng.random(N_BUS),
        "Region": "TEST",
        "Type": ["Gen" if i in GEN_ROWS else "Load" for i in range(N_BUS)],
        "MWLoad": np.where(np.isin(np.arange(N_BUS), GEN_ROWS), 0.0,
                           8.0 + 12.0 * rng.random(N_BUS)),
        "MvarLoad": 3.0,
    }).to_parquet(d / "bus_input_data.parquet", index=False)
    # generator groups (12 positional columns, MATLAB GenVarNames order)
    pd.DataFrame({
        "c1": [r + 1 for r in GEN_ROWS], "c2": [2, 1, 2],       # BusNum, NumGen
        "c3": [10.0, 5, 8], "c4": [120.0, 90, 150],             # Min/MaxMW
        "c5": [-40.0, -30, -50], "c6": [60.0, 45, 70],          # Min/MaxMVar
        "c7": ["Gas", "Hydro", "Coal"], "c8": ["CT", "HY", "ST"],
        "c9": [100.0, 50, 120], "c10": [25.0, 5, 20],           # A, B
        "c11": [0.02, 0.001, 0.015], "c12": [138.0, 138, 138],  # C, POIkV
    }).to_parquet(d / "generator_import.parquet", index=False)
    # real conductor catalog + EI stats (migrated data when present)
    if (REAL_DATA / "conductor_catalog.parquet").exists():
        pd.read_parquet(REAL_DATA / "conductor_catalog.parquet") \
            .to_parquet(d / "conductor_catalog.parquet", index=False)
    else:
        pytest.skip("run the migration first (conductor catalog)")
    counts = np.array([10047, 20537, 10178, 2575, 1340, 651, 349, 174, 90,
                       50, 31, 12, 13, 3, 3, 1, 1, 0, 1, 0, 0, 0, 1], float)
    dist = counts / counts.sum()
    (d / "ei_stats.json").write_text(json.dumps({
        "node_degree": {str(i + 1): int(c) for i, c in enumerate(counts)},
        "node_degree_dist": dist.tolist(),
        "node_degree_avg": float((np.arange(1, 24) * dist).sum()),
    }))
    return d


def _profile_fn(region, data_dir):
    """6-hour profiles for the 17 load buses (no bus-number column)."""
    rng = np.random.default_rng(5)
    base = 8.0 + 12.0 * rng.random((N_BUS - len(GEN_ROWS), 1))
    shape = 0.7 + 0.3 * np.sin(np.linspace(0, np.pi, 6))[None, :]
    p = base * shape
    return p, 0.3 * p


def _uced_stub(fleet, hydro_scaling, load_mw, reserve, gurobi_env=None):
    total = float(np.asarray(load_mw).sum())
    cap = fleet["MaxMW"].to_numpy(float)
    return total * cap / cap.sum(), fleet


def _candidate_params_stub(model, catalog):
    """Stands in for gridforge CreateOLMatrices: 138 kV class, X ∝ length."""
    cand = model.candidates
    c138 = catalog[catalog["kv"] == 138]
    mid = len(c138) // 2
    cand["reactance_x"] = 0.0006 * cand["length_km"]
    cand["l1_cost"] = (cand["length_km"] / cand["length_km"].mean())
    cand["capacity_mva"] = float(c138["mva"].iloc[mid])
    cand["conductor_idx"] = float(c138.index[mid])
    cand["max_conductor_idx"] = float(c138.index.max())
    cand["min_conductor_idx"] = float(c138.index.min())
    cand["min_kv"] = 69.0
    cand["voltage_kv_from"] = 138.0
    cand["voltage_kv_to"] = 138.0
    cand["kcmil"] = 0.0
    cand["cond_per_phase"] = 1.0


def _run(gc: GridCrucible, **kwargs):
    """gc.run(**kwargs), skipping when the OHM QPs exceed the size-limited
    licence bundled with pip gurobipy (these runs need a full licence)."""
    try:
        return gc.run(**kwargs)
    except gp.GurobiError as exc:
        if "size-limited" in str(exc):
            pytest.skip("needs a full Gurobi licence (size-limited licence)")
        raise


def _config(tmp: Path) -> CrucibleConfig:
    cfg = CrucibleConfig()
    cfg.case.regions = ["TEST"]
    cfg.case.rng_seed = 7
    cfg.io.data_dir = _fake_data_dir(tmp)
    cfg.io.checkpoint_dir = tmp / "runs"
    cfg.dispatch.use_cd = False               # ED-only keeps the run fast
    cfg.ohm.gamma_grid = (-4.0, -1.0, 4)
    cfg.ohm.inner_loop_terminate = 0.15
    cfg.trim.percent_remove = 0.08
    cfg.trim.max_iterations = 120
    return cfg


def _crucible(cfg, **kw) -> GridCrucible:
    return GridCrucible(cfg, uced_fn=_uced_stub,
                        candidate_params_fn=_candidate_params_stub,
                        profile_fn=_profile_fn, **kw)


# ---------------------------------------------------------------------------


class TestCheckpointRegistry:
    def test_roundtrip(self, config, tiny_model, tmp_path):
        config.io.checkpoint_dir = tmp_path
        tiny_model.meta["node_degree"] = np.array([3] * 6)
        tiny_model.meta["realism"] = {"nested": np.zeros(3)}   # non-serializable
        reg = CheckpointRegistry(config, run_id="t1")
        with pytest.warns(UserWarning, match="realism"):
            reg.save(tiny_model, "stage_x")
        loaded = reg.load("stage_x")
        pd.testing.assert_frame_equal(loaded.substations, tiny_model.substations)
        pd.testing.assert_frame_equal(loaded.candidates, tiny_model.candidates)
        assert np.array_equal(loaded.scenarios.tiers["full"],
                              tiny_model.scenarios.tiers["full"])
        assert loaded.scenarios.slack_bus == 0
        assert np.array_equal(loaded.meta["node_degree"], [3] * 6)
        assert "realism" not in loaded.meta
        assert reg.labels() == ["stage_x"]

    def test_latest_restricted(self, config, tiny_model, tmp_path):
        config.io.checkpoint_dir = tmp_path
        reg = CheckpointRegistry(config, run_id="t2")
        reg.save(tiny_model, "a")
        reg.save(tiny_model, "b")
        label, _ = reg.latest()
        assert label == "b"
        label, _ = reg.latest(among=["a"])
        assert label == "a"
        assert reg.latest(among=["zzz"]) is None


class TestConfigTOML:
    def test_from_toml_overrides(self, tmp_path):
        p = tmp_path / "run.toml"
        p.write_text('stages = ["load_inputs", "setup_substations"]\n'
                     "[case]\nregions = [\"TEST\"]\nrng_seed = 9\n"
                     "[trim]\npercent_remove = 0.1\n")
        cfg = CrucibleConfig.from_toml(p)
        assert cfg.case.rng_seed == 9
        assert cfg.trim.percent_remove == 0.1
        assert cfg.stages == ["load_inputs", "setup_substations"]

    def test_unknown_key_rejected(self, tmp_path):
        p = tmp_path / "bad.toml"
        p.write_text("[trim]\npercent_removed = 0.1\n")   # typo
        with pytest.raises(ValueError, match="percent_removed"):
            CrucibleConfig.from_toml(p)


class TestEndToEnd:
    @pytest.mark.filterwarnings("ignore:scenario window")
    @pytest.mark.filterwarnings("ignore:TUNELoop has no impedance_fn")
    def test_stages_through_trim(self, tmp_path):
        cfg = _config(tmp_path)
        gc = _crucible(cfg)
        model = _run(gc, until="trim_iterations")

        # stage outputs
        assert model.n_substations == N_BUS
        assert len(model.generators) == 5            # sum NumGen
        assert set(model.scenarios.tiers) == {"full", "complete", "advanced",
                                              "intermediate", "basic"}
        # TRIM actually converged below the substation threshold
        nd = model.node_degree("candidates")
        assert nd[nd > 0].mean() < 2.6
        assert len(model.candidates) < model.meta["initial_paths"]
        # the surviving network is one connected component with clean schema
        assert model.validate() == []
        # milestone checkpoints exist
        labels = gc.registry.labels()
        assert "trim_final" in labels
        assert any(lb.startswith("bbr_") for lb in labels)
        assert "build_candidates" in labels

    @pytest.mark.filterwarnings("ignore:scenario window")
    @pytest.mark.filterwarnings("ignore:TUNELoop has no impedance_fn")
    def test_resume_skips_completed_stages(self, tmp_path):
        cfg = _config(tmp_path)
        gc = _crucible(cfg)
        model1 = _run(gc, until="build_candidates")
        n_cand = len(model1.candidates)
        # a fresh orchestrator with the same run_id resumes, not recomputes
        gc2 = _crucible(cfg, run_id=gc.registry.run_id)
        model2 = _run(gc2, until="build_candidates")
        assert len(model2.candidates) == n_cand
        pd.testing.assert_frame_equal(model2.substations, model1.substations)

    def test_missing_migration_message(self, tmp_path):
        cfg = CrucibleConfig()
        cfg.io.data_dir = tmp_path / "empty"
        cfg.io.checkpoint_dir = tmp_path / "runs"
        with pytest.raises(FileNotFoundError, match="matlab_import"):
            GridCrucible(cfg).run(until="load_inputs")


class TestWriteOutputsStage:
    def test_stage_on_pre_bus_model(self, tiny_model, config, tmp_path):
        """Stage 11 wiring: on a pre-create_buses model the PowerWorld CSVs
        are skipped with a warning, everything else writes."""
        config.io.output_dir = tmp_path / "out"
        config.io.checkpoint_dir = tmp_path / "runs"
        config.io.data_dir = tmp_path / "nodata"    # no EI/WECC targets
        gc = GridCrucible(config, run_id="wo_test")
        gc.model = tiny_model
        with pytest.warns(UserWarning, match="PowerWorld"):
            gc.write_outputs()
        out = config.io.output_dir / "wo_test"
        assert (out / "SubstationGenLoadData.csv").exists()
        assert (out / "SubstationData.xlsx").exists()
        region = "".join(config.case.regions)
        assert (out / f"KCL-KVL {region} Statistics.xlsx").exists()
        assert not list(out.glob("BusImport*.csv"))     # skipped, not broken
