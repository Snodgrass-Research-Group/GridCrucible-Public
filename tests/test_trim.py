"""tests: TRIM Loop (scoring, protection, guarded removal)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gridcrucible.loops.trim import TRIMLoop
from gridcrucible.model import CrucibleModel

EI_STATS = {"node_degree_dist": [0.216, 0.442, 0.219, 0.055],
            "node_degree_avg": 2.4}


@pytest.fixture
def trim(config):
    return TRIMLoop(config, np.random.default_rng(42))


def _flows(chords_low: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """9 paths × 4 scenarios: ring edges (0-5) busy; chords low in a VARIED
    number of scenarios (4/3/2) — MATLAB's strict `> quantile` threshold
    marks nothing when all low-flow counts are identical (uniform data is
    degenerate for the tier logic, in MATLAB too)."""
    rng = np.random.default_rng(0)
    flows = 80.0 + 20.0 * rng.random((9, 4))
    if chords_low:
        flows[6, :] = 0.001          # low in 4 scenarios
        flows[7, :3] = 0.001         # low in 3
        flows[8, :2] = 0.001         # low in 2
    cutoffs = np.full(4, 0.01)
    dcpf = flows * 1.1
    return flows, dcpf, cutoffs


def _snet(model) -> np.ndarray:
    p = -model.substations["load_mw"].to_numpy(float)
    p[0] += -p.sum() * 0.45
    p[3] += -p.sum()  # rough balance; only magnitudes matter for scoring
    return np.tile(p[:, None], (1, 4))


def _prepare(model) -> None:
    model.meta["ei_stats"] = EI_STATS
    # realistic SRTV start: everything at 500 kV with real mileage so the
    # HV-target lock does NOT fire spuriously
    model.candidates["voltage_kv_from"] = 500
    model.candidates["voltage_kv_to"] = 500
    model.candidates["length_km"] = 3000.0  # km each -> way above HV target


class TestScoreAndMark:
    def test_scoring_columns_and_sort(self, trim, tiny_model):
        _prepare(tiny_model)
        flows, dcpf, cutoffs = _flows()
        trim.score_and_mark(tiny_model, flows, dcpf, cutoffs, _snet(tiny_model))
        cand = tiny_model.candidates
        # the two persistently-low chords (counts 4 and 3) sort to the front;
        # the half-busy chord averages ~45 MW and lands mid-table
        assert {3, 4} <= set(cand["low_flow_count"].iloc[:3])
        assert cand["mean_norm"].is_monotonic_increasing
        assert cand["max_dcpf"].gt(0).all()

    def test_keep_zeroes_l1(self, trim, tiny_model):
        _prepare(tiny_model)
        flows, dcpf, cutoffs = _flows()
        trim.score_and_mark(tiny_model, flows, dcpf, cutoffs, _snet(tiny_model))
        cand = tiny_model.candidates
        assert (cand.loc[cand["keep"], "l1_cost"] == 0).all()
        assert tiny_model.validate() == []

    def test_low_flow_chords_get_marked(self, trim, tiny_model):
        _prepare(tiny_model)
        flows, dcpf, cutoffs = _flows()
        trim.score_and_mark(tiny_model, flows, dcpf, cutoffs, _snet(tiny_model))
        cand = tiny_model.candidates
        marked = cand[cand["remove_prob"] > 0]
        assert len(marked) >= 1
        # only paths above the low-flow quantile threshold (counts 4 and 3)
        assert (marked["low_flow_count"] >= 3).all()
        assert (~marked["keep"]).all()

    def test_radial_protection(self, trim, tiny_model):
        _prepare(tiny_model)
        # sever bus 5 down to one line (drop 5-0 and 2-5, keep 4-5)
        cand = tiny_model.candidates
        drop = ((cand["from_bus"] == 5) & (cand["to_bus"] != 4)) | \
               ((cand["to_bus"] == 5) & (cand["from_bus"] != 4))
        tiny_model.candidates = cand[~drop].reset_index(drop=True)
        flows, dcpf, cutoffs = _flows()
        flows = flows[:len(tiny_model.candidates)]
        dcpf = dcpf[:len(tiny_model.candidates)]
        trim.score_and_mark(tiny_model, flows, dcpf, cutoffs, _snet(tiny_model))
        cand = tiny_model.candidates
        radial_row = cand[(cand["from_bus"].isin([4, 5])) & (cand["to_bus"].isin([4, 5]))]
        assert radial_row["is_radial"].all()
        assert radial_row["keep"].all()
        assert (radial_row["remove_prob"] == 0).all()

    def test_hv_target_lock_fires_when_reached(self, trim, tiny_model):
        _prepare(tiny_model)
        tiny_model.candidates["length_km"] = 8.0  # tiny mileage -> under target
        flows, dcpf, cutoffs = _flows()
        marking = trim.score_and_mark(tiny_model, flows, dcpf, cutoffs,
                                      _snet(tiny_model))
        assert marking.create_buses_signal
        assert tiny_model.candidates["keep"].all()  # all rows are >=230 kV
        assert not trim.normalize_length


class TestRemove:
    def _marked_model(self, trim, tiny_model) -> CrucibleModel:
        _prepare(tiny_model)
        flows, dcpf, cutoffs = _flows()
        trim.score_and_mark(tiny_model, flows, dcpf, cutoffs, _snet(tiny_model))
        return tiny_model

    def test_removes_and_stays_connected(self, trim, tiny_model):
        model = self._marked_model(trim, tiny_model)
        n0 = len(model.candidates)
        res = trim.remove(model)
        assert res.n_removed == 1          # ceil(0.05 * 9)
        assert len(model.candidates) == n0 - 1
        assert model.validate() == []      # still one component

    def test_never_removes_kept(self, trim, tiny_model):
        model = self._marked_model(trim, tiny_model)
        kept_pairs = set(map(tuple, model.candidates.loc[
            model.candidates["keep"], ["from_bus", "to_bus"]].to_numpy()))
        trim.remove(model)
        pairs = set(map(tuple, model.candidates[["from_bus", "to_bus"]].to_numpy()))
        assert kept_pairs <= pairs

    def test_endpoint_stranding_guard(self, config):
        """Path graph 0-1-2: removing the middle line strands bus 2 — the
        stranding PRE-filter catches it (before any connectivity check)."""
        subs = pd.DataFrame({"sub_id": range(3), "load_mw": [1.0, 1, 1]})
        cand = pd.DataFrame({
            "from_bus": [0, 1], "to_bus": [1, 2],
            "reactance_x": [0.1, 0.1], "l1_cost": [1.0, 1.0],
            "length_km": [50.0, 50.0], "capacity_mva": [100.0, 100.0],
            "voltage_kv_from": [138, 138], "voltage_kv_to": [138, 138],
            "keep": [False, False], "is_radial": [False, False],
            "remove_prob": [0.0, 1.0],
        })
        model = CrucibleModel(substations=subs, candidates=cand,
                              meta={"s_base": 100.0})
        config.trim.percent_remove = 0.5
        trim = TRIMLoop(config, np.random.default_rng(1))
        res = trim.remove(model)
        assert res.n_removed == 0 and res.n_protected == 1
        assert not res.went_sequential          # stranding guard, not fallback
        assert len(model.candidates) == 2
        assert model.candidates.loc[1, "keep"]

    def test_bridge_triggers_sequential_fallback(self, config):
        """Two triangles joined by one marked bridge: endpoints keep degree
        ≥ 2 (stranding guard passes) but removal disconnects — rollback,
        sequential retry, bridge re-kept."""
        subs = pd.DataFrame({"sub_id": range(6), "load_mw": 1.0})
        edges = [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3), (2, 3)]
        cand = pd.DataFrame({
            "from_bus": [e[0] for e in edges], "to_bus": [e[1] for e in edges],
            "reactance_x": 0.1, "l1_cost": 1.0, "length_km": 50.0,
            "capacity_mva": 100.0, "voltage_kv_from": 138,
            "voltage_kv_to": 138, "keep": False, "is_radial": False,
            "remove_prob": [0.0] * 6 + [1.0],   # only the bridge is marked
        })
        model = CrucibleModel(substations=subs, candidates=cand,
                              meta={"s_base": 100.0})
        config.trim.percent_remove = 0.15       # ceil(0.15*7) = 2 attempts
        trim = TRIMLoop(config, np.random.default_rng(1))
        res = trim.remove(model)
        assert res.n_removed == 0
        assert res.n_protected == 1
        assert res.went_sequential
        assert len(model.candidates) == 7       # nothing actually removed
        assert model.candidates.loc[6, "keep"]  # bridge locked


class TestConverged:
    def test_substation_level_threshold(self, trim, tiny_model):
        tiny_model.meta["ei_stats"] = EI_STATS
        assert not trim.converged(tiny_model)   # mean degree 3 >= 2.6

    def test_bus_level_threshold(self, trim, tiny_model):
        tiny_model.meta["ei_stats"] = {"node_degree_avg": 3.5}
        tiny_model.buses = pd.concat(
            [tiny_model.substations] * 2, ignore_index=True)  # bus-level
        assert trim.converged(tiny_model)       # mean degree 3 < 3.5... but
        # n_buses doubled -> degrees halved relative to buses; recompute:
        # candidates reference buses 0-5 only, so mean over 12 buses = 1.5
