"""Shared fixtures for GridCrucible property-based tests.

Validation strategy (ARCHITECTURE.md §2.8): (a) property tests — KCL and KVL hold,
distributions vs Form 715 targets, PF convergence, connectivity invariants;
(b) distribution-level regression vs the author's archived SRTV outputs
(place under tests/reference/, git-ignored if large); (c) MATLAB-generated
function-level I/O pairs on request.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gridcrucible.config import CrucibleConfig
from gridcrucible.model import CrucibleModel, ScenarioSet


@pytest.fixture
def config() -> CrucibleConfig:
    """Default (SRTV) config with a tiny deterministic seed."""
    cfg = CrucibleConfig()
    cfg.case.rng_seed = 42
    return cfg


@pytest.fixture
def tiny_model() -> CrucibleModel:
    """Hand-built 6-substation / 9-candidate model exercising the schema —
    small enough to verify OHM/TRIM/TUNE math by hand.

    Topology (0-based): ring 0-1-2-3-4-5-0 plus chords 0-3, 1-4, 2-5.
    """
    import pandas as pd

    from gridcrucible.model import ScenarioSet

    n = 6
    subs = pd.DataFrame({
        "sub_id": range(n), "external_num": [101, 102, 103, 104, 105, 106],
        "name": [f"SUB{i}" for i in range(n)],
        "lat": [35.0 + 0.1 * i for i in range(n)],
        "lon": [-87.0 - 0.1 * i for i in range(n)],
        "region": "SRTV",
        "load_mw": [50.0, 80.0, 0.5, 120.0, 60.0, 30.0],
        "load_mvar": [10.0, 16.0, 0.5, 24.0, 12.0, 6.0],
        "bus_type": [2, 1, 1, 2, 1, 1],
        "gen_max_mw": [200.0, 0, 0, 250.0, 0, 0],
        "gen_min_mw": [20.0, 0, 0, 25.0, 0, 0],
    })
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0),
             (0, 3), (1, 4), (2, 5)]
    m = len(edges)
    cand = pd.DataFrame({
        "from_bus": [e[0] for e in edges], "to_bus": [e[1] for e in edges],
        "from_orig": [e[0] for e in edges], "to_orig": [e[1] for e in edges],
        "length_km": [40.0 + 5 * i for i in range(m)],
        "reactance_x": [0.02 + 0.005 * i for i in range(m)],
        "l1_cost": [1.0] * (m - 1) + [0.0],
        "capacity_mva": [230.0] * m,
        "max_conductor_idx": [31] * m, "conductor_idx": [10] * m,
        "voltage_kv_from": [138] * m, "voltage_kv_to": [138] * m,
        "n_circuits": [1] * m, "tap": [0.0] * m,
        "gmd_index": list(range(1, m + 1)), "gmd_ft": [0.0] * m,
        "keep": [False] * (m - 1) + [True],
        "is_radial": [False] * m,
        "low_flow_count": [0] * m, "mean_norm": [0.0] * m,
        "std_norm": [0.0] * m, "max_flow": [0.0] * m, "max_dcpf": [0.0] * m,
        "remove_prob": [0.0] * m,
    })
    gens = pd.DataFrame({
        "BusNum": [0, 3], "Index": [1, 2], "Sub": [0, 3], "ID": [1, 1],
        "MinMW": [20.0, 25.0], "MaxMW": [200.0, 250.0],
        "MinMVar": [-50.0, -60.0], "MaxMVar": [80.0, 90.0],
        "FuelType": ["Gas", "Coal"], "UnitType": ["CT", "ST"],
        "A": [100.0, 150.0], "B": [22.0, 19.0], "C": [0.01, 0.008],
        "POIkV": [138.0, 138.0],
    })
    snet = np.zeros((n, 2), dtype=complex)
    load = subs["load_mw"].to_numpy() + 1j * subs["load_mvar"].to_numpy()
    snet[:, 0] = -load
    snet[:, 1] = -0.6 * load
    for col in range(2):  # balance each scenario at the two gen buses
        gen_total = -snet[:, col].real.sum()
        snet[0, col] += gen_total * 0.45
        snet[3, col] += gen_total * 0.55
    scen = ScenarioSet(tiers={"full": snet}, slack_bus=0,
                       gen_buses=np.array([3]), load_buses=np.array([1, 2, 4, 5]),
                       v_setpoints=np.where(np.isin(np.arange(n), [0, 3]), 1.02, 1.0))
    return CrucibleModel(substations=subs, buses=pd.DataFrame(),
                         candidates=cand, generators=gens, scenarios=scen,
                         meta={"s_base": 100.0, "regions": ["SRTV"]})


@pytest.fixture
def split_model(config) -> CrucibleModel:
    """4 substations: sub1/sub3 have two voltage levels (138+230); gens at
    sub1 (2 units, POI 230) and sub3 (1 unit, POI 115 -> nearest 138)."""
    n = 4
    subs = pd.DataFrame({
        "sub_id": range(n), "external_num": [11, 12, 13, 14],
        "name": [f"S{i}" for i in range(n)],
        "lat": [35.0, 35.1, 35.2, 35.3], "lon": [-87.0, -87.1, -87.2, -87.3],
        "region": "TEST", "load_mw": [40.0, 0.0, 55.0, 0.0],
        "load_mvar": [8.0, 0.0, 11.0, 0.0], "bus_type": [1, 2, 1, 2],
        "gen_max_mw": 0.0, "gen_min_mw": 0.0,
        "voltage_69": 0.0, "voltage_138": [138.0] * 4,
        "voltage_230": [0.0, 230.0, 0.0, 230.0], "voltage_500": 0.0,
    })
    edges = [(0, 1, 138), (1, 2, 138), (2, 3, 138), (3, 0, 138), (1, 3, 230)]
    cand = pd.DataFrame({
        "from_bus": [e[0] for e in edges], "to_bus": [e[1] for e in edges],
        "from_orig": [e[0] for e in edges], "to_orig": [e[1] for e in edges],
        "length_km": 60.0, "reactance_x": 0.03, "l1_cost": 0.0,
        "capacity_mva": 300.0, "max_conductor_idx": 100.0,
        "conductor_idx": 50.0, "min_conductor_idx": 1.0, "min_kv": 69.0,
        "voltage_kv_from": [float(e[2]) for e in edges],
        "voltage_kv_to": [float(e[2]) for e in edges],
        "n_circuits": 1.0, "tap": 0.0, "gmd_index": 1, "gmd_ft": 0.0,
        "keep": True, "is_radial": False, "low_flow_count": 0,
        "mean_norm": 0.0, "std_norm": 0.0, "max_flow": 0.0, "max_dcpf": 0.0,
        "remove_prob": 0.0, "kcmil": 0.0, "cond_per_phase": 1.0,
        "resize": False,
    })
    fleet = pd.DataFrame({
        "BusNum": [1, 1, 3], "Index": [1, 2, 3], "Sub": [1, 1, 3],
        "ID": [1, 2, 1], "MinMW": [10.0, 10, 5],
        "MaxMW": [150.0, 150, 400],                # largest unit at sub 3
        "MinMVar": -40.0, "MaxMVar": 60.0,
        "FuelType": ["Gas", "Gas", "Coal"], "UnitType": ["CT", "CT", "ST"],
        "A": 100.0, "B": 20.0, "C": 0.01,
        "POIkV": [230.0, 230.0, 115.0],
    })
    config.case.voltage_levels = [69, 138, 230, 500]
    config.case.max_length_miles = [25, 65, 90, 140]
    return CrucibleModel(substations=subs, candidates=cand, generators=fleet,
                         scenarios=ScenarioSet(slack_bus=1,
                                               gen_buses=np.array([3])),
                         meta={"s_base": 100.0})
