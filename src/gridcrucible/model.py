"""CrucibleModel: the single shared data structure every GridCrucible stage
consumes and returns.

Replaces the MATLAB workspace quintet ``Sub`` / ``Bus`` / ``Branch`` /
``OLBranch`` / ``Gen`` plus the ``PF`` struct (SNet scenario tiers, gen/load
bus lists) and the mutable parts of ``params``. Wraps the same conventions as
``gridforge.model.GridModel`` (0-based indices, km, ±1 sparse incidence,
MW/MVAr columns) — see docs/ARCHITECTURE.md §4 for the full
contract and MATLAB→Python naming glossary (§10).

Substations vs buses: MATLAB tracks ``params.buses`` (#substations) and
``params.nodes`` (#buses). Here ``substations`` and ``buses`` are separate
DataFrames that COINCIDE (same row count, buses.sub_id == index) until
``network.buses.create_buses`` splits generators/loads onto their own buses.
"""

from __future__ import annotations

import copy as _copy
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse import csgraph

# ---------------------------------------------------------------------------
# Column schemas (constants so io/checkpoints and tests can validate)
# ---------------------------------------------------------------------------

SUBSTATION_COLUMNS = [
    "sub_id",         # 0-based internal index (row position == sub_id); MATLAB Sub.NumInt-1
    "external_num",   # original EPIGRIDS/national bus number; MATLAB Sub.Num
    "name",
    "lat", "lon",     # WGS84 degrees
    "region",         # e.g. 'SRTV'
    "load_mw", "load_mvar",   # MATLAB Sub.MW / Sub.MVar (floored at 0.5 for tiny loads)
    "bus_type",       # 1=PQ, 2=PV (MATLAB Sub.Type); slack tracked in scenarios
    "gen_max_mw", "gen_min_mw",   # MATLAB Sub.MaxMW / Sub.MinMW
    # Sub.Voltages (n_subs × n_levels) and Sub.XFMVA live as prefixed columns:
    # voltage_69, voltage_138, ... and xf_mva_0, xf_mva_1, ... (see helpers)
]

CANDIDATE_COLUMNS = [
    "from_bus", "to_bus",         # 0-based CURRENT endpoints; MATLAB OLBranch.From/To
    "from_orig", "to_orig",       # original substation pair (survives bus splitting)
    "length_km",                  # haversine × LengthModifier; MATLAB OLBranch.Length
    "reactance_x",                # per-unit X; MATLAB OLBranch.X
    "l1_cost",                    # OHM regularizer weight (0 when keep=True!)
    "capacity_mva",               # MATLAB RateA
    "max_conductor_idx",          # largest allowed conductor (per kV table)
    "conductor_idx",
    "voltage_kv_from", "voltage_kv_to",
    "n_circuits", "tap",
    "gmd_index",                  # 1..1000 persisted draw (reproducibility!)
    "gmd_ft",
    "keep",                       # bool: protected from TRIM removal
    "is_radial",                  # bool: touches a degree-1 bus
    # --- TRIM scoring columns (appended by TRIMLoop.score_and_mark) ---
    "low_flow_count", "mean_norm", "std_norm", "max_flow", "max_dcpf",
    "remove_prob",
]

BRANCH_COLUMNS = CANDIDATE_COLUMNS[:17]  # kept/favored lines share the schema
                                         # minus TRIM scoring columns

# De-facto schema decided in /2b: the fleet KEEPS the MATLAB column
# names (UCCD, build_snet, and the Gen_reference regression all consume
# them) — one of the few places the glossary maps MATLAB->MATLAB.
GENERATOR_COLUMNS = [
    "BusNum",         # 0-based INTERNAL bus id after setup_generators
    "Index",          # 1-based running unit number (national fleet order)
    "Sub",            # == BusNum until CreateBuses splits gen buses
    "ID",             # per-bus unit counter (1-based)
    "MinMW", "MaxMW", "MinMVar", "MaxMVar",
    "FuelType", "UnitType",
    "A", "B", "C",    # cost coefficients (A fixed, B linear, C quadratic)
    "POIkV",
]

#: SNet scenario tiers, ordered dense-network → thin-network (BBR descending).
SCENARIO_TIERS = ("basic", "intermediate", "advanced", "complete", "full")


@dataclass
class ScenarioSet:
    """Net-injection scenario tiers + bus-role bookkeeping (MATLAB ``PF``).

    ``tiers[name]`` is a complex ndarray (n_buses × n_scenarios): net
    injection = generation − load per bus, in MW/MVAr (NOT per-unit; OHM
    divides by s_base). Built by ``data.scenarios.build_snet``;
    invalidated (must be rebuilt at bus granularity) by ``create_buses``.
    """

    tiers: dict[str, np.ndarray] = field(default_factory=dict)
    slack_bus: int = 0                 # MATLAB PF.SlackList (first gen bus)
    gen_buses: np.ndarray = field(default_factory=lambda: np.empty(0, int))
    load_buses: np.ndarray = field(default_factory=lambda: np.empty(0, int))
    v_setpoints: np.ndarray = field(default_factory=lambda: np.empty(0))
    # MATLAB PF.VBusSet: 1.0 everywhere, 1.02 at slack+gen buses

    def tier_for_density(self, density: float, thresholds: dict[str, float]) -> str:
        """Map candidate density (BBR = n_candidates/n_substations) to a tier.

        MATLAB MultiLevelProgram: >high(20)→basic, >mid(12)→intermediate,
        >midlow(6)→advanced, <low(3)→full, else→complete. (Note the MATLAB
        branch ORDER: the 'complete' tier is the low..midlow band because the
        <low test comes before the fall-through.)
        """
        if density > thresholds["high"]:
            return "basic"
        if density > thresholds["mid"]:
            return "intermediate"
        if density > thresholds["midlow"]:
            return "advanced"
        if density < thresholds["low"]:
            return "full"
        return "complete"


@dataclass
class ACPFResult:
    """AC power-flow result bundle (MATLAB ``ACPF`` struct).

    Produced by power/power_flow.run_ac_pf (pandapower); consumed by
    TUNELoop.run_ac, network.shunts, metrics, and io.writers (loss/stats
    sheets).
    """

    v_mag: np.ndarray | None = None       # per bus, pu
    v_angle: np.ndarray | None = None     # per bus, rad
    flows_mva: np.ndarray | None = None   # per branch (from-end apparent)
    losses: pd.DataFrame | None = None    # MATLAB ACPF.Losses table
    stats: pd.DataFrame | None = None     # MATLAB ACPF.Stats table
    converged: bool = False


@dataclass
class CrucibleModel:
    """The pipeline state. Every stage takes and returns one of these."""

    substations: pd.DataFrame = field(default_factory=pd.DataFrame)
    buses: pd.DataFrame = field(default_factory=pd.DataFrame)
    branches: pd.DataFrame = field(default_factory=pd.DataFrame)
    candidates: pd.DataFrame = field(default_factory=pd.DataFrame)
    generators: pd.DataFrame = field(default_factory=pd.DataFrame)
    scenarios: ScenarioSet = field(default_factory=ScenarioSet)
    acpf: ACPFResult | None = None
    meta: dict[str, Any] = field(default_factory=dict)
    # meta keys (grow the list, document here): s_base, regions, rng_seed,
    # outer_iter, snet_scaling, node_degree (ndarray), node_degree_dist,
    # ei_stats (Form 715 targets), bbr_thresholds, stage_history

    # -- derived quantities -------------------------------------------------

    @property
    def n_substations(self) -> int:
        """Number of substations (rows in the ``substations`` table)."""
        return len(self.substations)

    @property
    def n_buses(self) -> int:
        """MATLAB ``params.nodes`` (falls back to substation count)."""
        return len(self.buses) if len(self.buses) else len(self.substations)

    @property
    def density(self) -> float:
        """Candidate branch-bus ratio (BBR): drives scenario-tier selection."""
        return len(self.candidates) / max(self.n_substations, 1)

    def incidence(self, table: str = "candidates") -> sp.csc_matrix:
        """Signed incidence A (n_buses × n_paths): +1 at from, −1 at to.

        MATLAB: ``AOL = sparse([From;To],[i;i],[1;-1],nodes,paths)``.
        Rebuild after EVERY candidate mutation — never cache across TRIM
        removals (MATLAB recomputed it constantly for the same reason).
        """
        df = getattr(self, table)
        m = len(df)
        rows = np.concatenate([df["from_bus"].to_numpy(np.int64),
                               df["to_bus"].to_numpy(np.int64)])
        cols = np.tile(np.arange(m, dtype=np.int64), 2)
        vals = np.concatenate([np.ones(m), -np.ones(m)])
        return sp.csc_matrix((vals, (rows, cols)), shape=(self.n_buses, m))

    def node_degree(self, table: str = "candidates") -> np.ndarray:
        """Per-bus degree = diag(A·Aᵀ) of the requested table's graph
        (parallel edges count; computed directly via endpoint counts)."""
        df = getattr(self, table)
        ends = np.concatenate([df["from_bus"].to_numpy(np.int64),
                               df["to_bus"].to_numpy(np.int64)])
        return np.bincount(ends, minlength=self.n_buses).astype(np.int64)

    def validate(self) -> list[str]:
        """Schema + invariant check; returns a list of violations (empty=ok).

        Checks: column schemas above; 0-based index ranges; no self-loops;
        candidates connected (single component); keep/l1_cost consistency
        (keep→l1_cost==0); scenario tier shapes == n_buses.
        """
        v: list[str] = []
        for tbl, required in (("substations", SUBSTATION_COLUMNS),
                              ("candidates", CANDIDATE_COLUMNS),
                              ("generators", GENERATOR_COLUMNS)):
            df = getattr(self, tbl)
            if len(df):
                missing = [c for c in required if c not in df.columns]
                if missing:
                    v.append(f"{tbl}: missing columns {missing}")
        cand = self.candidates
        if len(cand) and not {"from_bus", "to_bus"} - set(cand.columns):
            fb = cand["from_bus"].to_numpy(np.int64)
            tb = cand["to_bus"].to_numpy(np.int64)
            n = self.n_buses
            if fb.min(initial=0) < 0 or tb.min(initial=0) < 0 \
                    or fb.max(initial=-1) >= n or tb.max(initial=-1) >= n:
                v.append(f"candidates: endpoint out of range [0, {n})")
            if (fb == tb).any():
                v.append(f"candidates: {(fb == tb).sum()} self-loop(s)")
            else:
                A = self.incidence("candidates")
                adj = (A @ A.T).tocsr()
                adj.setdiag(0)
                adj.eliminate_zeros()
                n_comp = csgraph.connected_components(
                    abs(adj), directed=False, return_labels=False)
                if n_comp > 1:
                    v.append(f"candidates: graph has {n_comp} components (must be 1)")
            if {"keep", "l1_cost"} <= set(cand.columns):
                bad = cand["keep"].astype(bool) & (cand["l1_cost"] != 0.0)
                if bad.any():
                    v.append(f"candidates: {int(bad.sum())} kept path(s) with "
                             "nonzero l1_cost (TRIM must zero them)")
        for name, arr in self.scenarios.tiers.items():
            if arr.shape[0] != self.n_buses:
                v.append(f"scenarios[{name}]: {arr.shape[0]} rows != "
                         f"{self.n_buses} buses")
        return v

    def copy(self) -> "CrucibleModel":
        """Return a deep copy — safe to mutate without affecting the original
        (used to snapshot state before a stage that edits tables in place)."""
        return _copy.deepcopy(self)
