"""GridCrucible: the end-to-end orchestrator.

MATLAB origin: SRTV/main_SRTV.m (764 loc, driver) + Matlab Functions/
MultiLevelProgram.m (396 loc, TRIM-iteration engine).
Algorithm doc: docs/algorithms/GridCrucible.md (stage table + iteration
pseudocode — every rule wired below is specified there).

Composition (ARCHITECTURE.md §2.4): GridCrucible OWNS one OHMLoop, one TUNELoop and
one TRIMLoop; state flows through a CrucibleModel; run control comes from a
CrucibleConfig; every stage and TRIM milestone checkpoints via
io.checkpoints (replaces the QNCAResults*/PFData*/BBR*.mat cascades).

GRIDFORGE-OWNED dependencies are injected (ARCHITECTURE.md §3) and resolved lazily
with clear errors until their conversions land:
* ``uced_fn``            — economic dispatch (gridforge EconomicDispatcher)
* ``candidate_params_fn``— CreateOLMatrices (X / l1_cost / capacities)
* ``impedance_fn``       — RXB impedance recompute for TUNE
* haversine distances    — OLPathDistances (internal fallback, verbatim math)
"""

from __future__ import annotations

import warnings
from typing import Callable

import numpy as np
import pandas as pd

from .config import STAGES, CrucibleConfig
from .io.checkpoints import CheckpointRegistry
from .loops import OHMLoop, TRIMLoop, TUNELoop
from .model import CrucibleModel, ScenarioSet


def _haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Great-circle distance (OLPathDistances is gridforge-owned; this is
    the standard formula as a documented fallback — prefer the gridforge
    import once converted)."""
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = p2 - p1, np.radians(lon2) - np.radians(lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


class GridCrucible:
    """Builds a KCL+KVL-compliant synthetic grid for the configured regions.

    Usage::

        cfg = CrucibleConfig.from_toml("runs/srtv.toml")
        gc = GridCrucible(cfg)
        model = gc.run()                            # all configured stages
        model = gc.run(until="trim_iterations")     # partial run
    """

    def __init__(self, config: CrucibleConfig, *,
                 uced_fn: Callable | None = None,
                 candidate_params_fn: Callable | None = None,
                 profile_fn: Callable | None = None,
                 impedance_fn: Callable | None = None,
                 gurobi_env=None, run_id: str | None = None) -> None:
        self.config = config
        self.rng = np.random.default_rng(config.case.rng_seed)
        if gurobi_env is None:
            # machine-tuned Threads/MemLimit (hardware.py; None when
            # gurobipy/license absent — Gurobi defaults then apply)
            from .hardware import make_gurobi_env
            gurobi_env = make_gurobi_env(verbose=config.verbose)
        self.ohm = OHMLoop(config, gurobi_env)
        self.trim = TRIMLoop(config, self.rng)
        self.tune: TUNELoop | None = None          # built in load_inputs
        self.model = CrucibleModel()
        self.registry = CheckpointRegistry(config, run_id)
        self._uced_fn = uced_fn
        self._candidate_params_fn = candidate_params_fn
        self._profile_fn = profile_fn
        self._impedance_fn = impedance_fn
        self._env = gurobi_env
        self._bus_input: pd.DataFrame | None = None
        self._catalog: pd.DataFrame | None = None
        self._ei_stats: dict | None = None

    # ------------------------------------------------------------------ run

    def run(self, until: str | None = None) -> CrucibleModel:
        """Execute config.stages (through ``until``), auto-resuming from the
        newest completed stage checkpoint of this run_id."""
        stages = list(self.config.stages)
        if until is not None:
            if until not in stages:
                raise ValueError(f"stage {until!r} not in config.stages")
            stages = stages[:stages.index(until) + 1]
        done = [s for s in stages if s in self.registry.labels()]
        if done:
            self.model = self.registry.load(done[-1])
            self.load_inputs()                     # cheap caches, always fresh
            stages = stages[stages.index(done[-1]) + 1:]
        for stage in stages:
            method = getattr(self, stage, None) or getattr(self, f"run_{stage}")
            method()
            if stage != "load_inputs":             # nothing to snapshot yet
                self.registry.save(self.model, stage)
        return self.model

    # ---------------------------------------------------------- stages 0..4

    def load_inputs(self) -> None:
        """Stage 0: cache the migrated inputs (bus data, conductor catalog,
        EI targets, transformer tables) and build the TUNE loop."""
        import json

        from .network.conductors import load_conductor_catalog

        d = self.config.io.data_dir
        if not (d / "manifest.json").exists():
            raise FileNotFoundError(
                f"{d}/manifest.json — run `python -m "
                "gridcrucible.io.matlab_import` first (migration)")
        self._bus_input = pd.read_parquet(d / "bus_input_data.parquet")
        self._catalog = load_conductor_catalog(d)
        self._ei_stats = json.loads((d / "ei_stats.json").read_text())
        xfdata = pd.read_parquet(d / "xfdata.parquet") \
            if (d / "xfdata.parquet").exists() else None
        xfxr = pd.read_parquet(d / "xfxr.parquet") \
            if (d / "xfxr.parquet").exists() else None
        self.tune = TUNELoop(self.config, self._catalog, xfdata, xfxr,
                             impedance_fn=self._impedance_fn, rng=self.rng)
        self.model.meta.setdefault("s_base", self.config.case.s_base)
        self.model.meta["regions"] = list(self.config.case.regions)
        self.model.meta["ei_stats"] = self._ei_stats

    def setup_substations(self) -> None:
        """Stage 1 (main_SRTV 'Initializing parameters'): region filter,
        external numbering = BusInputData row+1, PQ/PV typing, import buses."""
        bi = self._bus_input
        rows = np.concatenate([
            np.flatnonzero((bi["Region"].astype(str) == r).to_numpy())
            for r in self.config.case.regions])
        n = len(rows)
        region_col = bi["Region"].astype(str).to_numpy()[rows]
        subs = pd.DataFrame({
            "sub_id": np.arange(n),
            "external_num": rows + 1,              # MATLAB find() row numbers
            "name": bi["BusName"].to_numpy()[rows],
            "lat": bi["latitude"].to_numpy(float)[rows],
            "lon": bi["longitude"].to_numpy(float)[rows],
            "region": region_col,
            "load_mw": bi["MWLoad"].to_numpy(float)[rows],
            "load_mvar": bi["MvarLoad"].to_numpy(float)[rows],
            "bus_type": 1 + (bi["Type"].astype(str).to_numpy()[rows] == "Gen"),
            "gen_max_mw": 0.0, "gen_min_mw": 0.0,
        })
        self.model.substations = subs
        imp = np.asarray(self.config.case.import_buses_ext, int)
        self.model.meta["import_buses_int"] = \
            np.flatnonzero(np.isin(subs["external_num"].to_numpy(), imp))

    def setup_generators(self) -> None:
        """Stage 2 (main_SRTV §3.1–3.2): regional fleet, internal renumber,
        slack = first gen bus, tiny-load floor, bus-role lists."""
        from .data.generators import (create_generator_data_national,
                                      regional_fleet)

        subs = self.model.substations
        gen_nat = create_generator_data_national(self.config.io.data_dir)
        fleet = regional_fleet(gen_nat, subs["external_num"].to_numpy())
        ext_to_int = pd.Series(subs.index.to_numpy(),
                               index=subs["external_num"].to_numpy())
        fleet["BusNumExt"] = fleet["BusNum"]
        fleet["BusNum"] = ext_to_int[fleet["BusNum"]].to_numpy()
        fleet["Sub"] = fleet["BusNum"]
        fleet["Region"] = subs["region"].to_numpy()[fleet["BusNum"]]
        self.model.generators = fleet

        gen_int = np.sort(fleet["BusNum"].unique())
        slack, gen_buses = int(gen_int[0]), gen_int[1:]
        load_buses = np.setdiff1d(np.arange(len(subs)),
                                  np.concatenate([[slack], gen_buses]))
        v = np.ones(len(subs))
        v[np.concatenate([[slack], gen_buses])] = 1.02
        self.model.scenarios = ScenarioSet(
            tiers={}, slack_bus=slack, gen_buses=gen_buses,
            load_buses=load_buses, v_setpoints=v)
        low = load_buses[subs["load_mw"].to_numpy()[load_buses] < 0.5]
        subs.loc[low, ["load_mw", "load_mvar"]] = 0.5   # main_SRTV floor

    def build_scenarios(self) -> None:
        """Stage 3: data.scenarios.build_snet → SNet tiers."""
        from .data.scenarios import build_snet

        from .data.load_profiles import make_load_profiles
        self.model.scenarios = build_snet(
            self.model, self.config, self._bus_input,
            profile_fn=self._profile_fn or make_load_profiles,
            uced_fn=self._uced_fn, gurobi_env=self._env)

    def build_candidates(self) -> None:
        """Stage 4 (main_SRTV §5–6): all-pairs candidates, haversine length
        filter, persistent GMD draw, then the electrical parameters
        (X / l1_cost / capacities) via CreateOLMatrices (gridforge-owned,
        injected)."""
        subs = self.model.substations
        n = len(subs)
        fi, ti = np.triu_indices(n, 1)
        lat = subs["lat"].to_numpy(float)
        lon = subs["lon"].to_numpy(float)
        length_km = _haversine_km(lat[fi], lon[fi], lat[ti], lon[ti]) \
            * self.config.case.length_modifier
        keep = length_km < max(self.config.case.max_length_miles) * 1.61
        if keep.sum() < len(fi):
            pass                                    # MATLAB warns; lengths ok
        fi, ti, length_km = fi[keep], ti[keep], length_km[keep]
        m = len(fi)

        cand = pd.DataFrame({
            "from_bus": fi, "to_bus": ti, "from_orig": fi, "to_orig": ti,
            "length_km": length_km,
            "keep": False, "is_radial": False,
            "n_circuits": 1.0, "tap": 0.0,
            "low_flow_count": 0, "mean_norm": 0.0, "std_norm": 0.0,
            "max_flow": 0.0, "max_dcpf": 0.0, "remove_prob": 0.0,
        })
        self.model.candidates = cand
        params_fn = self._candidate_params_fn or self._resolve_ol_matrices()
        params_fn(self.model, self._catalog)        # adds reactance_x, l1_cost,
        # capacity_mva, conductor_idx, max/min_conductor_idx, voltage columns;
        # MAY drop gen-gen rows and assign gmd_index (GMDIndex.mat semantics)
        cand = self.model.candidates
        if "gmd_index" not in cand.columns:         # simple stubs: draw here
            cand["gmd_index"] = self._load_or_draw_gmd_index(len(cand))
            cand["gmd_ft"] = 0.0
        self.model.meta["initial_paths"] = len(cand)

    @staticmethod
    def _resolve_ol_matrices() -> Callable:
        try:
            from gridcrucible._gridforge.network.topology import create_ol_matrices  # type: ignore
            return create_ol_matrices
        except (ImportError, AttributeError) as e:
            raise ImportError(
                "CreateOLMatrices is gridforge-owned; the vendored "
                "gridcrucible._gridforge exposes it as build_candidates (a "
                "different signature), so there is no drop-in here — pass "
                "candidate_params_fn=... (fills reactance_x / l1_cost / "
                "capacity_mva / conductor_idx / max_conductor_idx / "
                "voltage_kv_from/to on model.candidates), e.g. via "
                "gridforge_bridge.wire_gridcrucible") from e

    def _load_or_draw_gmd_index(self, m: int) -> np.ndarray:
        """Reuse the author's persisted reference-run GMD draws when they match
        the candidate count (reproducibility — GMDIndex.mat semantics),
        else a seeded draw."""
        ref = self.config.io.data_dir / "reference_run" / "gmd_index.npz"
        if not self.config.io.reset_gmd_index and ref.exists():
            arr = np.load(ref)["gmd_index"]
            if len(arr) == m:
                return arr.astype(int)
        return self.rng.integers(1, 1001, size=m)

    # ------------------------------------------------- stage 5: TRIM engine

    def run_trim_iterations(self) -> None:
        """Stage 5 — the MultiLevelProgram while-loop (algorithm doc §Stage 5):
        BBR-tier scenario selection + SNetScaling retune at crossings → OHM
        per scenario → TRIM score/remove → substation-voltage recompute →
        TUNE resize → near-convergence metrics → milestone checkpoints."""
        from .metrics.realism import realism_metrics

        model = self.model
        cfg = self.config
        meta = model.meta
        ei_avg = float(meta.get("ei_stats", {}).get("node_degree_avg", 2.4))
        bbr = {"high": cfg.trim.bbr_high, "mid": cfg.trim.bbr_mid,
               "midlow": cfg.trim.bbr_midlow, "low": cfg.trim.bbr_low}
        meta.setdefault("snet_scaling", 1.0)
        meta.setdefault("outer_iter", 0)
        last_tier = meta.get("last_tier")
        nd2_saved = "nd2" in self.registry.labels()

        while True:
            meta["outer_iter"] += 1
            it = meta["outer_iter"]

            max_cap = self._max_capacity_mva()
            tier = model.scenarios.tier_for_density(model.density, bbr)
            snet = np.real(model.scenarios.tiers[tier])
            if tier != last_tier:
                meta["snet_scaling"] = self.ohm.tune_snet_scaling(
                    model, snet, max_cap)
                meta["last_tier"] = last_tier = tier
                self.registry.save(model, f"bbr_{tier}")
            snet_scaled = meta["snet_scaling"] * snet

            results = self.ohm.solve_all(model, snet_scaled, max_cap)
            flows = np.column_stack([r.flows_abs for r in results])
            dcpf = np.column_stack([r.dcpf_flows for r in results])
            cutoffs = np.array([r.cutoff for r in results])

            marking = self.trim.score_and_mark(model, flows, dcpf, cutoffs,
                                               snet_scaled)
            if marking.inner_terminate_override is not None:
                self.ohm.terminate_frac = marking.inner_terminate_override
            self.trim.remove(model)
            self._recompute_substation_voltages()
            self.tune.run_dc(model, snet_scaled)

            nd = model.node_degree("candidates")
            mean_nd = float(nd[nd > 0].mean()) if (nd > 0).any() else 0.0
            if mean_nd < 2 * ei_avg:
                if not nd2_saved:
                    self.registry.save(model, "nd2")
                    nd2_saved = True
                realism_metrics(model, cfg)
            self.registry.save(model, f"trim_iter_{it % 5}")

            if self.trim.converged(model):
                realism_metrics(model, cfg)
                self.registry.save(model, "trim_final")
                break
            if it >= cfg.trim.max_iterations:
                warnings.warn(f"TRIM stopped at max_iterations="
                              f"{cfg.trim.max_iterations} without converging "
                              f"(mean node degree {mean_nd:.2f})", stacklevel=2)
                break

    def _max_capacity_mva(self) -> np.ndarray:
        """MaxRateA = ConductorData.MVA(MaxCondIndex) — 1-based master-table
        lookup (cond_index semantics resolved in conductors.md)."""
        idx = self.model.candidates["max_conductor_idx"].astype(int)
        return self._catalog.loc[idx, "mva"].to_numpy(float)

    def _recompute_substation_voltages(self) -> None:
        """MultiLevelProgram 318-323: Sub.Voltages ← the kV levels present on
        surviving candidates at each substation (voltage_<kv> columns)."""
        subs = self.model.substations
        cand = self.model.candidates
        kvf = cand["voltage_kv_from"].to_numpy(float)
        fo = cand["from_orig"].to_numpy(int)
        to = cand["to_orig"].to_numpy(int)
        for kv in self.config.case.voltage_levels:
            col = np.zeros(len(subs))
            hit = kvf == kv
            touched = np.unique(np.concatenate([fo[hit], to[hit]]))
            col[touched] = kv
            subs[f"voltage_{kv}"] = col

    # --------------------------------------------------------- stages 6..11

    def create_buses(self) -> None:
        """Stage 6: substation → bus expansion (CreateBuses steps 1-7), then
        the bus-level SNet rebuild ('Subs' mode) and DC-PF transformer sizing
        (step 8)."""
        from .data.load_profiles import make_load_profiles
        from .data.scenarios import build_snet
        from .network.buses import (create_branch, create_buses,
                                    size_transformers_by_dcpf)

        create_buses(self.model, self.config,
                     xfdata=self.tune.xfdata if self.tune else None,
                     xfxr=self.tune.xfxr if self.tune else None)
        self.model.scenarios = build_snet(
            self.model, self.config, self._bus_input,
            profile_fn=self._profile_fn or make_load_profiles,
            uced_fn=self._uced_fn, gurobi_env=self._env, bus_frame="buses")
        self.model.meta.pop("scenarios_invalidated", None)
        size_transformers_by_dcpf(self.model, self.config, self.tune)
        create_branch(self.model)

    def shunt_compensation(self) -> None:
        """Stage 7: synchronous-condenser winnowing → shunt placement."""
        from .network.shunts import place_reactive_compensation

        place_reactive_compensation(self.model, self.config)

    def ac_refinement(self) -> None:
        """Stage 8: MiddleLoopAC on the bus-level Complete tier."""
        self.tune.run_ac(self.model, self.model.scenarios.tiers["complete"])

    def final_acpf(self) -> None:
        """Stage 9: final converged AC power flow → model.acpf."""
        from .power.power_flow import run_ac_pf

        self.model.acpf = run_ac_pf(self.model, table="candidates")

    def metrics(self) -> None:
        """Stage 10: in-loop realism metrics (comparisons arrive in )."""
        from .metrics.realism import realism_metrics

        realism_metrics(self.model, self.config)

    def write_outputs(self) -> None:
        """Stage 11 (main_SRTV §7–§11): the io/writers.py exports, gated by
        the config.io flags — PowerWorld import CSVs + load time series
        (write_final_results), substation/bus dumps (write_buses), the
        5-sheet metrics workbook (write_metrics; EI/WECC targets used when
        their migrated data is present, computed-and-cached on first use),
        and the ACPF stats workbook. Outputs land under
        config.io.output_dir/<run_id>/."""
        from pathlib import Path

        from .io import writers

        cfg = self.config
        out = Path(cfg.io.output_dir) / self.registry.run_id
        model = self.model
        region = "".join(cfg.case.regions)

        if cfg.io.write_final_results:
            if len(model.buses) and len(model.branches):
                writers.write_powerworld(model, model.acpf, out, "KCL-KVL")
            else:
                warnings.warn("write_outputs: skipping PowerWorld CSVs — "
                              "model has no bus-level network yet (run "
                              "create_buses)", stacklevel=2)
            if "complete" in model.scenarios.tiers:
                writers.write_loads(model, model.acpf, out)
        if cfg.io.write_buses:
            writers.write_substation_data(model, out)
            writers.write_bus_tables(model, out)
        if cfg.io.write_metrics:
            from .metrics.comparison import realism_metrics_actual
            from .metrics.realism import realism_metrics

            realism = model.meta.get("realism") \
                or realism_metrics(model, cfg, table=(
                    "branches" if len(model.branches) else "candidates"))
            targets: dict = {}
            for case in ("EI", "WECC"):
                try:
                    targets[case] = realism_metrics_actual(
                        case, cfg.io.data_dir)
                except FileNotFoundError:
                    pass                       # real case not migrated here
            writers.write_metrics(realism, None, targets, cfg, out)
        if model.acpf is not None:
            writers.write_acpf_stats(model.acpf, out, region)


def main() -> None:
    """CLI: ``gridcrucible <run.toml> [--until STAGE] [--run-id ID]``.
    Wires the gridforge bridge (real UCED / CreateOLMatrices / RXB) when
    gridforge is importable; otherwise the bare constructor's lazy resolvers
    raise with instructions at the stage that needs them."""
    import argparse

    ap = argparse.ArgumentParser(description="GridCrucible pipeline runner")
    ap.add_argument("config", help="TOML run configuration")
    ap.add_argument("--until", default=None, choices=STAGES,
                    help="stop after this stage")
    ap.add_argument("--run-id", default=None,
                    help="checkpoint namespace (resume by reusing an id)")
    args = ap.parse_args()
    cfg = CrucibleConfig.from_toml(args.config)
    try:
        from .gridforge_bridge import wire_gridcrucible
        gc = wire_gridcrucible(cfg, run_id=args.run_id)
    except ImportError:
        gc = GridCrucible(cfg, run_id=args.run_id)
    model = gc.run(until=args.until)
    print(f"done: {model.n_substations} substations, "
          f"{len(model.candidates)} candidates, "
          f"{len(model.branches)} branches")


if __name__ == "__main__":
    main()
