"""Net-injection scenario construction (the SNet tiers).

MATLAB origin: CreateSNet.m (585 loc — , still shell) and
CreateLoadScenarios.m (74 loc — converted here).
Algorithm docs: docs/algorithms/MakeLoadProfiles.md (§CreateLoadScenarios),
docs/algorithms/CreateSNet.md (todo — write before implementing build_snet).
/ 1c
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from ..config import CrucibleConfig
from ..model import CrucibleModel, ScenarioSet
from .load_profiles import make_load_profiles


def _stable_intersect_positions(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """MATLAB [~, ia] = intersect(A, B, 'stable'): positions into `a` of the
    elements present in `b`, in a's order."""
    return np.flatnonzero(np.isin(a, b))


def build_load_scenarios(bus_input: pd.DataFrame, config: CrucibleConfig,
                         model_bus_external: np.ndarray | None = None,
                         profile_fn: Callable = make_load_profiles,
                         ) -> tuple[np.ndarray, np.ndarray]:
    """CreateLoadScenarios.m port: hourly (P, Q) matrices for every bus of
    the configured regions, in EXTERNAL (BusInputData row, 1-based) order.

    Per region: fetch the regional profiles, then
    * if the profiles' first column is a mostly-consecutive bus-number
      sequence (MATLAB `mode(diff(col0)) == 1`): strip it; when the
      numbering has gaps (≥2 distinct increments), SCATTER the rows onto the
      region's full bus list; otherwise PREPEND zero rows for the region's
      generator buses (profiles cover load buses only).
    * if `model_bus_external` is given and smaller than the region, keep
      only rows for buses actually in the model (stable order).

    bus_input: the migrated BusInputData (row number + 1 == external bus
    number — the EPIGRIDS convention every table in this pipeline shares).
    """
    p_final: list[np.ndarray] = []
    q_final: list[np.ndarray] = []
    for region in config.case.regions:
        region_mask = bus_input["Region"].astype(str) == region
        local_buses = np.flatnonzero(region_mask.to_numpy()) + 1   # external, 1-based
        n_gen = int((region_mask
                     & (bus_input["Type"].astype(str) == "Gen")).sum())

        p, q = profile_fn(region, config.io.data_dir)
        p, q = np.asarray(p, float), np.asarray(q, float)

        col0 = p[:, 0]
        diffs = np.diff(col0)
        has_bus_col = diffs.size > 0 and _mode(diffs) == 1
        if has_bus_col:
            bus_nums = col0.astype(int)
            transitions = np.unique(diffs)
            p, q = p[:, 1:], q[:, 1:]
            if len(transitions) >= 2:          # gapped numbering: scatter
                pos = _stable_intersect_positions(local_buses, bus_nums)
                p_full = np.zeros((len(local_buses), p.shape[1]))
                q_full = np.zeros((len(local_buses), q.shape[1]))
                p_full[pos, :] = p
                q_full[pos, :] = q
                p, q = p_full, q_full
            else:                              # contiguous: pad gen buses
                p = np.vstack([np.zeros((n_gen, p.shape[1])), p])
                q = np.vstack([np.zeros((n_gen, q.shape[1])), q])
        else:
            p = np.vstack([np.zeros((n_gen, p.shape[1])), p])
            q = np.vstack([np.zeros((n_gen, q.shape[1])), q])

        if model_bus_external is not None \
                and len(model_bus_external) < len(local_buses):
            pos = _stable_intersect_positions(local_buses,
                                              np.asarray(model_bus_external))
            p, q = p[pos, :], q[pos, :]

        p_final.append(p)
        q_final.append(q)
    return np.vstack(p_final), np.vstack(q_final)


def _mode(arr: np.ndarray) -> float:
    vals, counts = np.unique(arr, return_counts=True)
    return float(vals[counts.argmax()])


def _select_scenario_hours(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """CreateSNet hour selection: min-load hour ±60 + peak hour ±60 → 242
    complex columns. (WinterWeek/PeakRamp are DEAD CODE in MATLAB — see the
    algorithm doc.) Windows clamped to the year with a warning (MATLAB would
    index-error)."""
    import warnings

    totals = p.sum(axis=0)
    n_hours = p.shape[1]

    def window(center: int) -> np.ndarray:
        lo, hi = center - 60, center + 61
        if lo < 0 or hi > n_hours:
            warnings.warn(f"scenario window at hour {center} clamped to the "
                          "year (MATLAB would error here)", stacklevel=3)
            lo, hi = max(lo, 0), min(hi, n_hours)
        return np.arange(lo, hi)

    # MATLAB max/min SKIP NaN totals (real SRTV raw data has NaN at KY hours
    # 1587/7298 and TN hour 699); numpy arg{min,max} would return the first
    # NaN position instead — nanarg* is the faithful port.
    cols = np.concatenate([window(int(np.nanargmin(totals))),
                           window(int(np.nanargmax(totals)))])
    scen = p[:, cols] + 1j * q[:, cols]
    return np.nan_to_num(scen)


def _diff_num_ed(n_buses: int, regional: bool) -> int:
    d = 0 if n_buses < 10000 else 1 if n_buses < 15000 else \
        2 if n_buses < 20000 else 3
    if regional and n_buses > 10000:
        d += 1
    return d


def _diff_num_cd(n_buses: int, regional: bool, gen_load_ratio: float,
                 regions: list[str], ed_diff: int) -> int:
    if gen_load_ratio < 1.1 or any(r.upper() == "RFCW" for r in regions):
        d = 0
    elif n_buses < 2000 or gen_load_ratio < 1.2 \
            or (len(regions) > 2 and regional):
        d = 2
    elif n_buses < 9000:
        d = 4
    elif n_buses < 12000:
        d = 6
    else:
        d = 8
    if regional and n_buses > 10000:
        d = ed_diff + 1
    return d


def _commitment_subsample(block: np.ndarray, gen_rows: np.ndarray,
                          diff_num: int) -> np.ndarray:
    """Greedy plant-commitment dedup (the tier engine): committed = gen rows
    with real injection > 0.1; scenarios whose commitment differs from the
    first remaining scenario by ≤ diff_num plants collapse to the one with
    max total generation."""
    remaining = block.copy()
    kept = []
    while remaining.shape[1]:
        pf_gen = remaining[gen_rows, :].real
        current = (pf_gen[:, 0] > 0.1).astype(int)
        committed = (pf_gen > 0.1).astype(int)
        differing = (current[:, None] - committed).sum(axis=0)
        matching = np.flatnonzero(np.abs(differing) <= diff_num)
        best = matching[int(pf_gen[:, matching].sum(axis=0).argmax())]
        kept.append(remaining[:, best])
        remaining = np.delete(remaining, matching, axis=1)
    return np.column_stack(kept) if kept else np.empty((block.shape[0], 0),
                                                       complex)


def _dispatch_block(loads: np.ndarray, fleet: pd.DataFrame,
                    dispatch_fn, reserve: float, gen_load_ratio: float,
                    n_buses: int, gurobi_env=None) -> np.ndarray:
    """One UC block: per scenario column, dispatch the fleet against the
    total load and form SNet col = per-bus dispatch − load. Hydro scheduled
    by scenario third (⅔ / 1 / ⅓) when GenLoadRatio > 1.2."""
    n_scen = loads.shape[1]
    out = np.zeros((n_buses, n_scen), complex)
    for i in range(n_scen):
        if gen_load_ratio > 1.2:
            hydro = (2 / 3) if i < n_scen / 3 else \
                1.0 if i < 2 * n_scen / 3 else (1 / 3)
        else:
            hydro = 1.0
        x, fleet_out = dispatch_fn(fleet, hydro, loads[:, i].real, reserve,
                                   gurobi_env=gurobi_env)
        gen_bus = np.bincount(fleet_out["BusNum"].to_numpy(int),
                              weights=x, minlength=n_buses)
        out[:, i] = gen_bus - loads[:, i]
    return out


def _resolve_uced(uced_fn):
    """UCED is GRIDFORGE-OWNED (ARCHITECTURE.md §3): resolve it lazily so this module
    isn't import-coupled to gridforge's conversion progress."""
    if uced_fn is not None:
        return uced_fn
    try:
        from gridcrucible._gridforge.dispatch.economic_dispatch import uced  # type: ignore
        return uced
    except (ImportError, AttributeError) as e:
        raise ImportError(
            "UCED (economic dispatch) is gridforge-owned; the vendored "
            "gridcrucible._gridforge exposes it as EconomicDispatcher."
            "solve_uced (adapted by gridforge_bridge.uced), so there is no "
            "drop-in module-level symbol here — pass uced_fn=... (signature "
            "(fleet, hydro_scaling, load_mw, reserve, gurobi_env=None) -> "
            "(x, fleet_out)), e.g. via gridforge_bridge.wire_gridcrucible") from e


def build_snet(model: CrucibleModel, config: CrucibleConfig,
               bus_input: pd.DataFrame,
               profile_fn: Callable = make_load_profiles,
               uced_fn: Callable | None = None,
               gurobi_env=None,
               bus_frame: str = "substations") -> ScenarioSet:
    """CreateSNet.m port (docs/algorithms/CreateSNet.md — read it; every
    guard below is specified there, including the MATLAB MaxMVar←MaxMW bug
    this port fixes).

    Preconditions: model.generators filled (setup_generators — internal
    0-based BusNum + MATLAB-style MinMW/MaxMW/A/B/C/FuelType columns) and
    model.scenarios slack/gen/load bus lists populated. Currently implements
    dispatch='Full' and the EPIGRIDS bus mode (the InitOptions defaults);
    'Regional'/'Subs' paths raise until a case needs them."""
    if config.dispatch.mode.lower() != "full":
        raise NotImplementedError("dispatch='Regional' — port when a "
                                  "multi-region case needs it (doc §Pipeline)")
    fleet = model.generators
    scen_meta = model.scenarios

    # bus_frame='buses' is MATLAB's Buses='Subs' rebuild inside CreateBuses:
    # substation load profiles scatter onto each sub's LOAD bus (the
    # minimum-voltage bus), dispatch maps through the bus-level fleet ids.
    bus_level = bus_frame == "buses"
    subs = model.buses if bus_level else model.substations
    n_buses = len(subs)

    # 1. per-bus gen aggregates (MVar from MVar — MATLAB bug fixed, doc §1)
    subs["gen_max_mw"] = np.bincount(fleet["BusNum"].to_numpy(int),
                                     weights=fleet["MaxMW"], minlength=n_buses)
    subs["gen_min_mw"] = np.bincount(fleet["BusNum"].to_numpy(int),
                                     weights=fleet["MinMW"], minlength=n_buses)

    # 2. load hours -> 242 scenario columns (profiles are per SUBSTATION;
    # bus-level mode scatters each sub's row onto its load bus)
    sub_ext = model.substations["external_num"].to_numpy()
    p, q = build_load_scenarios(bus_input, config,
                                model_bus_external=sub_ext,
                                profile_fn=profile_fn)
    if bus_level:
        load_bus_of_sub = np.zeros(len(model.substations), int)
        for s_id in range(len(model.substations)):
            rows = subs.index[(subs["sub_id"] == s_id)
                              & (subs["load_mw"].to_numpy() > 0)]
            if len(rows) == 0:
                rows = subs.index[subs["sub_id"] == s_id]
            load_bus_of_sub[s_id] = int(rows[0])
        p_b = np.zeros((n_buses, p.shape[1]))
        q_b = np.zeros((n_buses, q.shape[1]))
        p_b[load_bus_of_sub, :] = p
        q_b[load_bus_of_sub, :] = q
        p, q = p_b, q_b
    scen = _select_scenario_hours(p, q)
    n_scen = scen.shape[1]

    # 3. import-bus schedule (thirds): 0 / export / import
    import_rows = np.asarray(model.meta.get("import_buses_int", []), int)
    if import_rows.size:
        gen_mean = subs["gen_max_mw"][subs["gen_max_mw"] > 0].mean()
        thirds = np.floor(np.array([1, 2]) / 3 * n_scen).astype(int)
        scen[import_rows, :thirds[0]] = 0.0
        scen[import_rows, thirds[0]:thirds[1]] = -gen_mean
        scen[import_rows, thirds[1]:] = gen_mean

    # 4. low-load floor in the max scenario (0.2 + 0.12j)
    max_scen = int(scen.real.sum(axis=0).argmax())
    load_rows = scen_meta.load_buses if scen_meta.load_buses.size \
        else np.arange(n_buses)
    low = load_rows[scen.real[load_rows, max_scen] < 0.2]
    low = low[scen.real[low, max_scen] != 0]      # untouched zero-load buses
    scen[low, :] = 0.2 + 0.12j

    # 5. load-scaling guards
    hydro_cap = float(fleet.loc[fleet["FuelType"].astype(str)
                                .str.contains("Hydro", case=False), "MaxMW"].sum())
    peak = float(scen.real[:, max_scen].sum())
    cap, cap_min = float(fleet["MaxMW"].sum()), float(fleet["MinMW"].sum())
    if cap - (2 / 3) * hydro_cap <= 1.01 * peak:
        scen *= cap / (1.01 * peak)
    elif cap_min >= 0.99 * peak:
        scen *= 1.01 * cap_min / peak

    gen_load_ratio = cap / float(scen.real[:, max_scen].sum())

    # 6. dispatch blocks -> SNetFull (scen holds POSITIVE demand throughout,
    # matching MATLAB's LoadScenarios; SNet col = dispatch - load)
    loads = scen
    blocks: list[np.ndarray] = []
    model.meta["ed_block_width"] = n_scen if config.dispatch.use_ed else 0
    if config.dispatch.use_ed:
        blocks.append(_dispatch_block(loads, fleet, _resolve_uced(uced_fn),
                                      config.dispatch.reserve, gen_load_ratio,
                                      n_buses, gurobi_env))
    if config.dispatch.use_cd:
        from .dispatch import uccd
        blocks.append(_dispatch_block(loads, fleet, uccd,
                                      config.dispatch.reserve, gen_load_ratio,
                                      n_buses, gurobi_env))
    if config.dispatch.gen_n_minus_1:
        big = subs.index[subs["gen_max_mw"]
                         > config.dispatch.gen_contingency_min_mw]
        order = subs.loc[big, "gen_max_mw"].sort_values(ascending=False).index
        for bus in order:
            n1 = fleet[fleet["BusNum"] != bus]
            l1 = loads.copy()
            p1 = float(l1.real[:, max_scen].sum())
            if n1["MaxMW"].sum() <= 1.001 * p1:
                l1 = l1 * (fleet["MaxMW"].sum() / (1.001 * p1))
            elif n1["MinMW"].sum() >= 0.999 * p1:
                l1 = l1 * (1.001 * fleet["MinMW"].sum() / p1)
            blocks.append(_dispatch_block(l1, n1, _resolve_uced(uced_fn),
                                          config.dispatch.reserve,
                                          gen_load_ratio, n_buses, gurobi_env))
    snet_full = np.hstack(blocks)

    # 7. Bus.MW/MVar <- peak scenario loads (post-guards)
    subs["load_mw"] = loads[:, max_scen].real
    subs["load_mvar"] = loads[:, max_scen].imag

    # 8. tier subsampling per UC block
    gen_rows = np.concatenate([[scen_meta.slack_bus], scen_meta.gen_buses]) \
        if scen_meta.gen_buses.size else np.arange(n_buses)
    regional = False
    ed_diff = _diff_num_ed(n_buses, regional)
    tiers: dict[str, list[np.ndarray]] = {t: [] for t in
                                          ("complete", "advanced",
                                           "intermediate", "basic")}
    for bi, block in enumerate(blocks):
        if bi == 0 and config.dispatch.use_ed:
            d = ed_diff
        elif bi <= 1 and config.dispatch.use_cd:
            d = _diff_num_cd(n_buses, regional, gen_load_ratio,
                             config.case.regions, ed_diff)
        else:
            d = 0
        tiers["complete"].append(_commitment_subsample(block, gen_rows, d))
        adv = _commitment_subsample(block, gen_rows, d + 1)
        tiers["advanced"].append(adv)
        tiers["intermediate"].append(
            adv if n_buses < 4000
            else _commitment_subsample(block, gen_rows, d + 2))
        tiers["basic"].append(
            tiers["intermediate"][-1] if n_buses < 2000
            else _commitment_subsample(block, gen_rows, d + 3))

    return ScenarioSet(
        tiers={"full": snet_full,
               **{t: np.hstack(v) for t, v in tiers.items()}},
        slack_bus=scen_meta.slack_bus, gen_buses=scen_meta.gen_buses,
        load_buses=scen_meta.load_buses, v_setpoints=scen_meta.v_setpoints)
