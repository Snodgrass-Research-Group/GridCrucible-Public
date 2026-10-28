"""Substation → bus expansion and branch-table construction.

MATLAB origin: CreateBuses.m (460 loc), CreateBranch.m (11),
TrimQNCALines.m (40). Algorithm doc: docs/algorithms/CreateBuses.md —
including the slack-rule change (largest unit's bus) and the DEFERRED
step 8 (bus-level SNet rebuild + DC-PF transformer sizing, blocked on
build_snet's 'Subs' mode).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ..config import CrucibleConfig
from ..model import CrucibleModel, ScenarioSet
from .conductors import transformer_impedances

GEN_BUS_KV = 12.9      # MATLAB: generator buses
LOAD_BUS_KV = 13.8     # MATLAB: load buses (CreateLoadBuses branch)


def _sub_voltage_sets(subs: pd.DataFrame, levels: list[int]) -> list[list[int]]:
    """Per-substation ASCENDING list of present voltage levels (MATLAB
    Sub.Voltages rows, compacted)."""
    out = []
    for _, row in subs.iterrows():
        present = [kv for kv in sorted(levels)
                   if row.get(f"voltage_{kv}", 0) > 0]
        out.append(present)
    return out


def _new_xf_row(template_cols, **kw) -> dict:
    row = {c: 0.0 for c in template_cols}
    row.update({"tap": 1.0, "keep": True, "is_radial": True,
                "n_circuits": 1.0, "length_km": 1.0, "l1_cost": 0.0,
                "gmd_index": 1, "gmd_ft": 0.0, "resize": True,
                "reactance_x": 0.05,            # placeholder until sized
                "remove_prob": 0.0})
    row.update(kw)
    return row


def create_buses(model: CrucibleModel, config: CrucibleConfig,
                 xfdata: pd.DataFrame | None = None,
                 xfxr: pd.DataFrame | None = None) -> None:
    """CreateBuses.m port (steps 1-7 of the doc): voltage-split substations
    into buses, move candidate endpoints, create per-unit generator buses +
    GSUs and inter-voltage transmission transformers, land loads on each
    sub's minimum-voltage bus, re-derive bus roles (slack = LARGEST unit's
    bus), size transformer impedances, and INVALIDATE the scenario tiers."""
    if config.tune.change_voltages is False:
        pass                                     # no special casing needed
    if getattr(config, "create_load_buses", False) or \
            config.dispatch.buses.lower() == "load-buses":
        raise NotImplementedError("CreateLoadBuses branch — deferred (doc §5)")

    subs = model.substations
    cand = model.candidates
    fleet = model.generators
    levels = list(config.case.voltage_levels)
    n_sub = len(subs)
    sub_kvs = _sub_voltage_sets(subs, levels)

    # ---- 1. one bus per substation at its MINIMUM present voltage ---------
    buses = pd.DataFrame({
        "bus_id": np.arange(n_sub), "sub_id": np.arange(n_sub),
        "external_num": subs["external_num"].to_numpy(),
        "name": subs["name"].to_numpy(),
        "lat": subs["lat"].to_numpy(float), "lon": subs["lon"].to_numpy(float),
        "region": subs["region"].to_numpy(),
        "voltage_kv": [kvs[0] if kvs else max(levels) for kvs in sub_kvs],
        "bus_type": 1, "load_mw": 0.0, "load_mvar": 0.0,
        "gen_max_mw": 0.0, "gen_min_mw": 0.0,
    })
    cand = cand.reset_index(drop=True).copy()
    cand["from_bus"] = cand["from_orig"].astype(int)
    cand["to_bus"] = cand["to_orig"].astype(int)
    cand = cand[cand["tap"] == 0].reset_index(drop=True)  # re-run: drop XFs

    def _append_bus(**kw) -> int:
        nonlocal buses
        row = {"bus_id": len(buses), "external_num": buses["external_num"].max() + 1,
               "bus_type": 1, "load_mw": 0.0, "load_mvar": 0.0,
               "gen_max_mw": 0.0, "gen_min_mw": 0.0}
        row.update(kw)
        buses = pd.concat([buses, pd.DataFrame([row])], ignore_index=True)
        return len(buses) - 1

    # ---- 2. voltage splitting: one new bus per EXTRA level ----------------
    kv_from = cand["voltage_kv_from"].to_numpy(float)
    kv_to = cand["voltage_kv_to"].to_numpy(float)
    fo = cand["from_orig"].to_numpy(int)
    to = cand["to_orig"].to_numpy(int)
    for s in range(n_sub):
        for extra_kv in sub_kvs[s][1:]:          # ascending extras
            b = _append_bus(sub_id=s, name=subs.at[s, "name"],
                            lat=subs.at[s, "lat"], lon=subs.at[s, "lon"],
                            region=subs.at[s, "region"], voltage_kv=extra_kv)
            move_f = (fo == s) & (kv_from == extra_kv)
            move_t = (to == s) & (kv_to == extra_kv)
            cand.loc[move_f, "from_bus"] = b
            cand.loc[move_t, "to_bus"] = b

    # ---- 3. generator buses (one per UNIT) + GSUs --------------------------
    xf_rows: list[dict] = []
    fleet = fleet.copy()
    fleet["BuskV"] = 0.0
    for sub in np.sort(fleet["Sub"].unique()):
        sub = int(sub)
        units = fleet.index[fleet["Sub"] == sub]
        for poi_kv in pd.unique(fleet.loc[units, "POIkV"]):
            grp = [u for u in units if fleet.at[u, "POIkV"] == poi_kv]
            classes = sub_kvs[sub] or [max(levels)]
            poi_class = classes[int(np.argmin([abs(poi_kv - c)
                                               for c in classes]))]
            fleet.loc[grp, "BuskV"] = poi_class
            poi_bus = buses.index[(buses["sub_id"] == sub)
                                  & (buses["voltage_kv"] == poi_class)]
            poi_bus = int(poi_bus[0]) if len(poi_bus) else sub
            for u in grp:
                gb = _append_bus(sub_id=sub, name=subs.at[sub, "name"],
                                 lat=subs.at[sub, "lat"],
                                 lon=subs.at[sub, "lon"],
                                 region=subs.at[sub, "region"],
                                 voltage_kv=GEN_BUS_KV, bus_type=2,
                                 gen_max_mw=float(fleet.at[u, "MaxMW"]),
                                 gen_min_mw=float(fleet.at[u, "MinMW"]))
                fleet.at[u, "BusNum"] = gb
                max_mw = float(fleet.at[u, "MaxMW"])
                xf_rows.append(_new_xf_row(
                    cand.columns,
                    from_bus=gb, to_bus=poi_bus, from_orig=gb, to_orig=poi_bus,
                    voltage_kv_from=GEN_BUS_KV, voltage_kv_to=float(poi_class),
                    capacity_mva=math.ceil(1.2 * max_mw),
                    min_conductor_idx=math.ceil(max_mw),
                    max_conductor_idx=(1.2 * float(xfxr["MVA"].max())
                                       if xfxr is not None else 2400.0),
                    conductor_idx=math.ceil(1.2 * max_mw)))

    # ---- 4. transmission transformers between consecutive levels ----------
    for s in range(n_sub):
        for kv_lo, kv_hi in zip(sub_kvs[s][:-1], sub_kvs[s][1:]):
            b_lo = int(buses.index[(buses["sub_id"] == s)
                                   & (buses["voltage_kv"] == kv_lo)][0])
            b_hi = int(buses.index[(buses["sub_id"] == s)
                                   & (buses["voltage_kv"] == kv_hi)][0])
            xf_rows.append(_new_xf_row(
                cand.columns, transmission_xf=True,
                from_bus=b_lo, to_bus=b_hi, from_orig=b_lo, to_orig=b_hi,
                voltage_kv_from=float(kv_lo), voltage_kv_to=float(kv_hi),
                capacity_mva=2000.0,            # placeholder; sized in step 8
                min_conductor_idx=20.0,
                max_conductor_idx=(1.2 * float(xfxr["MVA"].max())
                                   if xfxr is not None else 2400.0),
                conductor_idx=2000.0, is_radial=False))

    if xf_rows:
        cand = pd.concat([cand, pd.DataFrame(xf_rows)], ignore_index=True)
    if "transmission_xf" in cand.columns:
        cand["transmission_xf"] = cand["transmission_xf"] \
            .fillna(False).astype(bool)
    # NOMINAL voltages for PowerWorld (MATLAB kVFromNom/kVToNom):
    # transformer_impedances SNAPS the working kV columns to XFData Pri/Sec
    # (12.9 -> 69 for GSUs, faithfully) — the nominals preserve the truth.
    cand["voltage_kv_from_nom"] = cand["voltage_kv_from"]
    cand["voltage_kv_to_nom"] = cand["voltage_kv_to"]

    # ---- 5. loads on each sub's FIRST (minimum-voltage) bus ----------------
    buses.loc[:n_sub - 1, "load_mw"] = subs["load_mw"].to_numpy()
    buses.loc[:n_sub - 1, "load_mvar"] = subs["load_mvar"].to_numpy()

    # ---- 6. bus roles: slack = LARGEST unit's bus --------------------------
    largest = int(fleet["MaxMW"].idxmax())
    slack = int(fleet.at[largest, "BusNum"])
    gen_buses = np.setdiff1d(np.sort(fleet["BusNum"].unique()), [slack])
    load_buses = np.setdiff1d(np.arange(len(buses)),
                              np.concatenate([[slack], gen_buses]))
    v = np.ones(len(buses))
    v[np.concatenate([[slack], gen_buses])] = 1.02
    model.scenarios = ScenarioSet(tiers={}, slack_bus=slack,
                                  gen_buses=gen_buses, load_buses=load_buses,
                                  v_setpoints=v)   # tiers INVALIDATED
    model.meta["scenarios_invalidated"] = True

    # ---- 7. transformer impedances -----------------------------------------
    if xfdata is not None and xfxr is not None:
        transformer_impedances(cand, xfdata, xfxr)

    model.buses = buses
    model.candidates = cand
    model.generators = fleet
    model.meta["node_degree"] = model.node_degree("candidates")


def size_transformers_by_dcpf(model: CrucibleModel, config: CrucibleConfig,
                              tune) -> None:
    """CreateBuses.m step 8 (after the bus-level SNet rebuild): two DC-PF
    passes on the Complete tier — pass 1 rates the RateA==2000 transmission-
    transformer placeholders from the per-path max flows (+ impedances),
    pass 2 feeds ResizeBranches / AssignConductors / the impedance recompute.
    MATLAB wrapped this in a silently-degrading try/catch; Python fails loud."""
    cand = model.candidates
    snet = np.real(model.scenarios.tiers["complete"])
    # raw bus-level tier: NOT OHM-scaled (tune.py _max_dc_flows docstring)
    max1 = tune._max_dc_flows(model, snet, ohm_scaled=False)
    cand["max_dcpf"] = max1
    if "transmission_xf" in cand.columns:
        placeholder = cand["transmission_xf"].to_numpy(bool) \
            & (cand["capacity_mva"].to_numpy(float) == 2000.0)
        cand.loc[placeholder, "capacity_mva"] = \
            np.maximum(max1[placeholder], 1.0)
        cand.loc[placeholder, "resize"] = True
        if tune.xfdata is not None:
            transformer_impedances(cand, tune.xfdata, tune.xfxr)
    max2 = tune._max_dc_flows(model, snet, ohm_scaled=False)
    cand["max_dcpf"] = max2
    tune._resize(cand, max2)
    tune._reassign_and_recompute(model)


def create_branch(model: CrucibleModel) -> None:
    """CreateBranch port: model.branches = candidates + EXTERNAL numbering
    columns (from_ext/to_ext) — the reporting view metrics/writers consume."""
    bus_df = model.buses if len(model.buses) else model.substations
    ext = bus_df["external_num"].to_numpy()
    branch = model.candidates.copy()
    branch["from_ext"] = ext[branch["from_bus"].to_numpy(int)]
    branch["to_ext"] = ext[branch["to_bus"].to_numpy(int)]
    model.branches = branch


def trim_generator_lines(model: CrucibleModel) -> int:
    """TrimQNCALines port (resume/add path): at each generator bus, drop
    gen-to-gen lines ≥ 80 mi, or (when the bus has > 6 / > 4 incident
    lines) ≤ 60 MVA lines ≥ 10 / ≥ 30 mi. NOTE the MATLAB mile factor here
    is 1.6, not 1.61 (ported as written). Returns rows removed."""
    cand = model.candidates
    gens = np.unique(model.generators["BusNum"].to_numpy(int))
    removed = 0
    for bus in gens:
        fb = cand["from_bus"].to_numpy(int)
        tb = cand["to_bus"].to_numpy(int)
        incident = np.flatnonzero((fb == bus) | (tb == bus))
        gen_gen = incident[np.isin(fb[incident], gens)
                           & np.isin(tb[incident], gens)]
        L = cand["length_km"].to_numpy(float)[gen_gen]
        cap = cand["capacity_mva"].to_numpy(float)[gen_gen]
        drop = L >= 80 * 1.6
        if len(incident) > 6:
            drop |= (cap <= 60) & (L >= 10 * 1.6)
        elif len(incident) > 4:
            drop |= (cap <= 60) & (L >= 30 * 1.6)
        if drop.any():
            cand = cand.drop(index=cand.index[gen_gen[drop]]) \
                .reset_index(drop=True)
            removed += int(drop.sum())
            model.candidates = cand
    return removed
