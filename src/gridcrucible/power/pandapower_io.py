"""CrucibleModel ↔ pandapower network bridge.

MATLAB origin: Table2Matpower.m (84 loc) — see
docs/algorithms/power_flow_adapters.md for the full mapping rationale.

Every created branch element carries its model row index in a
``crucible_row`` column on net.line / net.trafo (pandapower preserves custom
element columns through to_json) — flows are ALWAYS harvested back in model
row order via that column, never by table position.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pandapower as pp

from ..model import ACPFResult, CrucibleModel

_F_HZ = 60.0
_VMAX, _VMIN = 1.06, 0.94        # Table2Matpower VMAX/VMIN
_DEFAULT_KV = 138.0              # fallback when no voltage info exists yet


def _bus_frame(model: CrucibleModel) -> pd.DataFrame:
    return model.buses if len(model.buses) else model.substations


def _bus_vm(model: CrucibleModel, rows: np.ndarray) -> np.ndarray:
    """PV setpoints for the given bus rows: the shunt stage's v_set where
    finite, else scenarios.v_setpoints, else 1.02."""
    bus_df = _bus_frame(model)
    n = len(bus_df)
    vm = np.full(n, 1.02)
    vset = model.scenarios.v_setpoints
    if len(vset) == n:
        vm = np.asarray(vset, float).copy()
    if "v_set" in bus_df.columns:
        vs = bus_df["v_set"].to_numpy(float)
        ok = np.isfinite(vs) & (vs > 0)
        vm[ok] = vs[ok]
    return vm[np.asarray(rows, int)]


def _bus_voltages_kv(model: CrucibleModel, branch_df: pd.DataFrame) -> np.ndarray:
    """Per-bus nominal kV: buses.voltage_kv when present/nonzero, else the max
    voltage_kv_from/to over incident branches, else 138 (only affects the
    pu→ohm→pu round trip, which cancels when consistent)."""
    n = model.n_buses
    vn = np.zeros(n)
    bus_df = _bus_frame(model)
    if "voltage_kv" in bus_df.columns:
        vn = bus_df["voltage_kv"].to_numpy(float).copy()
    if (vn <= 0).any() and {"voltage_kv_from", "voltage_kv_to"} <= set(branch_df.columns):
        for end, col in (("from_bus", "voltage_kv_from"), ("to_bus", "voltage_kv_to")):
            b = branch_df[end].to_numpy(np.int64)
            kv = branch_df[col].to_numpy(float)
            np.maximum.at(vn, b, kv)
    vn[vn <= 0] = _DEFAULT_KV
    return vn


def _branch_params(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(r_pu, x_pu, b_pu) on the system base; candidates carry reactance_x
    only (r=b=0 there — legal for DC, asserted against AC by callers)."""
    x = df["x_pu"] if "x_pu" in df.columns else df["reactance_x"]
    r = df["r_pu"] if "r_pu" in df.columns else pd.Series(0.0, index=df.index)
    b = df["b_pu"] if "b_pu" in df.columns else pd.Series(0.0, index=df.index)
    return (r.to_numpy(float), x.to_numpy(float), b.to_numpy(float))


def to_pandapower(model: CrucibleModel, table: str = "branches",
                  injections_mw: np.ndarray | None = None,
                  scenario: int | None = None,
                  slack_bus: int | None = None,
                  sync_cond_gens: bool = True,
                  pv_dispatch: bool = False) -> "pp.pandapowerNet":
    """Build a pandapower net (Table2Matpower port).

    table: 'branches' or 'candidates'. Exactly one injection source:
    * ``injections_mw`` — a per-bus real P vector (DC engines); positive =
      generation. Modeled as sgens (P>0) and loads (P<0); with
      ``pv_dispatch`` the positive injections at GEN-ROLE buses
      (scenarios slack/gen lists) become PV gens at their setpoints
      instead — MATLAB's Bus.Type semantics, required for AC solves
      (Q-less sgens at floating PQ unit buses leave the Jacobian
      near-singular, measured 2026-07-11).
    * ``scenario`` — a column of the active SNet tier (complex, P+jQ).
    * neither — physical loads (load_mw/mvar) + PV gens from the generator
      table (vm from v_setpoints, default 1.02) + poly costs for later OPF.

    sync_cond_gens: when the bus frame carries the shunt stage's
    ``sync_cond`` column, model those buses as PV condensers (p=0,
    vm=v_set, the MATLAB ACPF.HighQ/LowQ bank limits). The shunt stage
    itself passes False (it manages its own condenser set while
    winnowing). Without them the post-stage-7 network has no reactive
    support and AC solves genuinely collapse (0/94 scenarios, 2026-07-11).

    slack_bus: force the ext_grid bus (run_dc_pf parity with MATLAB's
    slack=bus 0); default follows Table2Matpower — scenarios.slack_bus if
    set, else the max-capacity generator bus, else bus 0.
    """
    branch_df = getattr(model, table)
    bus_df = _bus_frame(model)
    n = model.n_buses
    s_base = float(model.meta.get("s_base", 100.0))

    net = pp.create_empty_network(sn_mva=s_base, f_hz=_F_HZ)
    vn = _bus_voltages_kv(model, branch_df)
    pp.create_buses(net, n, vn_kv=vn, max_vm_pu=_VMAX, min_vm_pu=_VMIN,
                    geodata=None)

    # --- branches: tap==0 & same end-kV -> line, else trafo ----------------
    r_pu, x_pu, b_pu = _branch_params(branch_df)
    fb = branch_df["from_bus"].to_numpy(np.int64)
    tb = branch_df["to_bus"].to_numpy(np.int64)
    cap = (branch_df["capacity_mva"].to_numpy(float)
           if "capacity_mva" in branch_df.columns else np.full(len(branch_df), 9999.0))
    tap = (branch_df["tap"].to_numpy(float)
           if "tap" in branch_df.columns else np.zeros(len(branch_df)))
    kv_f, kv_t = vn[fb], vn[tb]
    is_trafo = (tap != 0) | (kv_f != kv_t)
    rows = np.arange(len(branch_df))

    li = ~is_trafo
    if li.any():
        z_base = kv_f[li] ** 2 / s_base
        omega = 2 * np.pi * _F_HZ
        idx = pp.create_lines_from_parameters(
            net, from_buses=fb[li], to_buses=tb[li], length_km=1.0,
            r_ohm_per_km=np.maximum(r_pu[li] * z_base, 0.0),
            x_ohm_per_km=x_pu[li] * z_base,
            c_nf_per_km=b_pu[li] / (omega * z_base) * 1e9,
            max_i_ka=cap[li] / (np.sqrt(3.0) * kv_f[li]))
        net.line.loc[idx, "crucible_row"] = rows[li]

    tr = is_trafo
    if tr.any():
        hv_is_f = kv_f >= kv_t
        hv = np.where(hv_is_f, fb, tb)[tr]
        lv = np.where(hv_is_f, tb, fb)[tr]
        sn = np.where(cap[tr] > 0, cap[tr], s_base)
        # x_pu is on the SYSTEM base; vk% is on the TRAFO base.
        vk = np.maximum(x_pu[tr] * (sn / s_base) * 100.0, 0.01)
        vkr = np.clip(r_pu[tr] * (sn / s_base) * 100.0, 0.0, vk - 1e-6)
        # NOTE: tap RATIO modeling deferred — the KVL pipeline sets Tap=0
        # everywhere today (main_SRTV initializes zeros); revisit if TUNE
        # starts assigning off-nominal taps.
        idx = pp.create_transformers_from_parameters(
            net, hv_buses=hv, lv_buses=lv, sn_mva=sn,
            vn_hv_kv=np.maximum(kv_f, kv_t)[tr], vn_lv_kv=np.minimum(kv_f, kv_t)[tr],
            vk_percent=vk, vkr_percent=vkr, pfe_kw=0.0, i0_percent=0.0)
        net.trafo.loc[idx, "crucible_row"] = rows[tr]

    # --- slack --------------------------------------------------------------
    if slack_bus is None:
        if model.scenarios.gen_buses.size or model.scenarios.slack_bus:
            slack_bus = int(model.scenarios.slack_bus)
        elif "gen_max_mw" in bus_df.columns and bus_df["gen_max_mw"].max() > 0:
            slack_bus = int(bus_df["gen_max_mw"].to_numpy().argmax())
        else:
            slack_bus = 0
    vset = model.scenarios.v_setpoints
    slack_vm = float(vset[slack_bus]) if len(vset) == n else 1.02
    pp.create_ext_grid(net, bus=slack_bus, vm_pu=slack_vm)

    # --- injections ---------------------------------------------------------
    if injections_mw is not None and scenario is not None:
        raise ValueError("pass injections_mw OR scenario, not both")
    if injections_mw is not None or scenario is not None:
        if injections_mw is not None:
            inj = np.asarray(injections_mw)      # complex allowed: Q = imag
            p, q = np.real(inj).astype(float), np.imag(inj).astype(float)
        else:
            col = next(iter(model.scenarios.tiers.values()))[:, scenario]
            p, q = np.real(col), np.imag(col)
        src = (p > 0) & (np.arange(n) != slack_bus)
        snk = p < 0
        if pv_dispatch and model.scenarios.gen_buses.size:
            gen_role = np.zeros(n, bool)
            gen_role[np.asarray(model.scenarios.gen_buses, int)] = True
            gen_role[int(model.scenarios.slack_bus)] = True
            pv = src & gen_role
            if pv.any():
                rows = np.flatnonzero(pv)
                pp.create_gens(net, buses=rows, p_mw=p[pv],
                               vm_pu=_bus_vm(model, rows))
            src &= ~gen_role
        if src.any():
            pp.create_sgens(net, buses=np.flatnonzero(src), p_mw=p[src], q_mvar=q[src])
        if snk.any():
            pp.create_loads(net, buses=np.flatnonzero(snk), p_mw=-p[snk], q_mvar=-q[snk])
    else:
        if "load_mw" in bus_df.columns:
            has = bus_df["load_mw"].to_numpy(float) != 0
            if has.any():
                pp.create_loads(net, buses=np.flatnonzero(has),
                                p_mw=bus_df["load_mw"].to_numpy(float)[has],
                                q_mvar=(bus_df["load_mvar"].to_numpy(float)[has]
                                        if "load_mvar" in bus_df.columns else 0.0))
        gens = model.generators
        for _, g in gens.iterrows():
            if int(g["BusNum"]) == slack_bus:
                continue  # slack is the ext_grid
            gi = pp.create_gen(net, bus=int(g["BusNum"]), p_mw=0.0,
                               vm_pu=(float(vset[int(g["BusNum"])]) if len(vset) == n else 1.02),
                               min_p_mw=float(g["MinMW"]), max_p_mw=float(g["MaxMW"]),
                               min_q_mvar=float(g.get("MinMVar", 0.0)),
                               max_q_mvar=float(g.get("MaxMVar", 0.0)))
            if {"A", "B", "C"} <= set(gens.columns):
                # MATLAB gencost (Table2Matpower): C quadratic, B linear, A fixed
                pp.create_poly_cost(net, gi, "gen", cp0_eur=float(g["A"]),
                                    cp1_eur_per_mw=float(g["B"]),
                                    cp2_eur_per_mw2=float(g["C"]))

    # --- synchronous condensers (shunt stage output) -------------------------
    if sync_cond_gens and "sync_cond" in bus_df.columns:
        sc = bus_df["sync_cond"].fillna(False).to_numpy(bool).copy()
        sc[slack_bus] = False
        sc[net.gen["bus"].to_numpy(int)] = False   # never double-control a bus
        if sc.any():
            rows = np.flatnonzero(sc)
            # MATLAB ACPF.HighQ/LowQ bank sizes (network/shunts.py constants;
            # sign flip: MATLAB QNet is load-positive)
            pp.create_gens(net, buses=rows, p_mw=0.0,
                           vm_pu=_bus_vm(model, rows),
                           min_q_mvar=-800.0, max_q_mvar=1200.0)
    return net


def branch_flows_mva(net, n_branches: int, dc: bool = False) -> np.ndarray:
    """Per-branch apparent (or DC active) flow in MODEL ROW ORDER, taking
    max(|from|, |to|) like ACPowerFlowSimple. NaN for rows absent from the
    net (should not happen when the net came from to_pandapower)."""
    flows = np.full(n_branches, np.nan)
    if len(net.line):
        rows = net.line["crucible_row"].to_numpy(int)
        res = net.res_line
        if dc:
            s = np.maximum(np.abs(res["p_from_mw"]), np.abs(res["p_to_mw"]))
        else:
            s = np.maximum(np.hypot(res["p_from_mw"], res["q_from_mvar"]),
                           np.hypot(res["p_to_mw"], res["q_to_mvar"]))
        flows[rows] = s.to_numpy(float)
    if len(net.trafo):
        rows = net.trafo["crucible_row"].to_numpy(int)
        res = net.res_trafo
        if dc:
            s = np.maximum(np.abs(res["p_hv_mw"]), np.abs(res["p_lv_mw"]))
        else:
            s = np.maximum(np.hypot(res["p_hv_mw"], res["q_hv_mvar"]),
                           np.hypot(res["p_lv_mw"], res["q_lv_mvar"]))
        flows[rows] = s.to_numpy(float)
    return flows


def from_pandapower(net, n_branches: int) -> ACPFResult:
    """Harvest a solved net into an ACPFResult (voltages in pu/rad, flows in
    model row order, loss + voltage-stat frames)."""
    losses = pd.DataFrame({
        "element": ["line", "trafo"],
        "pl_mw": [float(net.res_line["pl_mw"].sum()) if len(net.line) else 0.0,
                  float(net.res_trafo["pl_mw"].sum()) if len(net.trafo) else 0.0],
        "ql_mvar": [float(net.res_line["ql_mvar"].sum()) if len(net.line) else 0.0,
                    float(net.res_trafo["ql_mvar"].sum()) if len(net.trafo) else 0.0],
    })
    vm = net.res_bus["vm_pu"].to_numpy(float)
    va = np.deg2rad(net.res_bus["va_degree"].to_numpy(float))
    stats = pd.DataFrame({"vm_max": [float(np.nanmax(vm))],
                          "vm_min": [float(np.nanmin(vm))],
                          "va_max_deg": [float(np.nanmax(np.rad2deg(va)))],
                          "total_loss_mw": [float(losses["pl_mw"].sum())]})
    return ACPFResult(v_mag=vm, v_angle=va,
                      flows_mva=branch_flows_mva(net, n_branches),
                      losses=losses, stats=stats,
                      converged=bool(net.converged))


def load_reference_case(name: str, data_dir: Path) -> "pp.pandapowerNet":
    """Load a migrated real-network case ('EIBranchSanitized',
    'EILargeIslandSanitized', 'WECCBranchSanitized', 'WECCLargeIslandSanitized')
    for the comparison metrics (replaces loadcase())."""
    path = Path(data_dir) / "reference_cases" / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} — run `python -m gridcrucible.io.matlab_import` first")
    return pp.from_json(path)
