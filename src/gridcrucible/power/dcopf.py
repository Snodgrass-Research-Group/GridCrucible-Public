"""DC-OPF benchmark statistics (MatpowerDCOPF.m port).

MATLAB origin: MatpowerDCOPF.m (143 loc) — per injection scenario, run a DC
OPF; when infeasible, relax all branch ratings uniformly, then TIGHTEN back
down re-solving until infeasible, recording the tightest feasible pass.
Feeds ElectricalComparison's `DCOPFStats` (validation targets:
data/reference_run/pf_dcopfstats.parquet).
Algorithm doc: docs/algorithms/ElectricalComparison.md — every rule and
quirk below is specified there; the MOSEK→MIPS solver chain collapses to
pandapower's ``rundcopp`` (ARCHITECTURE.md §2.2 "pandapower everywhere").
· consumed by metrics/comparison.py.
"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd

from ..model import CrucibleModel
from . import pandapower_io as ppio
from .power_flow import angle_differences_deg

#: MATLAB's magic per-unit sniff threshold (max column Σ|SNet| below this is
#: taken to be per-unit and multiplied by SBase).
_PU_SNIFF = 35e3

#: DCOPFStats columns, MATLAB names (reference-parquet compatible).
_STAT_COLUMNS = ("Network", "Loads", "LoadMW", "LimitRelaxPercent", "GenMW",
                 "MaxAngleDiff", "MeanAngleDiff", "Losses", "LossPercent",
                 "PercentLoadedMean", "PercentLoadedStd", "success")


def _opf_exceptions() -> tuple:
    import pandapower as pp
    excs: list[type] = [pp.powerflow.LoadflowNotConverged]
    try:
        from pandapower.optimal_powerflow import OPFNotConverged
        excs.append(OPFNotConverged)
    except ImportError:
        pass
    return tuple(excs)


def _base_net(model: CrucibleModel, table: str):
    """to_pandapower physical net (costed gens), stripped of bus-table loads
    (scenario columns supply the loads) and with the slack generator's
    limits/cost attached to the ext_grid (MATLAB keeps the slack unit in
    mpc.gen; pandapower moves it to ext_grid, which OPF dispatches only
    within explicit limits + cost)."""
    import pandapower as pp

    net = ppio.to_pandapower(model, table=table)
    net.load = net.load.iloc[0:0]
    slack = int(model.scenarios.slack_bus)
    fleet = model.generators
    at_slack = fleet.index[fleet["BusNum"].astype(int) == slack]
    if len(at_slack):
        g = fleet.loc[at_slack]
        net.ext_grid["min_p_mw"] = float(g["MinMW"].sum())
        net.ext_grid["max_p_mw"] = float(g["MaxMW"].sum())
        if {"A", "B", "C"} <= set(fleet.columns):
            r = g.iloc[0]
            pp.create_poly_cost(net, int(net.ext_grid.index[0]), "ext_grid",
                                cp0_eur=float(r["A"]),
                                cp1_eur_per_mw=float(r["B"]),
                                cp2_eur_per_mw2=float(r["C"]))
    return net


def run_dcopf_stats(model: CrucibleModel, snet: np.ndarray, *,
                    network: str = "KCL/KVL", loads: str = "KCL/KVL",
                    dc_overload_percent_max: np.ndarray | None = None,
                    table: str = "branches") -> pd.DataFrame:
    """MatpowerDCOPF.m port: one `DCOPFStats` row per scenario column.

    snet: complex (n_buses × n_scen); MW, with the MATLAB per-unit sniff
    kept verbatim (max column Σ|snet| < 35e3 → multiplied by SBase).
    dc_overload_percent_max: DCStats.OverloadPercentMax per scenario (from
    the plain DC-PF stats on the SAME scenarios) — sizes the first rating
    relaxation `max(1, 1 + |x|/100)`; None → 1 (straight to the 3× step).

    Per scenario: loads at Re<0 buses; generators OUT at buses without
    positive injection; PMIN scaled by 1/(1.2·ΣPMIN/ΣPD) when ΣPMIN > ΣPD;
    infeasible → relax ratings ×RelaxMulti → ×3·RelaxMulti → give up; on
    relaxed feasibility, tighten (×0.90/0.95/0.98/0.99 by current relax%)
    re-solving until infeasible, keeping the tightest feasible pass.
    `LimitRelaxPercent` reports the uniform multiplier directly (MATLAB
    read it off branch 1 — identical for uniform scaling) and floors at 0.
    """
    import pandapower as pp

    snet = np.atleast_2d(np.asarray(snet))
    if snet.shape[0] != model.n_buses:
        snet = snet.T
    s_base = float(model.meta.get("s_base", 100.0))
    if np.abs(snet).sum(axis=0).max() < _PU_SNIFF:
        snet = snet * s_base

    base = _base_net(model, table)
    branch_df = getattr(model, table)
    m = len(branch_df)
    A = model.incidence(table)
    r_pu = (branch_df["r_pu"] if "r_pu" in branch_df.columns
            else pd.Series(0.0, index=branch_df.index)).to_numpy(float)
    cap = branch_df["capacity_mva"].to_numpy(float)
    fails = _opf_exceptions()

    rows = []
    for i in range(snet.shape[1]):
        p_inj = snet[:, i].real
        row: dict = {"Network": network, "Loads": loads, "success": False,
                     "LimitRelaxPercent": np.nan, "GenMW": np.nan,
                     "MaxAngleDiff": np.nan, "MeanAngleDiff": np.nan,
                     "Losses": np.nan, "LossPercent": np.nan,
                     "PercentLoadedMean": np.nan, "PercentLoadedStd": np.nan}

        net = copy.deepcopy(base)
        load_buses = np.flatnonzero(p_inj < 0)
        pp.create_loads(net, buses=load_buses, p_mw=-p_inj[load_buses],
                        q_mvar=np.abs(snet[load_buses, i].imag),
                        controllable=False)
        row["LoadMW"] = float(-p_inj[load_buses].sum())

        gen_on = p_inj > 0
        if len(net.gen):
            net.gen["in_service"] = gen_on[net.gen["bus"].to_numpy(int)]
            off = ~net.gen["in_service"].to_numpy(bool)
            net.gen.loc[off, "min_p_mw"] = 0.0
            has_eg_lims = "min_p_mw" in net.ext_grid.columns
            pmin_sum = float(net.gen.loc[~off, "min_p_mw"].sum()) \
                + (float(net.ext_grid["min_p_mw"].fillna(0).sum())
                   if has_eg_lims else 0.0)
            if pmin_sum > row["LoadMW"] > 0:
                scale = 1.0 / (1.2 * pmin_sum / row["LoadMW"])
                net.gen["min_p_mw"] *= scale
                if has_eg_lims:
                    net.ext_grid["min_p_mw"] *= scale

        def attempt(net_i, relax: float) -> bool:
            net_i.line["max_loading_percent"] = 100.0 * relax
            net_i.trafo["max_loading_percent"] = 100.0 * relax
            try:
                pp.rundcopp(net_i)
                return True
            except fails:
                return False

        opm = 0.0 if dc_overload_percent_max is None \
            else float(dc_overload_percent_max[i])
        relax_first = max(1.0, 1.0 + abs(opm) / 100.0)
        relax = 1.0
        ok = attempt(net, relax)
        if not ok:
            for relax in (relax_first, 3.0 * relax_first):
                ok = attempt(net, relax)
                if ok:
                    break
        if not ok:
            rows.append(row)
            continue

        def record(net_i, relax: float) -> None:
            flows = ppio.branch_flows_mva(net_i, m, dc=True)
            va = np.deg2rad(net_i.res_bus["va_degree"].to_numpy(float))
            ad = angle_differences_deg(A, va)
            util = np.abs(flows) / cap
            losses = float(s_base * np.sum((np.abs(flows) / s_base) ** 2
                                           * r_pu))
            row.update({
                "success": True,
                "LimitRelaxPercent": max(0.0, 100.0 * (relax - 1.0)),
                "GenMW": float((net_i.res_gen["p_mw"].sum() if len(net_i.gen)
                                else 0.0) + net_i.res_ext_grid["p_mw"].sum()),
                "MaxAngleDiff": float(np.nanmax(ad)),
                "MeanAngleDiff": float(np.nanmean(ad)),
                "Losses": losses,
                "LossPercent": 100.0 * losses / (row["LoadMW"] + losses),
                "PercentLoadedMean": 100.0 * float(np.nanmean(util[util < 1])),
                "PercentLoadedStd": 100.0 * float(np.nanstd(util[util < 1],
                                                            ddof=1)),
            })

        record(net, relax)
        while relax > 1.0:                      # tightening ladder
            pct = 100.0 * (relax - 1.0)
            relax *= 0.90 if pct > 100 else 0.95 if pct > 20 \
                else 0.98 if pct > 10 else 0.99
            if not attempt(net, relax):
                break
            record(net, relax)
            if relax <= 1.0:                    # at/below base: floor at 0
                break
        rows.append(row)

    return pd.DataFrame(rows, columns=list(_STAT_COLUMNS))
