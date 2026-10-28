"""DC / AC power-flow adapters (pandapower-backed, with a direct sparse fast
path for the DC solves inside OHM).

MATLAB origin: DCPowerFlowSimple.m, DCPowerFlowFull.m, ACPowerFlowSimple.m,
YBusCalcDC.m + the gridforge-owned ACPowerFlowFinal/YBusCalc call sites.
Algorithm doc: docs/algorithms/power_flow_adapters.md (formulas verbatim).

Conventions: injections in MW (converted to per-unit internally, matching
DCInnerLoop's SNet/SBase ÷ 100·flows round trip); slack for DC solves is
**bus 0 unconditionally** (MATLAB DCPowerFlowSimple), angles in radians,
angle *differences* helper in degrees (MATLAB compatibility).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse import csgraph
from scipy.sparse.linalg import splu

from ..model import ACPFResult, CrucibleModel
from . import pandapower_io as ppio


class SingularNetworkError(RuntimeError):
    """Replaces MATLAB's SingularB/SingularY flags: raised when the network
    is disconnected or a bus is isolated; TRIM/TUNE catch it."""


@dataclass
class DCPFResult:
    """A DC power-flow solution: bus voltage angles and branch flows."""
    theta_rad: np.ndarray     # per bus, slack (bus 0) = 0
    flows_mw: np.ndarray      # signed sending-end MW per branch row


def b_matrix(A: sp.spmatrix, x_pu: np.ndarray,
             contingency: int | None = None) -> sp.csc_matrix:
    """YBusCalcDC port: B = A·diag(1/x)·Aᵀ. contingency (0-based branch
    index) zeroes that branch's primitive susceptance (line open)."""
    b = 1.0 / np.asarray(x_pu, float)
    if contingency is not None:
        b = b.copy()
        b[contingency] = 0.0
    return (A @ sp.diags(b) @ A.T).tocsc()


def check_connected(A: sp.spmatrix) -> None:
    """Exact port of the SingularB guards (zero-degree bus / condest /
    sprank): a connected graph with no isolated bus is exactly the
    non-singular condition for the reduced B."""
    deg = np.abs(A).sum(axis=1).A1 if hasattr(np.abs(A).sum(axis=1), "A1") \
        else np.asarray(np.abs(A).sum(axis=1)).ravel()
    if (deg == 0).any():
        raise SingularNetworkError(
            f"{int((deg == 0).sum())} isolated bus(es)")
    adj = (A @ A.T).tocsr()
    adj.setdiag(0)
    adj.eliminate_zeros()
    n_comp = csgraph.connected_components(abs(adj), directed=False,
                                          return_labels=False)
    if n_comp > 1:
        raise SingularNetworkError(f"network has {n_comp} components")


def angle_differences_deg(A: sp.spmatrix, theta_rad: np.ndarray) -> np.ndarray:
    """MATLAB AngleDifference = |Aᵀ·Va_degrees| (per branch, DEGREES)."""
    return np.abs(A.T @ np.degrees(theta_rad))


def _table_x(df: pd.DataFrame) -> np.ndarray:
    return (df["x_pu"] if "x_pu" in df.columns else df["reactance_x"]).to_numpy(float)


class DCSeedFactor:
    """Pre-factorized DC PF: the reduced B-matrix LU depends only on the
    topology and reactances, NOT the injections — factor ONCE and solve
    many RHS. On the real-SRTV candidate graph (2.49M branches → an
    effectively dense 4,625² B) the factorization is ~45 s while each
    solve is milliseconds; OHM's per-scenario seeding was refactorizing it
    23× per TRIM iteration (the measured ~55 s/solve overhead)."""

    def __init__(self, A: sp.spmatrix, x_pu: np.ndarray) -> None:
        check_connected(A)
        self.A = A
        self.b = 1.0 / np.asarray(x_pu, float)
        B = b_matrix(A, x_pu)
        self.n = B.shape[0]
        self._lu = splu(B[1:, 1:].tocsc())

    def solve(self, p_pu: np.ndarray) -> DCPFResult:
        """Solve for angles and branch flows given a per-bus injection vector
        ``p_pu`` (p.u., slack at bus 0) using the pre-factorized reduced B."""
        theta = np.zeros(self.n)
        theta[1:] = self._lu.solve(np.asarray(p_pu, float)[1:])
        return DCPFResult(theta_rad=theta,
                          flows_mw=self.b * (self.A.T @ theta))


def dc_pf_arrays(A: sp.spmatrix, x_pu: np.ndarray, p_pu: np.ndarray,
                 contingency: int | None = None) -> DCPFResult:
    """θ = B_red⁻¹·P_red (slack = bus 0), flows = (Aᵀθ)/x in PER-UNIT — the
    exact DCPowerFlowSimple computation via sparse LU, on raw arrays (used by
    OHM's Newton seeding where no CrucibleModel round trip is wanted).
    For repeated solves on ONE topology use DCSeedFactor."""
    if contingency is None:
        return DCSeedFactor(A, x_pu).solve(p_pu)
    check_connected(A)
    B = b_matrix(A, x_pu, contingency)
    n = B.shape[0]
    theta = np.zeros(n)
    theta[1:] = splu(B[1:, 1:].tocsc()).solve(np.asarray(p_pu, float)[1:])
    b = 1.0 / np.asarray(x_pu, float)
    b = b.copy()
    b[contingency] = 0.0
    flows_pu = b * (A.T @ theta)
    return DCPFResult(theta_rad=theta, flows_mw=flows_pu)  # caller rescales


def robust_runpp(net, *, enforce_q_lims: bool = False,
                 nr_tolerance_mva: float = 0.05,
                 allow_gs: bool = True,
                 gs_tolerance_mva: float = 1.0,
                 gs_max_iteration: int = 5000) -> str:
    """AC solve ladder for the full-scale synthetic networks; returns the
    method that converged ('nr' | 'gs') or raises LoadflowNotConverged.

    Rationale (real-SRTV shunt stage, 2026-07-11): the synthetic network's
    impedances span ~6 decades (X 4e-5..18 pu) so the NR Jacobian is
    numerically near-singular (σ_min ≈ 3e-13 measured AT a converged
    point) — NR stalls above ~1 MVA mismatch even from a converged start,
    and pandapower's enforce_q_lims PV↔PQ switching collapses weak buses
    to V=0 (exactly-singular Jacobian). Gauss-Seidel (Jacobian-free)
    converges to a sane profile at 1 MVA ≈ 2e-5 relative on the 55-GW
    system. MATLAB's ACPowerFlowFinal recorded PF.Converge in a metrics
    table and PROCEEDED regardless (pfsolve.m is lost; no evidence of a
    tighter criterion) — accepting the GS solution is at least as strict
    as as-run MATLAB. Q limits are the caller's job (explicit
    PVPQSwitching clamping — pandapower enforce_q_lims only attempted on
    the NR rung)."""
    import pandapower as pp

    try:
        pp.runpp(net, enforce_q_lims=enforce_q_lims, init="dc",
                 max_iteration=50, tolerance_mva=nr_tolerance_mva)
        return "nr"
    except Exception:
        if not allow_gs:
            # envelope-style callers: a GS attempt costs ~15 min per
            # scenario and (measured overnight 2026-07-12: 8 passes,
            # converged count pinned at 42-43/94) rescues nothing —
            # fail fast, the scenario contributes zero to the envelope
            raise
    pp.runpp(net, algorithm="gs", init="dc",
             max_iteration=gs_max_iteration, tolerance_mva=gs_tolerance_mva)
    return "gs"


def run_dc_pf(model: CrucibleModel, snet_mw: np.ndarray,
              table: str = "candidates", engine: str = "pandapower",
              contingency: int | None = None) -> DCPFResult:
    """DC power flow (DCPowerFlowSimple port). snet_mw: per-bus real net
    injection, MW. Slack = bus 0 (MATLAB convention; the slack absorbs the
    imbalance). Engines agree to ~1e-8 MW (pinned by tests):

    * 'direct'      — sparse LU on b_matrix (used inside OHM seeding).
    * 'pandapower'  — to_pandapower + rundcpp (the §2.6 default).
    """
    df = getattr(model, table)
    s_base = float(model.meta.get("s_base", 100.0))
    p = np.asarray(snet_mw, float)
    if engine == "direct":
        A = model.incidence(table)
        res = dc_pf_arrays(A, _table_x(df), p / s_base, contingency)
        return DCPFResult(theta_rad=res.theta_rad, flows_mw=s_base * res.flows_mw)
    if engine != "pandapower":
        raise ValueError(f"unknown engine {engine!r}")
    import pandapower as pp
    net = ppio.to_pandapower(model, table=table, injections_mw=p, slack_bus=0)
    if contingency is not None:
        for elm in ("line", "trafo"):
            tbl = getattr(net, elm)
            hit = tbl.index[tbl["crucible_row"] == contingency]
            tbl.loc[hit, "in_service"] = False
    try:
        pp.rundcpp(net)
    except Exception as e:  # pandapower raises LoadflowNotConverged and friends
        raise SingularNetworkError(str(e)) from e
    theta = np.deg2rad(net.res_bus["va_degree"].to_numpy(float))
    theta = theta - theta[0]  # reference at bus 0 like MATLAB
    flows = _signed_dc_flows(net, len(df))
    return DCPFResult(theta_rad=theta, flows_mw=flows)


def _signed_dc_flows(net, n_branches: int) -> np.ndarray:
    """Signed sending-end MW per model row (from-end for lines, hv for
    trafos), matching PLineSend's sign convention."""
    flows = np.full(n_branches, np.nan)
    if len(net.line):
        flows[net.line["crucible_row"].to_numpy(int)] = \
            net.res_line["p_from_mw"].to_numpy(float)
    if len(net.trafo):
        flows[net.trafo["crucible_row"].to_numpy(int)] = \
            net.res_trafo["p_hv_mw"].to_numpy(float)
    return flows


def run_dc_pf_full(model: CrucibleModel, snet_mw: np.ndarray,
                   table: str = "branches", network_label: str = "KCL-KVL",
                   loads_label: str = "") -> tuple[np.ndarray, pd.DataFrame]:
    """DCPowerFlowFull port: DC PF per scenario column + the DCStats frame
    (formulas verbatim — they feed ElectricalComparison's Excel output).

    snet_mw: (n_buses × n_scenarios) real injections in MW.
    Returns (flows_mw matrix, DCStats DataFrame)."""
    df = getattr(model, table)
    s_base = float(model.meta.get("s_base", 100.0))
    cap = df["capacity_mva"].to_numpy(float)
    r_pu = (df["r_pu"] if "r_pu" in df.columns
            else pd.Series(0.0, index=df.index)).to_numpy(float)
    A = model.incidence(table)
    snet_mw = np.atleast_2d(np.asarray(snet_mw, float))
    if snet_mw.shape[0] != model.n_buses:
        snet_mw = snet_mw.T
    n_scen = snet_mw.shape[1]
    flows = np.zeros((len(df), n_scen))
    rows = []
    for i in range(n_scen):
        p = snet_mw[:, i]
        res = run_dc_pf(model, p, table=table, engine="direct")
        flows[:, i] = res.flows_mw
        adiff = angle_differences_deg(A, res.theta_rad)
        util = np.abs(res.flows_mw) / cap
        p_pu = p / s_base
        losses = s_base * float(((flows[:, i] / s_base) ** 2 * r_pu).sum())
        over = util[util > 1]
        under = util[util < 1]
        rows.append({
            "Network": network_label, "Loads": loads_label,
            "GenMW": float(p_pu[p_pu > 0].sum()) * s_base,
            "LoadMW": float(np.abs(p_pu[p_pu < 0].sum())) * s_base,
            "MaxAngleDiff": float(adiff.max()),
            "MeanAngleDiff": float(adiff.mean()),
            "Losses": losses,
            "LossPercent": 100.0 * losses / (np.abs(p_pu).sum() * s_base / 2 + losses),
            "PercentLoadedMean": 100.0 * float(under.mean()) if under.size else 0.0,
            "PercentLoadedStd": 100.0 * float(under.std(ddof=1)) if under.size > 1 else 0.0,
            "LinesOverloaded": int(over.size),
            "PercentLinesOverloaded": 100.0 * over.size / util.size,
            "OverloadPercentMax": float(util.max() * 100 - 100) if over.size else 0.0,
            "OverloadPercentMean": float(over.mean() * 100 - 100) if over.size else 0.0,
            "OverloadPercentStd": float((over * 100).std(ddof=1)) if over.size > 1 else 0.0,
        })
    return flows, pd.DataFrame(rows)


def run_ac_pf(model: CrucibleModel, scenario: int | None = None,
              enforce_q_limits: bool = True,
              injections_mva: np.ndarray | None = None,
              table: str | None = None,
              allow_gs: bool = True) -> ACPFResult:
    """AC Newton-Raphson via pandapower runpp (replaces ACPowerFlowSimple /
    ACPowerFlowFinal call sites; MATLAB's pfsolve.m was never committed, so
    this is the only implementation). enforce_q_limits reproduces
    PVPQSwitching. injections_mva: optional complex per-bus injection vector
    (MVA) overriding physical loads/gens — the MiddleLoopAC scenario mode.
    Raises SingularNetworkError on non-convergence."""
    import pandapower as pp
    if table is None:
        table = "branches" if len(model.branches) else "candidates"
    df = getattr(model, table)
    if "r_pu" not in df.columns:
        raise ValueError("AC power flow needs r_pu/b_pu — run impedance "
                         "assignment (RXB) before run_ac_pf; X-only "
                         "data is DC territory")
    net = ppio.to_pandapower(model, table=table, scenario=scenario,
                             injections_mw=injections_mva, pv_dispatch=True)

    def _plausible() -> bool:
        # NR can converge to the spurious low-voltage branch (measured on
        # the real case: vmin 0.001 / vmax 1.815 "converged") — reject
        # solutions no real network could hold so callers treat them as
        # non-convergence (MATLAB's LowVoltageThreshold idea)
        vm = net.res_bus["vm_pu"].to_numpy(float)
        return bool(np.nanmin(vm) > 0.5 and np.nanmax(vm) < 1.6)

    try:
        pp.runpp(net, enforce_q_lims=enforce_q_limits, init="flat")
        ok = _plausible()
    except Exception:
        ok = False
    if not ok:
        # full-scale conditioning fallback (see robust_runpp docstring);
        # q-lims only on the NR rung — GS accepts the unclamped solution
        try:
            robust_runpp(net, enforce_q_lims=enforce_q_limits,
                         allow_gs=allow_gs)
        except Exception as e:
            raise SingularNetworkError(f"AC PF did not converge: {e}") from e
        if not _plausible():
            raise SingularNetworkError(
                "AC PF converged to an implausible voltage profile "
                f"(vmin={float(net.res_bus['vm_pu'].min()):.3f}, "
                f"vmax={float(net.res_bus['vm_pu'].max()):.3f})")
    return ppio.from_pandapower(net, len(df))
