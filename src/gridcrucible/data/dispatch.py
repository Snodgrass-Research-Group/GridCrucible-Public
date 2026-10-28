"""UCCD: complementary dispatch (gurobipy QP).

MATLAB origin: UCCD.m (94 loc, cplexqp). Algorithm doc:
docs/algorithms/UCCD.md — the decommitment/reserve/cleanup loops are ported
step-for-step, including the lb→0 infeasibility fallback used throughout. Companion UCED (economic dispatch) is gridforge-owned.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class DispatchError(RuntimeError):
    """QP infeasible even with the lb=0 fallback."""


def _solve_qp(c2: np.ndarray, c1: np.ndarray, lb: np.ndarray, ub: np.ndarray,
              total_load: float, env=None) -> np.ndarray | None:
    """min ½xᵀdiag(c2)x + c1ᵀx  s.t. Σx = total_load, lb ≤ x ≤ ub.
    Returns None when infeasible (MATLAB's empty fval)."""
    import gurobipy as gp

    m = gp.Model("uccd", env=env)
    m.Params.OutputFlag = 0
    x = m.addMVar(len(c1), lb=lb, ub=ub)
    m.addConstr(x.sum() == total_load)
    m.setObjective(0.5 * (x @ np.diag(c2) @ x) + c1 @ x, gp.GRB.MINIMIZE)
    m.optimize()
    if m.Status != gp.GRB.OPTIMAL:
        return None
    return x.X


def _solve_with_fallback(fleet: pd.DataFrame, total_load: float,
                         env=None) -> np.ndarray:
    """The QP with MATLAB's ubiquitous lb→0 retry."""
    c2, c1 = fleet["C"].to_numpy(float), fleet["B"].to_numpy(float)
    lb, ub = fleet["MinMW"].to_numpy(float), fleet["MaxMW"].to_numpy(float)
    if lb.sum() <= total_load:
        x = _solve_qp(c2, c1, lb, ub, total_load, env)
        if x is not None:
            return x
    x = _solve_qp(c2, c1, np.zeros(len(lb)), ub, total_load, env)
    if x is None:
        raise DispatchError(
            f"UCCD QP infeasible even with lb=0 (load {total_load:.1f} MW, "
            f"capacity {ub.sum():.1f} MW)")
    return x


def uccd(fleet: pd.DataFrame, hydro_scaling: float, load_mw: np.ndarray,
         reserve: float, gurobi_env=None) -> tuple[np.ndarray, pd.DataFrame]:
    """UCCD.m port. fleet: generator table with MinMW/MaxMW/B/C/A/FuelType
    (original index PRESERVED on the returned surviving fleet — SNet maps
    dispatch back to buses through it). load_mw: only the sum matters.

    Returns (x, surviving_fleet): per-unit MW dispatch aligned with
    surviving_fleet rows."""
    gens = fleet.copy()
    hydro = gens["FuelType"].astype(str).str.lower() == "hydro"
    gens.loc[hydro, "MaxMW"] *= hydro_scaling
    gens.loc[hydro, "MinMW"] *= hydro_scaling
    total_load = float(np.asarray(load_mw, float).sum())

    x = _solve_with_fallback(gens, total_load, gurobi_env)

    # rough decommitment of the near-max (cheap) units, ≤ 5 rounds ----------
    gens_out = gens
    for _ in range(5):
        turn_off = (gens["MaxMW"].to_numpy(float) - x) \
            < 0.05 * gens["MaxMW"].to_numpy(float)
        gens_out = gens                       # pre-deletion snapshot
        gens = gens[~turn_off]
        if gens["MaxMW"].sum() < total_load:  # over-decommitted: revert
            break
        gens_out = gens
        try:
            x = _solve_with_fallback(gens, total_load, gurobi_env)
        except DispatchError:
            break
        lb = gens["MinMW"].to_numpy(float)
        if x.min() > 1 or (x > lb + 1).all():
            break

    gens = gens_out
    x = _solve_with_fallback(gens, total_load, gurobi_env)

    # reserve trimming: drop the 3 cheapest-per-MW units while spinning
    # reserve is excessive --------------------------------------------------
    while gens["MaxMW"].sum() * (1 - reserve) > total_load and len(gens) > 5:
        with np.errstate(divide="ignore", invalid="ignore"):
            cost_per_mw = (gens["A"].to_numpy(float)
                           + gens["B"].to_numpy(float) * x
                           + gens["C"].to_numpy(float) * x ** 2) / x
        cost_per_mw = np.where(np.isfinite(cost_per_mw), cost_per_mw, np.inf)
        cheapest = np.argpartition(cost_per_mw, 3)[:3]
        if gens["MaxMW"].sum() - gens["MaxMW"].to_numpy(float)[cheapest].sum() \
                <= total_load:
            break
        gens = gens.drop(index=gens.index[cheapest])
        x = _solve_with_fallback(gens, total_load, gurobi_env)

    # cleanup: drop ≤0.1 MW units, rescale to rebalance ----------------------
    low = x <= 0.1
    if low.any():
        gens = gens[~low]
        x = x[~low]
        mismatch = total_load - x.sum()
        x = x * (1 + mismatch / x.sum())
    return x, gens
