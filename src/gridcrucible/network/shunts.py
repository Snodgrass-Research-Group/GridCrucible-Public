"""Reactive-power (shunt) placement via synchronous-condenser winnowing.

MATLAB origin: ReactivePowerPlacement.m (235 loc) + PVPQSwitching.m (47 —
ported EXPLICITLY inside _solve: pandapower's enforce_q_lims collapses
weak buses to V=0 at full scale, see power_flow.robust_runpp).
Algorithm doc: docs/algorithms/ReactivePowerPlacement.md. Pipeline stage 7.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import CrucibleConfig
from ..model import ACPFResult, CrucibleModel

HIGH_Q_MVAR = 800.0     # shunt reactor bank size (withdrawal)  [ACPF.HighQ]
LOW_Q_MVAR = -1200.0    # shunt capacitor bank size (injection) [ACPF.LowQ]
V_PLACE_LO, V_PLACE_HI = 0.94, 1.06   # permanent-placement voltage band


def correct_power_factors(snet: np.ndarray) -> np.ndarray:
    """MATLAB PF-correction: retarget load power factors by band
    (≤0.9→0.97, ≤0.95→0.98, ≤0.96→0.99) and rebuild Q from P."""
    snet = np.asarray(snet, complex)
    with np.errstate(invalid="ignore", divide="ignore"):
        pf = np.abs(np.real(snet) / np.abs(snet))
    pf = np.nan_to_num(pf, nan=1.0)
    target = pf.copy()
    target[pf <= 0.96] = 0.99
    target[pf <= 0.95] = 0.98
    target[pf <= 0.9] = 0.97
    q = np.abs(np.real(snet)) * np.sqrt(1.0 / target ** 2 - 1.0)
    return np.real(snet) + 1j * np.sign(np.imag(snet) + 1e-30) * q


def place_reactive_compensation(model: CrucibleModel, config: CrucibleConfig,
                                per_switch: float = 0.05) -> ACPFResult:
    """ReactivePowerPlacement port (winnowing loop of the algorithm doc).

    Consumes the bus-level Complete tier (max ED scenario); mutates
    model.buses (sync_cond, v_set, bus_type) and returns the final converged
    ACPFResult. Line charging b_pu is capped at 1 (series-compensation
    stand-in, as in MATLAB).
    """
    import pandapower as pp

    from ..power import pandapower_io as ppio

    buses = model.buses
    cand = model.candidates
    if "b_pu" in cand.columns:
        cand.loc[cand["b_pu"] > 1.0, "b_pu"] = 1.0

    tier = model.scenarios.tiers.get("complete")
    if tier is None:
        raise ValueError("shunt placement needs the bus-level Complete tier "
                         "(rebuild scenarios after create_buses)")
    ed_width = int(model.meta.get("ed_block_width", tier.shape[1]))
    snet = correct_power_factors(tier[:, :ed_width])
    max_scen = int(np.abs(snet.real).sum(axis=0).argmax())
    inj = snet[:, max_scen]

    n = len(buses)
    slack = int(model.scenarios.slack_bus)
    gen_buses = np.asarray(model.scenarios.gen_buses, int)
    real_gen = np.concatenate([[slack], gen_buses])
    candidates = np.setdiff1d(np.arange(n), real_gen)
    sync_cond = np.zeros(n, bool)

    def _solve(cand_set: np.ndarray) -> tuple[ACPFResult | None, np.ndarray]:
        """AC PF with condenser gens at cand_set; returns (result, |Q| per
        candidate) or (None, _) on non-convergence.

        Solver: power_flow.robust_runpp (NR → Gauss-Seidel ladder — see its
        docstring for the full-scale conditioning rationale). Q limits are
        applied via EXPLICIT PVPQSwitching passes (the MATLAB algorithm:
        clamp violators to the bank limit as PQ, two fixed re-solves) —
        pandapower's enforce_q_lims collapses weak buses to V=0 on this
        network, so the 'natively subsumed' shortcut only holds on the NR
        rung of small cases."""
        from ..power.power_flow import robust_runpp

        net = ppio.to_pandapower(model, table="candidates",
                                 injections_mw=inj, slack_bus=slack,
                                 sync_cond_gens=False)  # winnowing owns them
        gi = pp.create_gens(net, buses=cand_set, p_mw=0.0, vm_pu=1.0,
                            min_q_mvar=-HIGH_Q_MVAR, max_q_mvar=-LOW_Q_MVAR)
        gi = np.asarray(gi)
        q_clamped = np.zeros(len(cand_set))    # limit Q of demoted-to-PQ rows
        try:
            robust_runpp(net)
        except Exception:
            return None, np.zeros(len(cand_set))
        for _ in range(2):                     # PVPQSwitching.m: two passes
            active = net.gen.loc[gi, "in_service"].to_numpy(bool)
            q = np.where(active,
                         net.res_gen.loc[gi, "q_mvar"].to_numpy(float),
                         q_clamped)
            hi = active & (q > -LOW_Q_MVAR)    # sign flip: MATLAB QNet
            lo = active & (q < -HIGH_Q_MVAR)
            if not (hi.any() or lo.any()):
                break
            for mask, limit in ((hi, -LOW_Q_MVAR), (lo, -HIGH_Q_MVAR)):
                if mask.any():
                    rows = gi[mask]
                    net.gen.loc[rows, "in_service"] = False
                    q_clamped[mask] = limit
                    pp.create_sgens(net,
                                    buses=net.gen.loc[rows, "bus"].to_numpy(int),
                                    p_mw=0.0, q_mvar=limit)
            try:
                robust_runpp(net)
            except Exception:
                return None, np.zeros(len(cand_set))
        active = net.gen.loc[gi, "in_service"].to_numpy(bool)
        q = np.where(active, net.res_gen.loc[gi, "q_mvar"].to_numpy(float),
                     q_clamped)
        return ppio.from_pandapower(net, len(cand)), np.abs(q)

    result, q_abs = _solve(candidates)
    if result is None:
        raise RuntimeError("initial all-PV power flow did not converge — "
                           "network needs impedance review before shunt "
                           "placement")
    while True:
        # permanent placement at voltage violations
        v = result.v_mag
        violate = (v <= V_PLACE_LO) | (v >= V_PLACE_HI)
        sync_cond |= violate & np.isin(np.arange(n), candidates)

        if len(candidates) + sync_cond.sum() < 0.2 * n or per_switch < 1e-2:
            sync_cond[candidates] = True         # keep the survivors
            break

        # demote the smallest-|Q| per_switch share (never the permanent ones)
        removable = ~sync_cond[candidates]
        k = min(int(np.ceil(per_switch * n)), int(removable.sum()))
        if k == 0:
            sync_cond[candidates] = True
            break
        order = np.argsort(np.where(removable, q_abs, np.inf))
        trial = np.delete(candidates, order[:k])
        new_result, new_q = _solve(trial)
        if new_result is None:                    # revert + shrink rate
            per_switch *= 0.9
            continue
        candidates, result, q_abs = trial, new_result, new_q

    buses["sync_cond"] = sync_cond
    buses["v_set"] = np.where(sync_cond | np.isin(np.arange(n), real_gen),
                              1.0, 1.0)
    buses.loc[np.flatnonzero(np.isin(np.arange(n), real_gen) | sync_cond),
              "bus_type"] = 2
    buses.loc[~(np.isin(np.arange(n), real_gen) | sync_cond), "bus_type"] = 1
    largest = int(buses["gen_max_mw"].idxmax())
    buses.loc[largest, "bus_type"] = 3            # slack reasserted
    model.acpf = result
    return result


def pv_pq_switching(*args, **kwargs):
    """PVPQSwitching.m is natively subsumed by pandapower's
    enforce_q_lims=True (Q clamped at the limit → bus goes PQ; released when
    voltage recovers) — see the algorithm doc. Kept as a named reference for
    parity documentation; not callable."""
    raise NotImplementedError(
        "use run_ac_pf(..., enforce_q_limits=True) — PVPQSwitching is "
        "performed inside the pandapower solve")
