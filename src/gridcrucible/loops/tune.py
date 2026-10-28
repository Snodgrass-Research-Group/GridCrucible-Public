"""TUNE Loop — Thermal Updating of Network Elements (the middle loop).

MATLAB origin: MiddleLoop.m (126, DC) + MiddleLoopAC.m (161, AC).
Algorithm doc: docs/algorithms/MiddleLoop.md; leaf math in
docs/algorithms/conductors.md. (loops) / 1d (leaves)

Place in the flow: runs after TRIM removal each orchestrator iteration
(run_dc), and again as pipeline stage 8 after bus creation + shunt placement
(run_ac). Impedance recompute after conductor changes is gridforge-owned
(RXB) — injected via ``impedance_fn`` (see the doc; None = ratings-only
mode with a one-time warning).
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from ..config import CrucibleConfig
from ..model import CrucibleModel
from ..network.conductors import (assign_conductors, resize_branches,
                                  transformer_impedances)
from ..power.power_flow import dc_pf_arrays

ImpedanceFn = Callable[[pd.DataFrame, pd.DataFrame], None]

#: run_ac scenario-pool worker state (one CrucibleModel per process; the
#: pool is rebuilt each envelope pass because the refine loop mutates it).
_AC_POOL_MODEL = None


def _ac_pool_init(model_bytes: bytes) -> None:
    global _AC_POOL_MODEL
    import pickle
    _AC_POOL_MODEL = pickle.loads(model_bytes)


def _ac_pool_solve(col: np.ndarray):
    """One scenario ACPF in a pool worker; returns flows_mva or None.
    allow_gs=False: envelope scenarios that NR can't solve contribute zero
    (MATLAB semantics) — the ~15-min GS attempt rescues none of them
    (measured 2026-07-12)."""
    from ..power.power_flow import SingularNetworkError, run_ac_pf
    try:
        res = run_ac_pf(_AC_POOL_MODEL, injections_mva=col,
                        table="candidates", allow_gs=False)
    except SingularNetworkError:
        return None
    return res.flows_mva if res.converged else None


@dataclass
class TuneReport:
    """Summary of one TUNE middle-loop pass over a voltage tier."""
    tier: str = ""                     # voltage tier processed
    n_upsized: int = 0                 # branches given a larger conductor
    n_downsized: int = 0               # branches given a smaller conductor
    refine_iters: int = 0              # inner refine iterations run
    n_overloaded_final: int = 0        # branches still overloaded at exit
    term_criteria_final: int = 0       # termination code the loop exited on


class TUNELoop:
    """Middle-loop engine. Stateless between iterations except the conductor
    catalog and the injected impedance recompute."""

    def __init__(self, config: CrucibleConfig, conductor_catalog: pd.DataFrame,
                 xfdata: pd.DataFrame | None = None,
                 xfxr: pd.DataFrame | None = None,
                 impedance_fn: ImpedanceFn | None = None,
                 rng: np.random.Generator | None = None) -> None:
        self.config = config
        self.catalog = conductor_catalog
        self.xfdata = xfdata
        self.xfxr = xfxr
        self.impedance_fn = impedance_fn
        self.rng = rng or np.random.default_rng(config.case.rng_seed)
        self._warned_no_impedance = False

    # ------------------------------------------------------------ helpers

    def _resize(self, cand: pd.DataFrame, flows_mw: np.ndarray, *,
                upsize=True, downsize=True, downsize_keep=True,
                percent_resize: float | None = None) -> dict:
        cfg = self.config.tune
        return resize_branches(
            cand, flows_mw,
            percent_resize=percent_resize or cfg.percent_resize,
            percent_change=cfg.percent_change,
            conductor_margin=cfg.conductor_margin,
            downsize_threshold=cfg.downsize_threshold,
            upsize=upsize, downsize=downsize, downsize_keep=downsize_keep,
            random_resize=cfg.random_resize, rng=self.rng)

    def _reassign_and_recompute(self, model: CrucibleModel) -> None:
        """assign_conductors on lines, impedances via gridforge RXB (or warn),
        transformer_impedances on tap≠0 rows."""
        cand = model.candidates
        tap = (cand["tap"].to_numpy(float) if "tap" in cand.columns
               else np.zeros(len(cand)))
        lines = cand.index[tap == 0]
        line_view = cand.loc[lines]
        assign_conductors(line_view, self.catalog,
                          self.config.tune.change_voltages)
        cand.loc[lines] = line_view
        if self.impedance_fn is not None:
            # LINES ONLY: transformer rows carry MVA sizing values in
            # conductor_idx (CreateBuses convention, e.g. 2000), which are
            # NOT conductor-catalog indices — RXB on them is out-of-bounds.
            # MATLAB hid exactly this crash in CreateBuses step 8's
            # silently-degrading try/catch (algorithms/CreateBuses.md Open
            # questions); transformer_impedances below owns tap≠0 rows.
            line_imp = cand.loc[lines].copy()
            self.impedance_fn(line_imp, self.catalog)
            for col in ("r_pu", "reactance_x", "x_pu", "b_pu", "gmd_ft"):
                if col in line_imp.columns:
                    if col not in cand.columns:
                        cand[col] = 0.0
                    cand.loc[lines, col] = line_imp[col]
        elif not self._warned_no_impedance:
            warnings.warn("TUNELoop has no impedance_fn (gridforge RXB not "
                          "wired) — reactances NOT recomputed after conductor "
                          "changes; ratings-only mode", stacklevel=3)
            self._warned_no_impedance = True
        if (tap != 0).any() and self.xfdata is not None:
            transformer_impedances(cand, self.xfdata, self.xfxr,
                                   substations=None)

    def _max_dc_flows(self, model: CrucibleModel, snet_mw: np.ndarray,
                      ohm_scaled: bool = True) -> np.ndarray:
        """Per-path max |DC flow| over scenario columns (the B-matrix is
        factored ONCE and reused across columns — power_flow.DCSeedFactor).

        ohm_scaled=True (TRIM-era MiddleLoop): the injections arrive
        PRE-MULTIPLIED by meta['snet_scaling'] for the OHM QP — back it out
        (snet/(snet_scaling·s_base) per-unit). ohm_scaled=False
        (CreateBuses step 8): the raw bus-level tier is passed — divide by
        s_base only. Feeding raw injections through the scaled path
        under-rates every DC-PF-sized transformer by exactly snet_scaling
        (real-SRTV crash 2026-07-11: 4.57x -> 19 MVA GSU-side XFs carrying
        128 MW -> ACPF infeasible). SingularNetworkError propagates
        (MATLAB errored)."""
        from ..power.power_flow import DCSeedFactor

        A = model.incidence("candidates")
        x = model.candidates["reactance_x"].to_numpy(float)
        s_base = float(model.meta.get("s_base", 100.0))
        scaling = float(model.meta.get("snet_scaling", 1.0)) if ohm_scaled \
            else 1.0
        seed = DCSeedFactor(A, x)
        out = np.zeros(len(model.candidates))
        for i in range(snet_mw.shape[1]):
            res = seed.solve(np.real(snet_mw[:, i]) / (scaling * s_base))
            out = np.maximum(out, np.abs(res.flows_mw) * s_base)
        return out

    # ------------------------------------------------------------ run_dc

    def run_dc(self, model: CrucibleModel, snet_mw: np.ndarray) -> TuneReport:
        """MiddleLoop.m port. Uses candidates.max_flow (OHM envelope) and
        candidates.max_dcpf set by TRIM scoring; density tier picks the
        strategy (doc §run_dc)."""
        cfg = self.config.trim   # BBR thresholds live in [trim]
        cand = model.candidates
        density = model.density
        snet_mw = np.atleast_2d(np.asarray(snet_mw))
        if snet_mw.shape[0] != model.n_buses:
            snet_mw = snet_mw.T
        report = TuneReport()

        if density > cfg.bbr_high:
            report.tier = "flags-only"          # scaling guarantees headroom
            return report
        if density > cfg.bbr_mid:
            report.tier = "ohm-envelope"
            counts = self._resize(cand, cand["max_flow"].to_numpy(float))
        elif density > cfg.bbr_midlow:
            report.tier = "dcpf-envelope"
            counts = self._resize(cand, cand["max_dcpf"].to_numpy(float))
        else:
            report.tier = "refine"
            term = 5 if density < cfg.bbr_low else 20
            counts = self._resize(cand, cand["max_dcpf"].to_numpy(float))
            self._reassign_and_recompute(model)
            max_dcpf = self._max_dc_flows(model, snet_mw)
            cand["max_dcpf"] = max_dcpf
            margin = self.config.tune.conductor_margin
            it = 0
            while True:
                over = ((cand["capacity_mva"].to_numpy(float)
                         - max_dcpf * margin) < 0) \
                    & (cand["conductor_idx"].to_numpy(float)
                       < cand["max_conductor_idx"].to_numpy(float))
                headroom = (cand["max_conductor_idx"].to_numpy(float)
                            - cand["conductor_idx"].to_numpy(float)) > 1
                report.n_overloaded_final = int(over.sum())
                if int((over & headroom).sum()) <= term:
                    break
                c2 = self._resize(cand, max_dcpf)
                for k in ("upsized", "downsized", "downsized_keep"):
                    counts[k] += c2[k]
                self._reassign_and_recompute(model)
                max_dcpf = self._max_dc_flows(model, snet_mw)
                cand["max_dcpf"] = max_dcpf
                it += 1
                if it > 15:                      # MATLAB relaxation valve
                    term += 1
            report.refine_iters = it
            report.term_criteria_final = term

        if "resize" in cand.columns and cand["resize"].any():
            self._reassign_and_recompute(model)
        report.n_upsized = counts["upsized"]
        report.n_downsized = counts["downsized"] + counts["downsized_keep"]
        return report

    # ------------------------------------------------------------ run_ac

    def run_ac(self, model: CrucibleModel, snet_mva: np.ndarray) -> TuneReport:
        """MiddleLoopAC.m port (pipeline stage 8). snet_mva: complex
        (n_buses × n_scenarios) in MVA; column 0 is dropped (MATLAB treats it
        as a duplicate — doc open question #2). percent_resize forced to
        0.01. Non-converged scenarios contribute zero flow to the envelope."""
        import time

        from ..power.power_flow import SingularNetworkError, run_ac_pf

        cand = model.candidates
        if "r_pu" not in cand.columns:
            raise ValueError("run_ac needs r_pu/b_pu on candidates — run "
                             "impedance assignment first")
        snet = np.atleast_2d(np.asarray(snet_mva))[:, 1:]
        margin = self.config.tune.conductor_margin
        term, it = 5, 0
        report = TuneReport(tier="ac-refine")

        pcfg = self.config.parallel
        n_workers = pcfg.workers if pcfg.workers > 0 else (os.cpu_count() or 1)
        use_pool = pcfg.ac_parallel and n_workers > 1 and snet.shape[1] > 2

        def _max_ac_flows() -> np.ndarray:
            """Per-path max |AC flow| envelope over the scenario columns.
            Scenario ACPFs are independent single-threaded pandapower
            solves — with [parallel] ac_parallel they fan out over a
            process pool (fresh pool per pass: the refine loop mutates the
            candidates between passes). MATLAB used parfor here
            (MiddleLoopAC)."""
            t0 = time.perf_counter()
            out = np.zeros(len(model.candidates))
            n_ok = 0
            pooled = use_pool
            if pooled:
                import pickle
                from concurrent.futures import ProcessPoolExecutor
                from concurrent.futures.process import BrokenProcessPool

                cols = [snet[:, i] for i in range(snet.shape[1])]
                try:
                    with ProcessPoolExecutor(
                            max_workers=min(n_workers, len(cols)),
                            initializer=_ac_pool_init,
                            initargs=(pickle.dumps(model),)) as pool:
                        for flows in pool.map(_ac_pool_solve, cols):
                            if flows is not None:
                                out = np.maximum(out, np.nan_to_num(flows))
                                n_ok += 1
                except BrokenProcessPool:
                    # sporadic Windows spawn-worker death (observed
                    # 2026-07-12) — redo this pass serially rather than
                    # kill a multi-hour stage
                    warnings.warn("run_ac envelope pool broke — redoing "
                                  "this pass serially", stacklevel=2)
                    pooled = False
                    out = np.zeros(len(model.candidates))
                    n_ok = 0
            if not pooled:
                for i in range(snet.shape[1]):
                    try:
                        res = run_ac_pf(model, injections_mva=snet[:, i],
                                        table="candidates", allow_gs=False)
                    except SingularNetworkError:
                        continue                 # non-converged → zero flows
                    if res.converged:
                        out = np.maximum(out, np.nan_to_num(res.flows_mva))
                        n_ok += 1
            print(f"[run_ac] envelope pass {it}: {n_ok}/{snet.shape[1]} "
                  f"scenarios converged in {time.perf_counter() - t0:.0f}s "
                  f"({'pool' if use_pool else 'serial'})", flush=True)
            return out

        max_ac = _max_ac_flows()
        cand["max_acpf"] = max_ac
        while True:
            over = ((cand["capacity_mva"].to_numpy(float) - max_ac * margin) < 0) \
                & (cand["conductor_idx"].to_numpy(float)
                   < cand["max_conductor_idx"].to_numpy(float))
            headroom = (cand["max_conductor_idx"].to_numpy(float)
                        - cand["conductor_idx"].to_numpy(float)) > 1
            report.n_overloaded_final = int(over.sum())
            if int((over & headroom).sum()) <= term:
                break
            c = self._resize(cand, max_ac, percent_resize=0.01)
            report.n_upsized += c["upsized"]
            report.n_downsized += c["downsized"] + c["downsized_keep"]
            self._reassign_and_recompute(model)
            max_ac = _max_ac_flows()
            cand["max_acpf"] = max_ac
            it += 1
            if it > 20:
                term += 1
        report.refine_iters = it
        report.term_criteria_final = term
        if "resize" in cand.columns and cand["resize"].any():
            self._reassign_and_recompute(model)
        return report
