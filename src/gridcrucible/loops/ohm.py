"""OHM Loop — Optimization Honoring KVL Minimization (the inner loop).

MATLAB origin: DCInnerLoop.m (327 loc, converted here), ACInnerLoopOptim.m
(270 loc, follow-up — needs its algorithm doc first).
Algorithm doc: docs/algorithms/DCInnerLoop.md — read it; every rule below
(γ sweep, UpdateGamma feedback, stall handler, Newton fallback, the
TuneSNetScaling handshake and its clamp-to-1 quirks) is documented there.

Place in the flow: called by GridCrucible.run_trim_iterations once per
injection scenario per TRIM iteration. Solves

    min ½·pᵀdiag(X)p + γ·l1_costᵀ|p|   s.t.  A·p = snet_pu

whose optimality conditions enforce KVL on the nonzero-flow subgraph, with
the L1 term driving flow on unneeded candidate paths to ~0. Two solvers:

* **QP** (primary; MATLAB shipped this Gurobi formulation): split p = p⁺−p⁻,
  p± ≥ 0, minimize ½p±ᵀdiag([X;X])p± + γ[c;c]ᵀp± s.t. [A,−A]p± = snet_pu.
* **Newton** (fast path): the ε-smoothed unsplit KKT system, with the
  divergence guard (α halving, ε relaxation) and permanent fallback to QP
  when α ≤ 0.25 — port of the MATLAB solver-switch.

γ-range adaptation is STATEFUL (MATLAB mutated options.Lam /
options.InnerLoopTerminate): it lives on this class, never in CrucibleConfig.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

from ..config import CrucibleConfig
from ..model import CrucibleModel
from ..power.power_flow import DCSeedFactor, SingularNetworkError


@dataclass
class OHMResult:
    """Per-scenario output (MATLAB DCInnerLoop return tuple)."""

    flows_abs: np.ndarray        # |flow| per candidate path, MW
    dcpf_flows: np.ndarray       # plain DC-PF |flows|, MW (TRIM MaxDCPF)
    cutoff: float                # zero-flow threshold actually used (MW-scale)
    gamma_used: float
    update_gamma: int            # -1 shift γ down / 0 ok / +1 shift γ up
    solver_used: str             # 'newton' | 'qp'
    percent_low_flow: np.ndarray = field(default_factory=lambda: np.empty(0))


class OHMSolveError(RuntimeError):
    """QP infeasible/failed — MATLAB's 'ERROR! ... not solving correctly'."""


class _QPWorkspace:
    """One gurobipy model per topology (candidate set); re-solved across the
    γ sweep by updating the linear objective, and across scenarios by
    updating the constraint RHS — MATLAB rebuilt the model every call, this
    is the intended Python improvement (ARCHITECTURE.md §5)."""

    def __init__(self, X: np.ndarray, l1_cost: np.ndarray, A: sp.spmatrix,
                 env=None, verbose: bool = False) -> None:
        import gurobipy as gp

        m = len(X)
        self.m = m
        self.c = np.concatenate([l1_cost, l1_cost])
        self.model = gp.Model("ohm_dc", env=env)
        self.model.Params.OutputFlag = 1 if verbose else 0
        self.p = self.model.addMVar(2 * m, lb=0.0, name="p")
        aeq = sp.hstack([A, -A]).tocsr()
        self.constr = self.model.addMConstr(
            aeq, self.p, "=", np.zeros(A.shape[0]))
        # quadratic part is γ-independent: set ONCE; solve() only updates the
        # linear coefficients (Obj attr) — ~4x cheaper per solve at 5M vars
        # than rebuilding the MQuadExpr (measured 2.8s -> 0.7s at real-SRTV
        # scale)
        self.model.setMObjective(
            0.5 * sp.diags(np.concatenate([X, X])).tocsc(), None, 0.0,
            sense=gp.GRB.MINIMIZE)

    def solve(self, snet_pu: np.ndarray, gamma: float) -> np.ndarray:
        """Signed per-path flows (pu). Raises OHMSolveError on failure."""
        import gurobipy as gp

        self.constr.setAttr("RHS", np.asarray(snet_pu, float))
        self.p.Obj = gamma * self.c
        self.model.optimize()
        status = self.model.Status
        if status in (gp.GRB.INFEASIBLE, gp.GRB.INF_OR_UNBD, gp.GRB.UNBOUNDED):
            raise OHMSolveError(f"QP status {status} (infeasible/unbounded)")
        if status not in (gp.GRB.OPTIMAL, gp.GRB.SUBOPTIMAL):
            raise OHMSolveError(f"QP status {status}")
        x = self.p.X
        return x[:self.m] - x[self.m:]


class OHMLoop:
    """Inner-loop engine. One instance per GridCrucible run.

    Mutable state (reported upward, checkpointed in meta):
    * ``gammas`` — current γ grid (shifted by shift_gammas on UpdateGamma
      feedback; reset by the γ≥100 stall handler).
    * ``terminate_frac`` — zero-flow fraction ending the sweep (lowered by
      the stall handler, floor 0.05; TRIM lowers it to 0.2 near convergence).
    """

    def __init__(self, config: CrucibleConfig, gurobi_env=None) -> None:
        lo, hi, n = config.ohm.gamma_grid
        self.gammas: np.ndarray = np.logspace(lo, hi, n)
        self.terminate_frac: float = config.ohm.inner_loop_terminate
        self.config = config
        self.s_base = config.case.s_base
        self._env = gurobi_env
        self._qp: _QPWorkspace | None = None
        self._qp_key: tuple[int, int] | None = None  # (n_paths, X checksum)

    # -- public API (called by GridCrucible) --------------------------------

    def solve_all(self, model: CrucibleModel, snet_mw: np.ndarray,
                  max_capacity_mva: np.ndarray,
                  max_feedback_rounds: int = 10) -> list[OHMResult]:
        """Solve every scenario column of ``snet_mw`` (MW), then apply the
        γ feedback loop (MultiLevelProgram 264-285): scenarios reporting
        update_gamma≠0 are re-solved after shift_gammas until all return 0
        (bounded by max_feedback_rounds — MATLAB's loop was unbounded).

        max_capacity_mva: per-path largest-conductor rating (the MaxRateA =
        ConductorData.MVA[MaxCondIndex] lookup is the ORCHESTRATOR's job)."""
        A = model.incidence("candidates")
        cand = model.candidates
        X = cand["reactance_x"].to_numpy(float)
        c = cand["l1_cost"].to_numpy(float)
        snet_mw = np.atleast_2d(np.asarray(snet_mw, float))
        if snet_mw.shape[0] != model.n_buses:
            snet_mw = snet_mw.T
        cap_pu = np.asarray(max_capacity_mva, float) / self.s_base

        # ONE LU factorization per candidate topology, reused by every
        # scenario's DC seed (measured ~45 s each on the near-complete
        # 2.49M-candidate graph — was 23x per TRIM iteration)
        seed = DCSeedFactor(A, X)

        results: list[OHMResult] = [
            self._sweep_dc(X, c, A, snet_mw[:, i] / self.s_base, cap_pu, seed)
            for i in range(snet_mw.shape[1])]

        for _ in range(max_feedback_rounds):
            pending = [i for i, r in enumerate(results) if r.update_gamma != 0]
            if not pending:
                break
            dirs = {results[i].update_gamma for i in pending}
            self.shift_gammas(0 if len(dirs) > 1 else dirs.pop())
            for i in pending:
                results[i] = self._sweep_dc(X, c, A, snet_mw[:, i] / self.s_base,
                                            cap_pu, seed)
        else:
            import warnings
            warnings.warn("OHM γ feedback did not settle within "
                          f"{max_feedback_rounds} rounds", stacklevel=2)
        return results

    def solve_scenario(self, candidates, A: sp.spmatrix, snet_pu: np.ndarray,
                       max_capacity_pu: np.ndarray) -> OHMResult:
        """One scenario, full γ sweep (MATLAB DCInnerLoop body), per-unit I/O
        except the MW-scaled result fields."""
        X = candidates["reactance_x"].to_numpy(float)
        c = candidates["l1_cost"].to_numpy(float)
        return self._sweep_dc(X, c, A, np.asarray(snet_pu, float),
                              np.asarray(max_capacity_pu, float))

    def tune_snet_scaling(self, model: CrucibleModel, snet_mw: np.ndarray,
                          max_capacity_mva: np.ndarray) -> float:
        """BBR-tier-crossing handshake (codectrl.TuneSNetScaling): solve the
        max-Σ|snet| scenario once with the γ grid widened ±1 decade (+4
        points, this call only); scaling = max(capacity)/max|flow|, and —
        MATLAB quirk, port faithfully — RESET TO 1 (not clamped) when the
        ratio exceeds 1000, likewise floored to 1 below 1."""
        snet_mw = np.atleast_2d(np.asarray(snet_mw, float))
        if snet_mw.shape[0] != model.n_buses:
            snet_mw = snet_mw.T
        worst = int(np.abs(snet_mw).sum(axis=0).argmax())
        A = model.incidence("candidates")
        cand = model.candidates
        cap_pu = np.asarray(max_capacity_mva, float) / self.s_base

        saved = self.gammas
        lo, hi = np.log10(saved[0]), np.log10(saved[-1])
        self.gammas = np.logspace(lo - 1, hi + 1, len(saved) + 4)
        try:
            res = self._sweep_dc(cand["reactance_x"].to_numpy(float),
                                 cand["l1_cost"].to_numpy(float),
                                 A, snet_mw[:, worst] / self.s_base, cap_pu)
        finally:
            self.gammas = saved
        peak = float(np.abs(res.flows_abs).max())
        if peak <= 0:
            return 1.0
        scaling = float(max_capacity_mva.max()) / peak
        if scaling < 1.0 or scaling > 1000.0:
            return 1.0
        return scaling

    def shift_gammas(self, direction: int) -> None:
        """UpdateGamma feedback (MultiLevelProgram 266-272): ±0.5 decades;
        direction 0 = scenarios disagreed, widen BOTH ends (8 points)."""
        lo, hi = np.log10(self.gammas[0]), np.log10(self.gammas[-1])
        if direction > 0:
            self.gammas = np.logspace(lo + 0.5, hi + 0.5, 7)
        elif direction < 0:
            self.gammas = np.logspace(lo - 0.5, hi - 0.5, 7)
        else:
            self.gammas = np.logspace(lo - 0.5, hi + 0.5, 8)

    # -- internals -----------------------------------------------------------

    def _get_qp(self, X: np.ndarray, c: np.ndarray, A: sp.spmatrix) -> _QPWorkspace:
        key = (len(X), hash(X.tobytes()) ^ hash(c.tobytes()))
        if self._qp is None or self._qp_key != key:
            self._qp = _QPWorkspace(X, c, A, env=self._env,
                                    verbose=self.config.verbose)
            self._qp_key = key
        return self._qp

    def _sweep_dc(self, X, c, A, snet_pu, max_capacity_pu,
                  seed_factor: DCSeedFactor | None = None) -> OHMResult:
        """γ sweep with Newton fast path + QP fallback + stall handler.
        seed_factor: shared B-matrix LU (solve_all builds it once per
        candidate topology); None = factor here (single-scenario callers)."""
        cfg = self.config.ohm
        m = len(X)
        run_newton = cfg.solver.lower() == "newton"
        run_qp = (not run_newton) or cfg.use_qp_with_newton

        # seed from a plain DC power flow (also the dcpf_flows output)
        seed = (seed_factor or DCSeedFactor(A, X)).solve(snet_pu)
        dcpf_mw = np.abs(seed.flows_mw) * self.s_base
        p = seed.flows_mw.copy()             # pu, signed
        theta = seed.theta_rad[1:].copy()    # slack (bus 0) removed

        newton_cutoff = cfg.newton_zero_cutoff
        if run_newton and cfg.newton_recalc_cutoff:
            newton_cutoff = float(np.quantile(dcpf_mw, 0.25)) / self.s_base
        epsilon, alfa = 0.01, 1.0            # sweep-level Newton state

        gammas = self.gammas
        plf = np.zeros(len(gammas))
        flows_out: np.ndarray | None = None
        cutoff_out = 0.0
        gamma_used = float(gammas[-1])
        solver_used = "newton" if run_newton else "qp"
        best_flows = np.zeros(m)
        qp_failed = False

        for i, gamma in enumerate(gammas):
            flows_pu = None
            if run_newton:
                p, theta, epsilon, alfa, ok = self._newton_step(
                    X, c, A, snet_pu, float(gamma), p, theta, epsilon, alfa)
                if not ok:                    # α ≤ 0.25: abandon Newton
                    run_newton, run_qp = False, True
                    solver_used = "qp"
                else:
                    flows_pu = p
            if run_qp:
                try:
                    flows_pu = self._get_qp(X, c, A).solve(snet_pu, float(gamma))
                except OHMSolveError:
                    qp_failed = True          # MATLAB: break the sweep
                    break
            assert flows_pu is not None
            best_flows = flows_pu
            cutoff = cfg.qp_zero_cutoff if solver_used == "qp" else newton_cutoff
            plf[i] = float((np.abs(flows_pu) < cutoff).mean())
            if plf[i] >= self.terminate_frac and flows_out is None:
                flows_out = np.abs(flows_pu)
                cutoff_out = cutoff
                gamma_used = float(gamma)
                break

        if flows_out is not None:
            update = 0
        elif qp_failed or not np.any(best_flows):
            flows_out, cutoff_out, update = np.zeros(m), 0.0, +1
        else:                                 # best-effort column, ask for ↑γ
            flows_out = np.abs(best_flows)
            gamma_used = float(gammas[min(len(gammas) - 1,
                                          int(plf.argmax()))])
            cutoff_out, update = 0.0, +1
        positive = plf[plf > 0]
        if positive.size and positive.min() > 0.4:
            update = -1                       # overshoot: shift γ down

        # γ≥100 stall handler (MATLAB 319-325): reset grid, lower target
        if gamma_used >= 100.0:
            self.gammas = np.logspace(-4, -3, 8)
            self.terminate_frac = max(
                (np.floor(plf.max() * 100) - 2) / 100.0, 0.05)

        return OHMResult(flows_abs=flows_out * self.s_base,
                         dcpf_flows=dcpf_mw, cutoff=cutoff_out * self.s_base,
                         gamma_used=gamma_used, update_gamma=update,
                         solver_used=solver_used, percent_low_flow=plf)

    def _newton_step(self, X, c, A, snet_pu, gamma, p, theta, epsilon, alfa):
        """One γ's Newton solve on the ε-smoothed KKT system. Returns
        (p, theta, epsilon, alfa, still_newton). NOTE: implements the TRUE
        second derivative γ·c·ε²·(p²+ε²)^(−3/2) — MATLAB line 98 wrote
        exponent −2/3 (documented open question); the correct Hessian is
        positive-definite and converges at least as well."""
        m = len(X)
        A_red = A[1:, :].tocsr()
        snet_red = np.asarray(snet_pu, float)[1:]
        stop = 5e-4 + 10.0 * gamma
        Xd = sp.diags(X)

        def grad(pv, tv):
            smooth = np.sqrt(pv ** 2 + epsilon ** 2)
            vd = gamma * c * pv / smooth
            return np.concatenate([Xd @ pv + A_red.T @ tv + vd,
                                   A_red @ pv - snet_red])

        g = grad(p, theta)
        g_norm_prev = float(np.linalg.norm(g))
        for _ in range(7):                    # MATLAB: Iter < 8
            if np.linalg.norm(g) <= stop:
                break
            smooth2 = p ** 2 + epsilon ** 2
            d = gamma * c * epsilon ** 2 * smooth2 ** (-1.5)
            hes = sp.bmat([[Xd + sp.diags(d), A_red.T],
                           [A_red, None]], format="csc")
            delta = splu(hes).solve(-g)
            p = p + alfa * delta[:m]
            theta = theta + alfa * delta[m:]
            g = grad(p, theta)
            g_norm = float(np.linalg.norm(g))
            if g_norm - g_norm_prev > 0.2 * g_norm_prev:   # divergence guard
                if alfa <= 0.25:
                    return p, theta, epsilon, alfa, False  # switch to QP
                alfa *= 0.5
                epsilon += 0.005
                stop += epsilon
            g_norm_prev = g_norm
        return p, theta, epsilon, alfa, True

    def _solve_ac(self, model: CrucibleModel, snet_pu: np.ndarray) -> OHMResult:
        """AC-formulation inner loop — PERMANENTLY RETIRED (the author,
        2026-07-11): ACInnerLoopOptim.m was never-operational dev code
        (see docs/algorithms/ACInnerLoopOptim.md) and the AC OHM mode is
        not needed; DC OHM + TUNE's AC refinement + the final ACPF are the
        accepted design."""
        raise NotImplementedError(
            "AC OHM mode retired per the author 2026-07-11 — see "
            "docs/algorithms/ACInnerLoopOptim.md; use ohm.formulation='dc'")
