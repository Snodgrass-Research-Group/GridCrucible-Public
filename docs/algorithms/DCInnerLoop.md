# DCInnerLoop.m → `loops/ohm.py` (OHMLoop, DC mode)

**Source:** `KCL-KVL Compliant Grid Generator/Matlab Functions/DCInnerLoop.m` (327 loc)

## Purpose

One OHM-loop evaluation for ONE net-injection scenario: find sparse DC flows
on the candidate graph that satisfy KCL exactly and KVL through optimality,
with an L1 penalty (weight γ) that drives flow on "unnecessary" paths to ~0.
Sweeps γ upward until enough paths are at zero flow, and reports which γ
achieved it. TRIM later uses "how often was this path ~zero across scenarios"
as removal evidence.

## The optimization problem

Variables split as p = p⁺ − p⁻, p± ≥ 0 (size 2·n_paths):

    min  ½ p⁺ᵀdiag(X)p⁺ + ½ p⁻ᵀdiag(X)p⁻ + γ·L1Costᵀ(p⁺+p⁻)
    s.t. [A, −A]·[p⁺;p⁻] = SNet          (KCL at every bus, per-unit)

* X = candidate reactances; ½Xp² is the "energy" objective whose stationarity
  conditions reproduce the DC power-flow (KVL) equations — this is the
  KCL+KVL trick of the whole dissertation pipeline.
* L1Cost = per-path weight (length-based, zeroed for `Keep` paths by TRIM).
* Everything is per-unit: SNet/SBase in, flows × SBase out.
* MATLAB solves it two ways, controlled by `options.Solver`:

### Solver 1: QP ("Opti" path) — CPLEX / Gurobi / Mosek

MATLAB already contains a **working Gurobi formulation** (lines 153-179):
`Q = blkdiag(diag(X),diag(X))`, `obj = γ[L1Cost;L1Cost]`, `A=[AOL,−AOL]`,
`sense='='`, `lb=0` (default). Port THIS to gurobipy matrix API:

```python
m = gp.Model(env=env)
p = m.addMVar(2 * n_paths, lb=0.0)
m.addMConstr(sp.hstack([A, -A]).tocsr(), p, '=', snet_pu)
m.setObjective(0.5 * p @ Q @ p + gamma * c @ p)   # update gamma via p.Obj
```
Build the model ONCE per TRIM iteration; per-γ re-solve = update linear obj
coefficients only (warm start comes free). Per-scenario = update RHS
(`constr.RHS = snet_pu`). Zero-flow threshold: `options.OptiCutoff = 1e-5`.
Flows out: `SBase * (p⁺ − p⁻)`.

### Solver 2: custom Newton on the ε-smoothed L1 (the fast path — KEEP IT)

Replace |p| with sqrt(p² + ε²), ε=0.01, and solve the stationarity system of
the *unsplit* Lagrangian directly by Newton iteration on z = [p; θ₂..ₙ]:

    Grad = [ diag(X)p + Aᵀ₂..ₙ θ + γ·L1Cost·p/sqrt(p²+ε²) ;  A₂..ₙ p − SNet₂..ₙ ]
    Hes  = [ diag(X) + D(γ,ε) ,  Aᵀ₂..ₙ ;  A₂..ₙ , 0 ]     (sparse KKT)

* Initialized from a plain DC power flow (`DCPowerFlowSimple`) — flows p₀ and
  angles θ₀ (slack removed = bus index 0).
* Step: z += α·Δz, Δz = −Hes⁻¹Grad (sparse solve; use `scipy.sparse.linalg.spsolve`
  or `splu` refactorization per iteration since D changes).
* Max 8 iterations; stop when ‖Grad‖ < 5e-4 + 10γ.
* **Divergence guard:** if ‖Grad‖ grows > 1.2× between iterations, halve α and
  relax ε += 0.005 (and loosen the stop criterion by ε). If α drops ≤ 0.25,
  ABANDON Newton for this and all remaining γ: switch `options.Solver='Opti'`
  (QP) — port this fallback faithfully.
* Zero-flow threshold `options.NewtonCutoff` (0 by default; optional
  `RecalcCutoff` = 25% quantile of initial |DC flows|).
* NOTE MATLAB Hessian term uses exponent −2/3 where the true second derivative
  of p/sqrt(p²+ε²) needs −3/2: `(p²+ε²)^(−2/3)` vs `^(−3/2)`. Likely a typo
  that Newton tolerates (quasi-Newton direction). **Open question for the author:
  intentional?** Python: implement the correct −3/2 but keep a comment; verify
  convergence on reference cases.

## The γ sweep (per call)

For γ in `options.Lam` (logspace grid, default PC: logspace(−5,−4,5)):
1. Solve (Newton and/or QP per flags `UseOpti`, `Solver`).
2. PercentLowFlow(γ) = share of paths with |flow| < cutoff.
3. First γ with PercentLowFlow ≥ `options.InnerLoopTerminate` (0.3 default):
   emit |flows|, cutoff, `UpdateGamma=0`, DONE (break unless debugging).

If NO γ reached the target: return best-effort flows (max-PercentLowFlow
column) with `UpdateGamma=+1` (caller shifts Lam up). If min nonzero
PercentLowFlow > 0.4 (overshoot): `UpdateGamma=−1` (caller shifts Lam down).
Caller (orchestrator) widens/shifts `options.Lam` by ±0.5 decades and re-runs
just the failed scenarios (MultiLevelProgram lines 264-285).

**Stall handler (γ ≥ 100):** reset Lam to logspace(−4,−3,8) and LOWER
`InnerLoopTerminate` to (floor(max(PercentLowFlow)·100) − 2)/100, floor 0.05.
This mutates shared options — in Python it is OHMLoop state
(`self.gammas`, `self.terminate_frac`), reported upward in the result object.

**TuneSNetScaling handshake:** when the orchestrator crosses a BBR tier it
calls the OHM loop once on the max-|SNet| scenario with
`codectrl.TuneSNetScaling=1`; the loop then widens Lam by ±1 decade
(+4 points) for that call only. Result sets
`SNetScaling = max_line_capacity / max|flow|` clamped to [1, 1000].

## I/O contract (Python)

```python
@dataclass
class OHMResult:
    flows_abs: np.ndarray      # |flow| per candidate path, MW
    dcpf_flows: np.ndarray     # plain DC-PF |flows| (for TRIM MaxDCPF)
    cutoff: float              # zero-flow threshold actually used
    gamma_used: float
    update_gamma: int          # -1 / 0 / +1  (Lam range feedback)
    solver_used: str           # 'newton' | 'qp'

OHMLoop.solve_scenario(candidates, A, snet_pu, gamma_grid) -> OHMResult
OHMLoop.solve_all(model, tier) -> list[OHMResult]   # joblib over scenarios
```

Inputs per scenario: `X`, `l1_cost`, `max_capacity_pu`
(= ConductorData.MVA[max_conductor_idx]/SBase), sparse incidence A
(n_buses × n_paths), snet_pu (real part). The plain DC PF that seeds Newton
comes from `power/power_flow.py` (pandapower adapter).

## Parallelism

MATLAB runs scenarios via `parfor` (with RAM/size guards). Python:
`joblib.Parallel(n_jobs=config.workers)` over scenarios; each worker builds
its own Gurobi env (`OutputFlag=0`). Guards on problem size (5e5/5e6 paths)
become a config `parallel_min_free_ram` note, not hard-coded RAM reads.

## Validation

* KCL: `A @ p == snet_pu` to 1e-6 on every solve.
* KVL/optimality: for paths with |p|>cutoff, X·p ≈ −Δθ across the path
  (θ from the equality duals / Newton state).
* QP vs Newton agreement on small cases (MATLAB debugging code did exactly
  this — NewtonOptiAgree).
* Reference: the author can produce MATLAB (γ, PercentLowFlow) traces on demand.

## Open questions

1. Hessian exponent −2/3 vs −3/2 (above).
2. `options.UseOpti=0` + `Solver='gurobi'` default means Newton is OFF in the
   shipped config (`RunOpti=1`, `RunNewton=0`) — confirm Newton is still worth
   porting first-class (it is, per ARCHITECTURE.md §5).
3. `MaxLineCapacity` argument is accepted but the QP upper bound is commented
   out (line 160) — capacity is NOT enforced in the inner loop, only used for
   SNetScaling. Confirm intentional.
