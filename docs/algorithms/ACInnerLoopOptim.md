# ACInnerLoopOptim.m — the AC OHM loop (never-operational dev code)

> **RESOLVED 2026-07-11 (the author): the AC inner loop is NOT needed.**
> Option 1 below is the permanent answer — `OHMLoop._solve_ac` stays
> `NotImplementedError`, ACInnerLoopOptim.m joins the orphans in spirit,
> and DC OHM + TUNE AC refinement + final ACPF is the accepted design.
> The Open questions below are retained for the record only.

Source: `Matlab Functions/ACInnerLoopOptim.m` (270 loc). Target:
`loops/ohm.py` (`OHMLoop._solve_ac`).
Written 2026-07-10 from a full read of the source + its call site.

## HEADLINE FINDING: this function has never run to completion

Three independent, unconditional fatal defects mean the AC formulation was
**never operational** in the current source generation:

1. **`options` clobber → crash on first γ** — line 89 (inside the γ loop)
   overwrites the QNCA `options` struct with
   `optimoptions('fmincon',...)`; line 121 then reads `options.SBase`,
   which does not exist on an optimoptions object → MATLAB errors (the
   try/catch ends at line 106 and does not cover this).
2. **`Indxopt` undefined** — line 101 assigns the inductor-energy solve to
   `NLxopt`, but lines 154–163 consume `Indxopt`, which is never defined.
3. **`NewNLeLength` undefined** — line 254 (WriteInnerLoopResults branch).

And the **call site is stale**: `MultiLevelProgram.m:73` (guarded by
`options.Formulation == 'ac'`; the InitOptions default is `'dc'`) passes
**16 arguments to the 18-parameter signature** — `SNet2` is dropped, so
every argument from `VBusSet` on is shifted (`LoadList` receives
`OLBranch.X`, `params` receives `codectrl`, and `options`/`codectrl` are
never bound). It would crash at the first `options.Lam` access even before
defect 1.

Related: the sibling `options.Formulation=='dc'` branch has a
`RunDiodeCode` path calling `DCPFwDiodesInnerLoop(...)` — **that file does
not exist anywhere in the repo** (phantom function, unreachable behind the
hard-coded `RunDiodeCode = 0` at MultiLevelProgram.m:6).

The orphans `ACInnerLoop.m`, `ACInnerLoopV1.m`, `ACInnerLoopDebugging.m`
are earlier generations of the same idea (see Open questions).

## Intended algorithm (reverse-engineered)

For lossless networks (`LosslessLines = 1` hard-coded), the AC power-flow
equations are the stationarity conditions of the scalar **energy function**

    E(V, θ) = −½ · Vᵀ (BBus ∘ cos(θᵢ − θⱼ)) V − PNetᵀ θ − Σ_{i∈load} QNetᵢ · log(Vᵢ)

over z = [V; θ] (V magnitudes, θ angles) — the classic energy-function /
Bergen–Hill formulation. The function first runs a plain
`ACPowerFlowFinal` (gridforge-owned) to get `BBus` (bus susceptance),
`YPrim` (primitive branch admittances) and baseline voltages `VBusPolar`
for diagnostics, then for each γ in `options.Lam` solves up to FOUR
fmincon problems (`sqp`, x0 = flat start + U(0,0.01) noise):

| # | Objective | Constraints beyond bounds | Role |
|---|---|---|---|
| a | `ACfun`: E(z) + ‖γ·L1Cost‖₁ | none | baseline — NOTE the L1 term is a **constant** in z (no `·flows`): it does not affect the argmin. Written intent was presumably γ‖L1Cost ∘ flow‖₁ |
| b | `ACFunL1Lin`: E(z) + ‖γ·L1Cost ∘ (BPrim ∘ AOLᵀθ)‖₁ | `BBus·θ = −PNet` | **the primary variant** — its `fval`/exit flag gate the loop (lines 107–113); L1 on the DC-approximated flows |
| c | `ACFunL1NL`: E(z) + ‖γ·L1Cost ∘ (−V_f V_t BPrim sin(θ_f−θ_t))‖₁ | none | L1 on the true lossless AC flow |
| d | `Indfun`: −½ Vᵀ(BBus∘cosΔθ)V (inductor energy ½Li²) | `ceq: w(1) = w(2)` — an obvious **placeholder** (the commented-out ceq below it is the real nonlinear KCL constraint) | experimental |

Quirk to preserve knowledge of: variant (b)'s `beq` is written as the n×2
matrix `[zeros(n,1), -PNet]`, which MATLAB linearizes column-major into
the length-2n vector `[0; −PNet]` — combined with
`Aeq = [0 0; 0 BBus]` this yields n trivial rows and the n meaningful
rows `BBus·θ = −PNet` (a DC-PF equality on the angles).

Bounds: V ∈ [0.75, 1.4] pu with generator/slack buses PINNED to their
setpoints (lb = ub = GenVolt); θ ∈ [−0.7, 0.7] rad with slack pinned to 0.
Slack absorbs any injection imbalance > 0.01 pu (line 52). Per variant,
flows are recovered as `SBase · V_f · conj(YPrim · AOLᵀ V)` (complex,
real part kept), zeroed below `options.QNCACutoff`; `FlowsOut` is
**overwritten** by each variant in sequence, so the last computed one wins
(as written, (d) — which crashes first; with defects fixed the effective
return would be the inductor-energy flows, almost certainly NOT intended).
Returns `(FlowsOut, lambda)` with `lambda` = last γ — there is **no
termination criterion**, no `PercentLowFlow`, no `UpdateGamma` feedback:
the γ sweep just runs the full grid, unlike DCInnerLoop.

The rest of the file (lines 165–256) is diagnostics: percent differences
vs the plain-ACPF voltages, MVar quadratic-approximation checks, and
per-γ PowerWorld `LineImport<i>.csv` / `NLeImport<i>.csv` writers.

## Port decision (pending the author — see Open questions)

The Python port CANNOT be algorithm-for-algorithm faithful to code that
never executed. Options, in recommended order:

1. **Keep `OHMLoop._solve_ac` as `NotImplementedError` citing this doc**
   (status quo) until the author confirms the AC OHM loop is wanted for the
   pipeline (the DC loop + TUNE's AC refinement + final ACPF already give
   AC-feasible outputs downstream).
2. If wanted: implement the INTENT — variant (b) (energy function + L1 on
   linearized flows + `BBus·θ = −PNet`) as the single formulation, via
   `scipy.optimize.minimize(method='trust-constr')` with analytic
   gradient/Hessian (closest to fmincon semantics; no solver-license
   coupling; the ‖·‖₁ term splits into p⁺/p⁻ exactly as the DC QP does).
   gurobipy is NOT a natural fit: the cos(Δθ) coupling requires
   general-function constraints (piecewise-linear), losing the smooth
   NLP structure fmincon exploited.
3. Wire the same γ-sweep/termination machinery as `_sweep_dc` around it
   (zero-flow fraction vs `terminate_frac`), since the MATLAB version's
   missing termination logic is clearly unfinished, not a design choice.

## Open questions (the author)

1. Did ANY generation of the AC inner loop (ACInnerLoop.m / V1 /
   Debugging) ever produce results used in the dissertation or papers?
   If yes, which file should the port follow? If no, is AC OHM still a
   goal, or is DC-OHM + downstream AC refinement the accepted design
   (making this a permanent orphan)?
2. In variant (a) the L1 term is constant (no flow factor) — typo for
   `‖γ·L1Cost ∘ flows‖₁`, or deliberate baseline-without-L1?
3. Variant (d)'s `ceq: w(1) = w(2)` is a placeholder; the commented-out
   line below it is full nonlinear KCL. Was (d) abandoned?
4. `FlowsOut` last-writer-wins across the four variants — which variant's
   flows should the OHM loop actually hand to TRIM? (The gating logic at
   lines 107–113 suggests (b).)
5. `DCPFwDiodesInnerLoop` is called but does not exist in the repo — is
   that file lost (worth recovering?) or abandoned?
