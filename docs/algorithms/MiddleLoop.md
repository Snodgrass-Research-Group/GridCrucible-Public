# MiddleLoop.m + MiddleLoopAC.m → `loops/tune.py` (TUNELoop)

**Sources:** MiddleLoop.m (126), MiddleLoopAC.m (161)
Leaves in algorithms/conductors.md.

## run_dc (MiddleLoop.m) — per TRIM iteration

Density tiers (same BBR thresholds as scenario selection):
* **BBR > high (20)**: flags only (Upsize on, Downsize off) — NO resize call;
  SNet scaling guarantees no overloads while scaling > 1.
* **BBR > mid (12)**: single `resize_branches` pass using the **OHM flow
  envelope** (`candidates.max_flow`).
* **BBR > midlow (6)**: single pass using **max_dcpf** instead.
* **BBR ≤ midlow**: iterative refinement. term_criteria = 5 below low (3),
  else 20. Loop: resize (max_dcpf) → assign_conductors → impedance recompute
  (gridforge RXB) → transformer_impedances → DC PF per scenario with
  **UNSCALED injections** (snet/(snet_scaling·s_base), the scaling applied
  for OHM is backed out) → max_dcpf = per-path max |flow| → overloaded =
  (capacity − max_dcpf·margin) < 0 AND cond_idx < max_cond_idx. Continue
  while count(overloaded AND headroom > 1) > term_criteria; after 15 iters
  term_criteria += 1 per iter (guaranteed exit). Disconnection → error
  (SingularNetworkError propagates).
* Finally: if any `resize` flags remain set → assign + impedances once more.

## run_ac (MiddleLoopAC.m) — pipeline stage 8

Same refinement shape with AC power flows: percent_resize forced to 0.01
(protect the near-final network); term = 5, relax after 20 iters; scenarios =
SNet columns 2..end (col 1 is a duplicate — MATLAB comment), auto per-unit
detection ported as an explicit expectation (injections arrive in MVA);
non-converged scenarios contribute ZERO flows (MATLAB multiplies by the
converge flag) — i.e. they don't drive sizing; slack = max-capacity gen bus.
MATLAB warm-started VBusSet from the last converged scenario — pandapower
equivalent: `init="results"` chaining, deferred (flat start is more robust;
revisit if convergence suffers). Final per-scenario losses/stats came from
ACPowerFlowFinal — harvested from our ACPFResult instead.

## Impedance recompute = gridforge RXB (dependency injection)

After conductors change, R/X/B must be recomputed — that algorithm is
**gridforge-owned** (RXB.m + GMD tables). `TUNELoop(impedance_fn=...)`
accepts the recompute callable; the orchestrator wires
`gridforge.network.impedance` once its conversion lands. `impedance_fn=None`
leaves X unchanged (one warning) — fine for rating-only tests, NOT for
production runs.

## Open questions

1. MiddleLoop's two lowest tiers differ ONLY in term_criteria (5 vs 20) and
   the iter>15 vs >20 relax point (AC) — intentional asymmetry? (Ported as
   written.)
2. MiddleLoopAC drops scenario 1 as "duplicate" — verify against CreateSNet's
   column layout in .
