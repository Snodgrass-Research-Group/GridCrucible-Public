# UCCD.m → `data/dispatch.py` (complementary dispatch)

**Source:** UCCD.m (94 loc, cplexqp ×7 call sites). Companion UCED
(economic dispatch) is gridforge-owned. 
## Purpose

The **complementary dispatch** ("CD"): a dispatch that pushes output onto the
EXPENSIVE units, so the SNet scenario set exercises candidate paths in both
directions (the KVL pipeline's analogue of gridforge's Converse ED). It does
NOT maximize cost directly — it repeatedly solves the cheap-dispatch QP and
then DECOMMITS the units the QP loaded to their maximum (the cheap ones),
forcing later solves onto costlier units.

## Algorithm (ported step-for-step)

Inputs: fleet table (MinMW/MaxMW/A/B/C/FuelType), hydro_scaling, load vector
(only Σload matters), reserve.

1. Hydro Max/Min scaled by hydro_scaling.
2. Base QP: min ½xᵀdiag(C)x + Bᵀx s.t. Σx = Σload, Min ≤ x ≤ Max
   (gurobipy; every re-solve below is this QP on the surviving fleet).
   Infeasibility fallback used THROUGHOUT: if Σmin > Σload (or QP fails),
   retry with lb = 0.
3. **Rough decommitment** (≤ 5 rounds): units with (Max − x) < 0.05·Max
   (i.e. dispatched within 5% of max — the cheap ones) are REMOVED; re-solve.
   Exit when: surviving capacity < load (revert to pre-deletion fleet), or
   min(x) > 1 / all(x > lb+1), or infeasible.
4. Re-solve on the surviving fleet.
5. **Reserve trimming**: while capacity·(1 − reserve) > load and > 5 units:
   drop the 3 lowest cost-per-MW units (CostPerMW = (A + Bx + Cx²)/x),
   re-solve; stop if the removal would leave capacity ≤ load.
6. Cleanup: drop units dispatched ≤ 0.1 MW and rescale the remaining x by
   (1 + mismatch/Σx) to rebalance.

Returns (x, surviving fleet with original row identity preserved) — SNet
construction maps dispatch back to buses via the fleet's index.

## Python notes

* One gurobipy model per fleet subset (fleets shrink monotonically — rebuild
  per solve is fine at ~10³ units; the QP is diagonal).
* MATLAB's `mink(CostPerMW, 3)` → `np.argpartition`.
* The 5%-of-max threshold: MATLAB comment says 3%, code says 0.05 — code wins.
* Zero-division in CostPerMW when x=0: MATLAB produced Inf (never selected by
  mink) — replicate with np.errstate(divide='ignore').

## Validation

* Complementarity property: CD's cost-per-MW-weighted dispatch ≥ UCED's on
  the same fleet/load (expensive units carry more).
* Σx == Σload after every stage incl. the final rescale.
* Bounds respected (except the documented lb=0 fallback cases).
* Reference: the author can produce MATLAB (fleet, load) → (x, fleet_out)
  pairs on demand.
