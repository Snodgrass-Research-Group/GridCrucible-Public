# OuterLoop.m + RemovePaths.m → `loops/trim.py` (TRIMLoop)

**Sources:** `Matlab Functions/OuterLoop.m` (163 loc), `Matlab Functions/RemovePaths.m` (86 loc)

## Purpose

The TRIM Loop (Topology Reduction via Informed Metrics) consumes the
per-scenario OHM flow matrices and decides which candidate paths to delete
this iteration. Two phases: `score_and_mark` (OuterLoop.m — scoring,
protection, removal probabilities) and `remove` (RemovePaths.m — actually
delete, guarding connectivity).

## Phase 1 — score_and_mark (OuterLoop.m)

Inputs: `Flows` (n_paths × n_scenarios, |OHM flows|), `DCPowerFlows` (plain
DC-PF flows), per-scenario zero-flow `Cutoff`, SNet, EI Form-715 target stats
(`Data.EI.Stats`), node-degree stats.

1. **Per-path scenario statistics** appended to candidates:
   * `LowFlow` = # scenarios where flow ≤ that scenario's cutoff (the key
     sparsity evidence).
   * `Max`, `Mean` of raw flows; `MaxDCPF` = max plain-DC flow (used later by
     TUNE sizing).
   * `LineND` = max (or mean, per `options.Normalize.NodeDegreeMean`) of the
     from/to bus node degrees; also `LineNDMin`.
2. **Flow normalization** (multiplicative cascade — order matters, each step
   rescales `Flows` before the next):
   * by line rating (`Normalize.RateA`, default OFF),
   * by net injection (`Normalize.SNet`, ON): divide by
     `LineSNet = max(non-concurrent peak |SNet| at endpoints)`, zero→mean guard,
     scaled by `params.SNetScaling`,
   * by node degree (`Normalize.NodeDegree`, ON): divide by LineND — penalizes
     hub attachment,
   * by length (`Normalize.Length`, ON): divide by path length — favors short
     lines per MW.
   `MeanNorm` = mean of normalized flows; `Std` = std of normalized flows.
3. **Protection (Keep flags):**
   * Radial: any path touching a degree-1 bus → `Keep=1, Radial=1`.
   * Degree-tier locking vs EI targets: once synthetic degree-1 share ≥ 0.8 ×
     EI degree-1 share, lock all paths at degree-2 buses; nested check locks
     the next tier. **MATLAB BUG (probable): the inner block re-tests
     `NodeDegree == 2` where the comment says 3 (line 89). Open question.**
   * High-voltage stop: when total HV line-length ≤ `TargetLengthHV`, lock all
     paths with kVFrom ≥ 230 and set `options.CreateBuses=1`.
   * `L1Cost[Keep==1] = 0` — protected paths lose their L1 penalty (feeds back
     into the next OHM solve).
4. **Adaptive strategy switches** (mutate options — becomes TRIMLoop state):
   * near-converged (mean ND < 3× EI avg): enable one-line plots; toggle
     `Normalize.Length` by comparing HV total length vs a degree-scaled target;
     re-enable `Normalize.NodeDegree` while mean ND > EI avg.
   * mean ND < 3.5 → lower `InnerLoopTerminate` to 0.2 (OHM needs fewer
     zero-flow paths per scenario as the network thins).
5. **Removal probabilities** (`RemoveProb`, after sorting candidates by
   `MeanNorm` ascending — sorted order is what makes "first k candidates" in
   Phase 2 the weakest ones):
   * top-5%-LowFlow paths: RemoveProb = 0.25·U(0,1) (small chance regardless).
   * mid tier (MeanNorm ≤ median, Std ≤ 75% quantile, not Keep,
     LowFlow > max(25% quantile of positive LowFlow, 0.1·n_scenarios)):
     RemoveProb = U(0,1).
   * strict tier (MeanNorm ≤ 25% quantile, Std ≤ mean Std, not Keep,
     LowFlow > same threshold): RemoveProb = 1.
   * Keep paths: RemoveProb = 0.
   * `StochasticRemove=1`: if ≥ ceil(PercentRemove·n) paths have prob 1, take
     only those; else threshold against ONE shared uniform draw
     (`RemoveProb >= I2`). Non-stochastic: everything with prob > 0.

## Phase 2 — remove (RemovePaths.m)

Deletes up to `ceil(options.PercentRemove · n_paths)` (5%) of the marked
candidates (they are at the top of the MeanNorm-sorted table):

1. **Bus-stranding guard:** for the batch, count deletions per endpoint bus;
   any bus whose remaining degree would drop < 1 gets ALL its paths flagged
   `Keep=1` and pulled out of the batch.
2. **Connectivity guard:** delete the batch, rebuild incidence + DC B-matrix
   (slack removed) and test `condest(BBus) >= 1e16` (singular ⇒ disconnected).
   If disconnected: roll back and fall into one-at-a-time mode (and if density
   < MidBBR, latch `options.RemoveSequential=1` for future iterations), where
   each single removal is tested the same way; failures are re-kept
   (`Keep=1, RemoveProb=0`).

**Python improvement (behavior-preserving):** replace the condest singularity
test with an exact connectivity check
(`scipy.sparse.csgraph.connected_components` on the candidate graph) — same
intent, faster and not tolerance-dependent. Keep a `condest`-style fallback
note in tests to document equivalence. Batch guard #1 stays as-is (it is a
heuristic pre-filter, cheaper than per-line checks).

## I/O contract (Python)

```python
TRIMLoop.score_and_mark(model, flows, dcpf_flows, cutoffs) -> TrimMarking
    # appends low_flow_count / mean_norm / std_norm / remove_prob / keep /
    # is_radial to model.candidates; returns strategy-switch report
TRIMLoop.remove(model) -> TrimResult   # drops rows from model.candidates,
    # updates node-degree stats in model.meta; reports n_removed, n_protected
```

RNG: all U(0,1) draws from `numpy.random.Generator` seeded via config
(one child seed per TRIM iteration for reproducibility).

## Validation

* Invariants after every `remove`: graph connected; no bus with degree 0;
  `keep` paths never removed; ≤ ceil(5%·n) removed.
* Degree distribution drifts monotonically toward the EI target across
  iterations on reference runs.
* Reference: MATLAB per-iteration `OLBranch` snapshots (QNCAResultsBBR*.mat)
  give distribution-level regression targets.

## Open questions

1. The degree-3 locking block that re-tests `== 2` (OuterLoop.m:89) — bug or
   intentional double-lock? Python implements the comment (== 3) behind a
   config flag `legacy_nd3_bug=False`; ask the author.
2. `OLBranch.LowFlow` accumulates across the scenario loop but `Cutoff10`
   (bottom-10% quantile) is computed and only used under debugging — drop it?
3. `options.PercentRemove` is documented as "each iteration of the middle
   loop" but is used by TRIM — naming only, no behavior issue.
