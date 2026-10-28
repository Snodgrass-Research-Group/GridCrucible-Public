# main_SRTV.m + MultiLevelProgram.m + InitOptions.m → `core.py` (GridCrucible) + `config.py`

**Sources:** `SRTV/main_SRTV.m` (764 loc), `Matlab Functions/MultiLevelProgram.m` (396 loc),
`SRTV/User Input Data/InitOptions.m` (81 loc)

## Purpose

The GridCrucible orchestrator replaces both mains:

* `main_SRTV.m` — the region-specific driver: data loading, run control
  (codectrl flags), substation/bus/generator setup, candidate-graph creation,
  post-loop bus creation, shunt compensation, AC refinement, metrics, writers.
* `MultiLevelProgram.m` — the TRIM-iteration engine: the `while Go` loop that
  runs OHM (per scenario) → network stats → TRIM (score/remove) → TUNE
  (resize) → realism metrics → checkpoint, until mean node degree hits the
  EI target.

## Pipeline stages (TOML `stages = [...]`, checkpoint after each)

| # | Stage | MATLAB origin | Python target |
|---|---|---|---|
| 0 | `load_inputs` | main_SRTV §1–2 addpath + load cascade | `io/matlab_import` (pre-migrated data) + `CrucibleConfig` |
| 1 | `setup_substations` | main_SRTV §Initializing (Sub/Bus tables from BusInputData.csv, RegionStates.csv) | `GridCrucible.setup_substations` |
| 2 | `setup_generators` | §3.1–3.2 (GenNat subset → internal numbering, slack = first gen bus, low-load floor 0.5 MW) | `GridCrucible.setup_generators` |
| 3 | `build_scenarios` | §3.3 CreateSNet (SNet tiers via ED/UCED/UCCD dispatch) | `data/scenarios.build_snet` |
| 4 | `build_candidates` | §5–6 VChooseK pairs → length filter (< max(MaxLength)·1.61 km) → GMDIndex randi(1000) → CreateOLMatrices (gridforge) | `GridCrucible.build_candidates` |
| 5 | `trim_iterations` | §7 MultiLevelProgram | `GridCrucible.run_trim_iterations` |
| 6 | `create_buses` | §9 CreateBuses (+ CreateBranch, stats, RealismMetrics) | `network/buses` |
| 7 | `shunt_compensation` | §10 ReactivePowerPlacement | `network/shunts` |
| 8 | `ac_refinement` | §10 MiddleLoopAC (+ CreateBranch) | `loops/tune.TUNELoop.run_ac` |
| 9 | `final_acpf` | §10 ACPowerFlowFinal | `power/power_flow` |
| 10 | `metrics` | §8 RealismMetrics(+Actual EI/WECC), NetworkGraphComparison, ElectricalComparison, NetworkStatistics, PlotNodeDegrees* | `metrics/*` |
| 11 | `write_outputs` | §2.4, §11 (SubstationData.xlsx, BusData.xlsx, BranchTable.csv, WriteLoads, WritePowerWorld ×2, WriteMetrics, AC stats xlsx) | `io/writers` |

The MATLAB `codectrl` flags (ReRun/Resume/Add/Analyze/Load + ~25 feature
flags) map to: stage list (which stages run), checkpoint auto-resume (Resume),
and boolean config options (plots, writers, GenN_1, ShuntCompensation,
SubstationMetrics, Stats715*, ...). The try/load cascades
(QNCAResults*.mat → PFData*.mat fallbacks, lines 92–210) disappear: the
checkpoint registry stores schema-versioned bundles.

## Stage 5: run_trim_iterations (MultiLevelProgram) — the heart

```
while mean(node_degree) > threshold:                 # outer TRIM iteration
    tier = density_tier(n_candidates / n_substations)   # BBR: >20, >12, >6, >3, ≤3
    snet, scaling = pick_scenarios(tier)                # tier → SNet matrix + retune
    results = OHM.solve_all(snet)                       # per-scenario, joblib parallel
    handle_gamma_feedback(results)                      # re-solve failed scenarios w/ shifted Lam
    stats = network_statistics(candidates)              # gridforge NetworkStatistics
    TRIM.score_and_mark(candidates, results); TRIM.remove()
    recompute_voltages_per_substation()                 # Sub.Voltages from surviving kVFrom
    TUNE.run_dc(candidates, results)                    # MiddleLoop resize
    if mean_nd < 2 * EI_avg: realism_metrics(); checkpoint('ND2')
    checkpoint_if_tier_crossed()                        # BBR20/12/6/3 milestones
```

Key mechanics to port faithfully:

* **BBR tiers** (`options.HighBBR=20, MidBBR=12, MidLowBBR=6, LowBBR=3`)
  select the scenario tier: Basic → Intermediate → Advanced → Complete →
  Full (`PF.SNet*`). Dense networks use few scenarios (cost), thin networks
  use all.
* **SNetScaling retune at each tier crossing:** one OHM solve on the
  max-Σ|SNet| scenario with widened γ grid (TuneSNetScaling handshake), then
  `SNetScaling = max(MaxRateA)/max|flows|` clamped [1, 1000]; scenarios are
  multiplied by it. Initial scaling on PC for < 4000 buses:
  `0.1·mean(node_degree)`.
* **Parallel-pool resizing per tier** (MATLAB parpool 30%/60%/100% of cores,
  caps 16/16/32) → single `joblib` pool with `n_jobs` from config; document
  the MATLAB behavior, don't replicate pool churn.
* **γ feedback:** scenarios returning `update_gamma=±1` get re-solved after
  shifting `Lam` ±0.5 decades (both ends if mixed signals; loop until 0).
* **Termination:** when mean node degree < 2× EI average, start computing
  realism metrics + Branch table each iteration; terminate when mean ND <
  `Data.EI.Stats.NodeDegreeAvg` (bus-level) or 2.6 (substation-level).
  On termination: PlotNetwork, CreateBranch, RealismMetrics, final checkpoint.
* **Formulation switch:** `options.Formulation` 'dc' → DCInnerLoop per
  scenario; 'ac' → ACInnerLoopOptim on scenario 1 (AC mode is
  single-scenario in MATLAB — confirm with the author whether that's a
  limitation to keep).
* `params.buses` (#substations) vs `params.nodes` (#buses) duality — nodes
  updated whenever `height(Bus) > height(Sub)`.
* `BaseSNetScaling > 1` multiplies ALL tiers up front (small-case knob).
  Note MATLAB typo `PF.SNetBaisc` (line 52) would crash if that branch ran —
  config default keeps it 1; flag in doc, don't reproduce the typo.

## Candidate-graph construction details (stage 4)

* All bus pairs (VChooseK → `np.triu_indices`), keep pairs with haversine
  length < max(options.MaxLength)·1.61 km (MaxLength in miles per voltage:
  [25, 65, 90, 140] for [69, 138, 230, 500] kV).
* `LengthModifier` multiplies lengths (small-system knob, default 1).
* `GMDIndex = randi([1,1000])` per path, PERSISTED (GMDIndex.mat) and reused
  across runs; Python: seeded Generator + persisted parquet, `reset_gmd_index`
  config flag replaces codectrl.ResetGMDIndex/LoadGMDIndex.
* Then `CreateOLMatrices` (gridforge-owned) computes X, L1Cost, RateA,
  MaxCondIndex, incidence AOL. L1 weighting knobs: `L1ExistingLines`,
  `GenToGenPenalty`, `L1Weight`.

## InitOptions.m → CrucibleConfig defaults (SRTV case)

VoltageLevels [69,138,230,500]; MaxLength [25,65,90,140] mi;
TargetLengths table (Form 715/DOE totals by kV, TargetLengthHV = max(sum
230–765, 999-row)); Lam = logspace(−5,−4,5) (PC) / logspace(−7,−5,6) (Linux —
use as large-case preset); InnerLoopTerminate 0.3; PercentRemove 0.05;
PercentResize 0.05; PercentChange 1.3; ConductorMargin 1.0;
DownsizeThreshold 0.5; Reserve 0.1; LoadScenarios 24; UCScenarios 2 (ED+CD);
Dispatch 'Full'; Formulation 'dc'; Solver 'gurobi' (already the MATLAB
default!); ThetaMax 45°; SBase 100; OptiCutoff 1e-5; MinBranchBus 1.6;
StochasticRemove 1; RandomResize 1; ChangeVoltages 1; CreateGenBuses 1;
RoutingFactor 1. Sections: [case], [ohm], [trim], [tune], [dispatch],
[parallel], [io] in the TOML.

## Open questions

1. AC formulation single-scenario only (MultiLevelProgram line 73) —
   intentional?
2. `codectrl.ReRun || codectrl.ReRun` (line 13) — second term was probably
   meant to be `codectrl.Resume`; affects whether BBR/Normalize defaults are
   reset on resume. Ask.
3. main_SRTV §7 passes `DataBus` even when it may not exist (both branches of
   the `if exist('DataBus')` call MultiLevelProgram identically, line 586-589)
   — dead branch, ignore?
4. Mid-run "Section to change variables" (main_SRTV lines 211–223, e.g.
   ACParallel=0, RandomResize=1, PercentResize=0.05, Dispatch='Regional' on
   resume) contradicts InitOptions defaults (Dispatch='Full') — which wins for
   the Python defaults? Currently: InitOptions values, with resume overrides
   documented here.
