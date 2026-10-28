# GridCrucible architecture

How the package is organised, the data contract every stage shares, and the
design decisions made while porting the KCL-KVL Compliant Grid Generator (the
`main_SRTV.m` QNCA pipeline from Jonathan Snodgrass's dissertation) to Python.
Per-algorithm notes are in [algorithms/](algorithms/); the MATLAB call tree is
in [CALL_TREE.md](CALL_TREE.md).

## 1. What GridCrucible is

GridCrucible is the Python successor to the **KCL-KVL compliant** synthetic
transmission-grid generator from Jonathan Snodgrass's dissertation — the
MATLAB code calls it **QNCA** (QP Network Creation Algorithm). Unlike the
KCL-only FLO/SCALE pipeline (GridForge), QNCA produces networks that satisfy
**both** Kirchhoff's laws from the start: flows come from a quadratic program
whose optimality conditions enforce KVL, with an L1 regularizer that drives
flow on unneeded candidate paths to zero so they can be removed.

Three nested loops refine a fully connected candidate graph down to a
realistic network:

| Loop | Name | MATLAB origin | What it does |
|---|---|---|---|
| Inner | **OHM Loop** (*Optimization Honoring KVL Minimization*) | `DCInnerLoop.m`, `ACInnerLoopOptim.m` | Per injection scenario: solve min ½·pᵀdiag(X)p + γ·L1Costᵀ·\|p\| s.t. A·p = SNet (split p = p⁺−p⁻ ≥ 0). Sweeps γ upward (`options.Lam` logspace) until ≥ `InnerLoopTerminate` fraction of paths carry ~zero flow. Two solvers: QP (CPLEX/Gurobi/Mosek → **gurobipy**) and a custom Newton method on the ε-smoothed L1 KKT system (fast path — port it, don't drop it). |
| Middle | **TUNE Loop** (*Thermal Updating of Network Elements*) | `MiddleLoop.m`, `MiddleLoopAC.m` | Re-runs DC (later AC) power flows on surviving paths, reassigns conductors (`AssignConductors`), resizes branches (`ResizeBranches`), recomputes impedances (`RXB` — gridforge-owned) and transformer impedances (`TransformerImpedances`) so ratings track the evolving topology. |
| Outer | **TRIM Loop** (*Topology Reduction via Informed Metrics*) | `OuterLoop.m`, `RemovePaths.m` | Scores every candidate path across scenarios (low-flow counts, normalized mean/std of flows — normalizers: RateA, SNet, node degree, length), protects radial lines and locked node-degree tiers vs EI Form 715 targets, then removes a percentage stochastically (`RemoveProb` tiers). Terminates when mean node degree hits the EI average (~2.6). |

`GridCrucible` (class, `core.py`) is the orchestrator: it replaces `main_SRTV.m`
(setup, data, scenario creation, post-processing) **plus** the
`MultiLevelProgram.m` while-loop (BBR-tier scenario scaling → OHM per scenario
→ TRIM → TUNE → metrics → checkpoint, repeat until converged), then bus
creation, shunt compensation, AC refinement, metrics, and writers.

## 2. Design decisions

1. **Scope.** Everything reachable from the two MATLAB roots, `main_SRTV.m`
   and `MultiLevelProgram.m`, is ported: the generative pipeline and the
   realism metrics, EI/WECC comparisons and plots. `FavoringLines`,
   `StepSize`, `SizingTransformers`, `ResizeLines`, `MeshNetwork`, `NLKCL` and
   `SquentialDCPowerFlow` look like loop members but are orphans;
   `ResizeBranches` and `TransformerImpedances` are the live variants.
2. **Solvers.** All optimisation uses `gurobipy` (replacing `cplexqp`,
   `quadprog`, `fmincon`). All power flow and OPF uses **pandapower** (DC PF,
   AC PF, DC OPF), replacing the MATLAB power-flow functions and the MATPOWER
   wrappers.
3. **DC and AC OHM loops** were ported together (`DCInnerLoop` and
   `ACInnerLoopOptim`).
4. **Architecture.** Composition, not inheritance: `GridCrucible`
   instantiates `OHMLoop`, `TUNELoop` and `TRIMLoop`; state flows through
   `CrucibleModel`. Python ≥ 3.12, pinned to 3.13 in `.python-version`
   (numba has no 3.14 support yet). pandapower 3.5.x caps numpy, so uv
   resolves numpy 2.3.x.
5. **Run control.** A TOML config and stage checkpoints replace the
   `codectrl.*` flags (ReRun/Resume/Add/Analyze/Load). Every stage and each
   BBR density milestone inside the TRIM loop writes a checkpoint (replacing
   QNCAResultsBBR20/12/6/3.mat, QNCAResults*.mat, PFData*.mat); runs resume
   from the latest.
6. **Data.** The KVL `.mat`/`.xls` inputs are migrated once to parquet/npz
   (`io/matlab_import.py`). The large sanitized MATPOWER case scripts
   (`EIBranchSanitized.m`, `EILargeIslandSanitized.m`, `WECC*Sanitized.m`)
   are data, migrated to pandapower JSON + parquet.
7. **Conversion style.** Algorithm for algorithm: intent was reverse-engineered
   (documented in `docs/algorithms/*.md`) and written as idiomatic Python, not
   translated line by line.
8. **Validation.** Three tiers: (a) property-based (KCL and KVL satisfied,
   node-degree and length distributions against Form 715 targets, power-flow
   convergence, angle and voltage limits); (b) distribution-level regression
   against the archived MATLAB SRTV reference outputs in
   `data/matlab_exports/reference_run/`; (c) function-level input/output pairs
   from MATLAB for tricky helpers. Exact numeric equality is not expected
   (multiple optima, different random number generators).

## 3. Package layout

Package `gridcrucible` (src layout). **6 classes** (below); leaf algorithms are
plain functions in focused modules. NOT one class per MATLAB file; NOT a god
class.

```
src/gridcrucible/
├── core.py            class GridCrucible     - orchestrator (main_SRTV + MultiLevelProgram)
├── config.py          class CrucibleConfig   - TOML; replaces codectrl + options + InitOptions
├── model.py           class CrucibleModel    - THE shared data structure (see §4)
├── loops/
│   ├── ohm.py         class OHMLoop          - inner: QP+L1 / Newton, γ sweep (DC & AC)
│   ├── tune.py        class TUNELoop         - middle: conductor/transformer thermal updates
│   └── trim.py        class TRIMLoop         - outer: path scoring, locking, stochastic removal
├── power/
│   ├── pandapower_io.py   CrucibleModel <-> pandapower net (replaces Table2Matpower)
│   ├── power_flow.py      DC/AC PF adapters (replaces DCPowerFlowSimple/Full, ACPowerFlowSimple/Final call sites)
│   └── dcopf.py           DC OPF benchmark (replaces MatpowerDCOPF)
├── network/
│   ├── buses.py           CreateBuses / CreateBranch / TrimQNCALines
│   ├── conductors.py      AssignConductors / ResizeBranches / TransformerImpedances
│   └── shunts.py          ReactivePowerPlacement / PVPQSwitching
├── data/
│   ├── scenarios.py       CreateSNet + CreateLoadScenarios (+ dispatch orchestration)
│   ├── dispatch.py        UCCD (gurobipy QP; UCED comes from gridforge)
│   ├── load_profiles.py   MakeLoadProfiles_<REGION> family -> one builder + regional data
│   └── generators.py      CreateGeneratorDataNational (KVL variant)
├── metrics/
│   ├── graph_metrics.py   Sogol leaves + cyclebasisport (networkx/scipy)
│   ├── realism.py         RealismMetrics (in-loop, synthetic network)
│   └── comparison.py      RealismMetricsActual*/Comparison, NetworkGraphComparison, ElectricalComparison
├── io/
│   ├── matlab_import.py   one-time KVL data migration
│   ├── checkpoints.py     stage + BBR-milestone checkpoints (replaces SaveResults + QNCA*.mat)
│   └── writers.py         WriteLoads / WritePowerWorld / WriteMetrics / WriteSubstationData
└── viz/
    └── plots.py           PlotNetwork / PlotNodeDegrees* / PlotILFun
```

Shared with GridForge: `GridModel`, `EconomicDispatcher`/UCED, `RXB`
impedances + GMD handling, the Y-bus and candidate-path helpers. They come from
the vendored copy in `src/gridcrucible/_gridforge/` (six modules, see its
`__init__.py`) through the adapters in `gridforge_bridge.py`; call-tree nodes
tagged GRIDFORGE-OWNED are implemented there, not here.

## 4. The `CrucibleModel` contract

Every stage consumes and returns a `CrucibleModel`. It wraps/extends the
GridForge `GridModel` (§4 of GridForge's ARCHITECTURE.md) — same conventions
(0-based indexing, km, sparse incidence ±1, MW/MVar columns) — plus the
QNCA-specific state that MATLAB kept in `Sub`/`Bus`/`OLBranch`/`PF`/`params`:

```
CrucibleModel
├── substations: DataFrame   [sub_id(0-based), external_num, name, lat, lon, region,
│                             load_mw, load_mvar, bus_type, voltages(matrix->cols),
│                             xf_mva, gen_max_mw, gen_min_mw]          (MATLAB Sub)
├── buses: DataFrame         [bus_id, sub_id, voltage_kv, ...GridModel.buses cols]
│                             (MATLAB Bus; == substations until CreateBuses splits them)
├── branches: DataFrame      (MATLAB Branch — kept/favored lines; GridModel.branches cols
│                             + tap, n_circuits, voltage_kv_from/to, conductor_idx)
├── candidates: DataFrame    (MATLAB OLBranch — the living candidate set; adds:
│                             from_orig, to_orig, l1_cost, gmd_index, gmd, reactance_x,
│                             capacity_mva(RateA), max_conductor_idx, keep, is_radial,
│                             low_flow_count, mean_norm, std_norm, remove_prob)
├── generators: DataFrame    (MATLAB Gen; national ids preserved via bus_num_ext)
├── scenarios: ScenarioSet   (MATLAB PF.SNet* — injection matrices per tier:
│                             basic/intermediate/advanced/complete/full;
│                             + slack_bus, gen_buses, load_buses, v_setpoints)
├── acpf: ACPFResult | None  (voltages, flows, losses from last AC solve)
└── meta: dict               (params + options snapshot: s_base=100, region list,
                              outer_iter, snet_scaling, node-degree stats,
                              bbr thresholds {high:20, mid:12, midlow:6, low:3},
                              rng seed, stage history)
```

Conventions:
* **0-based** bus/branch/substation indices everywhere; MATLAB's
  `Num`/`NumInt` external-vs-internal dance is replaced by `external_num`
  column + positional index.
* **Substations vs buses**: `params.buses` = #substations, `params.nodes` =
  #buses. Until `CreateBuses` runs they coincide; afterwards buses > subs.
* **km everywhere**; MATLAB multiplies miles by 1.61 at I/O boundaries
  (`options.MaxLength` is in MILES from Form 1 fits).
* Complex `SNet` → separate P/Q arrays (`.real`/`.imag` of one complex ndarray
  is fine in memory; storage is parquet-friendly split).

## 5. Solver mapping

| MATLAB | Where | Python replacement |
|---|---|---|
| `cplexqp(blkdiag(X,X), γ[c;c], Aeq=[A,-A], beq=SNet, lb=0)` | DCInnerLoop | gurobipy QP, matrix API (`MVar`, `addMConstr`); model built ONCE per outer iter, γ and SNet updated in place (`obj` coefficient update / RHS update) — the sweep re-solves warm-started |
| custom Newton on ε-smoothed L1 KKT | DCInnerLoop | port to scipy.sparse (factorize the KKT with `splu`); keep solver-switch fallback Newton→QP on divergence (step halving αlfa, ε relaxation) |
| `fmincon` | ACInnerLoopOptim | gurobipy (nonconvex bilinear QCQP, `NonConvex=2`) or scipy `trust-constr` (see algorithms/ACInnerLoopOptim.md) |
| `cplexqp` | UCCD, UCED | gurobipy QP (UCED arrives via gridforge) |
| `rundcopf`/`mpoption`/`loadcase` | MatpowerDCOPF, ElectricalComparison | pandapower `rundcopp` + converted case files |
| DC/AC power flow (own solvers) | DCPowerFlowSimple/Full, ACPowerFlowSimple/Final call sites | pandapower `rundcpp`/`runpp` via `power/power_flow.py` adapters ("pandapower everywhere") |
| `eigs` | Sogol metrics, fieldervector | `scipy.sparse.linalg.eigsh` |
| `parpool`/`parfor` over scenarios | MultiLevelProgram, MiddleLoopAC, CreateOLMatrices | `joblib.Parallel` (loky) — one Gurobi env per worker; worker count from config (replaces the BBR-tier parpool resizing) |
| `VChooseK` MEX | main_SRTV | `np.triu_indices` (gridforge owns candidate generation) |

## 6. Porting notes

* **The QP is in per-unit**; flows are rescaled by `SBase=100` on output.
  MaxRateA comes from `ConductorData.MVA(MaxCondIndex)` — conductor table
  lookups must be migrated before OHM can run.
* **γ-range adaptation is stateful**: `options.Lam` (logspace) is widened/
  shifted by DCInnerLoop return code `UpdateGamma∈{-1,0,+1}` and by the
  γ≥100 stall handler (which also lowers `InnerLoopTerminate`). This mutation
  must live in OHMLoop state, not global config.
* **SNet scenario tiers + BBR thresholds**: MultiLevelProgram picks
  SNetBasic/Intermediate/Advanced/Complete/Full and re-tunes `SNetScaling`
  (max line capacity / max observed flow, clamped [1,1000]) as density
  (TotalPaths/buses) crosses 20/12/6/3. Checkpoint at each tier crossing.
* **Stochastic removal**: `OuterLoop` assigns `RemoveProb` by quantile tiers
  (see algorithms/OuterLoop.md); `RemovePaths` draws against it. Use
  `numpy.random.Generator` seeded from config for reproducibility. Same for
  `GMDIndex = randi([1,1000])` (persisted, reused across runs — keep that).
* **Radial protection & degree locking**: lines at degree-1 buses are never
  removed; once synthetic degree-1 share ≥ 0.8× EI target, degree-2 buses
  lock too (MATLAB has an apparent bug: the ND3 block re-checks `== 2`, not
  `== 3` — see algorithms/OuterLoop.md).
* **L1Cost zeroing**: `Keep==1` paths get `L1Cost=0` so the regularizer stops
  pushing them to zero — the QP objective changes as locking progresses.
* **Table sort as state**: MATLAB sorts OLBranch by `MeanNorm` mid-OuterLoop;
  downstream code relies on row order. In Python keep an explicit sort key
  column; never rely on DataFrame row order.
* **`exist('X')` / try-load cascades** in main_SRTV are resume logic —
  replaced entirely by the checkpoint registry (§2).
* **feval dynamic dispatch**: `GMDData<Region>` (14 files) and
  `MakeLoadProfiles_<Region>` (15 files, ~8.3k loc) become data-driven single
  functions (region → data file, not region → code file).
* MATLAB `quantile` uses linear interpolation ≈ numpy `quantile` default —
  but verify `method=` matches where thresholds gate behavior.
* `Sub.Type = 1 + strcmp(Type,'Gen')` → PQ=1/PV=2 coding; slack chosen as
  FIRST generator bus (`PF.SlackList = GenList(1)`), then removed from
  GenList. Loads < 0.5 MW are floored to 0.5 MW + 0.5 MVar.
* Excel writes (`SubstationData.xlsx`, `QNCATest.xlsx`, stats sheets) are
  *reports*; the PowerWorld CSV writer is the format-critical one (reuse
  gridforge `io/writers.py` conventions).

## 7. Naming glossary

| MATLAB name | GridCrucible name |
|---|---|
| QNCA (QP Network Creation Algorithm) | GridCrucible pipeline |
| Inner Loop | OHM Loop (`OHMLoop`) |
| Middle Loop | TUNE Loop (`TUNELoop`) |
| Outer Loop (incl. MultiLevelProgram while) | TRIM Loop (`TRIMLoop`) + orchestrator iteration |
| `OLBranch` (overload/potential paths) | `model.candidates` |
| `Sub` / `Bus` tables | `model.substations` / `model.buses` |
| `PF` struct (SNet*, GenList, SlackList, LoadList, VBusSet) | `model.scenarios` (ScenarioSet) |
| `SNet` (complex net injection matrix) | injection scenario matrix (tier) |
| `Lam` / `gama` | `gammas` / `gamma` (L1 weight sweep) |
| `L1Cost` | `candidates.l1_cost` |
| `Keep` / `Radial` | `candidates.keep` / `is_radial` |
| BBR (branch-bus ratio) | density = n_candidates / n_buses |
| `MeanNorm` | `candidates.mean_norm` (normalized mean scenario flow) |
| `RateA` | `capacity_mva` |
| `MaxCondIndex` / `CondIndex` | `max_conductor_idx` / `conductor_idx` |
| `codectrl` + `options` + `InitOptions` | `CrucibleConfig` (TOML) |
| QNCAResults*.mat / PFData*.mat / BBR snapshots | checkpoint registry (`io/checkpoints.py`) |
