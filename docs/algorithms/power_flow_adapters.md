# Power-flow adapters → `power/pandapower_io.py` + `power/power_flow.py`

**Sources:** Table2Matpower.m (84), DCPowerFlowSimple.m (64), DCPowerFlowFull.m
(101), ACPowerFlowSimple.m (45), YBusCalcDC.m (15); contracts from
gridforge-owned YBusCalc.m / ACPowerFlowFinal.m call sites.
"pandapower everywhere" (ARCHITECTURE.md §2.2).

## What the MATLAB does

### YBusCalcDC (→ `b_matrix`)
`BBus = A·diag(1./x)·Aᵀ` (n_buses² sparse). Contingency k > 0: zero row k of
the primitive susceptance (line open). NOTE: the returned `ACon` drops the
column but `BBus` is built from the FULL A with the zeroed primitive — port
the zeroed-primitive semantics.

### DCPowerFlowSimple (→ `run_dc_pf`)
1. Build/reuse BBus.
2. Singularity guards → `SingularB`: any zero-degree bus; `condest(BBus_red)
   > 1e16`; `sprank(A) < n−1`. Python: exact test — connected_components on
   the branch graph == 1 AND every bus has degree ≥ 1 (raises
   `SingularNetworkError`; TRIM catches it, matching MATLAB's flag checks).
3. **Slack = bus 0 unconditionally** (θ₀=0; `BBus(2:,2:)\P(2:)`) — NOT
   PF.SlackList. Keep exactly (candidate-graph solves rely on it).
4. `PLineSend = SBase·(B′Aᵀ)·θ` MW, signed, one per branch/candidate row.
5. Extras: VBusPolar (Vm≡1, Va in degrees), `AngleDifference = |Aᵀ·Va_deg|`
   (DEGREES — used by TUNE angle checks) → helper `angle_differences_deg`.

### DCPowerFlowFull (→ `run_dc_pf_full`)
Same solve per scenario column of PF.SNet, plus a `DCStats` row per scenario:
GenMW = SBase·Σ P⁺, LoadMW = SBase·Σ|P⁻|, Max/MeanAngleDiff (deg),
`Losses = SBase·Σ (p_pu²·R)` (I²R at V≈1), LossPercent = 100·L/(ΣMW/2 + L),
PercentLoadedMean/Std over NON-overloaded lines, LinesOverloaded count + %,
OverloadPercentMax/Mean/Std (overloaded lines only, minus 100). Reproduce the
formulas verbatim — they feed ElectricalComparison's Excel sheets.

### ACPowerFlowSimple / ACPowerFlowFinal call sites (→ `run_ac_pf`)
YBus (R/X/B, taps) → `pfsolve` Newton-Raphson → per-line
`flows = max(|V_f·conj(I)|, |V_t·conj(I)|)·SBase`. **pfsolve.m is a LOST
FILE** (never committed) — pandapower `runpp` is the only implementation
path, not merely the chosen one. `enforce_q_limits=True` reproduces
PVPQSwitching. Output harvested into ACPFResult: v_mag (pu), v_angle (rad),
flows_mva = max(from,to) apparent power per branch IN MODEL ROW ORDER,
losses/stats frames.

### Table2Matpower (→ `to_pandapower`)
mpc mapping worth keeping: bus types (1 PQ / 2 PV / 3 slack), slack = bus of
the MAX-capacity generator (fallback bus 0); Vm/Va start 1/0; VMAX/VMIN
1.06/0.94; loads PD/QD from Bus.MW else Sub.MW; branch R/X/B per-unit on
SBase=100, RateA, Tap; gen QMIN/QMAX, PMIN/PMAX, VG=1; gencost poly-2
(c₂=A, c₁=B, c₀=C). The `savecase(...)` side effect is NOT ported (writers
own exports).

## pandapower mapping decisions

* Buses: `vn_kv` from voltage_kv (candidates: kv_from). Geodata from lat/lon.
* Branches with `tap == 0` and same end-kV → `line` elements with per-unit →
  ohmic conversion: `r_ohm = r_pu·z_base`, `x_ohm = x_pu·z_base`,
  `c_nf = b_pu / (ω·z_base)·1e9`, `z_base = vn_kv²/s_base`, length 1 km,
  `max_i_ka = capacity_mva/(√3·vn_kv)`. Branches with tap ≠ 0 or differing
  end-kV → `trafo` elements (sn_mva = capacity, vk from x_pu on capacity
  base). Candidate tables (TRIM phase, X-only) get r_pu=0 lines — legal for
  `rundcpp`; `runpp` on candidates is not a supported combination (assert).
* **Row identity:** every created element carries its model row index in a
  `crucible_row` column (preserved by pp to_json). `from_pandapower` and flow
  harvesting ALWAYS reorder to model row order via that column — pandapower
  splits rows across line/trafo tables, order is NOT preserved otherwise.
* Slack: `ext_grid` at `scenarios.slack_bus` if set, else max-gen_max_mw bus
  (Table2Matpower rule), vm_pu from v_setpoints (default 1.02).
* Generators: `gen` (PV) with vm_pu = 1.02, p_mw from scenario dispatch or 0;
  poly costs attached via `pp.create_poly_cost` (cost_a/b/c) for later OPF.
* Loads: one `load` per bus with load_mw/mvar (or scenario column).
* `run_dc_pf` engines: `"pandapower"` (default; net build + `rundcpp`) and
  `"direct"` (sparse `splu` on b_matrix — bit-identical math, used inside OHM
  seeding where a net rebuild per call would dominate runtime). A test pins
  agreement ≤ 1e-8 MW between engines. This is the documented revisit point
  from the "pandapower everywhere" decision.

## Contracts (implemented)

```python
b_matrix(A, x_pu, contingency=None) -> csc_matrix          # YBusCalcDC
run_dc_pf(model, snet_mw, table="candidates", engine=...)  # -> DCPFResult(theta_rad, flows_mw)
run_dc_pf_full(model, snet_mw_matrix, ...) -> (flows, DCStats DataFrame)
run_ac_pf(model, scenario=None, enforce_q_limits=True) -> ACPFResult
angle_differences_deg(A, theta_rad) -> ndarray             # |Aᵀ·θ| in degrees
to_pandapower(model, table, scenario) -> pandapowerNet     # Table2Matpower
from_pandapower(net, n_branches) -> ACPFResult
load_reference_case(name, data_dir) -> pandapowerNet
SingularNetworkError                                        # replaces SingularB flag
```

## Validation

* tiny_model 2-bus hand calc + ring: KCL at every bus (A@p == P), flow
  antisymmetry, engine agreement (direct vs rundcpp).
* WECCLargeIsland reference case: rundcpp + runpp converge (done in 
  for DC; AC checked here).
* DCStats formulas: loss formula on a hand-solvable case.

## Open questions

1. DCPowerFlowSimple treats bus 0 as slack even when PF.SlackList differs —
   intentional for candidate graphs? (Assumed yes; flagged for the author.)
2. ACPowerFlowSimple returns max(|send|,|rec|) — ACPowerFlowFinal's exact
   output contract should be cross-checked against gridforge's power_flow.md
   when its conversion lands (adapter signatures kept compatible).
