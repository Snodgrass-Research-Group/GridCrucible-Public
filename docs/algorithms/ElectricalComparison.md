# ElectricalComparison.m + MatpowerDCOPF.m → `metrics/comparison.py` + `power/dcopf.py`

Sources: `Matlab Functions/ElectricalComparison.m` (213 loc),
`Matlab Functions/MatpowerDCOPF.m` (143 loc).
Written 2026-07-10 from a full read of both sources.

## Purpose

Cross-network electrical benchmark of the synthetic networks: serve each
network's DC scenarios on each topology and record (a) plain DC-PF
overload statistics (`DCStats`, computed by `DCPowerFlowFull` — already
converted as `power/power_flow.py` adapters) and (b) DC-OPF
feasibility/relaxation statistics (`DCOPFStats`, computed by
`MatpowerDCOPF`). Networks: **KCL/KVL** (this pipeline's output), **DC1**
(GridForge KCL-only homotopy) and **Garver** (GridForge KCL-only Garver
variant). Validation targets for the port exist in
`data/reference_run/pf_dcstats.parquet` / `pf_dcopfstats.parquet`.

## The comparison grid (ElectricalComparison.m)

Six network × load-scenario combos, each appending rows to the shared
`DCStats`/`DCOPFStats` tables, and **each wrapped in a bare try/catch that
silently swallows any failure** (`T1 = 1` no-op — port as logged skips):

| § | Network | Loads | Injection mapping |
|---|---|---|---|
| 1 | KCL/KVL | KCL/KVL | `PF.SNetFull` as-is |
| 2 | KCL/KVL | DC1 | if gen bus ids differ between pipelines: `accumarray(GarverBus.BusNumKVLInt, SNetComplete_col)` then zero-pad to the larger bus count; else direct copy + zero-pad |
| 3 | DC1 | DC1 | `PFDC1.SNetComplete` as-is |
| 4 | DC1 | KCL/KVL | mirror of §2 via `Bus.BusNumKCLInt`; when ids are equal it instead RELOADS `PFData.mat` (resume artifact — port reads the checkpoint) |
| 5 | Garver | KCL/KVL | same as §4 |
| 6 | Garver | DC1 | direct |

Common transforms per combo:
* **per-unit sniffing**: if `max(col sums of |SNet|) > 35e3` the matrix is
  taken to be MW and divided by SBase (magic threshold — keep verbatim,
  flag in port docstring).
* **zero-scenario drop**: columns with `Σ|SNet| < 100` deleted.
* combo labels carried as `Network` / `Loads` columns in both stats tables.
* After the grid: `CreateLoadScenarios` is re-run and `PF.SNet` reset to
  the full KCL/KVL set (state restoration, not analysis).
* An EI/WECC DCOPF section at the end is fully commented out (dead).

## MatpowerDCOPF.m per-scenario procedure (→ `power/dcopf.py`)

Setup: `ext2int`, incidence `ANet`, `YBusCalcDC` B-matrix + `condest`
printout (debug only), `find_islands` (result unused!), and a throwaway
"solver test" DCOPF with PMIN=0 / RATE_A=0 that only decides
MOSEK → MIPS fallback. Then for every scenario column i:

1. Load buses = `Re(SNet) < 0` → `PD = |Re|`, `QD = |Im|`; generators at
   buses with `Re(SNet) ≤ 0` are switched OUT (`GEN_STATUS=0, PMIN=0`).
2. Commitment feasibility guard: if `ΣPMIN > ΣPD`, scale all PMIN by
   `1/(1.2·ΣPMIN/ΣPD)`.
3. Solve chain on infeasibility: chosen solver → MIPS →
   relax all `RATE_A` by `RelaxMulti = max(1, 1 + |DCStats.OverloadPercentMax(i)|/100)`
   (NOTE the cross-coupling: the plain DC-PF stats for the SAME scenario
   must exist first) → MIPS again → `3×RelaxMulti` → give up.
4. If it took a relaxation to become feasible, a **tightening loop**
   shrinks `RATE_A` (×0.90 if current relax% > 100, ×0.95 if > 20,
   ×0.98 if > 10, else ×0.99) and re-solves until infeasible or the relax%
   goes ≤ 0; each SUCCESSFUL pass overwrites the recorded stats, so the
   final row reflects the **tightest feasible** limits found.
5. Recorded per scenario (`DCOPFStats` row): `Network`, `Loads`, `LoadMW`
   (ΣPD), `LimitRelaxPercent` (computed from **branch 1's** rating ratio
   only — all branches are scaled uniformly so it is representative),
   `GenMW` (ΣPG), `MaxAngleDiff`/`MeanAngleDiff` (|ANetᵀ·VA|, degrees),
   `Losses = SBase·Σ (|PF|/SBase)²·R` (I²R from DC flows — an
   approximation by construction), `LossPercent = 100·L/(ΣPD + L)`,
   `PercentLoadedMean/Std` = mean/std of `|PF|/RATE_A` **over branches
   with utilization < 1 only** (overloads excluded from the mean).
6. Solver reset to MOSEK before the next scenario.

## Bugs / quirks found (preserve or flag — do not silently fix)

1. The `if ~isfield(mpPF,'SNet')` branch (line 33) indexes with an
   undefined `i1` → would crash; unreachable in practice (SNet always set
   by callers). Port: require the scenario matrix, drop the branch.
2. `find_islands` result unused; `condest` printed and discarded.
3. `LimitRelaxPercent` uses branch 1 only (see above).
4. If even `3×RelaxMulti` fails, the scenario's stats row keeps whatever
   defaults the table fills (zeros) — no explicit failure marker. Port:
   record `success=False` explicitly.
5. §4/§5's `PFData.mat` reload when bus ids are equal — resume-artifact
   dependency; port reads the equivalent checkpoint/ScenarioSet instead.
6. try/catch swallowing entire combos: port logs the skip reason.

## Port mapping

* `power/dcopf.py — run_dcopf_stats(model_or_net, snet_mw, *, rate_a,
  labels, ...) -> pd.DataFrame`: pandapower `rundcopp` (ARCHITECTURE.md §2.2
  "pandapower everywhere"; the MOSEK/MIPS chain collapses to pandapower's
  solver + the relax/tighten ladder kept verbatim). Same-name stat columns
  as MATLAB so the reference parquets compare 1:1.
* `metrics/comparison.py — electrical_comparison(kvl_model, others:
  dict[str, ...], ...)`: the 6-combo grid with external-id remapping
  (`accumarray` → `np.bincount(weights=...)`), zero-scenario drop, pu
  sniffing; returns `(dc_stats, dcopf_stats)`.
* DC1/Garver inputs come from GridForge runs (GridModel/checkpoints);
  accept any object exposing buses/branches/generators + scenario matrix.

## Open questions (the author)

1. The 35e3 pu-sniff threshold and the Σ|SNet|<100 zero-scenario cutoff —
   confirm they can stay hard-coded (they interact with SBase=100).
2. `LimitRelaxPercent` from branch 1 only — intended (uniform scaling) or
   should the port report the actual uniform multiplier directly?
3. Are the DC1 AND Garver comparisons both still wanted, given GridForge
   is the only source of those cases now?
