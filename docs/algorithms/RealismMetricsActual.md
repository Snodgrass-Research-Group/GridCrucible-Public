# RealismMetricsActualEI.m / RealismMetricsActualWECC.m → `metrics/comparison.py`

Sources: `RealismMetricsActualEI.m` (919 loc), `RealismMetricsActualWECC.m`
(891 loc). Written 2026-07-10 from a full read of the
EI variant + a normalized diff of the WECC variant (region rename aside,
only 3 real differences — §WECC below). `RealismMetricsComparison.m` is a
separate 888-loc read (this doc will grow a section when it's read).

## Purpose

Compute the REAL-network statistics the synthetic-network metrics are
compared against, for two case flavors each:
* **Branch-level** (`Data.<R>.Stats`): the sanitized "Branch" case —
  non-realistic branches removed, largest island only (`find_islands` +
  `extract_islands` → our `reference_cases/<R>BranchSanitized`).
* **Bus-level** (`DataBus.<R>.Stats`): the "Bus" case — every branch
  element retained incl. transformer windings
  (`<R>LargeIslandSanitized`).

MATLAB caches everything in `Data.mat` (isfield guards + repeated saves);
the port computes-and-caches to parquet/npz under `data_dir` keyed by case.

## Computation blocks

1. **Setup per flavor**: largest island, incidence `ANet`, B-matrix +
   `condest` printout (debug only; NOTE the YBusCalcDC call passes
   `branch(:,T_BUS)` where reactances belong — a positional BUG, results
   discarded so harmless), `BBR = n_branch/n_bus`.
2. **Sogol metrics** (gated `codectrl.Stats715Sogol{Branch,Bus}`):
   eigencentrality, fiedler vector, TIER (PC-only guard), singular values
   + normalized laplacian (only when < 20k buses), nodedegre — ALL already
   converted in `metrics/graph_metrics.py`; the port just wires them.
3. **The ND-fit block** — one helper applied ~11×:
   `histcounts(deg, 1..max+1)` → dist = counts/Σ → truncate at the FIRST
   zero bin → `poly1` fit of `log10(dist)` vs degree → record
   `{Label, NumLines, GOF (r², sse), Coeff (slope, intercept),
   Area = trapz(x, fitline(x))}` appended to `Stats.ND`.
   Variants (Branch flavor): NodeDegree (incidence; fitted TWICE — first
   over the full range without zero-truncation into ND(1) whose Area is
   only set under the plot flag — port the truncated one, flag the dupe),
   SubDegreeLines (`Substations.CountLine` — from ei_substations.parquet),
   SubDegree (`CountConnections`), BusDegreeLines (`Bus.CountLine`, drop
   ≤0), BusDegree (`Bus.Neighbors`), BusDegreeInService
   (`Bus.NeighborsInservice`). Bus flavor: same list against the Bus case.
4. **Cycle-basis stats** per flavor: graph → `cyclebasisport` (converted:
   networkx `cycle_basis`) → cycle lengths → mean + histogram (MATLAB
   auto-binning — port with numpy 'auto' bins and document the bin-edge
   difference; validation is distribution-level, ARCHITECTURE.md §2.8) → bin
   MIDPOINTS → normalized over bins < 80 → `exp1` (a·e^(bx)) and `power2`
   (a·x^b + c) fits with GOF + confint (scipy `curve_fit` + delta-method
   CIs). `HistAreas` rows (trapz of fit curves) are only appended under
   `PlotMetricsRealNetworks` — quirk, port unconditionally.
5. **Form 715 MVA-limit histograms** per kV class (69/138/230/345/500/765):
   lines = `Data.<R>.Branch` rows with `IsXF=='No'`, zero/NaN lengths
   dropped, Nom kV snapped to the NEAREST class; per class, histogram of
   `LimitMVAA` within hard-coded outlier bounds LowMVA/HighMVA =
   [24..3601]/[154..9275], fixed bin counts MVABins = [27;20;20;12;8;6].
   **BUG (EI only)**: the transformer-MVA histogram loop that follows
   writes into the SAME `Form715.MVAHist(i1)` slots, silently overwriting
   the line histograms — port to separate `MVAHistXF`, flag for the author.
   Distribution transformers (kVFrom==69 & kVTo≠0 — sic, a truthiness
   filter) are removed before the XF histogram.
6. **Length stats per class** (`LineLengthXBMiles`): Total/Mean/Std/
   Q10/Median/Q90, NaN→0, round(…,2); the 999 row = Σ classes 230..765
   (indices 3:6). **Branch stats**: VoltPercent, GVAMiles =
   Σ(length·LimitMVAA)/1000, Count; 999 row likewise.
7. **Bus voltage-class counts** for both matpower cases (bus BASE_KV
   snapped to nearest class IN PLACE on the loaded case — the port must
   not mutate shared case objects).
8. **Form 1 stats**: blocks 6-7 applied to `Data.Form1`
   (form1.parquet; kV snapped to classes).
9. Excel writers (`EI Statistics.xlsx` 5 sheets, `Form 1 Statistics.xlsx`)
   → `io/writers.py`.

## WECC differences (the only real ones)

1. No transformer handling at all in the Form-715 blocks (its branch
   table has no IsXF/XF rows) — and `kVTo` is snapped from `NomkVFrom`
   (its table lacks NomkVTo).
2. `HistAreas` init guarded by the plot flag.
3. Everything else is the EI code with `EI`→`WECC`.

Port as ONE parameterized function
`realism_metrics_actual(case: 'EI'|'WECC', data_dir, *, sogol=..., plots=None)`
reading the migrated parquets (ei/wecc_substations, ei/wecc_branch,
ei/wecc_buses, form1) + `reference_cases/*.npz`.

## RealismMetricsComparison.m (888 loc, read 2026-07-10)

Runs on a SYNTHETIC network's branch table (called from
NetworkGraphComparison for the KCL-only Garver/DC1 networks; `params`
carries the KVL network's already-computed stats for overlays) and
produces the comparison-side stats + figures:

* **Cycle basis**: same pipeline as the Actual variants but with FIXED
  bins `2:2:80` (Actual uses MATLAB auto-binning) and a THIRD fit —
  `weibull` (a·b·x^(b−1)·e^(−a·x^b)) — alongside exp1/power2.
* **ND-fit block ×5** on: NodeDegree (incidence), SubDegreeLines
  (`Tap==0` rows only), SubDegree, BusDegreeLines (`Tap==0`), BusDegree
  (NOTE: this one bins with `histcounts(..., nodes-1)` — a bin-COUNT
  argument, unlike every other block's explicit edges).
* **MVA histograms** per kV class of `RateA` where `kVFrom==kVTo==class`,
  fixed MVABins counts; normalized values kept for the EI-overlay plots.
* **HistAreas** comparison table (rows: KCL/KVL, KCL-Only, EI All
  Branches, EI Positive Impedances, WECC All Branches, WECC Positive
  Impedances; cols: Exp_RSq/Area_Exp/Power_RSq/Area_Power) — filled
  inside plot-flag blocks (same quirk as the Actual variants).
* All overlay figures: cycle-fit curves and ND fit lines vs
  EI/WECC, MVA histogram overlays; figure code moves to `viz/plots.py`.

**Interface hazard for the port**: the ND overlay plots address the ND
struct arrays by FIXED POSITION (SynIndex/RealIndex pairs 1/2, 2/3, 3/4,
4/5, 5/6+7), assuming the exact append order of RealismMetrics (synthetic)
vs RealismMetricsActual* (real). The port must key ND fits by LABEL
(dict), never by position.

**Latent crash**: the `codectrl.StatsSogol` branch (line 21) calls
`Table2Matpower(Bus, GarverBranch, Gen, Sub, ...)` with `Bus`/`Gen`/`Sub`
UNDEFINED in this function's scope — enabling StatsSogol here would error
(same class of never-exercised path as ACInnerLoopOptim).

## Form715LimitFits provenance — RESOLVED 2026-07-10

The v4-exported **Form715LimitFits** normal distributions
(Fit69Form715…Fit765Form715, now form715limitfits_params.parquet) have
**no consumer in the KVL codebase** — the KVL folder's copy of the .mat is
vestigial. The real consumer is in the **KCL-only pipeline**
(gridforge's source): `KCL-Compliant Grid Generator/
NationalConductorSizing.m:147` loads the .mat and
`Matlab Functions/GenerateTargetDistribution.m` draws target line-rating
distributions from `Fit69Form715`/`Fit138Form715` (only those two classes;
138 kV draws sorted, floored at 1) for national conductor sizing.

**Discrepancy flagged for the gridforge side**: gridforge's
`data_store/form715/fits.json` REFITS normals over the RAW Form 715
samples, while the .mat fits were built on **outlier-filtered** limits —
the parameters differ substantially (69 kV: exported μ=52.6/σ=16.4 vs
refit μ=80.3/σ=255.8; every class disagrees). gridforge's own
`data_store/exports/Data/Form715LimitFits__Fit*.parquet` already carry the
EXACT exported parameters (verified equal to our v4 export to 4+ decimals)
— its NationalConductorSizing consumer should draw from those, not the
raw refit. GridForge-side fix; recorded here because the discovery came
from the KVL-side export cross-check.

## Open questions (the author)

1. ~~Where should the Form715LimitFits normal fits plug in?~~ RESOLVED —
   KCL-pipeline consumer (see §Form715LimitFits provenance); the open
   piece is the gridforge fits.json filtered-vs-raw discrepancy above.
2. EI transformer-MVA histograms overwrite the line-MVA histograms
   (same struct slots) — separate them in the port?
3. The YBusCalcDC call passes `branch(:,T_BUS)` as the reactance vector
   (result only feeds a `condest` printout) — confirm dead/debug.
4. MATLAB `histogram` auto-binning for cycle lengths: is bin-edge parity
   required, or is distribution-level equivalence enough (numpy 'auto')?
