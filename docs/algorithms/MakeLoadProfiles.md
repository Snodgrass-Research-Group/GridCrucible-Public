# MakeLoadProfiles_<Region>.m family + CreateLoadScenarios.m → `data/load_profiles.py` + `data/scenarios.py`

**Sources:** MakeLoadProfiles_<REGION>.m (15 regional files, ~8,350 loc;
SRTV = 670 loc analyzed in depth), smoothprofile.m (47 loc — ACTIVE, the
call-tree analyzer wrongly orphaned it because family pseudo-nodes hide
their callees; since fixed), CreateLoadScenarios.m (74 loc).

## What MakeLoadProfiles_SRTV does (the family skeleton)

Per state (AL/KY/MS/TN for SRTV):
1. **State hourly totals** from <State>.mat (1×8760) — migrated.
2. **NREL building shapes**: 17 reference-building CSVs per city
   (Birmingham/Louisville/...), each summed across columns (file 17 — the
   residential one — scaled ×1000). Day-normalized correlation +
   `clusterdata(R, 9)` is computed but **UNUSED** — the 17→10 aggregation
   is HARD-CODED per city (e.g. AL: {12,13,14}/3, {3,4,11}/3, ..., {17}).
   The Python port needs NO clustering — just the fixed group means.
3. **Sector construction** (from the xls hourly workbooks: AL_Houly_Load.xls
   etc.): state trend + hard-coded CIR (commercial/industrial/residential)
   fractions + selected NREL aggregate shapes + residential top-up →
   state_load(:, 1..3) per sector, 8760 hours.
4. **Bus distribution**: BusDist = xlsread('SRTV bus data.xlsx'), sector
   weights in cols 24-26: P_load(bus,:) = Σ_s (BusDist(bus,s)/CIR(s))
   state_load(:,s)ᵀ; Q via per-sector power factors (qfC/qfI/qfR).
5. **smoothprofile** per bus: a small QP that patches day-boundary jumps —
   quadratic correction pivoting over hours 20..24 / 1..5, minimizing
   (Σa)²+(Σb)² s.t. positive second derivative around midnight.
6. Output: [BusNums, P_load(8760)] and [BusNums, Q_load].

## Port strategy (two stages — decided 2026-07-08)

The builder's OUTPUT is deterministic per region: it is really a DATA
PRODUCT. So:
* **Bridge (now)**: tools/export_opaque_tables.m v3 runs
  `MakeLoadProfiles_<R>` in MATLAB and saves P_Load/Q_Load as **-v6 .mat**
  (plain doubles — scipy-readable, no CSV bloat); the migration converts to
  `data/load_profiles/<REGION>_{p,q}.parquet` (+ bus_nums). Runtime
  `make_load_profiles(region)` READS the cache. Exact parity with MATLAB by
  construction.
* **Full port (dedicated session, tracked)**: rebuild the pipeline in Python
  from the migrated inputs (state totals ✓, NREL CSVs, xls workbooks, bus
  xlsx) so the cache can be regenerated without MATLAB. The hard-coded
  aggregations + CIR constants get lifted into a per-region TOML/JSON spec
  (region → data, ARCHITECTURE.md §6). `smooth_profile` (the QP) is ported NOW —
  it is small, self-contained, and the only genuine algorithm in the chain.

## CreateLoadScenarios.m → `scenarios.build_load_scenarios`

Per region: call the profile builder; then
* if column 0 of P_Load is a contiguous bus-number sequence
  (`mode(diff(col0))==1`): strip it; if the numbering has ≥2 distinct gaps
  ("Transitions"), scatter rows into a zero matrix over the region's buses
  via `intersect(LocalRegionBuses, ZachLoadBuses, 'stable')`; else prepend
  zero rows for the region's GENERATOR buses (profiles cover load buses
  only).
* if the model's bus set is smaller than the region's, keep only rows for
  buses present (`intersect(LocalRegionBuses, Bus.Num, 'stable')`).
* concatenate regions → (n_buses × 8760) P and Q.

## SUPERSEDED: the pure-Python builder lives in `loadsmith` (2026-07-11)

The full MakeLoadProfiles conversion (all 13 regions) lives in the separate
**LoadSmith** package (https://github.com/Snodgrass-Research-Group/LoadSmith-Public:
stage modules ingest/clustering/fitting/distribution/smoothing/outputs plus
per-region configs and its own algorithm notes). This doc remains the spec for gridcrucible's BRIDGE (`make_load_profiles`
npz cache) and for CreateLoadScenarios; the builder Port-strategy section
below is kept for reference but must NOT be re-implemented here.
Feeding LoadSmith outputs into the npz cache format is not wired yet.

## Data quality (real SRTV bridge export, verified 2026-07-10)

The v4 bridge export (`SRTV_LoadProfiles.mat` → `load_profiles/SRTV.npz`,
4451 buses × [busnum + 8760 hours]) contains **NaNs inherited from the raw
state data**: Kentucky.mat `KY` is NaN at 1-based hours **1587 and 7298**,
Tennessee.mat `TN` at hour **699**. MakeLoadProfiles_SRTV propagates them
into every KY/TN bus row (2,611 buses, 3,726 cells each in P and Q) — so
the author's own MATLAB runs carried these NaNs too.

Downstream this is benign *by MATLAB semantics*, which the port mirrors:
* `CreateSNet` hour selection (`[~,h] = max/min(sum(P_Load))`) — MATLAB
  max/min OMIT NaN; `_select_scenario_hours` uses `np.nanargmin/nanargmax`
  (plain `argmin/argmax` would return the first NaN column — hour 699 — for
  BOTH windows). Real hours: peak = 4187 (0-based, late June), min = 7439
  (November); the NaN hours fall in neither ±60 window, so the selected
  scenarios are NaN-free on the real data. The `nan_to_num` guard on the
  selected columns stays as a safety net for other regions/years.

Additionally, a handful of cells are slightly **negative** (P: 798 cells of
39M, min −5.9 MW across 382 buses; Q: 987 cells, min −1.5 MVar): the
`smoothprofile` day-boundary corrections dip small night loads below zero.
This is genuine MATLAB output; CreateSNet's low-load floor (0.2 + 0.12j in
the max scenario) and the dispatch QPs tolerate it exactly as MATLAB did.

## Open questions

1. SRTV BusDist columns 24-26 are magic indices into 'SRTV bus data.xlsx' —
   capture the sheet layout in the per-region spec during the full port.
2. MakeLoadProfiles comment says NY/NYSEG profiles as an alternative source
   (commented out) — dead path, not ported.
3. smoothprofile hard-codes 3 channels; MakeLoadProfiles calls it on ONE
   row at a time (P then Q separately) — MATLAB implicitly broadcasts;
   ported as n-channel.
