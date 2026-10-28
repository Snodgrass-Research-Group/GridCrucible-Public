# CreateSNet.m → `data/scenarios.py` (build_snet)

**Source:** CreateSNet.m (585 loc). The last
prerequisite for the orchestrator.

## Purpose

Builds `PF.SNet*` — the five net-injection scenario tiers the whole TRIM
iteration consumes: **Full** (every dispatch column) and the subsampled
**Complete / Advanced / Intermediate / Basic** (BBR-tier selection, dense
networks use fewer scenarios).

## Pipeline (per dispatch region; 'Full' mode = one pass over all regions)

1. **Bus gen aggregates**: per-bus Max/Min MW/MVar via accumarray of the
   fleet; fuel/unit type from the first unit. **MATLAB BUG (lines 27-28):
   Bus.MaxMVar/MinMVar are assigned MaxMW/MinMW values** — Python implements
   the evident intent (MVar from MVar); flagged for the author.
2. **Load hours**: regional profiles (same routing as CreateLoadScenarios —
   shared helper), then scenario selection: **min-load hour ±60** (121 cols)
   + **peak hour ±60** (121 cols) → 242 complex columns.
   WinterWeek (hours 912:1079) and the max-ramp week are computed but
   **NEVER USED** (dead since the LoadScenariosRegional concat excludes
   them) — not ported; noted for the author. NaN→0.
   'sine' LoadData alternative: 0.3·sin+0.7 scaling of Bus loads.
3. **Import-bus schedule** (thirds of the scenario set): 0 / −mean(gen max)
   (export) / +mean (import).
4. **Low-load floor**: loads < 0.2 MW in the max scenario → 0.2 + 0.12j.
5. **Load-scaling guards**: capacity−⅔·hydro ≤ 1.01·peak → scale down to
   capacity; Σmin ≥ 0.99·peak → scale up to 1.01·Σmin.
6. **Dispatch blocks** into SNetFull (col = accumarray(commit bus, dispatch)
   − load): ED block (UCED — GRIDFORGE-OWNED, dependency-injected), CD block
   (UCCD, ours). Hydro scheduling by scenario third (⅔ / 1 / ⅓) when
   GenLoadRatio > 1.2. Optional **gen N-1 blocks**: for the NumCon largest
   gen buses (> GenConSize MW): drop that plant, rescale load if the guards
   trip, ED per scenario. UCScenarios bookkeeping ported verbatim.
7. **Bus.MW/MVar ← peak scenario** (with the same scaling guards) — the
   substation loads the rest of the pipeline sees.
8. GenImport.csv / LoadImportBase.csv writers are NOT ported here (
   io/writers — PowerWorld formats).

## Tier subsampling (the heart)

Per UC block (each LoadScenarios columns of SNetFull), greedy dedup by
**plant-commitment signature**: committed = gen-bus rows with injection
> 0.1; DifferingGen(s) = # plants differing from the block's first scenario;
all scenarios within `DiffNum` plants are "matching" → keep ONLY the one
with max total gen, drop the rest; repeat until the block is empty.

DiffNum (block 1 = ED): by system size 0 (<10k buses) /1/2/3 (≥20k), +1 for
Regional dispatch >10k. Block 2 (CD): 0 if GenLoadRatio<1.1 or RFCW; 2 if
<2000 buses or GLR<1.2 or >2 regions Regional; 4/6/8 by size; Regional>10k →
ED's+1. Other blocks (N-1): 0. Tiers: **Complete** = DiffNum, **Advanced** =
+1, **Intermediate** = +2 (alias of Advanced when <4000 buses), **Basic** =
+3 (alias of Intermediate when <2000 buses).

## Python interface

```python
build_snet(model, config, bus_input, profile_fn=make_load_profiles,
           uced_fn=None, gurobi_env=None) -> ScenarioSet
```
* Requires model.scenarios.slack_bus/gen_buses/load_buses pre-populated
  (setup_generators) — the commitment signature reads those rows.
* `uced_fn(fleet, hydro_scaling, load_mw, reserve) -> (x, fleet_out)`:
  gridforge's UCED once converted (resolved lazily with a clear error);
  tests inject a stub. UCCD comes from data.dispatch.
* Scenario-window clamp: min/peak ±60 h clamped to [0, 8760) with a warning
  (MATLAB would index-error on profiles peaking in the first/last 60 h).

## Open questions

1. The MaxMVar←MaxMW bug (above) — confirm intent.
2. Dead WinterWeek/PeakRamp selections — resurrect or delete?
3. `Bus.MaxMVar` bug also feeds the import-bus schedule magnitude
   (mean(Bus.MaxMW) — unaffected) — verified no other consumer of the MVar
   aggregates inside this file.
