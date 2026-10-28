# WritePowerWorld.m / WriteLoads.m / WriteSubstationData.m / WriteMetrics.m → `io/writers.py`

Sources (all under `KCL-KVL Compliant Grid Generator/Matlab Functions/`):
`WritePowerWorld.m` (186 loc, **FORMAT-CRITICAL**), `WriteLoads.m` (24),
`WriteSubstationData.m` (31), `WriteMetrics.m` (68), plus the inline
`writetable` blocks of `main_SRTV.m` §8/§10/§11 (electrical/ACPF stats
sheets, SubstationData.xlsx/BusData.xlsx).
Written 2026-07-10 from a full read of all four sources and
every live call site.

## Call sites (what actually runs)

| Caller | Call | Notes |
|---|---|---|
| main_SRTV §11 (`codectrl.WriteFinalResults`) | `WriteLoads(ACPF,Sub,options)` then `WritePowerWorld(Bus,Sub,Branch,Gen,ACPF,options)` (KCL-KVL) and `WritePowerWorld(DC1Bus,DC1Bus,DC1Branch,Generators,PFGarver,options)` (KCL-Only), falling back to the `PF` struct in the catch cascade | **6 args to a 7-param function** — see Open Q1 |
| main_SRTV §11 (`codectrl.WriteBuses`) | `writetable(Sub,'SubstationData.xlsx')`, `writetable(Bus,'BusData.xlsx')` | plain table dumps |
| main_SRTV §11 (`codectrl.WriteMetrics`) | `WriteMetrics(params,Garverparams,Data,DataBus,options)`; catch → `WriteMetrics(params,params,...)` (self-compare → KCL columns zeroed) | |
| main_SRTV §5 | `WriteSubstationData(BusInputData,GenNat)` | pre-pipeline data snapshot |
| main_SRTV §8 (`codectrl.ElectricalMetrics && WriteMetrics`) | `writetable(PF.DCStats / PF.DCOPFStats, 'Synthetic <Region> Electrical DC {Power Flow,OPF} Statistics.xlsx')` | GarverRegion label preferred, Region fallback |
| main_SRTV §10 (`codectrl.WriteMetrics`) | `writetable(ACPF.Losses / ACPF.Stats, 'KCL-KVL <Region> AC Power Flow Statistics.xlsx', Sheet 1/2)` | inside a bare `try` |
| MultiLevelProgram :393 | `WritePowerWorld(Bus,Sub,Branch,Gen)` | 4 args — crashes at the Load section (Open Q1) |

## WritePowerWorld.m — the PowerWorld import CSVs

Signature: `[PWBus,PWBranch] = WritePowerWorld(Bus,Sub,Branch,Gen,PF,ACPF,options)`.
Emits up to six **`writetable`-style CSVs** (ONE header row of column names,
then data — NOT the `cell2csv` aux-section layout used by gridforge's
NationalConductorSizing export and by the orphaned AC inner-loop
`LineImport<N>.csv` debug dump, which prepend a `Branch` section row and use
the "From Number"/…/"Length" 14-column header. Both layouts import into
PowerWorld; this file is the writetable one. See Open Q2.)

Filenames: `<Section>Import<Network><Region>.csv` with
`Network ∈ {KCL-KVL, KCL-Only}` and `Region` = `cell2mat(options.Region)`;
each write is wrapped in try/catch that falls back to the suffix-less name
when `options` is missing (which, per Open Q1, is ALWAYS the case in the
live call sites — the reference outputs on disk are `BusImport.csv` etc.).

### 1. BusImport (gated `exist('Bus')`)

Column order: `Number, Name, NomkV, AreaNumber, ZoneNumber, NomB, NomG,
Latitude, Longitude[, Vpu, Vangle], SubName`.
* Number = `Bus.Num` (EXTERNAL bus number), Name = `Bus.Name`,
  NomkV = `Bus.kV`, AreaNumber = ZoneNumber = 1, NomB = NomG = 0,
  Latitude/Longitude = `Bus.Lat/Lon`.
* Vpu/Vangle only when `Bus.Vm/Va` exist (a bare `try`) — i.e. after an AC
  power flow wrote voltages back onto the Bus table. They sit BEFORE
  SubName in the column order.
* SubName = `Bus.Name` (the Sub-number column is commented out).

### 2. SubstationImport (always)

`Number, Name, Latitude, Longitude` from the Sub table.

### 3. BranchImport (always)

Column order: `BusNumFrom, BusNumTo, Circuit, BranchDeviceType, IsXF, R, X,
B, LimitMVAA, LimitMVAB, LimitMVAC, LineLength`.
* From/To = `Branch.From/To` — external bus numbers (the same space as
  BusImport.Number; MATLAB Branch keeps `From/To` external and
  `FromInt/ToInt` internal).
* Circuit = 1 for every row.
* BranchDeviceType = `Branch.DeviceType` ('Line' / 'Transformer').
* IsXF = 'YES' unless `strcmpi(DeviceType,'Line')` → 'NO'.
* R/X/B = `Branch.R/X/B` (per-unit).
* Limits = RateA / 1.1·RateA / 1.15·RateA. (Gridforge's SCALE export uses
  1/1.05/1.1 for lines and 1/1.25/1.5 for XFs — different ladder, keep
  each writer's own.)
* LineLength = `Branch.Length` **as stored (km)** — NO /1.61 mile
  conversion, unlike the LineImport debug dump and gridforge (Open Q3).

### 4. PeakLoadImport (always; source precedence + sign quirk)

Base columns `BusNum (=Bus.Num), ID (=1), Status ('Closed'), SMW, SMvar`;
rows with `SMW == 0` are DELETED at the end.
Source precedence:
1. `ACPF` non-empty → per-unit sniff (`Σ|real(SNet(:,1))|·40 < Σ Bus.MW`
   → `SNet *= options.SBase`), LoadBuses = `real(SNet(:,1)) > 0`
   (**load-positive convention**), SMW/SMvar = |real|/|imag|; ALSO adds
   `MW, Mvar` columns (= SMW/SMvar on the surviving rows).
2. else `PF.SNet` exists → same pu sniff, LoadBuses =
   `real(SNet(:,1)) < 0` (**net-injection convention** — note the sign
   FLIP vs the ACPF branch); NO MW/Mvar columns in this mode.
3. else → SMW/SMvar (+MW/Mvar) from `Bus.MW/MVar` (catch: Sub.MW/MVar).
Only scenario COLUMN 1 is used in modes 1-2.

### 5. GenImport (gated: Bus table has a `PNet` column)

Column order: `BusNum, ID, Status, FuelType, UnitType, MWMin, MWMax,
MvarMin, MvarMax, FixedCostInd, CubicB, CubicC, CostModel, MWSetPoint,
MvarSetpoint, VoltSet, AGC, AVR, PartFact`.
* BusNum = `Gen.BusNumExt` (external), ID = `Gen.ID`.
* Committed buses: `GenBusCommit = find(Bus.GenMWSet > 0.1)` (internal bus
  ids). `setdiff/intersect(Gen.BusNumInt, GenBusCommit)` → decommitted /
  committed indices — MATLAB set-ops return the FIRST unit per bus only
  (Open Q4): at a multi-unit bus, only the first unit gets Status='Open'
  (if decommitted) or the WHOLE bus's setpoint (if committed).
* Status 'Closed' default; 'Open' at the decommitted first-units.
* MWSetPoint = `abs(Bus.PNet(GenBusCommit))` at committed first-units,
  floored to 0 when < 0.1; MvarSetpoint = `abs(Bus.QNet(GenBusCommit))`.
* Cost: FixedCostInd/CubicB/CubicC = `Gen.A/B/C`, CostModel = 'Cubic'.
* VoltSet = 1, AGC = AVR = 'YES'.
* PartFact = `MWMax / Σ MWMax` (fleet-wide share).

### 6. SwitchedShuntsImport (gated: Bus table has a `QNet` column)

Rows = `find(Bus.SyncCond)`. Column order: `BusNum, ID, Status, RegBusNum,
ShuntMode, RegulationType, RegHigh, RegLow, RegTarget, BlockMvarPerStep1,
BlockNumberStep1, BlockMvarPerStep2, BlockNumberStep2, MvarNom`.
* BusNum = RegBusNum = `Bus.Num` (external) of the SyncCond buses; ID = 1,
  Status = 'Closed', ShuntMode = 'Continuous', RegulationType = 'Volt',
  RegHigh/RegLow = 1.06/0.94, RegTarget = 1.
* Split by sign of `Bus.QNet` (PowerWorld injection-positive): B-set =
  QNet > 0, G-set = QNet < 0 (rows with QNet == 0 fall in NEITHER set and
  keep 0 in the Block columns). With Nom = 1.3·|QNet|:
  Step1 = −ceil(Nom/10) on G / −1 on B (small opposite-direction range),
  Step2 = +1 on G / +ceil(Nom/10) on B; both BlockNumberStep = 10.
* MvarNom = +ceil(|QNet|) on B, −ceil(|QNet|) on G.

### Python port (`io/writers.py::write_powerworld`)

`write_powerworld(model, acpf, out_dir, network_label, *, region=None,
snet=None, load_sign=-1) -> dict[str, Path]` — one function, one file per
section, exact writetable headers/order above.
* Bus/Sub/Branch sections from `model.buses` (bus_id/external_num/name/
  voltage_kv/lat/lon), `model.substations`, `model.branches`
  (from_ext/to_ext from `create_branch`; DeviceType from `tap` ≠ 0;
  r_pu/b_pu columns optional → 0.0 when absent, X from x_pu else
  reactance_x).
* Vpu/Vangle from `acpf.v_mag/v_angle` (ACPFResult) when given.
* Loads: `snet` argument (complex vector/matrix, col 0 used) with
  `load_sign=+1` reproducing the ACPF branch and `-1` the PF branch;
  default `snet=None` = mode 3 (buses load_mw/load_mvar). pu sniff ×40
  verbatim against Σ load_mw.
* Gen/SwitchedShunts sections are emitted IFF `model.buses` carries the
  MATLAB-equivalent columns (`gen_mw_set`+`p_net`+`q_net` /
  `sync_cond`+`q_net`) — the same presence-gating MATLAB uses. Those
  columns are written by the final-ACPF/shunt stages (pipeline hook;
  shunts.py already writes `sync_cond`).
* The first-unit-per-bus set-op quirk is reproduced verbatim
  (`drop_duplicates` on the fleet's bus column before the set ops).

## WriteLoads.m — time-series load matrices

`WriteLoads(PF,Bus,options)`: takes `PF.SNet` (pu sniff ×40 as above,
column-1 sum vs `Σ Bus.MW`) else `PF.SNetComplete` (NO pu sniff on this
fallback), writes `abs(real(SNet))'` → `TimeSeriesMWLoad.csv` and
`abs(imag(SNet))'` → `TimeSeriesMVarLoad.csv` — TRANSPOSED (rows =
scenarios, cols = buses), headerless (`writematrix`). Note it is called
with `Sub` as the `Bus` arg — only `Bus.MW` is read.

Port: `write_loads(model, acpf, out_dir, *, snet=None) -> tuple[Path, Path]`;
`snet=None` → `model.scenarios.tiers['complete']` (the SNetComplete
fallback, no sniff); explicit `snet` → pu sniff verbatim.

## WriteSubstationData.m — SubstationGenLoadData.csv

Builds a Sub table from the RAW inputs and writes
`SubstationGenLoadData.csv` (writetable, header row). Column order:
`Num, Lat, Lon, Name, Type, NumInt, MW, MVar, MaxMW, MinMW, GenFuelType,
GenUnitType`.
* Num = external bus number; NumInt = 1..n (1-BASED row index — kept
  1-based in the CSV for file parity, Open Q5).
* Type = 1 (PQ) + `strcmp(Type,'Gen')` → 2 at gen substations.
* MW/MVar = real/imag of ComplexLoads.
* MaxMW/MinMW = `accumarray` of GenNat.MaxMW/MinMW per bus.
* GenFuelType/UnitType = the FIRST unit's values per gen bus, empty
  elsewhere. MATLAB assigns `Sub.GenFuelType(GenBusesMax) =
  GenNat.FuelType(FirstGen)` — sizes only match when every gen bus has a
  nonzero MaxMW total (Open Q6).

Port: `write_substation_data(model, out_dir)` from `model.substations` +
`model.generators` (first unit per `BusNum`). Companions
`write_bus_tables(model, out_dir)` = the `codectrl.WriteBuses` dumps
(SubstationData.xlsx / BusData.xlsx — plain table dumps of
substations/buses).

## WriteMetrics.m — 'KCL-KVL <Region> Statistics.xlsx' (5 sheets)

Deletes then rewrites the workbook:
* Sheet1 `params.Stats.Branch`, Sheet2 `params.Stats.Bus`, Sheet3
  `params.Stats.Length` — produced by **NetworkStatistics.m
  (GRIDFORGE-OWNED)**: Length = Volt/Total/Mean/Std/Quantile10/Median/
  Quantile90 (999 row = Σ 230..765), Branch = Volt/VoltPercent/GVAMiles,
  Bus = Voltage/Number/Percent. The port takes these as DataFrame
  arguments — it does NOT recompute them.
* Sheet4 `params.Stats.HistAreas` — the cycle-fit comparison table
  (`metrics/comparison.hist_areas`).
* Sheet5 `NDAreas` — one row per synthetic ND-fit variant: `Label_ND,
  KVL_Area, KVL_RSq, KCL_Area, KCL_RSq, EIBranchArea, EIBranchRSq,
  EIBusArea, EIBusRSq, WECCBranchArea, WECCBranchRSq, WECCBusArea,
  WECCBusRSq`. When `params == Garverparams` (the self-compare fallback)
  KCL_Area/KCL_RSq are zeros. MATLAB slices the REAL networks' ND arrays
  positionally (`Temp1(2:end-1)`, r² loop indexed by the SYNTHETIC count)
  — the RealismMetricsActual.md "Interface hazard"; the port joins by
  LABEL (`node_degree, sub_degree_lines, sub_degree, bus_degree_lines,
  bus_degree` ↔ the same keys in `realism_metrics_actual()['branch'/'bus']
  ['degree_fits']`), which also fixes the off-by-slice at both ends
  (Open Q7).

Port: `write_metrics(synthetic, garver, targets, config, out_dir, *,
branch_stats=None, bus_stats=None, length_stats=None) -> Path` where
`synthetic`/`garver` are `realism_metrics()`/`comparison_stats()` dicts and
`targets = {'EI': realism_metrics_actual('EI',...), 'WECC': ...}`.
HistAreas rows (MATLAB order): KCL/KVL, KCL-Only, `<case> All Branches` =
`targets[case]['bus']`, `<case> Positive Impedances` =
`targets[case]['branch']`.

Companions (the main_SRTV inline blocks):
* `write_electrical_stats(dc_stats, dcopf_stats, out_dir, region)` →
  `Synthetic <region> Electrical DC Power Flow Statistics.xlsx` +
  `... DC OPF Statistics.xlsx` (one sheet each; delete-first semantics =
  overwrite).
* `write_acpf_stats(acpf, out_dir, region)` →
  `KCL-KVL <region> AC Power Flow Statistics.xlsx`, Sheet1 = Losses,
  Sheet2 = Stats (skipped silently when `acpf.losses/stats` is None —
  MATLAB's bare try).

## Open questions (the author)

1. **WritePowerWorld arity bug**: the function takes 7 args
   `(Bus,Sub,Branch,Gen,PF,ACPF,options)` but EVERY live call passes 6
   (`...,Gen,ACPF,options`) — so inside, `PF`=the ACPF struct,
   `ACPF`=options, `options`=undefined. Consequences as-run: (a) all
   filenames fall to the suffix-less catch (`BusImport.csv`); (b) the
   Load section enters `~isempty(ACPF)` with ACPF=options and crashes on
   `options.SNet` — Bus/Substation/Branch CSVs are written, Load/Gen/
   Shunt sections never execute, and the error is swallowed by the §11
   catch cascade (whose LAST fallback call would propagate). The
   MultiLevelProgram 4-arg call crashes the same way. The Python port
   implements the INTENDED 7-arg behavior — confirm that's what you want
   (and whether the suffixed filenames should be the default).
2. Two PowerWorld CSV layouts exist in the MATLAB codebase: writetable
   (this file, WritePowerWorld headers) vs cell2csv aux-sections
   ('Branch' row + 'From Number'... 14 columns; AC inner-loop LineImport
   debug + gridforge SCALE export). Port keeps WritePowerWorld on the
   writetable layout — OK?
3. BranchImport.LineLength is written in km (`Branch.Length` raw); the
   LineImport debug dump and gridforge write miles (km/1.61). Which does
   the PowerWorld import definition expect?
4. GenImport: at multi-unit buses only the FIRST unit is opened /
   receives the whole bus setpoint (setdiff/intersect first-occurrence
   semantics). Port reproduces this. Intended?
5. SubstationGenLoadData.csv NumInt kept 1-based for parity with MATLAB
   output files — confirm (internal ids elsewhere are 0-based).
6. WriteSubstationData: the FirstGen/GenBusesMax alignment breaks if any
   gen bus has ΣMaxMW == 0 (MATLAB size-mismatch error). Python groups by
   bus explicitly, so it can't crash — flagging the latent MATLAB bug.
7. WriteMetrics slices real-network ND structs positionally
   (`Temp1(2:end-1)`, synthetic-count r² loops over real arrays. The real
   arrays have 6-7 entries vs the synthetic 5) — label-keyed join in the
   port changes which real fits land in which row vs MATLAB when the
   append order differed. Distribution-level parity assumed OK?
