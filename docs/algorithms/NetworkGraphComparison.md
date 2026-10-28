# NetworkGraphComparison.m → `metrics/comparison.py`

Source: `Matlab Functions/NetworkGraphComparison.m` (453 loc).
Written 2026-07-10 from a full read.

## Purpose

Adapter + metrics driver for the **KCL-only pipeline's outputs** (what
GridForge produces): load the region's *Garver* network and *DC1*
(DC-homotopy-1) network with their ED/CD dispatch scenarios, rebuild them
as Branch/Bus tables in the KVL pipeline's schema, establish the KCL↔KVL
bus-number mappings that `ElectricalComparison` needs, and run the metrics
stack (`NetworkStatistics` → `RealismMetrics` → `RealismMetricsComparison`
→ `PlotNodeDegreesComparison`) on each KCL-only network. Everything is
wrapped in try/catch fallback cascades — the MATLAB function is
data-archaeology over ad-hoc .mat files; **the Python port replaces all of
that with reading a GridForge run directory/checkpoint** and only ports
the semantic content below.

## Inputs (MATLAB .mat cascade → GridForge-run equivalents)

| .mat | Content | Python source |
|---|---|---|
| `<R>Garver.mat` | Garver-stage lines: FromNet(/Regional/National), ToNet, LineWeight (km!), LineCapacity, optional Transformers | gridforge Garver checkpoint |
| `<R>DC1.mat` | DC1-stage `Branches` table (fully conductor-assigned: LengthMiles, GMDIndex/GMDFeet, CapacityMVA, ConductorIndex(Final), VoltagekV, SizeKcmil, R/X/Bpu, CondPerPhase, Circuit) + optional Transformers | gridforge DC1 checkpoint |
| `<R>SNetED/SNetCD.mat` | dispatch scenario vectors (SNetDC / SNetDCConverse; CD falls back to ED when absent/all-zero) | gridforge scenario outputs |
| `<R>BusNumName/BusData/NetInjection/GeneratorsRegional.mat` | bus id/name/coords, net injection, regional fleet | gridforge model tables |
| `BusMapping.mat` | KCLBus ↔ KVLBus pairs | must be exported/derived (open Q1) |

## Semantic content to port

1. **Garver branch parameterization** (Garver lines carry only capacity):
   assign voltages by length like CreateOLMatrices' descending-kV init
   (`options.MaxLength(end)` is first RAISED to cover the longest line
   +5 mi); `AssignConductors` picks the conductor with the MVA rating
   closest to the KCL-only line capacity; `RXB` computes impedances.
   GMD comes from DC1's table when available else
   `ceil(100·rand)` (NOTE: **100**, not the usual 1000 — flag).
2. **Internal renumbering**: BusesKCL = unique(From∪To), sequential
   NumInt; per-bus kV = max incident `kVFrom`; regional fleet filtered to
   buses present, `BusNumInt` remapped.
3. **Transformer append blocks** (both networks): copy-a-slice-of-branch
   -table pattern with `Tap=1`, `R/Xpu` from the Transformers table.
   **BUGS to flag, port the intent**: `To = Transformers.FromBus` (To
   should be the secondary bus) and `kVFrom` assigned twice
   (Primary then overwritten by Secondary) leaving `kVTo` stale.
4. **KCL↔KVL mappings**: `Sub.BusNumKCL = KVL2KCL(Sub.Num)`,
   `Bus.BusNumKCL/KVL`, `Gen.BusNumKCL/KVL`,
   `GarverBus/DC1Bus.BusNumKVL(Int)` — the id columns ElectricalComparison
   §2/§4/§5 remaps rely on.
5. **PF structs per network**: aggregate NetInjection/SNetED/SNetCD to the
   network's bus set when longer (accumarray via BusData.NumIntKCL);
   GenBuses = injection > 0, slack = FIRST gen bus, LoadList = rest;
   `SNetComplete = −[SNetED, SNetCD]` (sign flip to load convention);
   per-unit sniff: `Σ|SNetED| < 20000 → ×SBase` (**different threshold**
   than ElectricalComparison's 35e3 — keep both verbatim). Bus table gets
   Type (2 gen / 1 load), MW/MVar = |load part|, kV.
6. **Metrics driver**: params {Scenarios: 2, TotalLines/Paths, buses,
   nodes, Stats.NodeDegree = diag(A·Aᵀ)}; then NetworkStatistics
   (gridforge-owned), RealismMetrics (converted, `metrics/realism.py`),
   RealismMetricsComparison (this module), and
   PlotNodeDegreesComparison + Excel stat workbooks.
   `codectrl.PlotMetricsRealNetworks` forced 0 for the Garver pass.

## Port shape

```python
def load_kcl_network(run_dir_or_model, catalog, config, *, stage="dc1")
    -> tuple[CrucibleModel-like, ScenarioSet-like]   # per network
def network_graph_comparison(kvl_model, kcl_garver, kcl_dc1, config)
    -> dict[str, params-like]                        # metrics per network
```
Bus-mapping columns land on the KVL model's tables exactly as in MATLAB so
`electrical_comparison` can consume them unchanged.

## BusMapping — RESOLVED 2026-07-10

`BusMapping.mat` is **absent from the repo** (it lived in the MATLAB
working directory only), and it is **fully derivable**:
`BusInputDataGarver.csv` (KCL-pipeline numbering) and `BusInputData.csv`
(KVL numbering) contain the SAME 75,432 buses — names unique in both
files, 100% overlap, `Num == row+1` in both, State columns agree perfectly
on the name join, ~9.5% of buses sit at a different row (hence the need
for a mapping at all). The mapping IS the BusName join.

Implemented as `metrics/comparison.kcl_kvl_bus_map(data_dir)` →
DataFrame [KCLBus, KVLBus, BusName] (1-based external ids, the MATLAB
table's columns); `BusInputDataGarver.csv` migrates to
`bus_input_data_garver.parquet` (matlab_import `_CSV_SOURCES`, always-run
incremental pickup). The join validates one-to-one and raises if the two
files ever diverge.

## Open questions (the author)
2. Garver GMD draw uses `ceil(100·rand)` — 100 vs the 1000-row GMD
   tables: intentional (low-GMD bias) or typo?
3. Transformer-append bugs (To=FromBus, kVFrom double assignment): port
   the intent (To=secondary bus, kVFrom=primary/kVTo=secondary)?
4. The 20000 pu-sniff here vs 35e3 in MatpowerDCOPF — confirm keeping
   both magic numbers as-is.
