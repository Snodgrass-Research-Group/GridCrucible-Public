# CreateBuses.m (+CreateBranch, TrimQNCALines) → `network/buses.py`

**Sources:** CreateBuses.m (460 loc), CreateBranch.m (11), TrimQNCALines.m (40)
Pipeline stage 6 (after TRIM convergence).

## CreateBuses: substation → electrical-bus expansion

1. **Init**: one bus per substation; on re-run, lines restored to
   from_orig/to_orig and all transformers deleted first. Per-sub voltage
   sets come from the surviving candidates (orchestrator's
   `_recompute_substation_voltages` = MATLAB Sub.Voltages); each sub's
   FIRST bus carries its MINIMUM present voltage.
2. **Voltage splitting**: for each extra voltage level a sub has, append a
   new bus at that kV and MOVE the endpoints of candidates whose
   kVFrom/kVTo equals that level (matched via from_orig/to_orig) onto it.
3. **Generator buses** (CreateGenBuses=1): per gen substation, per distinct
   POIkV group — ONE NEW BUS PER UNIT (kV 12.9, type PV), units moved
   there; `BuskV` = the sub voltage class nearest the unit's POIkV. A GSU
   transformer per unit: gen bus → that POI-class bus, tap=1, keep=1,
   radial=1, length=1 km, RateA = ceil(1.2·MaxMW),
   MinCondIndex = ceil(MaxMW) (MVA-as-index overload, cf. conductors.md),
   MaxCondIndex = 1.2·max(XFXR.MVA), resize=1 (impedances via
   TransformerImpedances).
4. **Transmission transformers**: per sub with >1 level, a chain between
   CONSECUTIVE voltage buses (ascending), tap=1, keep=1, RateA=2000
   placeholder (later sized from DC-PF flows), MinCondIndex=20.
5. **Loads** (CreateLoadBuses=0, the shipped default): substation loads land
   on the sub's FIRST (minimum-voltage) bus. (The load-bus branch — 13.8 kV
   buses + step-down transformers for loads < 50 MW — is ported behind the
   flag but the default path matches MATLAB.)  [Load-bus branch: DEFERRED —
   flag raises until a case needs it.]
6. **Slack rule CHANGES**: largest-MaxMW unit's bus becomes slack (setup
   stage used the FIRST gen bus); PV list = gen buses minus slack; load
   list = the rest.
7. TransformerImpedances on all tap≠0 rows; scenario tiers INVALIDATED
   (must be rebuilt at bus granularity).
8. **DEFERRED (tracked)**: MATLAB then rebuilds SNet with
   CreateSNet(Buses='Subs') and sizes the RateA==2000 transmission
   transformers from Complete-tier DC-PF flows (two passes + ResizeBranches
   + AssignConductors + RXB), all inside a try/catch that silently degrades
   (options.NewPF=0). Needs build_snet's 'Subs' mode → follow-up item
   `size_transformers_by_dcpf` (not yet implemented).

## CreateBranch → `create_branch`

Branch = candidates with internal ids preserved (FromInt/ToInt) and
From/To mapped to EXTERNAL bus numbers — a reporting view. Python keeps
from_bus/to_bus internal and adds from_ext/to_ext columns instead.

## TrimQNCALines → `trim_generator_lines`

Resume/add-path utility: at each generator bus, drop gen-to-gen lines that
are ≥ 80 mi (128 km), or — when the bus has > 6 (resp. > 4) incident lines —
lines ≤ 60 MVA that are ≥ 10 mi (resp. ≥ 30 mi). NOTE the MATLAB factor is
1.6 here, not 1.61 (ported as written, flagged).

## Open questions

1. Gen buses at 12.9 kV vs load buses at 13.8 kV — intentional distinct
   nominal voltages? (Ported as written.)
2. **Step-8 impedance recompute never ran in MATLAB (found 2026-07-11 at
   full scale)**: transformer rows carry MVA sizing values in CondIndex
   (e.g. 2000 / ceil(1.2·MW)) which are NOT ConductorData indices, so the
   step-8 `RXB` call on the mixed Branch table indexes out-of-bounds — the
   surrounding try/catch swallowed it, silently skipping the whole
   AssignConductors/RXB refresh after XF sizing. The Python port crashed
   loudly at exactly this point on the first full-scale run (real-SRTV,
   2,491,428→5,880 lines) and now restricts the RXB recompute to line rows
   (tap==0; `transformer_impedances` owns tap≠0), i.e. it EXECUTES the
   step MATLAB always skipped. Confirm this improvement is wanted rather
   than reproducing the silent skip.
2. Step 8's silent try/catch degradation — Python will fail loudly instead
   once 'Subs' mode lands.
3. TrimQNCALines' 1.6 vs 1.61 mile factor.
