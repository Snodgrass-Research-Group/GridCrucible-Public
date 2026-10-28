# AssignConductors.m + ResizeBranches.m + TransformerImpedances.m → `network/conductors.py`

**Sources:** AssignConductors.m (35), ResizeBranches.m (67), TransformerImpedances.m (81)
These are the TUNE Loop's leaves.

## The conductor catalog (RESOLVED open question)

Verified against the migrated data: the union of `CondIndex` across the six
per-kV tables is **exactly 1..148, monotone in MVA** — i.e. the master
ConductorData table (the opaque .mat) is the MVA-sorted concatenation of the
per-kV tables, and its CondIndex column equals its row number. That is why
MATLAB can use `argmin` row indices and CondIndex column values
interchangeably (`ConductorData.kV(Branch.CondIndex)` row-indexes it).
`load_conductor_catalog` rebuilds the master (sort by cond_index, assert
contiguity, index = 1-based cond_index) — MATLAB row semantics preserved.

## ResizeBranches → `resize_branches` (rating steps, no conductor pick yet)

`percent_utilized = |flow|/capacity`. Three independent passes, each capped
at `ceil(percent_resize · n_paths)` rows and each setting `resize=True`:
* **Upsize**: util > 1/conductor_margin AND cond_idx < max_cond_idx →
  `capacity = ceil(capacity · percent_change)`. Ineligible overloads (already
  at max conductor) are flagged `overloaded` for manual review.
* **DownsizeKeep**: util < downsize_threshold AND cond_idx > min_cond_idx AND
  keep → `capacity = ceil(capacity / percent_change)`.
* **Downsize**: same for NOT-keep rows.
Candidate order: `random_resize=True` (default) shuffles the eligible set;
False takes them worst-first. NOTE the MATLAB False-path re-sorted the whole
table (sortrows side effect); Python orders index arrays only — row order is
owned by TRIM's mean_norm sort (deviation documented, behavior-equivalent
since MATLAB re-sorts every TRIM iteration anyway).

## AssignConductors → `assign_conductors` (snap ratings to real conductors)

For `resize`-flagged rows (lines only, tap==0): per-circuit rating =
capacity/n_circuits; pick the nearest-MVA row of the MASTER table
(`change_voltages=True`, may change the voltage class!) — but if that
conductor's kV < the row's `min_kv`, re-pick nearest-MVA within the min_kv
class. `change_voltages=False`: nearest-MVA within the row's own kV class.
Then: conductor_idx, voltage_kv ← catalog kV, kcmil ← Size, cond_per_phase ←
CondPhase, capacity = n_circuits × catalog MVA (exact table value).
Clears `resize`. DEVIATION: MATLAB updates only kVFrom (kVTo drifts stale
until later recompute); Python sets BOTH ends for tap==0 rows — physically
identical, and prevents spurious line→trafo classification in the pandapower
bridge.

## TransformerImpedances → `transformer_impedances`

Only tap≠0 rows (none exist until CreateBuses adds GSUs — the orchestrator wires the
substation XFMVA memory). Per resize-flagged transformer: snap end-kVs to the
nearest XFData Pri/Sec values; optional XFMVA memory reuses a previously
sized transformer's MVA; X/R: rating < 50 MVA → 25; > 2000 MVA → split into
ceil(rate/2000) parallel circuits and interpolate XFXR at the per-circuit
rating; else interpolate XFXR at the rating. Z_pu = (100/RateA)·(XHL/100)
(100 MVA base), X = √(Z²/(1+1/XR²)), R = X/XR. cond_index ← ceil(rate/
circuits) (for transformers cond_index stores per-unit MVA, NOT a catalog
row — MATLAB overload, kept). Clears `resize`.

## Schema additions to `candidates` (model.py note)

`resize` (bool), `percent_utilized`, `min_conductor_idx`, `min_kv`,
`kcmil`, `cond_per_phase`, `overloaded`/`undersized`/`undersized_keep`
(diagnostics), `max_acpf`, `from_sub` + `transmission_xf` (transformers,
).

## Open questions

1. ResizeBranches downsize comment says "reduce by 20%" but divides by
   percent_change (1.3 → −23%) — comment rot, formula ported.
2. TransformerImpedances `XFMin = RateA < MinCondIndex` compares MVA against
   an INDEX-named column — for transformers MinCondIndex evidently stores a
   minimum MVA. Confirm with the author.
