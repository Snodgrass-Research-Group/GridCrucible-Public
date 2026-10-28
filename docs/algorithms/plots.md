# PlotNetwork / PlotNodeDegrees* / PlotILFun + RealismMetrics figures → `viz/plots.py`

Sources (all under `KCL-KVL Compliant Grid Generator/Matlab Functions/`):
`PlotNetwork.m` (63), `PlotNodeDegrees.m` (212), `PlotNodeDegreesActual.m`
(87), `PlotNodeDegreesComparison.m` (183), `PlotILFun.m` (439), plus the
~400 loc of figure code inside `RealismMetrics.m` /
`RealismMetricsComparison.m` whose COMPUTE sides are already converted
(`metrics/realism.py`, `metrics/comparison.py`).
Written 2026-07-10.

## Conventions of the port

* **No pyplot**: every function builds `matplotlib.figure.Figure` directly
  (OO API) — backend-independent, Agg-safe, no `plt.show()`, no global
  figure registry to leak in long runs.
* Every function takes an optional `out` path; when given, the figure is
  saved (PNG via `fig.savefig`) and the path returned alongside; the
  `Figure` is ALWAYS returned so callers/tests can inspect axes.
* All metric inputs are the **label-keyed dicts** returned by
  `metrics/realism.realism_metrics`, `metrics/comparison.comparison_stats`
  and `metrics/comparison.realism_metrics_actual` — never positional
  structs (see RealismMetricsActual.md §Interface hazard). Arrays are
  `np.asarray`-ed defensively because `realism_metrics_actual` results may
  round-trip through the JSON cache (lists, not ndarrays).
* Real-network label mapping (fixed): `<case> All Branches` =
  `targets[case]['bus']` (LargeIslandSanitized), `<case> Positive
  Impedances` = `targets[case]['branch']` (BranchSanitized).
* MATLAB emitted dozens of separate `figure` windows; the port groups the
  same content into multi-panel figures (one Figure per function call) —
  a deliberate deviation, content per panel preserved.
* Fit-curve evaluation helpers (module-private): exp `a·e^{bx}`, power
  `a·x^b + c`, weibull `a·b·x^{b−1}·e^{−a·x^b}`, log-linear ND fit
  `slope·x + intercept` drawn in log10 space (MATLAB scatter/plot of
  `log10(dist)` with a poly1 line — the port plots the SAME log10-space
  values on a linear axis, as MATLAB did).

## PlotNetwork.m → `plot_network(model, out=None, iteration=None, table="candidates")`

1. If `height(OLBranch) < 5·height(Sub)` MATLAB draws the abstract
   `graph(From,To,Mean)` with rounded edge weights. Port: networkx
   spring-layout panel with edge labels from `mean_norm` when the column
   exists (MATLAB `OLBranch.Mean` = mean scenario flow; our TRIM column is
   the normalized variant — recorded, Open Q1); skipped for dense sets
   with the same 5× rule.
2. Geographic one-line: `usamap(options.States)` + `shaperead('usastatehi')`
   state polygons — DROPPED in the port (no basemap/cartopy dependency;
   Open Q2). Lines drawn as NaN-separated lat/lon polylines per voltage
   with the MATLAB color map {69:k, 138:r, 230:b, 345:g, 500:c, 765:m};
   endpoints resolved sub-first (`FromSub/ToSub` try → `From/To` catch;
   port: `from_orig/to_orig` if present else `from_bus/to_bus`, against
   `model.substations`). Substations 'ro' (size 1), gen substations
   (`MaxMW > 0`, catch `GenMax`; port `gen_max_mw > 0`) 'bs' (size 3),
   import buses (`params.ImportBusesInt`; port
   `model.meta['import_buses_int']` when present) 'bd' (size 4).
   Title carries `params.OuterLoopIter` → `iteration` arg.

## RealismMetrics.m figures → `plot_cycle_fits`, `plot_real_cycle_fits`, `plot_degree_fit`, `plot_mva_histograms`

The compute side (fits, histograms) lives in `metrics/realism.py`; these
functions ONLY draw from its dict.

* `plot_cycle_fits(synthetic, targets=None, out=None, region='SRTV',
  garver=None)` — panels (MATLAB FigNames in quotes):
  1. cycle-length histogram bars + exp-fit curve ('<R> CB KCL-KVL Exp
     Fit'; also covers 'CB Exp Fit Points <R>' — same data as
     scatter+curve),
  2. histogram bars + power-fit curve ('<R> CB KCL-KVL Power Fit',
     `ylim ≥ −0.005`),
  3./4. exp-fit comparison vs EI / WECC ('<R> CB Exp Comparison {EI,WECC}
     1'): real all-branches dashed, real positive-impedances dash-dot,
     synthetic solid (+ garver dotted when given — the
     RealismMetricsComparison overlay),
  5./6. power-fit comparison vs EI / WECC (`ylim ≥ −0.005/−0.01`).
  Panels 3-6 only when `targets` given. Weibull curves (comparison_stats
  adds `cycles['weibull_fit']`) overlay panels 1-2 when present.
* `plot_real_cycle_fits(targets, out=None)` — the
  `codectrl.PlotMetricsRealNetworks` block ('EI CB Fits', 'WECC CB
  Fits'): one panel per case, 4 curves (all-branches exp `--` / power
  `-.`, positive-impedances exp `-` / power `:`).
* `plot_degree_fit(fit_entry, out=None, label=..., region=...)` — the
  per-variant linear-fit figure ('<R> ND KCL-KVL Fit'): scatter of
  `log10(dist)` vs degree + fit line. `fit_entry` = one value of
  `synthetic['degree_fits']`; returns None-safe empty axes when the fit is
  degenerate (`fit is None` — MATLAB crashed there, realism.py guards).
* `plot_mva_histograms(synthetic, targets=None, out=None, region='SRTV')`
  — one panel per voltage level in `synthetic['mva_hist']`: synthetic
  normalized bars + (when targets given) the EI Form 715 normalized bars
  overlay (`targets['EI']['form715']['mva_hist'][kv]`) — MATLAB overlaid
  EI only; the port accepts any case key present (Open Q3).

## PlotNodeDegrees.m → `plot_node_degrees(synthetic, targets, out=None, region='SRTV')` + `plot_generator_degrees(model, out=None)`

MATLAB drew 12 semilogy figures: {EI, WECC} × {node degree (plain+log),
sub degree all/lines, bus degree all/lines}, each overlaying the synthetic
dist on the real case's bus-flavor and branch-flavor dists. Port: one
figure, grid of panels — rows = the five degree variants (label-keyed:
`node_degree, sub_degree, sub_degree_lines, bus_degree, bus_degree_lines`),
cols = cases ('EI', 'WECC' — whichever keys `targets` has); each panel
semilogy of synthetic + real-bus + real-branch dists (`degrees` vs `dist`
from the fit dicts).

The trailing generator-substation block (unique to PlotNodeDegrees):
degrees of generator substations counted over branches touching gen subs
→ fraction-of-gen-buses vs lines plot (+ semilogy twin). Port
`plot_generator_degrees(model, out=None)` computes gen-sub degrees from
`model.candidates`/`branches` endpoints restricted to
`unique(generators['Sub'])` (MATLAB `Gen.Sub` try / `Gen.BusNum` catch)
and draws both panels. The dead `RadialGenerators` table build (wrapped in
try with a debugging catch) is NOT ported.

## PlotNodeDegreesActual.m → `plot_node_degrees_actual(targets, out=None)`

Six figures → six panels: per case × flavor (EI-bus, EI-branch, WECC-bus,
WECC-branch) semilogy of the four real dist variants (`node_degree`
'All branches', `bus_degree` 'Num Neighbors', `bus_degree_in_service`,
`bus_degree_lines` 'Lines Only'); plus the two cross-case panels
(sub-degree EI vs WECC ×{all, lines}; bus-degree EI vs WECC).

## PlotNodeDegreesComparison.m → `plot_node_degrees_comparison(synthetic, garver, targets, out=None, region='SRTV')`

Same 12-figure grid as PlotNodeDegrees but each panel adds the KCL-Only
(garver) line, and the bus-degree-all panels add the real in-service
variants (6 curves). Port: same panel grid as `plot_node_degrees` with the
garver overlay + `bus_degree_in_service` curves where available. Legends:
'KCL/KVL <R>' / 'KCL-Only <R>' / '<case> all branches' / '<case>
non-negative impedances' (+ in-service pair).

The ND fit-line overlay figures of RealismMetricsComparison.m
(SynIndex/RealIndex blocks, lines 436-583) are covered by
`plot_degree_fit_comparison(synthetic, garver, targets, variant, out=None)`
— fit LINES (log10 space) of KVL/KCL-only/real-bus/real-branch for one
label-keyed variant. This replaces the positional SynIndex/RealIndex
contract (the §Interface hazard).

## PlotILFun.m → `plot_inner_loop_diagnostics(gammas, flows_by_gamma, lines_on_by_gamma, out=None, *, from_bus=None, to_bus=None, percent_low_flow=None)`

Called from DCInnerLoop behind `codectrl.PlotResultsIL`
(`config.io.plot_inner_loop`); MATLAB signature carries separate
CPLEX/Newton matrices + `PlotOpts` — the port takes ONE solver's matrices
(call twice for QP + Newton; OHMLoop keeps only the active solver's
per-γ matrices anyway).

Inputs: `gammas` (the Lam sweep actually run), `flows_by_gamma`
(n_paths × n_γ |flow| matrix), `lines_on_by_gamma` (same-shape bool).
Three figure groups (MATLAB PlotOpts.Hist/Graphs/NodeDegree):
1. **Histograms** (`PlotOpts.Hist`): per γ, two panels — |flow| ≥ 1
   (integer-width bins) and |flow| < 1 (Sturges) — grid n_γ × 2, titles
   `log(γ)` to 3 sig figs.
2. **Graph drawings** (`PlotOpts.Graphs`): per-γ networkx graph of the
   on-lines with rounded flow edge labels — only when
   `from_bus/to_bus` are given; capped to reasonable γ counts.
3. **γ-sweep stats bars** (`PlotOpts.NodeDegree`): 3×2 panels — node
   degree mean/std/max (needs from/to; graph of on-lines only), paths
   with flow above the zero cutoff, flow mean/std — plus the second 3×2
   `diff()` derivative figure. MATLAB switches the x-axis between raw γ
   (when `Lam` is linearly spaced, tested via `Lam(4)-Lam(3) vs
   Lam(2)-Lam(1)`) and `log10(γ)`; the port applies the same test
   guarded for < 4 gammas (Open Q4). `percent_low_flow` (OHMResult) adds
   an optional overlay line on the paths panel.
Returns a list of Figures (histograms figure, optional graphs figure,
stats figure, derivative figure); `out` = base path, suffixed
`_hist/_graphs/_stats/_dstats`.

## Open questions (the author)

1. PlotNetwork edge labels use `OLBranch.Mean` (raw mean scenario flow).
   The Python candidates table keeps `mean_norm` (normalized) — is the
   normalized label acceptable for this diagnostic, or should OHM's raw
   mean flows be threaded through?
2. PlotNetwork's `usamap`/`shaperead('usastatehi')` state-boundary
   backdrop needs a basemap package (cartopy) that is not in the
   dependency set — dropped (lat/lon lines only). Add cartopy, or is the
   plain plot fine?
3. RealismMetrics MVA-overlay compares synthetic vs EI only (WECC
   Form 715 hists exist but were never overlaid). Port accepts any case —
   keep EI-only as the default caller behavior?
4. PlotILFun BUGS (not ported, documented): (a) with
   `PlotOpts.Graphs=0, NodeDegree=1` MATLAB references undefined
   NonZeroFlows/NodeDegree* — crash; the stats are computed inside the
   Graphs loop only. (b) `Lam(4)-Lam(3)` spacing test errors when the
   sweep has < 4 gammas. (c) The NodeDegree stats use `graph(...)` degree
   (parallel edges collapse? MATLAB graph keeps multi-edges — verified
   equivalent to endpoint bincount). Port computes stats unconditionally
   and guards short sweeps.
5. PlotNodeDegrees.m:60/139 (and the Comparison variant :71-72/158-159)
   plot `y = BusDegreeDist` against `x = 1:length(BusDegreeDistInService)`
   — mismatched x/y lengths whenever the two dists differ in length
   (MATLAB errors or truncates misleadingly). The port always pairs each
   dist with its own support. Also: several WECC-section titles say
   "Eastern Intercoonect" [sic] — cosmetic, not reproduced.
6. PlotNodeDegrees legends label the two real curves 'EI all branches' /
   'EI non-negative branches' but the plotted order swaps between
   Data (branch-flavor) and DataBus (bus-flavor) from figure to figure —
   the port's legends are derived from the dict keys, so they are always
   correct; MATLAB's mislabeled panels are flagged here instead.
