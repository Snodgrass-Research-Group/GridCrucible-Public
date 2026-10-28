"""Result and diagnostic plots (matplotlib OO API — Agg-safe, no pyplot).

MATLAB origin: PlotNetwork.m (63 — geographic one-line), PlotNodeDegrees.m
(212 — synthetic degree dists vs real targets + generator degrees),
PlotNodeDegreesActual.m (87), PlotNodeDegreesComparison.m (183 — synthetic
vs Garver vs real), PlotILFun.m (439 — OHM gamma-sweep diagnostics), plus
the ~400 loc of figure code stripped out of RealismMetrics.m /
RealismMetricsComparison.m (compute sides: metrics/realism.py,
metrics/comparison.py — plot FROM their returned dicts; everything is
keyed BY LABEL, never by position: docs/algorithms/RealismMetricsActual.md
"Interface hazard").

Algorithm doc: docs/algorithms/plots.md (panel-by-panel mapping to the
MATLAB figures + flagged MATLAB bugs). Every function builds a
``matplotlib.figure.Figure`` directly (no global pyplot state, no
plt.show()), takes an optional ``out`` path (PNG saved when given) and
RETURNS the Figure (list of Figures for the ILFun diagnostics).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

from ..model import CrucibleModel

#: MATLAB PlotNetwork voltage → line-color map (69:k 138:r 230:b 345:g
#: 500:c 765:m).
VOLTAGE_COLORS = {69: "k", 138: "r", 230: "b", 345: "g", 500: "c", 765: "m"}

#: Display names for the label-keyed degree-fit variants (plot titles).
DEGREE_VARIANTS = {
    "node_degree": "Node degree (incidence)",
    "sub_degree": "Substation degree - all branches",
    "sub_degree_lines": "Substation degree - lines only",
    "bus_degree": "Bus degree - all branches",
    "bus_degree_lines": "Bus degree - lines only",
}


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def _save(fig: Figure, out: Path | None) -> Figure:
    if out is not None:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight")
    return fig


def _arr(x) -> np.ndarray:
    """Defensive ndarray coercion — realism_metrics_actual dicts may have
    round-tripped through the JSON cache (lists, not arrays)."""
    return np.asarray(x, dtype=float)


def _legend(ax, **kw) -> None:
    """Legend only when labeled artists exist (degenerate-input panels)."""
    if ax.get_legend_handles_labels()[0]:
        ax.legend(**kw)


def _eval_exp(fit: dict, x: np.ndarray) -> np.ndarray:
    return fit["a"] * np.exp(fit["b"] * x)


def _eval_power(fit: dict, x: np.ndarray) -> np.ndarray:
    return fit["a"] * x ** fit["b"] + fit.get("c", 0.0)


def _eval_weibull(fit: dict, x: np.ndarray) -> np.ndarray:
    a, b = fit["a"], fit["b"]
    return a * b * x ** (b - 1) * np.exp(-a * x ** b)


_FIT_EVAL = {"exp_fit": _eval_exp, "power_fit": _eval_power,
             "weibull_fit": _eval_weibull}


def _plot_cycle_fit(ax, cycles: dict | None, fit_key: str, style: str,
                    label: str) -> bool:
    """Draw one network's cycle fit curve on its OWN bin midpoints
    (MATLAB: each curve evaluated on its network's BinEdges)."""
    if not cycles:
        return False
    fit = cycles.get(fit_key)
    if not fit:
        return False
    x = _arr(cycles["hist"]["bin_mid"])
    keep = x < 80
    x = x[keep]
    ax.plot(x, _FIT_EVAL[fit_key](fit, x), style, lw=2, label=label)
    return True


def _dist_line(ax, entry: dict | None, label: str, **kw) -> bool:
    """One degree-distribution semilogy line: dist vs ITS OWN support
    (fixes the MATLAB x/y length-mismatch bug — doc Open Q5)."""
    if not entry:
        return False
    dist = _arr(entry["dist"])
    if dist.size == 0 or not (dist > 0).any():
        return False
    ax.semilogy(np.arange(1, dist.size + 1), np.where(dist > 0, dist, np.nan),
                lw=2, label=label, **kw)
    return True


def _real_flavors(targets: dict, case: str) -> list[tuple[str, dict]]:
    """[('<case> all branches', bus-flavor), ('<case> non-negative
    impedances', branch-flavor)] — present flavors only."""
    entry = targets.get(case) or {}
    out = []
    if entry.get("bus"):
        out.append((f"{case} all branches", entry["bus"]))
    if entry.get("branch"):
        out.append((f"{case} non-negative impedances", entry["branch"]))
    return out


# ---------------------------------------------------------------------------
# PlotNetwork
# ---------------------------------------------------------------------------

def plot_network(model: CrucibleModel, out: Path | None = None,
                 iteration: int | None = None,
                 table: str = "candidates") -> Figure:
    """PlotNetwork.m port: geographic one-line — NaN-separated lat/lon
    segments per voltage level (MATLAB color map), substation dots, gen
    substations as squares, import buses as diamonds. The usamap state
    backdrop is dropped (doc Open Q2); the sparse-network abstract graph
    panel is added when n_branches < 5*n_subs (MATLAB rule)."""
    df = getattr(model, table)
    subs = model.substations
    sparse = len(df) < 5 * len(subs)
    fig = Figure(figsize=(14, 6) if sparse else (8, 6))
    axes = fig.subplots(1, 2 if sparse else 1)
    ax_geo = axes[1] if sparse else axes

    fcol = "from_orig" if "from_orig" in df.columns else "from_bus"
    tcol = "to_orig" if "to_orig" in df.columns else "to_bus"
    f = df[fcol].to_numpy(int)
    t = df[tcol].to_numpy(int)
    lat = subs["lat"].to_numpy(float)
    lon = subs["lon"].to_numpy(float)
    if "voltage_kv_from" in df.columns:
        kv = df["voltage_kv_from"].to_numpy(float)
    elif "voltage_kv" in df.columns:
        kv = df["voltage_kv"].to_numpy(float)
    else:
        kv = np.zeros(len(df))

    for level, color in VOLTAGE_COLORS.items():
        rows = kv == level
        if not rows.any():
            continue
        seg_lat = np.column_stack([lat[f[rows]], lat[t[rows]],
                                   np.full(rows.sum(), np.nan)]).ravel()
        seg_lon = np.column_stack([lon[f[rows]], lon[t[rows]],
                                   np.full(rows.sum(), np.nan)]).ravel()
        ax_geo.plot(seg_lon, seg_lat, color, lw=0.7, label=f"{level} kV")
    ax_geo.plot(lon, lat, "ro", ms=1.5)
    gen_mask = subs["gen_max_mw"].to_numpy(float) > 0
    if gen_mask.any():
        ax_geo.plot(lon[gen_mask], lat[gen_mask], "bs", ms=3,
                    label="generator subs")
    imports = np.asarray(model.meta.get("import_buses_int", []), int)
    if imports.size:
        ax_geo.plot(lon[imports], lat[imports], "bd", ms=4,
                    label="import buses")
    ax_geo.set_xlabel("Longitude")
    ax_geo.set_ylabel("Latitude")
    region = "".join(model.meta.get("regions", ["?"]))
    title = f"{model.n_buses} bus {region} network"
    if iteration is not None:
        title += f", outer loop iteration: {iteration}"
    ax_geo.set_title(title)
    _legend(ax_geo, fontsize=8, loc="best")

    if sparse:                       # MATLAB graph(From,To,Mean) panel
        import networkx as nx

        g = nx.MultiGraph()
        g.add_nodes_from(range(model.n_buses))
        fb = df["from_bus"].to_numpy(int)
        tb = df["to_bus"].to_numpy(int)
        weights = (df["mean_norm"].to_numpy(float)
                   if "mean_norm" in df.columns else np.zeros(len(df)))
        for i in range(len(df)):
            g.add_edge(int(fb[i]), int(tb[i]), weight=float(weights[i]))
        pos = nx.spring_layout(g, seed=0)
        ax_g = axes[0]
        nx.draw_networkx(g, pos=pos, ax=ax_g, node_size=120, font_size=7,
                         node_color="#9ecae1")
        if "mean_norm" in df.columns:
            labels = {(int(fb[i]), int(tb[i])): f"{weights[i]:.0f}"
                      for i in range(len(df))}
            nx.draw_networkx_edge_labels(nx.Graph(g), pos=pos, ax=ax_g,
                                         edge_labels=labels, font_size=6)
        ax_g.set_title("Network graph (edge labels: mean flow)")
        ax_g.set_axis_off()
    return _save(fig, out)


# ---------------------------------------------------------------------------
# RealismMetrics figure code — cycle fits, ND fits, MVA histograms
# ---------------------------------------------------------------------------

def plot_cycle_fits(synthetic: dict, targets: dict | None = None,
                    out: Path | None = None, region: str = "SRTV",
                    garver: dict | None = None) -> Figure:
    """RealismMetrics/RealismMetricsComparison cycle-basis figures:
    synthetic histogram + exp/power(+weibull) fits, and the fit-curve
    comparisons vs the real cases (doc: plot_cycle_fits)."""
    cycles = synthetic.get("cycles")
    cases = [c for c in ("EI", "WECC") if targets and targets.get(c)]
    # row 0 = synthetic exp/power; then 2 comparison panels per case
    n_rows = 1 + len(cases)
    fig = Figure(figsize=(12, 4 * n_rows))
    axes = fig.subplots(n_rows, 2, squeeze=False)

    for ax, fit_key, fit_label in ((axes[0][0], "exp_fit", "Exponential"),
                                   (axes[0][1], "power_fit", "Power")):
        if cycles:
            x = _arr(cycles["hist"]["bin_mid"])
            y = _arr(cycles["hist"]["norm"])
            ax.bar(x, y, width=(x[1] - x[0]) if x.size > 1 else 1.0,
                   color="#c6dbef", label="Histogram of Data")
            _plot_cycle_fit(ax, cycles, fit_key, "-",
                            f"{fit_label} Curve Fit")
            if cycles.get("weibull_fit"):
                _plot_cycle_fit(ax, cycles, "weibull_fit", ":",
                                "Weibull Fit")
        ax.set_xlabel("Cycle Length (Number of Branches Per Cycle)")
        ax.set_ylabel("Decimal Percent of Cycles")
        ax.set_title(f"{region} CB KCL-KVL {fit_label} Fit")
        ax.grid(True)
        _legend(ax, fontsize=8)

    # comparison panels vs each real case: exp row then power row
    panel = 2
    for fit_key, fit_label in (("exp_fit", "Exp"), ("power_fit", "Power")):
        for case in cases:
            ax = axes[panel // 2][panel % 2]
            panel += 1
            for (label, flavor), style in zip(_real_flavors(targets, case),
                                              ("--", "-.")):
                _plot_cycle_fit(ax, flavor.get("cycles"), fit_key, style,
                                label)
            _plot_cycle_fit(ax, cycles, fit_key, "-", f"Synthetic {region}")
            if garver:
                _plot_cycle_fit(ax, garver.get("cycles"), fit_key, ":",
                                f"KCL-Only {region}")
            ax.set_xlabel("Cycle Length (Number of Branches Per Cycle)")
            ax.set_ylabel("Decimal Percent of Cycles")
            ax.set_title(f"{region} CB {fit_label} Comparison {case}")
            ax.set_ylim(bottom=-0.01)
            ax.grid(True)
            _legend(ax, fontsize=8)
    while panel < 2 * n_rows:                # blank any unused panel
        axes[panel // 2][panel % 2].set_axis_off()
        panel += 1
    fig.set_layout_engine("tight")
    return _save(fig, out)


def plot_real_cycle_fits(targets: dict, out: Path | None = None) -> Figure:
    """The PlotMetricsRealNetworks block ('EI CB Fits' / 'WECC CB Fits'):
    per case, exp+power curves for both sanitized flavors."""
    cases = [c for c in ("EI", "WECC") if targets.get(c)]
    fig = Figure(figsize=(6 * max(len(cases), 1), 4.5))
    axes = np.atleast_1d(fig.subplots(1, max(len(cases), 1)))
    for ax, case in zip(axes, cases):
        styles = {("bus", "exp_fit"): "--", ("bus", "power_fit"): "-.",
                  ("branch", "exp_fit"): "-", ("branch", "power_fit"): ":"}
        names = {"bus": "All Branches", "branch": "Positive Impedances"}
        for (flavor, fit_key), style in styles.items():
            entry = targets[case].get(flavor) or {}
            kind = "Exponential" if fit_key == "exp_fit" else "Power"
            _plot_cycle_fit(ax, entry.get("cycles"), fit_key, style,
                            f"{names[flavor]}: {kind} Fit")
        ax.set_xlabel("Cycle Length (Number of Branches Per Cycle)")
        ax.set_ylabel("Decimal Percent of Cycles")
        ax.set_title(f"{case} CB Fits")
        ax.set_ylim(bottom=-0.01)
        ax.grid(True)
        _legend(ax, fontsize=8)
    fig.set_layout_engine("tight")
    return _save(fig, out)


def plot_degree_fit(fit_entry: dict, out: Path | None = None,
                    label: str = "node_degree",
                    region: str = "SRTV") -> Figure:
    """One ND-fit figure ('<R> ND KCL-KVL Fit'): scatter of log10(dist)
    vs degree + the linear fit line. fit_entry = one value of
    realism_metrics()['degree_fits']; degenerate fits draw data only."""
    fig = Figure(figsize=(6, 4.5))
    ax = fig.subplots()
    degrees = _arr(fit_entry.get("degrees", []))
    dist = _arr(fit_entry.get("dist", []))[:degrees.size]
    if degrees.size:
        ax.scatter(degrees, np.log10(dist), s=12, c="k", zorder=3,
                   label="Node Degree Data")
        fit = fit_entry.get("fit")
        if fit:
            ax.plot(degrees, fit["slope"] * degrees + fit["intercept"],
                    "k-", lw=2, label="Linear Curve Fit")
    ax.set_xlabel("Number of branches connected to the bus")
    ax.set_ylabel("Log of decimal percent of buses")
    ax.set_title(f"{region} ND KCL-KVL Fit - "
                 f"{DEGREE_VARIANTS.get(label, label)}")
    ax.grid(True)
    _legend(ax, fontsize=8)
    return _save(fig, out)


def plot_degree_fit_comparison(synthetic: dict, garver: dict | None,
                               targets: dict, variant: str,
                               out: Path | None = None,
                               region: str = "SRTV") -> Figure:
    """RealismMetricsComparison ND overlay (SynIndex/RealIndex blocks),
    LABEL-keyed: fit LINES in log10 space for KVL / KCL-only / each real
    case's bus+branch flavors, one panel per case."""
    cases = [c for c in ("EI", "WECC") if targets.get(c)] or ["EI"]
    fig = Figure(figsize=(6 * len(cases), 4.5))
    axes = np.atleast_1d(fig.subplots(1, len(cases)))

    def fit_line(ax, source: dict | None, label: str, style: str) -> None:
        entry = (source or {}).get("degree_fits", {}).get(variant)
        fit = entry.get("fit") if entry else None
        if not fit:
            return
        x = _arr(entry["degrees"])
        ax.plot(x, fit["slope"] * x + fit["intercept"], style, lw=2,
                label=label)

    for ax, case in zip(axes, cases):
        fit_line(ax, synthetic, f"KCL/KVL {region}", "-")
        fit_line(ax, garver, f"KCL-Only {region}", ":")
        entry = targets.get(case) or {}
        fit_line(ax, entry.get("bus"), f"{case} all branches", "--")
        fit_line(ax, entry.get("branch"),
                 f"{case} non-negative impedances", "-.")
        ax.set_xlabel("Number of branches connected to each node")
        ax.set_ylabel("Log of decimal fraction of nodes")
        ax.set_title(f"{DEGREE_VARIANTS.get(variant, variant)} fits "
                     f"vs {case}")
        ax.grid(True)
        _legend(ax, fontsize=8)
    fig.set_layout_engine("tight")
    return _save(fig, out)


def plot_mva_histograms(synthetic: dict, targets: dict | None = None,
                        out: Path | None = None, region: str = "SRTV",
                        case: str = "EI") -> Figure:
    """RealismMetrics MVA-limit histograms: per voltage level, synthetic
    normalized bars + (when targets given) the real Form 715 normalized
    bars overlay (MATLAB overlaid EI only — doc Open Q3)."""
    mva = synthetic.get("mva_hist", {})
    levels = sorted(mva)
    n = max(len(levels), 1)
    ncol = min(n, 3)
    nrow = (n + ncol - 1) // ncol
    fig = Figure(figsize=(4.5 * ncol, 3.5 * nrow))
    axes = np.atleast_1d(fig.subplots(nrow, ncol, squeeze=False)).ravel()
    real_hists = ((targets or {}).get(case) or {}) \
        .get("form715", {}).get("mva_hist", {})
    for ax, kv in zip(axes, levels):
        h = mva[kv]
        x, y = _arr(h["bin_mid"]), _arr(h["norm"])
        w = (x[1] - x[0]) if x.size > 1 else 1.0
        ax.bar(x, y, width=w, alpha=0.7, label=f"Synthetic {region}")
        rh = real_hists.get(kv) or real_hists.get(str(kv))
        if rh:
            rx, ry = _arr(rh["bin_mid"]), _arr(rh["norm"])
            rw = (rx[1] - rx[0]) if rx.size > 1 else 1.0
            ax.bar(rx, ry, width=rw, alpha=0.5,
                   label="Eastern Interconnect" if case == "EI" else case)
        ax.set_title(f"{kv} kV MVA limits (normalized)")
        ax.set_xlabel("MVA Limit")
        ax.set_ylabel("Decimal Percent of Lines")
        _legend(ax, fontsize=7)
    for ax in axes[len(levels):]:
        ax.set_axis_off()
    fig.set_layout_engine("tight")
    return _save(fig, out)


# ---------------------------------------------------------------------------
# PlotNodeDegrees / Actual / Comparison
# ---------------------------------------------------------------------------

def plot_node_degrees(synthetic: dict, targets: dict,
                      out: Path | None = None, region: str = "SRTV",
                      garver: dict | None = None) -> Figure:
    """PlotNodeDegrees.m (garver=None) / PlotNodeDegreesComparison.m
    (garver given): rows = the five label-keyed degree variants, cols =
    real cases; each panel semilogy of synthetic (+garver) vs the case's
    bus/branch-flavor dists."""
    variants = [v for v in DEGREE_VARIANTS
                if v in synthetic.get("degree_fits", {})]
    cases = [c for c in ("EI", "WECC") if targets.get(c)] or ["EI"]
    fig = Figure(figsize=(6 * len(cases), 3.2 * len(variants)))
    axes = fig.subplots(len(variants), len(cases), squeeze=False)
    syn_label = (f"KCL/KVL {region}" if garver is not None
                 else f"Synthetic {region}")
    for i, variant in enumerate(variants):
        for j, case in enumerate(cases):
            ax = axes[i][j]
            _dist_line(ax, synthetic["degree_fits"].get(variant), syn_label)
            if garver is not None:
                _dist_line(ax, garver.get("degree_fits", {}).get(variant),
                           f"KCL-Only {region}")
            for label, flavor in _real_flavors(targets, case):
                _dist_line(ax, flavor.get("degree_fits", {}).get(variant),
                           label)
                if variant == "bus_degree":     # the 6-curve panel
                    _dist_line(ax, flavor.get("degree_fits", {})
                               .get("bus_degree_in_service"),
                               f"{label} (in service)")
            ax.set_title(f"{DEGREE_VARIANTS[variant]} vs {case}",
                         fontsize=9)
            ax.set_xlabel("Number of branches connected to each bus",
                          fontsize=8)
            ax.set_ylabel("Fraction of buses", fontsize=8)
            ax.grid(True)
            _legend(ax, fontsize=7)
    fig.set_layout_engine("tight")
    return _save(fig, out)


def plot_node_degrees_comparison(synthetic: dict, garver: dict,
                                 targets: dict, out: Path | None = None,
                                 region: str = "SRTV") -> Figure:
    """PlotNodeDegreesComparison.m port — plot_node_degrees with the
    KCL-Only overlay (kept as a named function to mirror the MATLAB
    file set)."""
    return plot_node_degrees(synthetic, targets, out=out, region=region,
                             garver=garver)


def plot_node_degrees_actual(targets: dict,
                             out: Path | None = None) -> Figure:
    """PlotNodeDegreesActual.m port: per case x flavor panels of the four
    real dist variants + the two cross-case comparison panels."""
    cases = [c for c in ("EI", "WECC") if targets.get(c)]
    combos = [(case, flavor,
               f"{case} {'bus' if flavor == 'bus' else 'branch'} case")
              for case in cases for flavor in ("bus", "branch")
              if (targets[case] or {}).get(flavor)]
    n_panels = len(combos) + 2
    ncol = 2
    nrow = (n_panels + 1) // 2
    fig = Figure(figsize=(12, 3.5 * nrow))
    axes = fig.subplots(nrow, ncol, squeeze=False).ravel()

    curve_labels = (("node_degree", "All branches"),
                    ("bus_degree", "Num Neighbors"),
                    ("bus_degree_in_service",
                     "Num Neighbors across in-service branches only"),
                    ("bus_degree_lines", "Lines Only"))
    for ax, (case, flavor, title) in zip(axes, combos):
        fits = targets[case][flavor].get("degree_fits", {})
        for key, label in curve_labels:
            _dist_line(ax, fits.get(key), label)
        ax.set_title(f"{title} node degree distribution", fontsize=9)
        ax.set_xlabel("Number of branches connected to each bus",
                      fontsize=8)
        ax.set_ylabel("Decimal fraction of buses", fontsize=8)
        ax.grid(True)
        _legend(ax, fontsize=6)

    # cross-case substation-degree panel (branch flavor)
    ax = axes[len(combos)]
    for case in cases:
        fits = (targets[case].get("branch") or {}).get("degree_fits", {})
        _dist_line(ax, fits.get("sub_degree"), f"{case} Num Neighbors")
        _dist_line(ax, fits.get("sub_degree_lines"), f"{case} Lines Only")
    ax.set_title("Substation degree distributions", fontsize=9)
    ax.set_xlabel("Number of branches connected to each bus", fontsize=8)
    ax.set_ylabel("Decimal fraction of buses", fontsize=8)
    ax.grid(True)
    _legend(ax, fontsize=7)

    # cross-case bus-degree panel (bus flavor)
    ax = axes[len(combos) + 1]
    for case in cases:
        fits = (targets[case].get("bus") or {}).get("degree_fits", {})
        _dist_line(ax, fits.get("bus_degree"), f"{case} Num Neighbors")
    ax.set_title("Bus degree distributions", fontsize=9)
    ax.set_xlabel("Number of branches connected to each bus", fontsize=8)
    ax.set_ylabel("Decimal fraction of buses", fontsize=8)
    ax.grid(True)
    _legend(ax, fontsize=7)
    for ax in axes[n_panels:]:
        ax.set_axis_off()
    fig.set_layout_engine("tight")
    return _save(fig, out)


def plot_generator_degrees(model: CrucibleModel, out: Path | None = None,
                           table: str = "candidates") -> Figure:
    """The PlotNodeDegrees generator-substation block: degree of each gen
    substation counted over branches touching gen subs → fraction plot +
    semilogy twin. (The dead RadialGenerators table build is not ported.)"""
    df = getattr(model, table)
    gens = model.generators
    gcol = "Sub" if "Sub" in gens.columns else "BusNum"
    gen_subs = np.unique(gens[gcol].to_numpy(int)) if len(gens) \
        else np.empty(0, int)
    fcol = "from_orig" if "from_orig" in df.columns else "from_bus"
    tcol = "to_orig" if "to_orig" in df.columns else "to_bus"
    ends = np.concatenate([df[fcol].to_numpy(int), df[tcol].to_numpy(int)])
    ends = ends[np.isin(ends, gen_subs)]
    if gen_subs.size:
        per_sub = np.bincount(
            ends, minlength=int(gen_subs.max()) + 1)[gen_subs]
        per_sub = per_sub[per_sub > 0]
    else:
        per_sub = np.empty(0, int)
    fig = Figure(figsize=(11, 4))
    ax1, ax2 = fig.subplots(1, 2)
    if per_sub.size:
        counts = np.bincount(per_sub)[1:]           # degrees 1..max
        frac = counts / counts.sum()
        x = np.arange(1, counts.size + 1)
        ax1.plot(x, frac, lw=2)
        ax2.semilogy(x, np.where(frac > 0, frac, np.nan), lw=2)
    for ax, name in ((ax1, "degree graph"), (ax2, "degree distribution")):
        ax.set_xlabel("Number of Lines connected to the Generator Buses")
        ax.set_ylabel("Fraction of Generator Buses")
        ax.set_title(f"Generator substation {name}")
        ax.grid(True)
    fig.set_layout_engine("tight")
    return _save(fig, out)


# ---------------------------------------------------------------------------
# PlotILFun — OHM gamma-sweep diagnostics
# ---------------------------------------------------------------------------

def plot_inner_loop_diagnostics(gammas, flows_by_gamma, lines_on_by_gamma,
                                out: Path | None = None, *,
                                from_bus: np.ndarray | None = None,
                                to_bus: np.ndarray | None = None,
                                percent_low_flow: np.ndarray | None = None,
                                solver: str = "QP") -> list[Figure]:
    """PlotILFun.m port for ONE solver's sweep (MATLAB carried CPLEX and
    Newton matrices side by side — call twice). Fixes the MATLAB
    Graphs/NodeDegree dependency crash and the <4-gamma spacing test
    (doc Open Q4): stats are computed unconditionally, log-x is used
    whenever the sweep is not linearly spaced.

    gammas: the gamma values actually swept; flows_by_gamma
    (n_paths x n_gamma) |flow| matrix; lines_on_by_gamma same-shape bool.
    from_bus/to_bus enable the node-degree stats. Returns [histograms,
    stats, derivatives]; ``out`` (base path) saves them with
    _hist/_stats/_dstats suffixes.
    """
    gammas = np.asarray(gammas, float)
    flows = np.atleast_2d(np.asarray(flows_by_gamma, float))
    on = np.atleast_2d(np.asarray(lines_on_by_gamma, bool))
    n_g = min(len(gammas), flows.shape[1], on.shape[1])
    gammas, flows, on = gammas[:n_g], flows[:, :n_g], on[:, :n_g]
    figs: list[Figure] = []
    out = Path(out) if out is not None else None

    # ---- 1. per-gamma flow histograms (PlotOpts.Hist) ----------------------
    fig_h = Figure(figsize=(10, 2.8 * n_g))
    axes = fig_h.subplots(n_g, 2, squeeze=False)
    for k in range(n_g):
        f = np.abs(flows[:, k])
        big, small = f[f >= 1.0], f[f < 1.0]
        if big.size:                                 # integer-width bins
            edges = np.arange(np.floor(big.min()) - 0.5,
                              np.ceil(big.max()) + 1.5)
            axes[k][0].hist(big, bins=edges)
        if small.size:                               # Sturges (panel 2)
            axes[k][1].hist(small, bins="sturges")
        for ax in axes[k]:
            ax.set_xlabel("Magnitude of flow on paths")
            ax.set_ylabel("Number of paths")
            ax.set_title(f"{solver} flows, log(gamma) = "
                         f"{np.log10(gammas[k]):.3g}", fontsize=9)
    fig_h.set_layout_engine("tight")
    figs.append(_save(fig_h, out.with_name(out.stem + "_hist" + out.suffix)
                      if out else None))

    # ---- stats per gamma (MATLAB computed these inside the Graphs loop) ----
    n_on = on.sum(axis=0)
    flow_mean = np.array([np.abs(flows[on[:, k], k]).mean()
                          if n_on[k] else 0.0 for k in range(n_g)])
    flow_std = np.array([np.abs(flows[on[:, k], k]).std(ddof=1)
                         if n_on[k] > 1 else 0.0 for k in range(n_g)])
    have_graph = from_bus is not None and to_bus is not None
    if have_graph:
        fb = np.asarray(from_bus, int)
        tb = np.asarray(to_bus, int)
        deg_mean = np.zeros(n_g)
        deg_std = np.zeros(n_g)
        deg_max = np.zeros(n_g)
        for k in range(n_g):
            ends = np.concatenate([fb[on[:, k]], tb[on[:, k]]])
            # MATLAB graph(F,T) numbers nodes 1..max(id) — zero-degree
            # nodes below the max COUNT toward mean/std (kept, 0-based)
            deg = np.bincount(ends) if ends.size else np.zeros(1)
            deg_mean[k] = deg.mean()
            deg_std[k] = deg.std(ddof=1) if deg.size > 1 else 0.0
            deg_max[k] = deg.max()

    linear = (n_g >= 4
              and abs((gammas[3] - gammas[2]) - (gammas[1] - gammas[0]))
              < 1e-12)
    x = gammas if linear else np.log10(gammas)
    xlab = "gamma" if linear else "log(gamma)"
    width = (np.min(np.diff(x)) * 0.8 if n_g > 1 else 0.1)

    def _stat_fig(xv, panels, title_prefix=""):
        fig = Figure(figsize=(11, 9))
        axs = fig.subplots(3, 2, squeeze=False).ravel()
        for ax, (yv, title, ylab) in zip(axs, panels):
            if yv is None:
                ax.set_axis_off()
                continue
            ax.bar(xv, yv, width=width)
            ax.set_title(title_prefix + title, fontsize=9)
            ax.set_xlabel(xlab, fontsize=8)
            ax.set_ylabel(ylab, fontsize=8)
        fig.set_layout_engine("tight")
        return fig

    panels = [
        (deg_mean if have_graph else None, "Node Degree Mean",
         "Mean of Node Degree"),
        (deg_std if have_graph else None, "Node Degree Standard Deviation",
         "Standard Deviation of Node Degree"),
        (deg_max if have_graph else None, "Maximum node degree",
         "Highest Node degree"),
        (n_on.astype(float), "Paths with flow above cutoff",
         "Number of Paths"),
        (flow_mean, "Mean of flows", "Mean of flows"),
        (flow_std, "Standard Deviation of flows", "Flows STD"),
    ]
    fig_s = _stat_fig(x, panels)
    if percent_low_flow is not None:
        plf = np.asarray(percent_low_flow, float)[:n_g]
        ax = fig_s.axes[3].twinx()
        ax.plot(x[:len(plf)], plf, "r-", lw=1.5)
        ax.set_ylabel("fraction low-flow", color="r", fontsize=8)
    figs.append(_save(fig_s, out.with_name(out.stem + "_stats" + out.suffix)
                      if out else None))

    # ---- derivative panel ---------------------------------------------------
    if n_g > 1:
        dpanels = [(None if p[0] is None else np.diff(p[0]),
                    p[1], p[2]) for p in panels]
        fig_d = _stat_fig(x[1:], dpanels, title_prefix="Derivative of ")
        figs.append(_save(fig_d,
                          out.with_name(out.stem + "_dstats" + out.suffix)
                          if out else None))
    return figs
