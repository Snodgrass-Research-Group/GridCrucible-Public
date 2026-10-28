"""Comparisons against real networks (EI/WECC Form 715) and the KCL-only twin.

MATLAB origin: RealismMetricsActualEI.m (919), RealismMetricsActualWECC.m
(891), RealismMetricsComparison.m (888), NetworkGraphComparison.m (453),
ElectricalComparison.m (213).
Algorithm docs: docs/algorithms/RealismMetricsActual.md (incl.
§RealismMetricsComparison), NetworkGraphComparison.md,
ElectricalComparison.md — every rule, magic number and flagged MATLAB bug
below is specified there.

Layout:
* real-network stats — ``realism_metrics_actual('EI'|'WECC', ...)``: ONE
  parameterized port of both Actual variants, reading the migrated parquets
  and reference-case npz files; results cached to
  data_dir/comparison_cache/.
* synthetic-side comparison stats — ``comparison_stats``: RealismMetrics
  (converted, reused) + the weibull cycle fit unique to
  RealismMetricsComparison.
* ``hist_areas`` — the cross-network fit-quality table (fits keyed by
  LABEL: the MATLAB positional ND-index contract is deliberately dropped,
  doc §Interface hazard).
* ``electrical_comparison`` — the network × loads DC-PF/DC-OPF grid.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import CrucibleConfig
from ..model import CrucibleModel
from . import graph_metrics as gm
from .realism import MVA_BINS, _degree_fit, realism_metrics

#: Form 715 voltage classes; 999 = the aggregate 230..765 row.
VOLTAGE_CLASSES = np.array([69, 138, 230, 345, 500, 765, 999])

#: Hard-coded Form 715 MVA-limit outlier bounds per class (doc §5).
LOW_MVA = {69: 24, 138: 118, 230: 366, 345: 736, 500: 1725, 765: 3601}
HIGH_MVA = {69: 154, 138: 650, 230: 1083, 345: 1959, 500: 4217, 765: 9275}
LOW_XF_MVA = dict.fromkeys(LOW_MVA, 1)
HIGH_XF_MVA = {69: 400, 138: 600, 230: 1000, 345: 1600, 500: 2500, 765: 4000}


# ---------------------------------------------------------------------------
# fit + snapping helpers
# ---------------------------------------------------------------------------

def weibull_fit(x: np.ndarray, y: np.ndarray) -> dict | None:
    """MATLAB fit(...,'weibull'): y = a·b·x^(b−1)·e^(−a·x^b). Same result
    convention as graph_metrics.exp_fit/power_fit ({a, b, r2, area} or
    None on degenerate/non-convergent input)."""
    from scipy.optimize import curve_fit

    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if x.size < 3 or (x <= 0).any():
        return None

    def f(x, a, b):
        return a * b * x ** (b - 1) * np.exp(-a * x ** b)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, _ = curve_fit(f, x, y, p0=(0.1, 1.0), maxfev=10000)
    except (RuntimeError, ValueError):
        return None
    y_hat = f(x, *popt)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float(((y - y_hat) ** 2).sum()) / ss_tot if ss_tot > 0 else 0.0
    return {"a": float(popt[0]), "b": float(popt[1]), "r2": r2,
            "area": float(np.trapezoid(y_hat, x))}


def nearest_voltage_class(kv: np.ndarray,
                          classes: np.ndarray | None = None) -> np.ndarray:
    """Snap nominal kV to the NEAREST Form 715 class (MATLAB min(|Δ|) loop,
    vectorized). The 999 aggregate row never wins a nearest match for real
    voltages ≤ 882 kV, matching MATLAB (it included 999 in the search)."""
    classes = VOLTAGE_CLASSES if classes is None else np.asarray(classes)
    kv = np.asarray(kv, float)
    return classes[np.argmin(np.abs(kv[:, None] - classes[None, :]), axis=1)]


def _round2_nan0(v: float) -> float:
    return 0.0 if np.isnan(v) else round(float(v), 2)


# ---------------------------------------------------------------------------
# real-case loading (largest island, both sanitized flavors)
# ---------------------------------------------------------------------------

def _load_case_island(data_dir: Path, stem: str) -> dict:
    """reference_cases/<stem>.npz → largest connected island (in-service
    branches only), 0-based consecutive bus numbering (ext2int +
    find_islands/extract_islands equivalent)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    z = np.load(Path(data_dir) / "reference_cases" / f"{stem}.npz")
    bus, branch = z["bus"], z["branch"]
    ext = bus[:, 0].astype(np.int64)
    pos = {b: i for i, b in enumerate(ext)}
    fb = np.array([pos[b] for b in branch[:, 0].astype(np.int64)])
    tb = np.array([pos[b] for b in branch[:, 1].astype(np.int64)])
    on = branch[:, 10] != 0 if branch.shape[1] > 10 else np.ones(len(fb), bool)

    n = len(bus)
    adj = coo_matrix((np.ones(on.sum()), (fb[on], tb[on])), shape=(n, n))
    _, labels = connected_components(adj, directed=False)
    main = np.bincount(labels).argmax()
    keep_bus = labels == main
    keep_br = keep_bus[fb] & keep_bus[tb] & on
    renum = np.cumsum(keep_bus) - 1
    return {"bus": bus[keep_bus], "branch": branch[keep_br],
            "fb": renum[fb[keep_br]], "tb": renum[tb[keep_br]],
            "n_buses": int(keep_bus.sum())}


# ---------------------------------------------------------------------------
# Form 715 / Form 1 statistic blocks
# ---------------------------------------------------------------------------

def _length_stats(length: pd.Series, kv_class: np.ndarray) -> pd.DataFrame:
    """Per-class Total/Mean/Std/Q10/Median/Q90 (rounded 2, NaN→0);
    999 row: Total = Σ classes 230..765."""
    rows = []
    for kv in VOLTAGE_CLASSES[:-1]:
        v = length[kv_class == kv].to_numpy(float)
        rows.append({
            "Volt": int(kv),
            "Total": _round2_nan0(v.sum()),
            "Mean": _round2_nan0(v.mean()) if v.size else 0.0,
            "Std": _round2_nan0(np.std(v, ddof=1)) if v.size > 1 else 0.0,
            "Quantile10": _round2_nan0(np.quantile(v, 0.1)) if v.size else 0.0,
            "Median": _round2_nan0(np.median(v)) if v.size else 0.0,
            "Quantile90": _round2_nan0(np.quantile(v, 0.9)) if v.size else 0.0,
        })
    total_999 = round(sum(r["Total"] for r in rows[2:6]), 2)
    rows.append({"Volt": 999, "Total": total_999, "Mean": 0.0, "Std": 0.0,
                 "Quantile10": 0.0, "Median": 0.0, "Quantile90": 0.0})
    return pd.DataFrame(rows)


def _branch_stats(length: pd.Series, limit: pd.Series,
                  kv_class: np.ndarray) -> pd.DataFrame:
    """Per-class VoltPercent / GVAMiles (Σ length·MVA / 1000) / Count;
    999 row: GVAMiles = Σ classes 230..765, Count = all rows."""
    n_all = len(length)
    rows = []
    for kv in VOLTAGE_CLASSES[:-1]:
        m = kv_class == kv
        gva = round(float((length[m] * limit[m]).sum()), 2) / 1000.0
        rows.append({"Volt": int(kv),
                     "VoltPercent": round(100.0 * m.sum() / n_all, 3),
                     "GVAMiles": gva, "Count": int(m.sum())})
    rows.append({"Volt": 999,
                 "VoltPercent": 0.0,
                 "GVAMiles": sum(r["GVAMiles"] for r in rows[2:6]),
                 "Count": n_all})
    return pd.DataFrame(rows)


def _mva_hist(limits: np.ndarray, kv: int, low: dict, high: dict,
              n_bins: int | None) -> dict | None:
    """One class's MVA-limit histogram inside the outlier bounds."""
    v = limits[(limits >= low[kv]) & (limits <= high[kv])]
    if v.size < 2 or v.min() == v.max():
        return None
    bins = n_bins if n_bins is not None else "auto"
    edges = np.histogram_bin_edges(v, bins=bins)
    return gm.histogram_normalized(v, edges)


def _cycle_stats(fb: np.ndarray, tb: np.ndarray) -> dict | None:
    """Cycle-basis lengths → auto-binned normalized histogram (< 80) →
    exp1/power2 fits (the Actual variants use MATLAB auto-binning; numpy
    'auto' is the documented distribution-level equivalent, doc Open Q4)."""
    lengths = gm.cycle_basis_lengths(fb, tb)
    if not lengths.size:
        return None
    edges = np.histogram_bin_edges(lengths, bins="auto")
    hist = gm.histogram_normalized(lengths, edges)
    keep = hist["bin_mid"] < 80
    x, y = hist["bin_mid"][keep], hist["norm"][keep]
    return {"lengths": lengths, "mean_length": float(lengths.mean()),
            "hist": hist, "exp_fit": gm.exp_fit(x, y),
            "power_fit": gm.power_fit(x, y), "weibull_fit": weibull_fit(x, y)}


# ---------------------------------------------------------------------------
# RealismMetricsActualEI/WECC — one parameterized port
# ---------------------------------------------------------------------------

def realism_metrics_actual(case: str, data_dir, *,
                           include_sogol: bool = False,
                           use_cache: bool = True) -> dict:
    """Real-network statistics for case 'EI' or 'WECC' (algorithm doc
    RealismMetricsActual.md — both MATLAB variants in one function; the
    WECC differences are data-driven: no XF columns → no XF stats).

    Returns {'branch': {...}, 'bus': {...}, 'form715': {...},
    'form1': {...}} where branch/bus mirror realism_metrics() output keys
    (degree_fits, cycles, bbr) computed on the two sanitized cases.
    Heavy graph work is cached under data_dir/comparison_cache/."""
    case = case.upper()
    if case not in ("EI", "WECC"):
        raise ValueError(f"case must be 'EI' or 'WECC', got {case!r}")
    data_dir = Path(data_dir)
    cache = data_dir / "comparison_cache" / f"{case.lower()}_actual.json"
    if use_cache and cache.exists():
        return json.loads(cache.read_text())

    subs = pd.read_parquet(data_dir / f"{case.lower()}_substations.parquet")
    buses = pd.read_parquet(data_dir / f"{case.lower()}_buses.parquet")
    br = pd.read_parquet(data_dir / f"{case.lower()}_branch.parquet")
    form1 = pd.read_parquet(data_dir / "form1.parquet")

    out: dict = {"case": case}
    for flavor, stem in (("branch", f"{case}BranchSanitized"),
                         ("bus", f"{case}LargeIslandSanitized")):
        isl = _load_case_island(data_dir, stem)
        deg = np.bincount(np.concatenate([isl["fb"], isl["tb"]]),
                          minlength=isl["n_buses"])
        stats = {
            "bbr": len(isl["branch"]) / isl["n_buses"],
            "degree_fits": {
                # table-driven variants (MATLAB Data.<R>.Substations/Bus
                # columns; sub CountConnections arrives migrated as
                # BusCountConnectionsSum)
                "node_degree": _degree_fit(deg),
                "sub_degree_lines": _degree_fit(
                    subs["CountLine"].to_numpy(float).astype(int)),
                "sub_degree": _degree_fit(
                    subs["BusCountConnectionsSum"].to_numpy(float).astype(int)),
                "bus_degree_lines": _degree_fit(
                    buses.loc[buses["CountLine"] > 0, "CountLine"]
                    .to_numpy(float).astype(int)),
                "bus_degree": _degree_fit(
                    buses["Neighbors"].to_numpy(float).astype(int)),
                "bus_degree_in_service": _degree_fit(
                    buses["NeighborsInservice"].to_numpy(float).astype(int)),
            },
            "cycles": _cycle_stats(isl["fb"], isl["tb"]),
            # bus voltage-class counts (BASE_KV = MATPOWER col 9, snapped —
            # on a COPY, the MATLAB in-place case mutation is not ported)
            "bus_kv_counts": {
                int(kv): int((nearest_voltage_class(isl["bus"][:, 9]) == kv).sum())
                for kv in VOLTAGE_CLASSES},
        }
        if include_sogol:
            n = isl["n_buses"]
            x = isl["branch"][:, 3]
            vec, val = gm.eigencentrality(isl["fb"], isl["tb"], n)
            lam2, fv = gm.fiedler_vector(isl["fb"], isl["tb"], n)
            stats["sogol"] = {
                "eigencentrality": vec, "dominant_eigenvalue": val,
                "fiedler_lambda2": lam2, "fiedler_vector": fv,
                "normalized_laplacian_eigs":
                    gm.normalized_laplacian_eigs(isl["fb"], isl["tb"], n),
                "b_singular_values":
                    gm.b_matrix_singular_values(isl["fb"], isl["tb"], x, n),
            }
        out[flavor] = stats

    # ---- Form 715 line/XF stats (branch table) -----------------------------
    lines = br[br["IsXF"].astype(str).str.strip().str.lower() == "no"].copy()
    lines = lines[lines["LineLengthXBMiles"].notna()
                  & (lines["LineLengthXBMiles"] >= 1e-5)]
    lines["kv_class"] = nearest_voltage_class(lines["NomkVFrom"].to_numpy(float))
    kvc = lines["kv_class"].to_numpy()
    form715: dict = {
        "length": _length_stats(lines["LineLengthXBMiles"], kvc),
        "branch": _branch_stats(lines["LineLengthXBMiles"],
                                lines["LimitMVAA"], kvc),
        "mva_hist": {int(kv): _mva_hist(
            lines.loc[kvc == kv, "LimitMVAA"].to_numpy(float),
            int(kv), LOW_MVA, HIGH_MVA, MVA_BINS.get(int(kv)))
            for kv in VOLTAGE_CLASSES[:-1]},
    }
    has_xf_to = br.columns.intersection(["NomkVTo"]).size > 0
    xf = br[br["IsXF"].astype(str).str.strip().str.lower() == "yes"].copy()
    if len(xf) and has_xf_to:
        xf["kv_from"] = nearest_voltage_class(xf["NomkVFrom"].to_numpy(float))
        xf["kv_to"] = nearest_voltage_class(xf["NomkVTo"].to_numpy(float))
        xf = xf[~((xf["kv_from"] == 69) & (xf["kv_to"] != 0))]  # dist. XFs out
        # kept SEPARATE from the line histograms — the MATLAB EI variant
        # overwrote them (doc Open Q2)
        form715["mva_hist_xf"] = {int(kv): _mva_hist(
            xf.loc[(xf["kv_from"] == kv) | (xf["kv_to"] == kv),
                   "LimitMVAA"].to_numpy(float),
            int(kv), LOW_XF_MVA, HIGH_XF_MVA, None)
            for kv in VOLTAGE_CLASSES[:-1]}
    out["form715"] = form715

    # ---- Form 1 stats -------------------------------------------------------
    f1_kv = nearest_voltage_class(form1["kV"].to_numpy(float))
    out["form1"] = {"length": _length_stats(form1["Length"], f1_kv),
                    "branch": _branch_stats(form1["Length"],
                                            form1["Capacity"], f1_kv)}

    if use_cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(out, default=_jsonable))
    return out


def _jsonable(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, pd.DataFrame):
        return o.to_dict(orient="list")
    if isinstance(o, (np.integer, np.floating)):
        return o.item()
    raise TypeError(f"not JSON-serializable: {type(o)}")


# ---------------------------------------------------------------------------
# RealismMetricsComparison — synthetic-side stats + the HistAreas table
# ---------------------------------------------------------------------------

def comparison_stats(model: CrucibleModel, config: CrucibleConfig,
                     table: str = "branches",
                     include_sogol: bool = False) -> dict:
    """RealismMetricsComparison compute side: RealismMetrics (converted,
    reused verbatim) + the weibull cycle fit unique to the comparison
    variant. Works on any CrucibleModel-shaped network — the KVL output or
    a KCL-only (gridforge) network adapted by network_graph_comparison."""
    out = realism_metrics(model, config, table=table,
                          include_sogol=include_sogol)
    if out["cycles"] is not None:
        hist = out["cycles"]["hist"]
        keep = hist["bin_mid"] < 80
        out["cycles"]["weibull_fit"] = weibull_fit(hist["bin_mid"][keep],
                                                   hist["norm"][keep])
    return out


def hist_areas(cycle_fits: dict[str, dict | None]) -> pd.DataFrame:
    """The HistAreas comparison table: one row per network label (MATLAB
    rows: KCL/KVL, KCL-Only, EI All Branches, EI Positive Impedances, WECC
    All Branches, WECC Positive Impedances), columns Exp_RSq / Area_Exp /
    Power_RSq / Area_Power — the areas are trapz over each fit curve
    evaluated on its own bin midpoints. Fits are keyed by LABEL (the MATLAB
    positional contract is deliberately not ported — doc §Interface
    hazard)."""
    rows = []
    for label, c in cycle_fits.items():
        row: dict = {"Label_CB": label, "Exp_RSq": np.nan, "Area_Exp": np.nan,
                     "Power_RSq": np.nan, "Area_Power": np.nan}
        if c is not None:
            e, p = c.get("exp_fit"), c.get("power_fit")
            if e:
                row["Exp_RSq"], row["Area_Exp"] = e["r2"], e["area"]
            if p:
                row["Power_RSq"], row["Area_Power"] = p["r2"], p["area"]
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# ElectricalComparison — the network × loads DC-PF / DC-OPF grid
# ---------------------------------------------------------------------------

def electrical_comparison(networks: dict[str, CrucibleModel],
                          scenario_sets: dict[str, np.ndarray],
                          combos: list[tuple[str, str]] | None = None,
                          table: str = "branches") -> tuple[pd.DataFrame,
                                                            pd.DataFrame]:
    """ElectricalComparison.m port (doc §The comparison grid): run each
    (network, loads) combo's scenarios through the plain DC PF (DCStats)
    and the DC OPF (DCOPFStats), concatenating rows.

    networks: label → CrucibleModel (bus-level ids already harmonized —
    the KCL↔KVL remapping is network_graph_comparison's job).
    scenario_sets: label → complex (n_buses × n_scen) injection matrix in
    the NETWORK's bus numbering, MW (pu sniffed downstream, verbatim).
    combos: (network_label, loads_label) pairs; default = the MATLAB grid
    (every network × every loads set present). Failed combos are skipped
    with a warning (MATLAB try/catch, but logged — doc quirk 6)."""
    from ..power.dcopf import run_dcopf_stats
    from ..power.power_flow import run_dc_pf_full

    if combos is None:
        combos = [(n, l) for n in networks for l in scenario_sets]
    dc_frames, opf_frames = [], []
    for net_label, loads_label in combos:
        model = networks[net_label]
        s_base = float(model.meta.get("s_base", 100.0))
        snet = np.atleast_2d(np.asarray(scenario_sets[loads_label]))
        if snet.shape[0] != model.n_buses:
            snet = snet.T
        # zero-scenario drop, MATLAB verbatim two-step: the Σ|SNet| < 100
        # threshold applies to the matrix AFTER the 35e3 pu-sniff (per-unit
        # for real-size systems, raw MW for toy ones — doc §Common
        # transforms)
        w = snet / s_base if np.abs(snet).sum(axis=0).max() > 35e3 else snet
        snet = snet[:, np.abs(w).sum(axis=0) >= 100]
        if not snet.shape[1]:
            continue
        try:
            _, dc_stats = run_dc_pf_full(model, snet.real, table=table,
                                         network_label=net_label,
                                         loads_label=loads_label)
            opm = (dc_stats["OverloadPercentMax"].to_numpy(float)
                   if "OverloadPercentMax" in dc_stats.columns else None)
            # hand the DCOPF per-unit so its own 35e3 sniff restores MW
            # exactly (MatpowerDCOPF's re-scale step)
            opf_stats = run_dcopf_stats(model, snet / s_base,
                                        network=net_label,
                                        loads=loads_label, table=table,
                                        dc_overload_percent_max=opm)
        except Exception as e:  # MATLAB swallowed these silently; we log
            warnings.warn(f"electrical_comparison combo ({net_label!r}, "
                          f"{loads_label!r}) skipped: {type(e).__name__}: {e}",
                          stacklevel=2)
            continue
        dc_frames.append(dc_stats)
        opf_frames.append(opf_stats)
    empty = pd.DataFrame()
    return (pd.concat(dc_frames, ignore_index=True) if dc_frames else empty,
            pd.concat(opf_frames, ignore_index=True) if opf_frames else empty)


def realism_metrics_comparison(model: CrucibleModel,
                               garver_model: CrucibleModel,
                               targets: dict) -> dict:
    """Aggregate: comparison_stats on both synthetic networks + the real
    targets → labeled cycle-fit dict + HistAreas table (figures → 
    viz/plots.py)."""
    raise NotImplementedError(
        "assemble after network_graph_comparison lands (needs the gridforge "
        "run-dir adapter; see algorithms/NetworkGraphComparison.md)")


def kcl_kvl_bus_map(data_dir) -> pd.DataFrame:
    """The BusMapping.mat replacement (NetworkGraphComparison.md
    §BusMapping — RESOLVED 2026-07-10): the .mat is absent from the repo
    but fully derivable. BusInputDataGarver.csv (KCL numbering) and
    BusInputData.csv (KVL numbering) contain the SAME 75,432 uniquely-named
    buses in different row orders (9.5% of rows moved; Num == row+1 in
    both), so the mapping is exactly the BusName join.

    Returns a DataFrame [KCLBus, KVLBus, BusName] — 1-based external bus
    numbers on both sides, matching the MATLAB table's columns."""
    data_dir = Path(data_dir)
    kvl = pd.read_parquet(data_dir / "bus_input_data.parquet")
    garver_pq = data_dir / "bus_input_data_garver.parquet"
    if not garver_pq.exists():
        raise FileNotFoundError(
            f"{garver_pq} — re-run `python -m gridcrucible.io.matlab_import` "
            "(BusInputDataGarver.csv was added to the migration 2026-07-10)")
    kcl = pd.read_parquet(garver_pq)
    m = pd.merge(
        kvl[["Num", "BusName"]].rename(columns={"Num": "KVLBus"}),
        kcl[["Num", "BusName"]].rename(columns={"Num": "KCLBus"}),
        on="BusName", validate="one_to_one")
    if len(m) != len(kvl):
        raise ValueError(f"bus-name join incomplete: {len(m)} of {len(kvl)} "
                         "— the two BusInputData files diverged")
    return m[["KCLBus", "KVLBus", "BusName"]]


def network_graph_comparison(model: CrucibleModel, config: CrucibleConfig,
                             data_dir) -> dict:
    """NetworkGraphComparison port: adapt a GridForge (KCL-only) run into
    CrucibleModel-shaped networks + harmonized bus mappings. The bus
    mapping is SOLVED (kcl_kvl_bus_map above); the remaining dependency is
    gridforge Garver/DC1 run artifacts to adapt (their checkpoint format —
    gridforge's pipeline hasn't produced a full run yet)."""
    raise NotImplementedError(
        "needs gridforge Garver/DC1 run artifacts to adapt (bus mapping is "
        "solved: kcl_kvl_bus_map) — see algorithms/NetworkGraphComparison.md")
