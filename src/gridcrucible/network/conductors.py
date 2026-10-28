"""Conductor assignment and thermal resizing leaves (TUNE loop helpers).

MATLAB origin: AssignConductors.m (35), ResizeBranches.m (67),
TransformerImpedances.m (81). Algorithm doc: docs/algorithms/conductors.md —
including the catalog discovery (master table = MVA-sorted concatenation,
CondIndex == row number) and the two documented deviations (no table-sort
side effect; both end-kVs updated on lines).

Impedance (R/X/B) recompute after conductor changes is gridforge-owned (RXB)
— these helpers pick CONDUCTORS and capacities only.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd


def load_conductor_catalog(data_dir: Path) -> pd.DataFrame:
    """Rebuild the MATLAB master ConductorData table: MVA-sorted, indexed by
    the 1-based cond_index (verified contiguous 1..N — MATLAB row-indexing
    semantics like `ConductorData.kV(CondIndex)` map to `catalog.loc[idx]`)."""
    cat = pd.read_parquet(Path(data_dir) / "conductor_catalog.parquet")
    cat = cat.sort_values("cond_index").reset_index(drop=True)
    ci = cat["cond_index"].astype(int)
    if not (ci == np.arange(1, len(cat) + 1)).all():
        raise ValueError("conductor catalog cond_index is not contiguous "
                         "1..N — master-table assumption violated (see "
                         "docs/algorithms/conductors.md)")
    cat.index = pd.Index(ci, name="cond_index")
    return cat


def _nearest_mva_row(catalog: pd.DataFrame, target: float,
                     kv: float | None = None) -> int:
    """1-based cond_index of the nearest-MVA conductor (optionally within a
    kV class). MATLAB: min(abs(ConductorData.MVA - rate))."""
    sub = catalog if kv is None else catalog[catalog["kv"] == kv]
    if sub.empty:
        raise ValueError(f"no conductors at {kv} kV in the catalog")
    return int((sub["mva"] - target).abs().idxmin())


def resize_branches(candidates: pd.DataFrame, flows_mw: np.ndarray, *,
                    percent_resize: float, percent_change: float,
                    conductor_margin: float, downsize_threshold: float,
                    upsize: bool, downsize: bool, downsize_keep: bool,
                    random_resize: bool, rng: np.random.Generator) -> dict:
    """ResizeBranches.m port: step capacities up/down by percent_change on at
    most ceil(percent_resize·n) rows per pass; flags `resize` for
    assign_conductors. Mutates `candidates` in place; returns pass counts."""
    n = len(candidates)
    cap_rows = math.ceil(percent_resize * n)
    util = np.abs(np.asarray(flows_mw, float)) / \
        candidates["capacity_mva"].to_numpy(float)
    candidates["percent_utilized"] = util
    cond = candidates["conductor_idx"].to_numpy(float)
    max_cond = candidates["max_conductor_idx"].to_numpy(float)
    min_cond = (candidates["min_conductor_idx"].to_numpy(float)
                if "min_conductor_idx" in candidates.columns else np.ones(n))
    keep = candidates["keep"].to_numpy(bool)
    if "resize" not in candidates.columns:
        candidates["resize"] = False
    counts = {"upsized": 0, "downsized": 0, "downsized_keep": 0}

    def _pick(mask: np.ndarray) -> np.ndarray:
        idx = np.flatnonzero(mask)
        if random_resize:
            idx = rng.permutation(idx)
        else:  # worst-first (MATLAB sorted the whole table; we sort indices)
            idx = idx[np.argsort(-util[idx])]
        return idx[:cap_rows]

    if upsize:
        over_all = util > 1.0 / conductor_margin
        eligible = over_all & (cond < max_cond)
        candidates["overloaded"] = over_all & ~eligible   # manual-review flag
        rows = _pick(eligible)
        candidates.iloc[rows, candidates.columns.get_loc("capacity_mva")] = \
            np.ceil(candidates["capacity_mva"].to_numpy(float)[rows]
                    * percent_change)
        candidates.iloc[rows, candidates.columns.get_loc("resize")] = True
        counts["upsized"] = len(rows)

    for flag, key, keep_mask in ((downsize_keep, "downsized_keep", keep),
                                 (downsize, "downsized", ~keep)):
        if not flag:
            continue
        low_all = util < downsize_threshold
        eligible = low_all & (cond > min_cond) & keep_mask
        col = "undersized_keep" if key == "downsized_keep" else "undersized"
        candidates[col] = low_all & ~eligible
        rows = _pick(eligible)
        candidates.iloc[rows, candidates.columns.get_loc("capacity_mva")] = \
            np.ceil(candidates["capacity_mva"].to_numpy(float)[rows]
                    / percent_change)
        candidates.iloc[rows, candidates.columns.get_loc("resize")] = True
        counts[key] = len(rows)
    return counts


def assign_conductors(candidates: pd.DataFrame, catalog: pd.DataFrame,
                      change_voltages: bool) -> int:
    """AssignConductors.m port (lines only — caller masks tap==0 rows).

    For resize-flagged rows: nearest-MVA conductor on the per-circuit rating
    (globally when change_voltages — may move the row to a new voltage class,
    floored at min_kv — else within the row's own class); write conductor_idx
    / voltage class / kcmil / cond_per_phase; snap capacity to n_circuits ×
    exact table MVA. Clears `resize`. Returns rows updated."""
    if "resize" not in candidates.columns:
        return 0
    for col in ("kcmil", "cond_per_phase"):
        if col not in candidates.columns:
            candidates[col] = 0.0
    rows = candidates.index[candidates["resize"].astype(bool)]
    n_circ = (candidates["n_circuits"] if "n_circuits" in candidates.columns
              else pd.Series(1.0, index=candidates.index))
    min_kv = (candidates["min_kv"] if "min_kv" in candidates.columns
              else pd.Series(catalog["kv"].min(), index=candidates.index))
    for r in rows:
        per_circuit = candidates.at[r, "capacity_mva"] / n_circ[r]
        if change_voltages:
            ci = _nearest_mva_row(catalog, per_circuit)
            if catalog.at[ci, "kv"] < min_kv[r]:
                ci = _nearest_mva_row(catalog, per_circuit, kv=min_kv[r])
        else:
            ci = _nearest_mva_row(catalog, per_circuit,
                                  kv=candidates.at[r, "voltage_kv_from"])
        candidates.at[r, "conductor_idx"] = ci
        kv = catalog.at[ci, "kv"]
        candidates.at[r, "voltage_kv_from"] = kv
        candidates.at[r, "voltage_kv_to"] = kv   # documented deviation
        candidates.at[r, "kcmil"] = catalog.at[ci, "size"]
        candidates.at[r, "cond_per_phase"] = catalog.at[ci, "cond_phase"]
        candidates.at[r, "capacity_mva"] = n_circ[r] * catalog.at[ci, "mva"]
    candidates["resize"] = False
    return len(rows)


def transformer_impedances(branches: pd.DataFrame, xfdata: pd.DataFrame,
                           xfxr: pd.DataFrame,
                           substations: pd.DataFrame | None = None) -> int:
    """TransformerImpedances.m port for tap≠0 rows (none exist before
    CreateBuses — the Sub.XFMVA memory hooks arrive with ; passing
    substations=None skips them, faithful to a fresh sizing).

    Returns rows updated. Mutates x_pu / r_pu / n_circuits / conductor_idx
    (which for transformers stores per-unit MVA — MATLAB overload, kept)."""
    xf_rows = branches.index[branches["tap"].to_numpy(float) != 0] \
        if "tap" in branches.columns else branches.index[:0]
    if len(xf_rows) == 0:
        return 0
    pri = np.sort(xfdata["Pri"].unique())
    sec = np.sort(xfdata["Sec"].unique())
    mva_grid = xfxr["MVA"].to_numpy(float)
    xr_grid = xfxr["XR"].to_numpy(float)

    def _snap(val: float, grid: np.ndarray) -> float:
        return float(grid[np.abs(grid - val).argmin()])

    updated = 0
    resize_rows = [r for r in xf_rows if branches.at[r, "resize"]] \
        if "resize" in branches.columns else list(xf_rows)
    for r in resize_rows:
        rate = float(branches.at[r, "capacity_mva"])
        kv_f = _snap(float(branches.at[r, "voltage_kv_from"]), pri)
        kv_t = _snap(float(branches.at[r, "voltage_kv_to"]), sec)
        branches.at[r, "voltage_kv_from"] = kv_f
        branches.at[r, "voltage_kv_to"] = kv_t
        circuits = float(branches.at[r, "n_circuits"]) or 1.0
        if rate < 50:
            xr = 25.0
        elif rate > 2000:
            circuits = math.ceil(rate / 2000)
            branches.at[r, "n_circuits"] = circuits
            xr = float(np.interp(rate / circuits, mva_grid, xr_grid))
        else:
            xr = float(np.interp(rate, mva_grid, xr_grid))
        pair = xfdata[(xfdata["Pri"] == kv_f) & (xfdata["Sec"] == kv_t)]
        if pair.empty:
            pair = xfdata[(xfdata["Sec"] == kv_f) & (xfdata["Pri"] == kv_t)]
        if pair.empty:
            raise ValueError(f"no XFData entry for {kv_f}/{kv_t} kV")
        z_pu = (100.0 / rate) * (float(pair["XHL"].iloc[0]) / 100.0)
        x = math.sqrt(z_pu ** 2 / (1.0 + 1.0 / xr ** 2))
        branches.at[r, "x_pu" if "x_pu" in branches.columns
                    else "reactance_x"] = x
        if "r_pu" in branches.columns:
            branches.at[r, "r_pu"] = x / xr
        updated += 1
    branches.loc[list(xf_rows), "conductor_idx"] = np.ceil(
        branches.loc[list(xf_rows), "capacity_mva"].to_numpy(float)
        / branches.loc[list(xf_rows), "n_circuits"].to_numpy(float))
    if "resize" in branches.columns:
        branches.loc[list(xf_rows), "resize"] = False
    return updated
