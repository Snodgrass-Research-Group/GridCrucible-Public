"""In-loop realism metrics for the synthetic network.

MATLAB origin: RealismMetrics.m (611 loc; the ~400 lines of figure code
belong to viz/plots.py — this module computes and returns).
Algorithm doc: docs/algorithms/RealismMetrics.md.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import CrucibleConfig
from ..model import CrucibleModel
from . import graph_metrics as gm

#: MATLAB MVABins: fixed histogram bin counts per voltage level.
MVA_BINS = {69: 27, 138: 20, 230: 20, 345: 12, 500: 8, 765: 6}


def _degree_fit(deg: np.ndarray) -> dict:
    """Distribution over degrees 1..max, truncated at the FIRST ZERO entry
    (MATLAB rule), + log-linear fit (None when degenerate)."""
    deg = deg[deg > 0]
    if deg.size == 0:
        return {"degrees": np.empty(0, int), "dist": np.empty(0), "fit": None}
    counts = np.bincount(deg.astype(int))[1:]        # degrees 1..max
    dist = counts / counts.sum()
    zeros = np.flatnonzero(dist == 0)
    cut = zeros[0] if zeros.size else len(dist)
    degrees = np.arange(1, cut + 1)
    return {"degrees": degrees, "dist": dist,
            "fit": gm.log_linear_fit(degrees, dist[:cut])}


def realism_metrics(model: CrucibleModel, config: CrucibleConfig,
                    table: str = "candidates",
                    include_sogol: bool = False) -> dict:
    """RealismMetrics.m port. Computes cycle-basis stats + fits, the five
    node-degree fit variants, per-kV MVA histograms, and (opt-in) the Sogol
    spectral block. Result stored in model.meta['realism'] and returned."""
    df = getattr(model, table)
    fb = df["from_bus"].to_numpy(np.int64)
    tb = df["to_bus"].to_numpy(np.int64)
    n = model.n_buses
    tap = (df["tap"].to_numpy(float) if "tap" in df.columns
           else np.zeros(len(df)))
    lines = tap == 0
    out: dict = {}

    # 1. cycle basis + fits ---------------------------------------------------
    lengths = gm.cycle_basis_lengths(fb, tb)
    if lengths.size:
        hist = gm.histogram_normalized(lengths, np.arange(2, 81, 2))
        keep = hist["bin_mid"] < 80
        out["cycles"] = {
            "lengths": lengths, "mean_length": float(lengths.mean()),
            "hist": hist,
            "exp_fit": gm.exp_fit(hist["bin_mid"][keep], hist["norm"][keep]),
            "power_fit": gm.power_fit(hist["bin_mid"][keep], hist["norm"][keep]),
        }
    else:
        out["cycles"] = None

    # 2. five node-degree fit variants ---------------------------------------
    sub_f = (df["from_orig"] if "from_orig" in df.columns else df["from_bus"]) \
        .to_numpy(np.int64)
    sub_t = (df["to_orig"] if "to_orig" in df.columns else df["to_bus"]) \
        .to_numpy(np.int64)

    def endpoint_degree(f, t, mask):
        return np.bincount(np.concatenate([f[mask], t[mask]]), minlength=n)

    all_mask = np.ones(len(df), bool)
    out["degree_fits"] = {
        "node_degree": _degree_fit(model.node_degree(table)),
        "sub_degree_lines": _degree_fit(endpoint_degree(sub_f, sub_t, lines)),
        "sub_degree": _degree_fit(endpoint_degree(sub_f, sub_t, all_mask)),
        "bus_degree_lines": _degree_fit(endpoint_degree(fb, tb, lines)),
        "bus_degree": _degree_fit(endpoint_degree(fb, tb, all_mask)),
    }
    nd = model.node_degree(table)
    out["node_degree"] = nd
    out["mean_node_degree"] = float(nd[nd > 0].mean()) if (nd > 0).any() else 0.0

    # 3. MVA-limit histograms per voltage level -------------------------------
    out["mva_hist"] = {}
    if {"capacity_mva", "voltage_kv_from", "voltage_kv_to"} <= set(df.columns):
        kvf = df["voltage_kv_from"].to_numpy(float)
        kvt = df["voltage_kv_to"].to_numpy(float)
        cap = df["capacity_mva"].to_numpy(float)
        for kv in config.case.voltage_levels:
            rows = cap[(kvf == kv) & (kvt == kv)]
            if rows.size:
                n_bins = MVA_BINS.get(kv, 20)
                edges = np.linspace(rows.min(), rows.max() + 1e-9, n_bins + 1)
                out["mva_hist"][kv] = gm.histogram_normalized(rows, edges)

    # 4. Sogol spectral block (opt-in — codectrl.StatsSogol) ------------------
    if include_sogol:
        x = (df["x_pu"] if "x_pu" in df.columns
             else df["reactance_x"]).to_numpy(float)
        gen = np.concatenate([[model.scenarios.slack_bus],
                              model.scenarios.gen_buses]).astype(int)
        vec, val = gm.eigencentrality(fb, tb, n)
        lam2, fv = gm.fiedler_vector(fb, tb, n)
        tier_sorted, tier = gm.tier_efficiency(fb, tb, x, gen, n)
        out["sogol"] = {
            "eigencentrality": vec, "dominant_eigenvalue": val,
            "fiedler_lambda2": lam2, "fiedler_vector": fv,
            "normalized_laplacian_eigs": gm.normalized_laplacian_eigs(fb, tb, n),
            "b_singular_values": gm.b_matrix_singular_values(fb, tb, x, n),
            "tier": tier, "tier_sorted": tier_sorted,
            "node_degree_simple": gm.node_degree_simple(fb, tb, n),
        }
    model.meta["realism"] = out
    return out
