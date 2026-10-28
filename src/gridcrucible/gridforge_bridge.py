"""Adapters wiring the shared GridForge implementations into GridCrucible's
injection points.

The shared modules are vendored under :mod:`gridcrucible._gridforge` (a pinned
copy of just the GridForge helpers GridCrucible reuses), so this repository is
self-contained; these adapters are the single place that reshapes GridCrucible
data into those helpers' signatures and back. Provided adapters:
* ``uced``                — EconomicDispatcher.solve_uced behind the
                            (fleet, hydro_scaling, load, reserve) signature
* ``make_candidate_params_fn`` — the KVL CreateOLMatrices.m parameterization
                            (descending-kV mean-capacity init, gen-gen
                            removal) with impedances from gridforge
                            compute_rxb [RXB.m]
* ``make_impedance_fn``   — TUNE's post-conductor-change RXB recompute
* ``make_gmd_tables``     — regional GMD npz (gridforge GMDTables format)
                            built from the gmd_tables.parquet
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from .config import CrucibleConfig
from .model import CrucibleModel

#: MATLAB fleet schema (ours) <-> gridforge GridModel generator schema.
_TO_GF = {"BusNum": "bus", "MinMW": "min_mw", "MaxMW": "max_mw",
          "MinMVar": "min_mvar", "MaxMVar": "max_mvar",
          "FuelType": "fuel", "UnitType": "unit_type",
          "A": "cost_a", "B": "cost_b", "C": "cost_c"}
_FROM_GF = {v: k for k, v in _TO_GF.items()}


def uced(fleet: pd.DataFrame, hydro_scaling: float, load_mw: np.ndarray,
         reserve: float, gurobi_env=None) -> tuple[np.ndarray, pd.DataFrame]:
    """Economic dispatch via gridforge's EconomicDispatcher (UCED.m)."""
    from gridcrucible._gridforge.dispatch.economic_dispatch import EconomicDispatcher

    g = fleet.rename(columns=_TO_GF)
    res = EconomicDispatcher(g).solve_uced(
        float(np.asarray(load_mw, float).sum()),
        reserve=reserve, hydro_scaling=hydro_scaling)
    out = res.generators.rename(columns=_FROM_GF)
    return np.asarray(res.dispatch_mw, float), out


def make_gmd_tables(config: CrucibleConfig, region: str | None = None):
    """Build (and cache) the region's GMD npz in gridforge GMDTables format
    (gmd_table (1000×6) + xd + xd_prime) from the parquets."""
    from gridcrucible._gridforge.network.impedance import GMDTables

    region = region or config.case.regions[0]
    d = Path(config.io.data_dir)
    npz = d / f"gmd_{region.lower()}.npz"
    if not npz.exists():
        gmd = pd.read_parquet(d / "gmd_tables.parquet")
        gmd = gmd[gmd["region"] == region]
        if gmd.empty:
            raise ValueError(f"no GMD table for region {region!r}")
        table = gmd.pivot(index="gmd_index", columns="kv", values="gmd_ft") \
            .sort_index().to_numpy()
        xd = pd.read_parquet(d / "gmd_generator_reactances.parquet")
        xd = xd[xd["region"] == region].sort_values("n")
        np.savez_compressed(npz, gmd_table=table,
                            xd=xd["xd"].to_numpy(),
                            xd_prime=xd["xd_prime"].to_numpy())
    return GMDTables(npz)


def _kvl_variant_dir(config: CrucibleConfig) -> Path:
    """Build (and cache) a gridforge-format conductor variant dir from OUR
    KVL master catalog — the 148-row KVL set differs from gridforge's
    264-row national set, and conductor_idx values must index the SAME
    pooled table everywhere."""
    from .network.conductors import load_conductor_catalog

    out = Path(config.io.data_dir) / "conductors_kvl"
    pooled = out / "pooled.parquet"
    if not pooled.exists():
        out.mkdir(parents=True, exist_ok=True)
        cat = load_conductor_catalog(config.io.data_dir).reset_index(drop=True)
        cat = cat.rename(columns={
            "size": "kcmil", "cond_phase": "cond_per_phase",
            "ra": "ra_ohm_per_mi", "xa": "xa_ohm_per_mi",
            "xap": "xap_mohm_mi"})
        cat.to_parquet(pooled, index=False)
        for kv, grp in cat.groupby("kv"):
            grp.to_parquet(out / f"kv_{int(kv)}.parquet", index=False)
    return out


def make_impedance_fn(config: CrucibleConfig) -> Callable:
    """TUNE's impedance_fn: recompute per-unit R/X/B after conductor changes
    (gridforge compute_rxb, pooled-table indexing = our master cond_index)."""
    from gridcrucible._gridforge.network.conductors import ConductorCatalog
    from gridcrucible._gridforge.network.impedance import compute_rxb

    gf_catalog = ConductorCatalog(_kvl_variant_dir(config))
    gmd_tables = make_gmd_tables(config)

    def impedance_fn(cand: pd.DataFrame, catalog_df: pd.DataFrame) -> None:
        rxb = compute_rxb(
            cand["length_km"].to_numpy(float),
            cand["gmd_index"].to_numpy(int),
            cand["gmd_ft"].to_numpy(float),
            cand["conductor_idx"].to_numpy(int),
            cand["voltage_kv_from"].to_numpy(float),
            gf_catalog, gmd_tables, per_voltage=False,
            s_base=config.case.s_base)
        r, x, b = rxb[0], rxb[1], rxb[2]
        cand["r_pu"] = r
        cand["reactance_x"] = x
        if "x_pu" in cand.columns:
            cand["x_pu"] = x
        cand["b_pu"] = b
        if len(rxb) > 3:
            cand["gmd_ft"] = rxb[3]

    return impedance_fn


def wire_gridcrucible(config: CrucibleConfig, gurobi_env=None,
                      run_id: str | None = None):
    """A GridCrucible with ALL gridforge-owned injection points wired to the
    real bridged implementations (uced / CreateOLMatrices / RXB). This is
    how real runs are constructed — the bare ``GridCrucible(cfg)`` falls back
    to gridforge module paths that are still shells upstream."""
    from .core import GridCrucible

    return GridCrucible(
        config, uced_fn=uced,
        candidate_params_fn=make_candidate_params_fn(config),
        impedance_fn=make_impedance_fn(config),
        gurobi_env=gurobi_env, run_id=run_id)


def make_candidate_params_fn(config: CrucibleConfig) -> Callable:
    """The KVL CreateOLMatrices.m (xpul==0 path) parameterization:

    1. drop candidate pairs DIRECTLY connecting two generator buses when
       80 mi < length < max (the KCL-KVL gen-gen trim),
    2. descending-kV init: each path lands at the LOWEST voltage whose
       max-length admits it, with the MEAN-capacity conductor of that class
       (NOT gridforge's assign_by_length, which is the KCL
       AssignCondParameters semantics),
    3. l1_cost = l1_weight × length; MaxCondIndex = global catalog max,
    4. persistent GMD draw (reference-run reuse), then gridforge compute_rxb.
    """
    impedance_fn = make_impedance_fn(config)

    def params_fn(model: CrucibleModel, catalog_df: pd.DataFrame) -> None:
        cand = model.candidates
        gens = np.sort(model.generators["BusNum"].unique())
        fb = cand["from_bus"].to_numpy(int)
        tb = cand["to_bus"].to_numpy(int)
        gen_gen = np.isin(fb, gens) & np.isin(tb, gens)
        L = cand["length_km"].to_numpy(float)
        max_len_km = max(config.case.max_length_miles) * 1.61
        drop = gen_gen & (L > 80 * 1.61) & (L < max_len_km)
        if drop.any():
            model.candidates = cand = cand[~drop].reset_index(drop=True)
        n = len(cand)
        L = cand["length_km"].to_numpy(float)

        cond_idx = np.zeros(n)
        kv_col = np.zeros(n)
        min_cond = np.ones(n)
        kcmil = np.zeros(n)
        cpp = np.zeros(n)
        rate = np.zeros(n)
        levels = sorted(config.case.voltage_levels, reverse=True)
        max_mi = dict(zip(config.case.voltage_levels,
                          config.case.max_length_miles))
        for kv in levels:                       # descending: shortest → lowest kV
            mask = L <= max_mi[kv] * 1.61
            cls = catalog_df[catalog_df["kv"] == kv]
            mid = int((cls["mva"] - cls["mva"].mean()).abs().idxmin())
            cond_idx[mask] = mid
            kv_col[mask] = kv
            min_cond[mask] = cls.index.min()
            kcmil[mask] = catalog_df.at[mid, "size"]
            cpp[mask] = catalog_df.at[mid, "cond_phase"]
            rate[mask] = catalog_df.at[mid, "mva"]
        cand["conductor_idx"] = cond_idx
        cand["voltage_kv_from"] = kv_col
        cand["voltage_kv_to"] = kv_col
        cand["min_kv"] = kv_col
        cand["min_conductor_idx"] = min_cond
        cand["max_conductor_idx"] = float(catalog_df.index.max())
        cand["kcmil"] = kcmil
        cand["cond_per_phase"] = cpp
        cand["capacity_mva"] = rate
        cand["l1_cost"] = config.tune.l1_weight * L
        cand["resize"] = False

        # persistent GMD draw AFTER the gen-gen trim (GMDIndex.mat semantics)
        ref = Path(config.io.data_dir) / "reference_run" / "gmd_index.npz"
        if not config.io.reset_gmd_index and ref.exists() \
                and len(np.load(ref)["gmd_index"]) == n:
            cand["gmd_index"] = np.load(ref)["gmd_index"].astype(int)
        else:
            cand["gmd_index"] = np.random.default_rng(
                config.case.rng_seed).integers(1, 1001, size=n)
        cand["gmd_ft"] = 0.0
        impedance_fn(cand, catalog_df)

    return params_fn
