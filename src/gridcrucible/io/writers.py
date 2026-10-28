"""Report/export writers.

MATLAB origin: WriteLoads.m (24), WritePowerWorld.m (186 — FORMAT-CRITICAL:
writetable header rows + column order must import into PowerWorld),
WriteMetrics.m (68), WriteSubstationData.m (31), plus the inline writetable
calls in main_SRTV §8/§10/§11 (SubstationData.xlsx, BusData.xlsx,
'Synthetic <region> Electrical DC {Power Flow,OPF} Statistics.xlsx',
'KCL-KVL <region> AC Power Flow Statistics.xlsx').

Algorithm doc: docs/algorithms/writers.md — every column order, gating rule
and flagged MATLAB quirk below is specified there (incl. the 6-vs-7-arg
call-site bug: the port implements the INTENDED 7-arg behavior).
NOTE this file's CSVs use MATLAB's writetable layout (one header row of
column names); the cell2csv 'Branch'-section aux layout belongs to
gridforge io/writers.py (SCALE export) and the orphaned AC inner-loop
LineImport debug dumps — not here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..config import CrucibleConfig
from ..model import ACPFResult, CrucibleModel

#: Emergency / load-shed MVA rating ladder (WritePowerWorld lines 62-63).
#: Gridforge's SCALE export uses 1/1.05/1.1 (lines) and 1/1.25/1.5 (XFs);
#: each MATLAB writer keeps its own ladder — do not unify.
RATE_B_FACTOR = 1.10
RATE_C_FACTOR = 1.15

#: The five synthetic ND-fit variants, in the MATLAB Stats.ND append order
#: (WriteMetrics sheet-5 rows). Keys of realism_metrics()['degree_fits'].
ND_LABELS = ("node_degree", "sub_degree_lines", "sub_degree",
             "bus_degree_lines", "bus_degree")

#: HistAreas row labels in MATLAB order → (case key, flavor key) into the
#: realism_metrics_actual() result ('bus' = LargeIslandSanitized = all
#: branches; 'branch' = BranchSanitized = positive impedances).
HIST_AREA_REAL_ROWS = (
    ("EI All Branches", "EI", "bus"),
    ("EI Positive Impedances", "EI", "branch"),
    ("WECC All Branches", "WECC", "bus"),
    ("WECC Positive Impedances", "WECC", "branch"),
)


def _region(model: CrucibleModel) -> str:
    regions = model.meta.get("regions") or ["SRTV"]
    return "".join(regions) if isinstance(regions, (list, tuple)) \
        else str(regions)


def _write_csv(df: pd.DataFrame, path: Path) -> Path:
    """MATLAB writetable: one header row of column names, no index."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, lineterminator="\n")
    return path


def _pu_sniff(snet: np.ndarray, total_load_mw: float,
              s_base: float) -> np.ndarray:
    """WritePowerWorld/WriteLoads per-unit sniff: if the first scenario's
    |P| sum ×40 is below the table's MW total, the matrix is per-unit →
    rescale by SBase (verbatim, incl. the ×40 magic factor)."""
    snet = np.atleast_2d(np.asarray(snet, complex))
    if snet.shape[0] == 1:
        snet = snet.T
    if np.abs(snet[:, 0].real).sum() * 40.0 < total_load_mw:
        snet = snet * s_base
    return snet


# ---------------------------------------------------------------------------
# WritePowerWorld — the PowerWorld import CSVs (format-critical)
# ---------------------------------------------------------------------------

def write_powerworld(model: CrucibleModel, acpf: ACPFResult | None,
                     out_dir: Path, network_label: str, *,
                     region: str | None = None,
                     snet: np.ndarray | None = None,
                     load_sign: int = -1) -> dict[str, Path]:
    """WritePowerWorld.m port (INTENDED 7-arg behavior — doc Open Q1).

    network_label ∈ {'KCL-KVL', 'KCL-Only'}: main_SRTV exports BOTH the
    crucible network and the Garver twin. Filenames are the MATLAB
    suffixed form ``<Section>Import<Network><Region>.csv``.

    Loads: ``snet`` (complex, buses × scenarios; column 0 used) with
    ``load_sign=+1`` for ACPF-style load-positive matrices and ``-1``
    (default) for net-injection matrices (MATLAB's two branches);
    ``snet=None`` falls back to the bus-table MW/MVar columns.
    Gen / SwitchedShunts sections are emitted iff ``model.buses`` carries
    the MATLAB-equivalent columns (gen_mw_set+p_net+q_net / sync_cond+
    q_net) — the same presence-gating as MATLAB.
    Returns {section name: written path}.
    """
    if not len(model.buses):
        raise ValueError("write_powerworld needs model.buses (run "
                         "create_buses first; MATLAB gated on exist('Bus'))")
    if not len(model.branches):
        raise ValueError("write_powerworld needs model.branches (run "
                         "create_branch first)")
    out_dir = Path(out_dir)
    region = _region(model) if region is None else region
    suffix = f"{network_label}{region}"
    s_base = float(model.meta.get("s_base", 100.0))
    buses = model.buses
    subs = model.substations
    br = model.branches
    ext = buses["external_num"].to_numpy()
    written: dict[str, Path] = {}

    # -- 1. BusImport --------------------------------------------------------
    n = len(buses)
    pw_bus = pd.DataFrame({
        "Number": ext,
        "Name": buses["name"].to_numpy(),
        "NomkV": buses["voltage_kv"].to_numpy(float),
        "AreaNumber": np.ones(n, int),
        "ZoneNumber": np.ones(n, int),
        "NomB": np.zeros(n),
        "NomG": np.zeros(n),
        "Latitude": buses["lat"].to_numpy(float),
        "Longitude": buses["lon"].to_numpy(float),
    })
    if acpf is not None and acpf.v_mag is not None and acpf.v_angle is not None:
        pw_bus["Vpu"] = np.asarray(acpf.v_mag, float)     # before SubName,
        pw_bus["Vangle"] = np.asarray(acpf.v_angle, float)  # as in MATLAB
    pw_bus["SubName"] = buses["name"].to_numpy()
    written["bus"] = _write_csv(pw_bus, out_dir / f"BusImport{suffix}.csv")

    # -- 2. SubstationImport -------------------------------------------------
    pw_sub = pd.DataFrame({
        "Number": subs["external_num"].to_numpy(),
        "Name": subs["name"].to_numpy(),
        "Latitude": subs["lat"].to_numpy(float),
        "Longitude": subs["lon"].to_numpy(float),
    })
    written["substation"] = _write_csv(
        pw_sub, out_dir / f"SubstationImport{suffix}.csv")

    # -- 3. BranchImport -----------------------------------------------------
    m = len(br)
    if {"from_ext", "to_ext"} <= set(br.columns):
        fb_ext, tb_ext = br["from_ext"].to_numpy(), br["to_ext"].to_numpy()
    else:
        fb_ext = ext[br["from_bus"].to_numpy(int)]
        tb_ext = ext[br["to_bus"].to_numpy(int)]
    is_line = (br["tap"].to_numpy(float) == 0) if "tap" in br.columns \
        else np.ones(m, bool)
    x = (br["x_pu"] if "x_pu" in br.columns
         else br["reactance_x"]).to_numpy(float)
    r = br["r_pu"].to_numpy(float) if "r_pu" in br.columns else np.zeros(m)
    b = br["b_pu"].to_numpy(float) if "b_pu" in br.columns else np.zeros(m)
    rate_a = br["capacity_mva"].to_numpy(float)
    pw_branch = pd.DataFrame({
        "BusNumFrom": fb_ext,
        "BusNumTo": tb_ext,
        "Circuit": np.ones(m, int),
        "BranchDeviceType": np.where(is_line, "Line", "Transformer"),
        "IsXF": np.where(is_line, "NO", "YES"),
        "R": r, "X": x, "B": b,
        "LimitMVAA": rate_a,
        "LimitMVAB": RATE_B_FACTOR * rate_a,
        "LimitMVAC": RATE_C_FACTOR * rate_a,
        # MATLAB writes Branch.Length RAW (km) — no /1.61 (doc Open Q3)
        "LineLength": br["length_km"].to_numpy(float),
    })
    written["branch"] = _write_csv(
        pw_branch, out_dir / f"BranchImport{suffix}.csv")

    # -- 4. PeakLoadImport ---------------------------------------------------
    total_mw = float(buses["load_mw"].sum()) if "load_mw" in buses.columns \
        else 0.0
    if snet is not None:
        w = _pu_sniff(snet, total_mw, s_base)
        col0 = w[:, 0]
        # +1 → ACPF load-positive branch (real > 0 are loads);
        # -1 → net-injection branch (real < 0 are loads)
        load_rows = (col0.real * load_sign) > 0
        smw = np.where(load_rows, np.abs(col0.real), 0.0)
        smvar = np.where(load_rows, np.abs(col0.imag), 0.0)
        pw_load = pd.DataFrame({
            "BusNum": ext, "ID": 1, "Status": "Closed",
            "SMW": smw, "SMvar": smvar})
        if load_sign > 0:   # ACPF branch adds MW/Mvar columns (doc §4)
            pw_load["MW"] = smw
            pw_load["Mvar"] = smvar
    else:                   # Bus.MW/MVar fallback (has MW/Mvar columns)
        smw = buses["load_mw"].to_numpy(float)
        smvar = buses["load_mvar"].to_numpy(float)
        pw_load = pd.DataFrame({
            "BusNum": ext, "ID": 1, "Status": "Closed",
            "SMW": smw, "SMvar": smvar, "MW": smw, "Mvar": smvar})
    pw_load = pw_load[pw_load["SMW"] != 0.0].reset_index(drop=True)
    written["load"] = _write_csv(
        pw_load, out_dir / f"PeakLoadImport{suffix}.csv")

    # -- 5. GenImport (gated on bus-level dispatch columns) ------------------
    if {"gen_mw_set", "p_net", "q_net"} <= set(buses.columns) \
            and len(model.generators):
        gen = model.generators
        gen_bus = gen["BusNum"].to_numpy(int)
        committed = set(np.flatnonzero(
            buses["gen_mw_set"].to_numpy(float) > 0.1))
        # MATLAB setdiff/intersect return the FIRST unit per bus only
        # (doc Open Q4) — reproduced via first-occurrence masks.
        first_unit = ~pd.Series(gen_bus).duplicated().to_numpy()
        is_committed_bus = np.isin(gen_bus, list(committed))
        com_first = first_unit & is_committed_bus
        decom_first = first_unit & ~is_committed_bus

        p_net = buses["p_net"].to_numpy(float)
        q_net = buses["q_net"].to_numpy(float)
        max_mw = gen["MaxMW"].to_numpy(float)
        status = np.where(decom_first, "Open", "Closed")
        mw_set = np.where(com_first, np.abs(p_net[gen_bus]), 0.0)
        mw_set[mw_set < 0.1] = 0.0                # numerical-tolerance fix
        mvar_set = np.where(com_first, np.abs(q_net[gen_bus]), 0.0)
        pw_gen = pd.DataFrame({
            "BusNum": ext[gen_bus],               # EXTERNAL (Gen.BusNumExt)
            "ID": gen["ID"].to_numpy(),
            "Status": status,
            "FuelType": gen["FuelType"].to_numpy(),
            "UnitType": gen["UnitType"].to_numpy(),
            "MWMin": gen["MinMW"].to_numpy(float),
            "MWMax": max_mw,
            "MvarMin": gen["MinMVar"].to_numpy(float),
            "MvarMax": gen["MaxMVar"].to_numpy(float),
            "FixedCostInd": gen["A"].to_numpy(float),
            "CubicB": gen["B"].to_numpy(float),
            "CubicC": gen["C"].to_numpy(float),
            "CostModel": "Cubic",
            "MWSetPoint": mw_set,
            "MvarSetpoint": mvar_set,
            "VoltSet": 1,
            "AGC": "YES",
            "AVR": "YES",
            "PartFact": max_mw / max_mw.sum(),
        })
        written["gen"] = _write_csv(
            pw_gen, out_dir / f"GenImport{suffix}.csv")

    # -- 6. SwitchedShuntsImport (gated on sync_cond + q_net) ----------------
    if {"sync_cond", "q_net"} <= set(buses.columns):
        sc = np.flatnonzero(buses["sync_cond"].to_numpy(bool))
        if sc.size:
            q = buses["q_net"].to_numpy(float)[sc]
            is_b = q > 0                    # injection-positive convention
            is_g = q < 0                    # q == 0 rows fall in NEITHER
            nom = 1.3 * np.abs(q)
            step1 = np.zeros(len(sc))
            step1[is_g] = -np.ceil(nom[is_g] / 10.0)
            step1[is_b] = -1.0              # small opposite-direction range
            step2 = np.zeros(len(sc))
            step2[is_g] = 1.0
            step2[is_b] = np.ceil(nom[is_b] / 10.0)
            mvar_nom = np.zeros(len(sc))
            mvar_nom[is_b] = np.ceil(np.abs(q[is_b]))
            mvar_nom[is_g] = -np.ceil(np.abs(q[is_g]))
            pw_ss = pd.DataFrame({
                "BusNum": ext[sc], "ID": 1, "Status": "Closed",
                "RegBusNum": ext[sc],
                "ShuntMode": "Continuous", "RegulationType": "Volt",
                "RegHigh": 1.06, "RegLow": 0.94, "RegTarget": 1,
                "BlockMvarPerStep1": step1, "BlockNumberStep1": 10,
                "BlockMvarPerStep2": step2, "BlockNumberStep2": 10,
                "MvarNom": mvar_nom,
            })
            written["switched_shunts"] = _write_csv(
                pw_ss, out_dir / f"SwitchedShuntsImport{suffix}.csv")

    return written


# ---------------------------------------------------------------------------
# WriteLoads — headerless time-series load matrices
# ---------------------------------------------------------------------------

def write_loads(model: CrucibleModel, acpf: ACPFResult | None,
                out_dir: Path, *,
                snet: np.ndarray | None = None) -> tuple[Path, Path]:
    """WriteLoads.m port: TimeSeriesMWLoad.csv / TimeSeriesMVarLoad.csv —
    TRANSPOSED (rows = scenarios, cols = buses) |real| / |imag| matrices,
    headerless (writematrix). Explicit ``snet`` = the PF.SNet branch (pu
    sniff applied); None = the PF.SNetComplete fallback (no sniff)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if snet is not None:
        total_mw = float(model.buses["load_mw"].sum()) if len(model.buses) \
            else float(model.substations["load_mw"].sum())
        w = _pu_sniff(snet, total_mw, float(model.meta.get("s_base", 100.0)))
    else:
        w = model.scenarios.tiers.get("complete")
        if w is None:
            raise ValueError("write_loads: no snet given and no 'complete' "
                             "tier (MATLAB PF.SNetComplete) available")
        w = np.atleast_2d(np.asarray(w, complex))
    mw_path = out_dir / "TimeSeriesMWLoad.csv"
    mvar_path = out_dir / "TimeSeriesMVarLoad.csv"
    np.savetxt(mw_path, np.abs(w.real).T, delimiter=",", fmt="%.10g")
    np.savetxt(mvar_path, np.abs(w.imag).T, delimiter=",", fmt="%.10g")
    return mw_path, mvar_path


# ---------------------------------------------------------------------------
# WriteMetrics — 'KCL-KVL <Region> Statistics.xlsx' (5 sheets)
# ---------------------------------------------------------------------------

def _fit_row(fits: dict | None, label: str) -> tuple[float, float]:
    """(area, r²) of one label-keyed ND fit; NaN when missing/degenerate."""
    if not fits:
        return (np.nan, np.nan)
    entry = fits.get(label)
    fit = entry.get("fit") if entry else None
    if not fit:
        return (np.nan, np.nan)
    return (float(fit["area"]), float(fit["r2"]))


def hist_areas_table(synthetic: dict, garver: dict | None,
                     targets: dict) -> pd.DataFrame:
    """The Sheet-4 HistAreas table, rows in MATLAB order, label-keyed
    (doc §WriteMetrics): KCL/KVL, KCL-Only, then the four real rows."""
    from ..metrics.comparison import hist_areas

    fits: dict[str, dict | None] = {"KCL/KVL": synthetic.get("cycles")}
    fits["KCL-Only"] = garver.get("cycles") if garver else None
    for label, case, flavor in HIST_AREA_REAL_ROWS:
        entry = targets.get(case)
        fits[label] = (entry.get(flavor) or {}).get("cycles") if entry \
            else None
    return hist_areas(fits)


def nd_areas_table(synthetic: dict, garver: dict | None,
                   targets: dict) -> pd.DataFrame:
    """The Sheet-5 NDAreas table. MATLAB sliced the real ND structs
    positionally (Temp1(2:end-1)) — the port joins by LABEL (doc Open Q7).
    garver=None reproduces the params==Garverparams zero columns."""
    syn = synthetic.get("degree_fits", {})
    gar = garver.get("degree_fits") if garver else None
    rows = []
    for label in ND_LABELS:
        area, r2 = _fit_row(syn, label)
        row = {"Label_ND": label, "KVL_Area": area, "KVL_RSq": r2}
        if gar is not None:
            row["KCL_Area"], row["KCL_RSq"] = _fit_row(gar, label)
        else:
            row["KCL_Area"], row["KCL_RSq"] = 0.0, 0.0
        for case in ("EI", "WECC"):
            entry = targets.get(case) or {}
            for flavor, col in (("branch", "Branch"), ("bus", "Bus")):
                a, r = _fit_row((entry.get(flavor) or {}).get("degree_fits"),
                                label)
                row[f"{case}{col}Area"] = a
                row[f"{case}{col}RSq"] = r
        rows.append(row)
    return pd.DataFrame(rows)


def write_metrics(synthetic: dict, garver: dict | None, targets: dict,
                  config: CrucibleConfig, out_dir: Path, *,
                  branch_stats: pd.DataFrame | None = None,
                  bus_stats: pd.DataFrame | None = None,
                  length_stats: pd.DataFrame | None = None) -> Path:
    """WriteMetrics.m port: the 5-sheet statistics workbook.

    synthetic / garver: realism_metrics() / comparison_stats() dicts;
    targets: {'EI': realism_metrics_actual('EI',...), 'WECC': ...} (either
    may be absent). Sheets 1-3 (Branch/Bus/Length) come from gridforge
    NetworkStatistics — passed in, not recomputed (empty when None)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    region = "".join(config.case.regions)      # MATLAB: options.Region
    path = out_dir / f"KCL-KVL {region} Statistics.xlsx"
    with pd.ExcelWriter(path) as xl:           # overwrite == MATLAB delete
        (branch_stats if branch_stats is not None else pd.DataFrame()) \
            .to_excel(xl, sheet_name="Branch", index=False)
        (bus_stats if bus_stats is not None else pd.DataFrame()) \
            .to_excel(xl, sheet_name="Bus", index=False)
        (length_stats if length_stats is not None else pd.DataFrame()) \
            .to_excel(xl, sheet_name="Length", index=False)
        hist_areas_table(synthetic, garver, targets) \
            .to_excel(xl, sheet_name="HistAreas", index=False)
        nd_areas_table(synthetic, garver, targets) \
            .to_excel(xl, sheet_name="NDAreas", index=False)
    return path


def write_electrical_stats(dc_stats: pd.DataFrame, dcopf_stats: pd.DataFrame,
                           out_dir: Path, region: str) -> list[Path]:
    """main_SRTV §8 inline block: the two electrical-comparison workbooks
    (delete-first == overwrite)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for df, kind in ((dc_stats, "DC Power Flow"), (dcopf_stats, "DC OPF")):
        p = out_dir / f"Synthetic {region} Electrical {kind} Statistics.xlsx"
        df.to_excel(p, index=False)
        paths.append(p)
    return paths


def write_acpf_stats(acpf: ACPFResult, out_dir: Path,
                     region: str) -> Path | None:
    """main_SRTV §10 inline block: Losses (sheet 1) + Stats (sheet 2).
    Returns None when the result carries neither table (MATLAB bare try)."""
    if acpf is None or (acpf.losses is None and acpf.stats is None):
        return None
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"KCL-KVL {region} AC Power Flow Statistics.xlsx"
    with pd.ExcelWriter(path) as xl:
        (acpf.losses if acpf.losses is not None else pd.DataFrame()) \
            .to_excel(xl, sheet_name="Losses", index=False)
        (acpf.stats if acpf.stats is not None else pd.DataFrame()) \
            .to_excel(xl, sheet_name="Stats", index=False)
    return path


# ---------------------------------------------------------------------------
# WriteSubstationData + the codectrl.WriteBuses dumps
# ---------------------------------------------------------------------------

def write_substation_data(model: CrucibleModel, out_dir: Path) -> Path:
    """WriteSubstationData.m port → SubstationGenLoadData.csv.

    Column order per the MATLAB assignment sequence; NumInt stays 1-BASED
    for output-file parity (doc Open Q5). GenFuelType/UnitType = the FIRST
    unit per gen bus (explicit groupby — the MATLAB FirstGen/GenBusesMax
    alignment can crash, doc Open Q6)."""
    subs = model.substations
    n = len(subs)
    fuel = np.full(n, "", dtype=object)
    unit = np.full(n, "", dtype=object)
    gen = model.generators
    if len(gen):
        first = gen.loc[~gen["BusNum"].duplicated()]
        by_bus = gen.groupby("BusNum")["MaxMW"].sum()
        for _, g in first.iterrows():
            b = int(g["BusNum"])
            if 0 <= b < n and by_bus.get(b, 0.0) != 0.0:
                fuel[b] = g["FuelType"]
                unit[b] = g["UnitType"]
    out = pd.DataFrame({
        "Num": subs["external_num"].to_numpy(),
        "Lat": subs["lat"].to_numpy(float),
        "Lon": subs["lon"].to_numpy(float),
        "Name": subs["name"].to_numpy(),
        "Type": subs["bus_type"].to_numpy(int),
        "NumInt": np.arange(1, n + 1),
        "MW": subs["load_mw"].to_numpy(float),
        "MVar": subs["load_mvar"].to_numpy(float),
        "MaxMW": subs["gen_max_mw"].to_numpy(float),
        "MinMW": subs["gen_min_mw"].to_numpy(float),
        "GenFuelType": fuel,
        "GenUnitType": unit,
    })
    return _write_csv(out, Path(out_dir) / "SubstationGenLoadData.csv")


def write_bus_tables(model: CrucibleModel, out_dir: Path) -> list[Path]:
    """main_SRTV §11 codectrl.WriteBuses: plain table dumps
    (SubstationData.xlsx / BusData.xlsx)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for df, name in ((model.substations, "SubstationData.xlsx"),
                     (model.buses, "BusData.xlsx")):
        if len(df):
            p = out_dir / name
            df.to_excel(p, index=False)
            paths.append(p)
    return paths
