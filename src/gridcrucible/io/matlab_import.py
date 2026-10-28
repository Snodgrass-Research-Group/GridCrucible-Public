"""One-time migration of KVL-folder MATLAB inputs to Python-native formats.
 Runtime code NEVER reads .mat/.xls/case-scripts — only
the parquet/npz/pandapower-JSON this module writes to config.io.data_dir.
Idempotent (skips existing outputs unless force=True); writes manifest.json.

Run:  python -m gridcrucible.io.matlab_import   (from gridcrucible/, in-venv)

Source inventory and formats (verified 2026-07-08 against the real files):

* Conductor Data Tables/ConductorData{69..765}.mat — PLAIN float arrays,
  columns (per CreateOLMatrices.m array2table): MVA, kV, Size, CondPhase,
  Ra, Xa, XaP, CondIndex. → conductor_catalog.parquet.
* GMD Statistics and lookup tables/GMDData<REGION>.m — data-only function
  files: per kV a step function over gmd_index 1..1000 (breakpoints I1,
  values GMD) + Xd/XdPrime vectors. → gmd_tables.parquet,
  gmd_generator_reactances.parquet.
* Default Input Data CSVs (Form1, RTEP, BusInputData, RegionStates, EI/WECC
  planning-case tables) → parquet, 1:1.
* Default Input Data/GeneratorImport.XLSX → generator_import.parquet (raw;
  the fleet-table build is data/generators.py).
* EI node-degree targets — hard-coded in main_SRTV.m lines 362-367 →
  ei_stats.json (with derived distribution + average).
* Network Data/{EI,WECC}{Branch,LargeIsland}Sanitized.m — PowerWorld-exported
  MATPOWER case literals (up to ~293k lines) → parsed to mpc arrays, saved
  BOTH as npz (raw, robust) and pandapower JSON (via converter.from_ppc).
* Master Load Data/<REGION>/*.mat — plain numeric arrays (verified: e.g.
  Alabama.mat AL (1,8760)) → master_load/<REGION>.parquet (long format).
  The .xls hourly workbooks stay in place for the load-profile port.

**MATLAB-table opaques** (scipy CANNOT read MATLAB ``table`` objects):
GenNat.mat, XFData.mat, XFXR.mat, SRTV ConductorData.mat, SRTV Gen.mat are
opaque. Resolution per file:
  - ConductorData.mat: REBUILT here from the plain per-kV arrays (verified
    identical construction in CreateOLMatrices.m).
  - GenNat.mat: regenerated in from GeneratorImport.XLSX
    (CreateGeneratorDataNational port) — not migrated.
  - XFData/XFXR (+ Gen.mat reference outputs): need a one-time MATLAB export;
    run gridcrucible/tools/export_opaque_tables.m in MATLAB, then re-run this
    migration — it picks up the CSVs from data_dir/matlab_exports/.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

#: Column names for the plain ConductorData<kV> arrays (CreateOLMatrices.m:8).
CONDUCTOR_COLUMNS = ["mva", "kv", "size", "cond_phase", "ra", "xa", "xap",
                     "cond_index"]
KV_LEVELS = (69, 138, 230, 345, 500, 765)

#: EI node-degree target histogram (main_SRTV.m:362, degree 1..23 counts).
EI_NODE_DEGREE = [
    (1, 10047), (2, 20537), (3, 10178), (4, 2575), (5, 1340), (6, 651),
    (7, 349), (8, 174), (9, 90), (10, 50), (11, 31), (12, 12), (13, 13),
    (14, 3), (15, 3), (16, 1), (17, 1), (18, 0), (19, 1), (20, 0), (21, 0),
    (22, 0), (23, 1),
]

#: Default Input Data CSVs → parquet stem (None = keep stem, snake-cased).
_CSV_SOURCES = {
    "Network Data/Form1.csv": "form1",
    "Network Data/RTEPBranchData.csv": "rtep_branch_data",
    "BusInputData.csv": "bus_input_data",
    # KCL-pipeline bus numbering (same 75,432 uniquely-named buses in a
    # different row order) — the derivable replacement for the repo-absent
    # BusMapping.mat (algorithms/NetworkGraphComparison.md §BusMapping)
    "BusInputDataGarver.csv": "bus_input_data_garver",
    "RegionStates.csv": "region_states",
    "Network Data/PlanningCaseSubstations.csv": "ei_substations",
    "Network Data/PlanningCaseBranch.csv": "ei_branch",
    "Network Data/PlanningCaseBuses.csv": "ei_buses",
    "Network Data/WECCSubstations.csv": "wecc_substations",
    "Network Data/WECCBranch.csv": "wecc_branch",
    "Network Data/WECCBuses.csv": "wecc_buses",
    "Network Data/WECCBranchData.csv": "wecc_branch_data",
}

_CASE_SCRIPTS = ("EIBranchSanitized", "EILargeIslandSanitized",
                 "WECCBranchSanitized", "WECCLargeIslandSanitized")

#: Opaque MATLAB tables awaiting tools/export_opaque_tables.m (stem → note).
#: (v2 audit 2026-07-08: all .mat under the three input folders classified;
#: everything else is plain-numeric, rebuilt from CSV/XLSX sources, or a
#: reference-run artifact handled below.)
_OPAQUE_EXPORTS = {
    "XFData": "transformer rating/impedance table",
    "XFXR": "transformer X/R table",
    "Form715LimitFits": "Form 715 line-limit distribution fits (metrics)",
    "PJMGenData": "PJM generator data (comparison metrics)",
    "WECCGenData": "WECC generator data (comparison metrics)",
}


def _write_parquet(df: pd.DataFrame, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    return out


# ---------------------------------------------------------------------------
# Conductor tables
# ---------------------------------------------------------------------------

def migrate_conductor_tables(matlab_root: Path, data_dir: Path) -> Path:
    """Per-kV plain arrays → one conductor_catalog.parquet (kv column is the
    class; asserts the embedded kV column agrees with the filename)."""
    src = matlab_root / "Conductor Data Tables"
    frames = []
    for kv in KV_LEVELS:
        mat = sio.loadmat(src / f"ConductorData{kv}.mat")
        arr = mat[f"ConductorData{kv}"]
        if arr.shape[1] != len(CONDUCTOR_COLUMNS):
            raise ValueError(f"ConductorData{kv}: expected "
                             f"{len(CONDUCTOR_COLUMNS)} columns, got {arr.shape[1]}")
        df = pd.DataFrame(arr, columns=CONDUCTOR_COLUMNS)
        if not (df["kv"] == kv).all():
            raise ValueError(f"ConductorData{kv}: kV column disagrees with filename")
        frames.append(df)
    catalog = pd.concat(frames, ignore_index=True)
    return _write_parquet(catalog, data_dir / "conductor_catalog.parquet")


# ---------------------------------------------------------------------------
# GMD lookup tables (regional feval family)
# ---------------------------------------------------------------------------

_GMD_BLOCK = re.compile(
    r"GMD(\d+)\s*=\s*zeros\(1000,1\);\s*"
    r"I1\s*=\s*\[([^\]]+)\];\s*"
    r"GMD\s*=\s*\[([^\]]+)\];", re.S)
_GMD_VEC = re.compile(r"(Xd|XdPrime)\s*=\s*\[([^\]]+)\];", re.S)


def _parse_gmd_file(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One GMDData<REGION>.m → (long gmd table, xd table). Reproduces the
    MATLAB step-fill loop exactly (inclusive, later segments overwrite the
    shared breakpoint)."""
    text = path.read_text()
    region = path.stem.removeprefix("GMDData")
    cols = {}
    for m in _GMD_BLOCK.finditer(text):
        kv = int(m.group(1))
        i1 = np.array([int(float(x)) for x in re.split(r"[;,\s]+", m.group(2).strip())])
        vals = np.array([float(x) for x in re.split(r"[;,\s]+", m.group(3).strip())])
        if len(vals) != len(i1) - 1:
            raise ValueError(f"{path.name}: GMD{kv} breakpoint/value mismatch")
        col = np.zeros(1000)
        for i in range(len(i1) - 1):
            col[i1[i] - 1:i1[i + 1]] = vals[i]   # MATLAB I1(i):I1(i+1), 1-based
        cols[kv] = col
    if sorted(cols) != sorted(KV_LEVELS):
        raise ValueError(f"{path.name}: kV levels {sorted(cols)} != {KV_LEVELS}")
    gmd = pd.DataFrame({
        "region": region,
        "gmd_index": np.tile(np.arange(1, 1001), len(KV_LEVELS)),
        "kv": np.repeat(list(KV_LEVELS), 1000),
        "gmd_ft": np.concatenate([cols[kv] for kv in KV_LEVELS]),
    })
    vecs = {name: np.array([float(x) for x in re.split(r"[;,\s]+", body.strip())])
            for name, body in _GMD_VEC.findall(text)}
    if set(vecs) != {"Xd", "XdPrime"} or len(vecs["Xd"]) != len(vecs["XdPrime"]):
        raise ValueError(f"{path.name}: bad Xd/XdPrime vectors")
    xd = pd.DataFrame({"region": region, "n": np.arange(len(vecs["Xd"])),
                       "xd": vecs["Xd"], "xd_prime": vecs["XdPrime"]})
    return gmd, xd


def migrate_gmd_tables(matlab_root: Path, data_dir: Path) -> Path:
    """Parse the ``GMDData*.m`` lookup scripts into parquet: the per-region GMD
    tables and the generator subtransient-reactance table. Returns the path of
    the GMD-tables parquet written under ``data_dir``."""
    src = matlab_root / "GMD Statistics and lookup tables"
    gmds, xds = [], []
    for path in sorted(src.glob("GMDData*.m")):
        g, x = _parse_gmd_file(path)
        gmds.append(g)
        xds.append(x)
    if not gmds:
        raise FileNotFoundError(f"no GMDData*.m under {src}")
    _write_parquet(pd.concat(xds, ignore_index=True),
                   data_dir / "gmd_generator_reactances.parquet")
    return _write_parquet(pd.concat(gmds, ignore_index=True),
                          data_dir / "gmd_tables.parquet")


# ---------------------------------------------------------------------------
# Default Input Data (CSVs, XLSX, EI targets, opaque hand-offs)
# ---------------------------------------------------------------------------

def ingest_default_csvs(matlab_root: Path, data_dir: Path,
                        force: bool = False) -> dict[str, str]:
    """Default Input Data CSVs → parquet. ALWAYS runs (never
    section-skipped) so newly-listed _CSV_SOURCES entries appear on
    already-migrated machines; per-file mtime-incremental like
    ingest_matlab_exports."""
    src = matlab_root / "Default Input Data"
    written: dict[str, str] = {}
    for rel, stem in _CSV_SOURCES.items():
        if stem is None:
            continue
        p = src / rel
        if not p.exists():
            written[rel] = "MISSING (skipped)"
            continue
        out = data_dir / f"{stem}.parquet"
        if force or not out.exists() or out.stat().st_mtime < p.stat().st_mtime:
            df = pd.read_csv(p, low_memory=False)
            written[rel] = str(_write_parquet(df, out))
        else:
            written[rel] = f"up to date: {out}"
    return written


def migrate_default_inputs(matlab_root: Path, data_dir: Path) -> dict[str, str]:
    """Migrate the ``Default Input Data`` folder (default-input CSVs, the
    generator-import workbook, and the baked-in EI node-degree distribution)
    into parquet. Returns a map of source name -> written parquet path."""
    src = matlab_root / "Default Input Data"
    written: dict[str, str] = ingest_default_csvs(matlab_root, data_dir)

    xlsx = src / "GeneratorImport.XLSX"
    if xlsx.exists():
        df = pd.read_excel(xlsx, sheet_name=0)
        written["GeneratorImport.XLSX"] = str(
            _write_parquet(df, data_dir / "generator_import.parquet"))

    counts = np.array([c for _, c in EI_NODE_DEGREE], float)
    dist = counts / counts.sum()
    ei = {
        "node_degree": dict(EI_NODE_DEGREE),
        "node_degree_dist": dist.tolist(),
        "node_degree_avg": float((np.arange(1, len(dist) + 1) * dist).sum()),
        "source": "main_SRTV.m:362 (EI planning case histogram)",
    }
    out = data_dir / "ei_stats.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(ei, indent=1))
    written["EI node-degree targets"] = str(out)

    # Reference-run artifacts (SRTV/Example Datasets) — validation targets.
    # GMDIndex.mat is plain uint16 and scipy-readable: migrating it lets the
    # Python pipeline REUSE the author's exact per-path GMD draws.
    gmd_idx = matlab_root / "SRTV" / "Example Datasets" / "GMDIndex.mat"
    if gmd_idx.exists():
        arr = sio.loadmat(gmd_idx)["GMDIndex"].ravel().astype(np.uint16)
        out = data_dir / "reference_run" / "gmd_index.npz"
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, gmd_index=arr)
        written["GMDIndex.mat (reference run)"] = f"{out} ({arr.size} draws)"
    return written


def ingest_matlab_exports(data_dir: Path, force: bool = False) -> dict[str, str]:
    """Pick up whatever tools/export_opaque_tables.m has produced under
    data_dir/matlab_exports/. ALWAYS runs (never section-skipped): the author
    re-runs the export script over time and each re-run of the migration must
    ingest new arrivals. Incremental per file — an output is rebuilt only when
    missing or older than its source (or force=True)."""
    exports = data_dir / "matlab_exports"
    written: dict[str, str] = {}

    def stale(out: Path, src: Path) -> bool:
        return force or not out.exists() or out.stat().st_mtime < src.stat().st_mtime

    # Opaque MATLAB tables exported as CSV → parquet.
    for stem, note in _OPAQUE_EXPORTS.items():
        hits = sorted(exports.glob(f"{stem}*.csv"))   # stem.csv OR stem_<var>.csv
        if hits:
            for csv in hits:
                out = data_dir / f"{csv.stem.lower()}.parquet"
                if stale(out, csv):
                    df = pd.read_csv(csv, low_memory=False)
                    written[f"{csv.name} (via MATLAB export)"] = str(_write_parquet(df, out))
                else:
                    written[f"{csv.name} (via MATLAB export)"] = f"up to date: {out}"
        else:
            written[f"{stem}.mat"] = (
                f"NEEDS MATLAB EXPORT ({note}): run "
                "gridcrucible/tools/export_opaque_tables.m, then re-run migration")

    # Reference-run CSV exports (SRTV outputs for validation) → parquet.
    for csv in sorted((exports / "reference_run").glob("*.csv")):
        out = data_dir / "reference_run" / f"{csv.stem.lower()}.parquet"
        if stale(out, csv):
            df = pd.read_csv(csv, low_memory=False)
            written[f"reference_run/{csv.name}"] = str(_write_parquet(df, out))
        else:
            written[f"reference_run/{csv.name}"] = f"up to date: {out}"

    # Load-profile bridge exports (<REGION>_LoadProfiles.mat, plain -v6):
    # MATLAB-built P/Q profiles cached until the pure-Python builder lands
    # (docs/algorithms/MakeLoadProfiles.md "Port strategy").
    for matfile in sorted(exports.glob("*_LoadProfiles.mat")):
        region = matfile.stem.removesuffix("_LoadProfiles")
        out = data_dir / "load_profiles" / f"{region}.npz"
        if stale(out, matfile):
            m = sio.loadmat(matfile)
            out.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(out, p_load=m["P_Load"], q_load=m["Q_Load"])
            written[f"load_profiles/{region}"] = f"{out} {m['P_Load'].shape}"
        else:
            written[f"load_profiles/{region}"] = f"up to date: {out}"
    return written


# ---------------------------------------------------------------------------
# MATPOWER case scripts (EI / WECC sanitized)
# ---------------------------------------------------------------------------

def _parse_matpower_case(path: Path) -> dict:
    """PowerWorld-exported MATPOWER case .m → dict of numpy arrays + scalars.
    Only numeric ``mpc.<field> = [ ... ];`` blocks are parsed (bus_name /
    cell blocks are irrelevant to the electrical model)."""
    text = path.read_text(errors="replace")
    mpc: dict = {}
    m = re.search(r"mpc\.baseMVA\s*=\s*([0-9.eE+-]+)", text)
    mpc["baseMVA"] = float(m.group(1)) if m else 100.0
    m = re.search(r"mpc\.version\s*=\s*'(\w+)'", text)
    mpc["version"] = m.group(1) if m else "2"
    for m in re.finditer(r"mpc\.(\w+)\s*=\s*\[(.*?)\];", text, re.S):
        name, body = m.group(1), m.group(2)
        body = body.replace(";", "\n")
        try:
            arr = np.loadtxt(StringIO(body), comments="%", ndmin=2)
        except ValueError:
            continue  # non-numeric block
        if arr.size:
            mpc[name] = arr
    for req in ("bus", "branch"):
        if req not in mpc:
            raise ValueError(f"{path.name}: no numeric mpc.{req} block found")
    return mpc


def migrate_network_cases(matlab_root: Path, data_dir: Path) -> dict[str, str]:
    """Parse each sanitized case; ALWAYS save raw arrays as npz; additionally
    attempt a pandapower net (converter.from_ppc → JSON). The npz is the
    dependable artifact; the pp JSON is the convenient one."""
    src = matlab_root / "Default Input Data" / "Network Data"
    out_dir = data_dir / "reference_cases"
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}
    for stem in _CASE_SCRIPTS:
        p = src / f"{stem}.m"
        if not p.exists():
            written[stem] = "MISSING (skipped)"
            continue
        mpc = _parse_matpower_case(p)
        arrays = {k: v for k, v in mpc.items() if isinstance(v, np.ndarray)}
        npz = out_dir / f"{stem}.npz"
        np.savez_compressed(npz, baseMVA=mpc["baseMVA"], **arrays)
        note = f"{npz}"
        try:
            import pandapower as pp
            from pandapower.converter.pypower import from_ppc
            # gencost is deliberately DROPPED: pandapower's _from_ppc_gencost
            # crashes on PowerWorld's higher-order polynomial rows (and warns
            # it cannot represent them anyway). These cases feed topology/PF
            # realism comparisons; OPF costs come from the fleet data
            # (metrics/comparison assigns them when needed).
            ppc = {"version": mpc.get("version", "2"), "baseMVA": mpc["baseMVA"],
                   "bus": mpc["bus"],
                   "gen": mpc.get("gen", np.empty((0, 21))),
                   "branch": mpc["branch"]}
            net = from_ppc(ppc, f_hz=60)
            jsn = out_dir / f"{stem}.json"
            pp.to_json(net, jsn)
            note += f" + {jsn} ({len(net.bus)} buses, {len(net.line) + len(net.trafo)} branches)"
        except Exception as e:  # keep the npz; record why pp conversion failed
            note += f" (pandapower conversion FAILED: {type(e).__name__}: {e})"
        written[stem] = note
    return written


# ---------------------------------------------------------------------------
# Master Load Data
# ---------------------------------------------------------------------------

def migrate_master_load_data(matlab_root: Path, data_dir: Path,
                             regions: list[str] | None = None) -> dict[str, str]:
    """Numeric arrays from each region's .mat files → one long parquet per
    region: [source, var, idx, value]. Non-numeric/opaque vars are recorded
    and skipped; .xls workbooks are deferred to the load-profile port."""
    src = matlab_root / "Master Load Data"
    written: dict[str, str] = {}
    region_dirs = [d for d in sorted(src.iterdir()) if d.is_dir()]
    if regions:
        region_dirs = [d for d in region_dirs if d.name in regions]
    for rdir in region_dirs:
        rows, skipped = [], []
        for matfile in sorted(rdir.rglob("*.mat")):
            try:
                m = sio.loadmat(matfile)
            except Exception as e:
                skipped.append(f"{matfile.name} (loadmat: {type(e).__name__})")
                continue
            for var, val in m.items():
                if var.startswith("__"):
                    continue
                if isinstance(val, np.ndarray) and val.dtype.kind in "fiu":
                    # states saved as uint16 (e.g. Mississippi MS) included
                    flat = val.ravel(order="F").astype(np.float64)  # MATLAB column order
                    rows.append(pd.DataFrame({
                        "source": matfile.stem, "var": var,
                        "idx": np.arange(flat.size), "value": flat}))
                else:
                    skipped.append(f"{matfile.name}:{var} (non-numeric)")
        if rows:
            out = _write_parquet(pd.concat(rows, ignore_index=True),
                                 data_dir / "master_load" / f"{rdir.name}.parquet")
            note = str(out)
            if skipped:
                note += f" (skipped: {', '.join(skipped)})"
            written[rdir.name] = note
        else:
            written[rdir.name] = f"no numeric .mat data ({', '.join(skipped) or 'empty'})"
    return written


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def migrate_all(matlab_root: Path, data_dir: Path, force: bool = False) -> dict:
    """Run every migration; skip whole sections whose primary output exists
    (unless force). Writes data_dir/manifest.json and returns the manifest."""
    matlab_root, data_dir = Path(matlab_root), Path(data_dir)
    manifest: dict = {"generated": datetime.now(timezone.utc).isoformat(),
                      "matlab_root": str(matlab_root)}

    have_matlab = matlab_root.is_dir()

    def section(name: str, primary: Path, fn):
        if primary.exists() and not force:
            manifest[name] = f"SKIPPED (exists): {primary}"
        elif not have_matlab:
            manifest[name] = f"SKIPPED (no MATLAB tree at {matlab_root})"
        else:
            manifest[name] = fn()

    section("conductor_catalog", data_dir / "conductor_catalog.parquet",
            lambda: str(migrate_conductor_tables(matlab_root, data_dir)))
    section("gmd_tables", data_dir / "gmd_tables.parquet",
            lambda: str(migrate_gmd_tables(matlab_root, data_dir)))
    section("default_inputs", data_dir / "ei_stats.json",
            lambda: migrate_default_inputs(matlab_root, data_dir))
    section("reference_cases", data_dir / "reference_cases" / "EIBranchSanitized.npz",
            lambda: migrate_network_cases(matlab_root, data_dir))
    section("master_load", data_dir / "master_load",
            lambda: migrate_master_load_data(matlab_root, data_dir))

    # matlab_exports + default-CSV pickup are NEVER section-skipped — the
    # export script is re-run and _CSV_SOURCES grows over time; each
    # migration run must ingest new arrivals (per-file mtime checks keep
    # this cheap).
    manifest["matlab_exports"] = ingest_matlab_exports(data_dir, force)
    if have_matlab:
        manifest["default_csvs"] = ingest_default_csvs(matlab_root, data_dir, force)

    out = data_dir / "manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=1))
    return manifest


def main(argv: list[str] | None = None) -> None:
    """CLI entry point: run the full one-time migration ``matlab_root`` ->
    ``data_dir`` and print the manifest as JSON.

    Invoke as ``python -m gridcrucible.io.matlab_import [--matlab-root DIR]
    [--data-dir DIR] [--force]``. Without the MATLAB tree only the committed
    ``data/matlab_exports/`` CSVs are ingested.
    """
    import argparse

    from ..config import CrucibleConfig
    cfg = CrucibleConfig()
    ap = argparse.ArgumentParser(description="One-time MATLAB data migration")
    ap.add_argument("--matlab-root", type=Path, default=cfg.io.matlab_root,
                    help="the 'KCL-KVL Compliant Grid Generator' folder "
                         "(default: $GRIDCRUCIBLE_MATLAB_ROOT or next to the clone)")
    ap.add_argument("--data-dir", type=Path, default=cfg.io.data_dir)
    ap.add_argument("--force", action="store_true",
                    help="redo sections whose output already exists")
    args = ap.parse_args(argv)
    manifest = migrate_all(args.matlab_root, args.data_dir, force=args.force)
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
