"""National generator fleet construction (KVL variant).

MATLAB origin: CreateGeneratorDataNational.m (79 loc): explode the
GeneratorImport.xlsx group rows (NumGen units per row) into per-unit rows
with MW/MVar split evenly, nuclear floored at max, hydro floored at 0.
Verified against the audit: this reads ONLY GeneratorImport.xlsx (the
PJM/WECC gen .mat tables belong to the comparison metrics).
Algorithm notes inline (small helper — no separate doc). Regression target: data/matlab_exports/Gen_reference.csv
(the 719-unit SRTV fleet from the author's real run).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

#: MATLAB GenVarNames — assigned POSITIONALLY to the sheet's 12 columns.
_GEN_VAR_NAMES = ["BusNum", "NumGen", "MinMW", "MaxMW", "MinMVar", "MaxMVar",
                  "FuelType", "UnitType", "A", "B", "C", "POIkV"]


def create_generator_data_national(data_dir: Path) -> pd.DataFrame:
    """Build the national per-unit fleet (MATLAB ``GenNat``) from the
    migrated generator_import.parquet.

    Per source row (a group of NumGen identical units at one bus): each unit
    gets MaxMW/NumGen etc.; MinMW rules — Nuclear: dispatched at full
    capacity (min = max/NumGen), Hydro: min = 0 (dispatchable to zero), else
    MinMW/NumGen; guard min ≥ max → min = 0.99·max. ID counts units within a
    BUS (across groups); Index is the running 1-based unit number.
    """
    raw = pd.read_parquet(Path(data_dir) / "generator_import.parquet")
    if raw.shape[1] != len(_GEN_VAR_NAMES):
        raise ValueError(f"GeneratorImport sheet has {raw.shape[1]} columns, "
                         f"expected {len(_GEN_VAR_NAMES)}")
    gd = raw.set_axis(_GEN_VAR_NAMES, axis=1)

    units: list[dict] = []
    for bus in np.sort(gd["BusNum"].unique()):
        unit_id = 1
        for _, grp in gd[gd["BusNum"] == bus].iterrows():
            n = int(grp["NumGen"])
            max_mw = float(grp["MaxMW"]) / n
            fuel = str(grp["FuelType"])
            if fuel == "Nuclear":
                min_mw = float(grp["MaxMW"]) / n
            elif fuel == "Hydro":
                min_mw = 0.0
            else:
                min_mw = float(grp["MinMW"]) / n
            if min_mw >= max_mw:
                min_mw = 0.99 * max_mw
            for _ in range(n):
                units.append({
                    "BusNum": int(bus), "Index": len(units) + 1,
                    "Sub": int(bus), "ID": unit_id,
                    "MinMW": min_mw, "MaxMW": max_mw,
                    "MinMVar": float(grp["MinMVar"]) / n,
                    "MaxMVar": float(grp["MaxMVar"]) / n,
                    "FuelType": fuel, "UnitType": str(grp["UnitType"]),
                    "A": float(grp["A"]), "B": float(grp["B"]),
                    "C": float(grp["C"]), "POIkV": float(grp["POIkV"]),
                })
                unit_id += 1
    return pd.DataFrame(units)


def regional_fleet(gen_nat: pd.DataFrame, external_bus_nums: np.ndarray) -> pd.DataFrame:
    """Subset the national fleet to the given external bus numbers (the
    main_SRTV §3.2 regional filter; internal renumbering happens in
    GridCrucible.setup_generators, not here)."""
    return gen_nat[gen_nat["BusNum"].isin(np.asarray(external_bus_nums))] \
        .reset_index(drop=True)
