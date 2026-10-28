"""tests: create_buses / create_branch / trim_generator_lines."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gridcrucible.config import CrucibleConfig
from gridcrucible.model import CrucibleModel, ScenarioSet
from gridcrucible.network.buses import (create_branch, create_buses,
                                        trim_generator_lines)

DATA_DIR = CrucibleConfig().io.data_dir


@pytest.fixture
def xf_tables():
    if not (DATA_DIR / "xfdata.parquet").exists():
        pytest.skip("run the migration first")
    return (pd.read_parquet(DATA_DIR / "xfdata.parquet"),
            pd.read_parquet(DATA_DIR / "xfxr.parquet"))


class TestCreateBuses:
    def test_topology_and_roles(self, config, split_model, xf_tables):
        xfdata, xfxr = xf_tables
        create_buses(split_model, config, xfdata, xfxr)
        buses = split_model.buses
        # 4 sub buses + 2 voltage-split buses (subs 1,3) + 3 gen buses
        assert len(buses) == 9
        # first buses carry the MINIMUM voltage
        assert (buses.loc[:3, "voltage_kv"] == 138).all()
        split = buses[(buses["voltage_kv"] == 230) & (buses["bus_type"] == 1)]
        assert set(split["sub_id"]) == {1, 3}
        gen_buses = buses[buses["voltage_kv"] == 12.9]
        assert len(gen_buses) == 3 and (gen_buses["bus_type"] == 2).all()

        cand = split_model.candidates
        # the 230 kV line 1-3 moved to the split buses
        hv = cand[(cand["voltage_kv_from"] == 230) & (cand["tap"] == 0)]
        assert set(hv[["from_bus", "to_bus"]].iloc[0]) == set(split.index)
        # transformers: 3 GSUs + 2 transmission (one per split sub)
        xf = cand[cand["tap"] != 0]
        assert len(xf) == 5
        assert xf["keep"].all()
        assert (cand.loc[cand["transmission_xf"] == True,       # noqa: E712
                         "capacity_mva"] == 2000).all()
        # GSU capacity = ceil(1.2 * MaxMW); GSUs are the radial transformers.
        # transformer_impedances SNAPS working kV (12.9 -> 69, like MATLAB) —
        # the NOMINAL column preserves the generator voltage.
        gsu = xf[xf["is_radial"]]
        assert sorted(gsu["capacity_mva"]) == [180, 180, 480]
        assert (gsu["voltage_kv_from_nom"] == 12.9).all()
        # POI routing: sub-1 units (POI 230) hit sub 1's 230 bus;
        # sub-3 unit (POI 115) hits sub 3's MIN bus (138)
        sub1_230 = int(split[split["sub_id"] == 1].index[0])
        assert (gsu[gsu["capacity_mva"] == 180]["to_bus"] == sub1_230).all()
        assert int(gsu[gsu["capacity_mva"] == 480]["to_bus"].iloc[0]) == 3

    def test_loads_slack_and_invalidation(self, config, split_model, xf_tables):
        xfdata, xfxr = xf_tables
        create_buses(split_model, config, xfdata, xfxr)
        buses = split_model.buses
        assert buses.at[0, "load_mw"] == 40.0      # loads on min-kV bus
        assert buses.at[2, "load_mw"] == 55.0
        # slack = LARGEST unit's bus (sub-3's 400 MW unit)
        s = split_model.scenarios
        assert buses.at[s.slack_bus, "voltage_kv"] == 12.9
        assert split_model.generators.at[2, "BusNum"] == s.slack_bus
        assert s.tiers == {}                       # scenarios invalidated
        assert split_model.meta["scenarios_invalidated"]
        # connected + schema-clean at the bus level
        assert split_model.validate() == []
        # GSU/XF impedances were sized from the real tables
        xf = split_model.candidates[split_model.candidates["tap"] != 0]
        assert (xf["reactance_x"] > 0).all()

    def test_create_branch_external_numbering(self, config, split_model,
                                              xf_tables):
        create_buses(split_model, config, *xf_tables)
        create_branch(split_model)
        br = split_model.branches
        assert {"from_ext", "to_ext"} <= set(br.columns)
        assert br.at[0, "from_ext"] == 11          # sub 0 external number


class TestTrimGeneratorLines:
    def test_long_gen_gen_line_dropped(self, config, split_model):
        cand = split_model.candidates
        # make the 1-3 line long and gen-gen (buses 1,3 host generators)
        cand.loc[4, "length_km"] = 140.0           # >= 80*1.6 km
        n = trim_generator_lines(split_model)
        assert n == 1
        assert len(split_model.candidates) == 4

    def test_short_lines_survive(self, config, split_model):
        n = trim_generator_lines(split_model)
        assert n == 0
        assert len(split_model.candidates) == 5
