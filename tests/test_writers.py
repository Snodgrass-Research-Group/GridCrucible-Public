"""Property tests for io/writers.py.

Format-critical assertions: exact writetable column headers/order per
docs/algorithms/writers.md (WritePowerWorld sections), the pu ×40 sniff,
the SMW==0 row drop, the load-sign conventions, the first-unit-per-bus
GenImport quirk, the WriteMetrics sheet layout with label-keyed NDAreas.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gridcrucible.io import writers
from gridcrucible.model import ACPFResult

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _bus_level(model):
    """Give the tiny substation model a bus table + branches (as
    create_buses/create_branch would) so write_powerworld can run."""
    subs = model.substations
    model.buses = pd.DataFrame({
        "bus_id": np.arange(len(subs)),
        "sub_id": np.arange(len(subs)),
        "external_num": subs["external_num"].to_numpy(),
        "name": subs["name"].to_numpy(),
        "lat": subs["lat"].to_numpy(float),
        "lon": subs["lon"].to_numpy(float),
        "voltage_kv": 138.0,
        "bus_type": subs["bus_type"].to_numpy(),
        "load_mw": subs["load_mw"].to_numpy(float),
        "load_mvar": subs["load_mvar"].to_numpy(float),
        "gen_max_mw": subs["gen_max_mw"].to_numpy(float),
        "gen_min_mw": subs["gen_min_mw"].to_numpy(float),
    })
    ext = model.buses["external_num"].to_numpy()
    br = model.candidates.copy()
    br["from_ext"] = ext[br["from_bus"].to_numpy(int)]
    br["to_ext"] = ext[br["to_bus"].to_numpy(int)]
    model.branches = br
    return model


BUS_HEADER = ["Number", "Name", "NomkV", "AreaNumber", "ZoneNumber",
              "NomB", "NomG", "Latitude", "Longitude", "SubName"]
BUS_HEADER_V = BUS_HEADER[:9] + ["Vpu", "Vangle", "SubName"]
BRANCH_HEADER = ["BusNumFrom", "BusNumTo", "Circuit", "BranchDeviceType",
                 "IsXF", "R", "X", "B", "LimitMVAA", "LimitMVAB",
                 "LimitMVAC", "LineLength"]
LOAD_HEADER = ["BusNum", "ID", "Status", "SMW", "SMvar"]
GEN_HEADER = ["BusNum", "ID", "Status", "FuelType", "UnitType", "MWMin",
              "MWMax", "MvarMin", "MvarMax", "FixedCostInd", "CubicB",
              "CubicC", "CostModel", "MWSetPoint", "MvarSetpoint",
              "VoltSet", "AGC", "AVR", "PartFact"]
SS_HEADER = ["BusNum", "ID", "Status", "RegBusNum", "ShuntMode",
             "RegulationType", "RegHigh", "RegLow", "RegTarget",
             "BlockMvarPerStep1", "BlockNumberStep1", "BlockMvarPerStep2",
             "BlockNumberStep2", "MvarNom"]


# ---------------------------------------------------------------------------
# WritePowerWorld
# ---------------------------------------------------------------------------

def test_powerworld_headers_and_files(tiny_model, tmp_path):
    model = _bus_level(tiny_model)
    written = writers.write_powerworld(model, None, tmp_path, "KCL-KVL")
    assert set(written) == {"bus", "substation", "branch", "load"}
    # suffixed MATLAB filenames (intended 7-arg behavior, doc Open Q1)
    assert written["bus"].name == "BusImportKCL-KVLSRTV.csv"
    assert written["load"].name == "PeakLoadImportKCL-KVLSRTV.csv"

    bus = pd.read_csv(written["bus"])
    assert list(bus.columns) == BUS_HEADER
    assert bus["Number"].tolist() == \
        model.buses["external_num"].tolist()
    assert (bus["AreaNumber"] == 1).all() and (bus["NomB"] == 0).all()

    sub = pd.read_csv(written["substation"])
    assert list(sub.columns) == ["Number", "Name", "Latitude", "Longitude"]

    br = pd.read_csv(written["branch"])
    assert list(br.columns) == BRANCH_HEADER
    assert (br["BranchDeviceType"] == "Line").all()  # tap==0 everywhere
    assert (br["IsXF"] == "NO").all()
    np.testing.assert_allclose(br["LimitMVAB"], 1.10 * br["LimitMVAA"])
    np.testing.assert_allclose(br["LimitMVAC"], 1.15 * br["LimitMVAA"])
    # LineLength written raw in km (doc Open Q3)
    np.testing.assert_allclose(
        br["LineLength"], model.branches["length_km"].to_numpy())


def test_powerworld_branch_external_numbering(tiny_model, tmp_path):
    model = _bus_level(tiny_model)
    written = writers.write_powerworld(model, None, tmp_path, "KCL-KVL")
    bus = pd.read_csv(written["bus"])
    br = pd.read_csv(written["branch"])
    assert set(br["BusNumFrom"]) <= set(bus["Number"])
    assert set(br["BusNumTo"]) <= set(bus["Number"])


def test_powerworld_transformer_flags(tiny_model, tmp_path):
    model = _bus_level(tiny_model)
    model.branches.loc[0, "tap"] = 1.0          # make row 0 a transformer
    written = writers.write_powerworld(model, None, tmp_path, "KCL-KVL")
    br = pd.read_csv(written["branch"])
    assert br.loc[0, "BranchDeviceType"] == "Transformer"
    assert br.loc[0, "IsXF"] == "YES"
    assert (br.loc[1:, "IsXF"] == "NO").all()


def test_powerworld_voltage_columns_with_acpf(tiny_model, tmp_path):
    model = _bus_level(tiny_model)
    n = len(model.buses)
    acpf = ACPFResult(v_mag=np.full(n, 1.01), v_angle=np.zeros(n),
                      converged=True)
    written = writers.write_powerworld(model, acpf, tmp_path, "KCL-KVL")
    bus = pd.read_csv(written["bus"])
    assert list(bus.columns) == BUS_HEADER_V    # Vpu/Vangle BEFORE SubName


def test_powerworld_load_fallback_drops_zero_rows(tiny_model, tmp_path):
    model = _bus_level(tiny_model)
    written = writers.write_powerworld(model, None, tmp_path, "KCL-KVL")
    load = pd.read_csv(written["load"])
    # bus-table fallback carries the MW/Mvar duplicate columns
    assert list(load.columns) == LOAD_HEADER + ["MW", "Mvar"]
    assert (load["SMW"] != 0).all()
    assert len(load) == int((model.buses["load_mw"] != 0).sum())
    assert (load["Status"] == "Closed").all()


def test_powerworld_load_sign_conventions(tiny_model, tmp_path):
    model = _bus_level(tiny_model)
    snet = model.scenarios.tiers["full"]        # injections: loads negative
    written = writers.write_powerworld(model, None, tmp_path / "inj",
                                       "KCL-KVL", snet=snet, load_sign=-1)
    load = pd.read_csv(written["load"])
    assert list(load.columns) == LOAD_HEADER    # PF branch: no MW/Mvar cols
    neg = np.flatnonzero(snet[:, 0].real < 0)
    assert set(load["BusNum"]) == \
        set(model.buses["external_num"].to_numpy()[neg])
    np.testing.assert_allclose(
        np.sort(load["SMW"]), np.sort(np.abs(snet[neg, 0].real)))

    # ACPF branch: load-positive, MW/Mvar columns present
    written2 = writers.write_powerworld(model, None, tmp_path / "acpf",
                                        "KCL-KVL", snet=-snet, load_sign=+1)
    load2 = pd.read_csv(written2["load"])
    assert list(load2.columns) == LOAD_HEADER + ["MW", "Mvar"]
    np.testing.assert_allclose(np.sort(load2["SMW"]), np.sort(load["SMW"]))


def test_powerworld_pu_sniff_rescales(tiny_model, tmp_path):
    model = _bus_level(tiny_model)
    snet = model.scenarios.tiers["full"] / 100.0    # per-unit
    written = writers.write_powerworld(model, None, tmp_path, "KCL-KVL",
                                       snet=snet, load_sign=-1)
    load = pd.read_csv(written["load"])
    ref = writers.write_powerworld(model, None, tmp_path / "mw", "KCL-KVL",
                                   snet=snet * 100.0, load_sign=-1)
    ref_load = pd.read_csv(ref["load"])
    np.testing.assert_allclose(np.sort(load["SMW"]),
                               np.sort(ref_load["SMW"]))


def test_powerworld_gen_section_gating_and_first_unit(tiny_model, tmp_path):
    model = _bus_level(tiny_model)
    # no dispatch columns -> no gen file (MATLAB gate on Bus.PNet)
    w0 = writers.write_powerworld(model, None, tmp_path / "a", "KCL-KVL")
    assert "gen" not in w0

    # add a second unit at bus 0 to exercise the first-unit quirk
    gen2 = model.generators.iloc[[0]].copy()
    gen2["ID"] = 2
    model.generators = pd.concat([model.generators, gen2],
                                 ignore_index=True)
    model.buses["gen_mw_set"] = 0.0
    model.buses.loc[0, "gen_mw_set"] = 150.0    # bus 0 committed
    model.buses["p_net"] = 0.0
    model.buses.loc[0, "p_net"] = 150.0
    model.buses["q_net"] = 0.0
    model.buses.loc[0, "q_net"] = 30.0
    written = writers.write_powerworld(model, None, tmp_path / "b",
                                       "KCL-KVL")
    gen = pd.read_csv(written["gen"])
    assert list(gen.columns) == GEN_HEADER
    # committed bus 0: FIRST unit takes the whole setpoint, second gets 0
    rows0 = gen[gen["BusNum"] == 101]
    assert len(rows0) == 2
    assert rows0.iloc[0]["MWSetPoint"] == pytest.approx(150.0)
    assert rows0.iloc[1]["MWSetPoint"] == 0.0
    assert (rows0["Status"] == "Closed").all()
    # decommitted bus 3 (external 104): first unit Open
    rows3 = gen[gen["BusNum"] == 104]
    assert rows3.iloc[0]["Status"] == "Open"
    np.testing.assert_allclose(gen["PartFact"].sum(), 1.0)
    assert (gen["CostModel"] == "Cubic").all()


def test_powerworld_switched_shunts(tiny_model, tmp_path):
    model = _bus_level(tiny_model)
    model.buses["sync_cond"] = False
    model.buses.loc[[1, 4], "sync_cond"] = True
    model.buses["q_net"] = 0.0
    model.buses.loc[1, "q_net"] = 55.0          # B-type (injection +)
    model.buses.loc[4, "q_net"] = -80.0         # G-type
    written = writers.write_powerworld(model, None, tmp_path, "KCL-KVL")
    ss = pd.read_csv(written["switched_shunts"])
    assert list(ss.columns) == SS_HEADER
    b_row = ss[ss["BusNum"] == 102].iloc[0]
    g_row = ss[ss["BusNum"] == 105].iloc[0]
    assert b_row["BlockMvarPerStep1"] == -1
    assert b_row["BlockMvarPerStep2"] == np.ceil(1.3 * 55.0 / 10)
    assert b_row["MvarNom"] == np.ceil(55.0)
    assert g_row["BlockMvarPerStep1"] == -np.ceil(1.3 * 80.0 / 10)
    assert g_row["BlockMvarPerStep2"] == 1
    assert g_row["MvarNom"] == -np.ceil(80.0)
    assert (ss["RegHigh"] == 1.06).all() and (ss["RegLow"] == 0.94).all()


def test_powerworld_requires_bus_level_model(tiny_model, tmp_path):
    with pytest.raises(ValueError, match="model.buses"):
        writers.write_powerworld(tiny_model, None, tmp_path, "KCL-KVL")


# ---------------------------------------------------------------------------
# WriteLoads
# ---------------------------------------------------------------------------

def test_write_loads_transposed_headerless(tiny_model, tmp_path):
    model = _bus_level(tiny_model)
    snet = model.scenarios.tiers["full"]
    mw_path, mvar_path = writers.write_loads(model, None, tmp_path,
                                             snet=snet)
    mw = np.loadtxt(mw_path, delimiter=",")
    mvar = np.loadtxt(mvar_path, delimiter=",")
    assert mw.shape == (snet.shape[1], snet.shape[0])   # scenarios × buses
    np.testing.assert_allclose(mw, np.abs(snet.real).T)
    np.testing.assert_allclose(mvar, np.abs(snet.imag).T)


def test_write_loads_pu_sniff(tiny_model, tmp_path):
    model = _bus_level(tiny_model)
    snet = model.scenarios.tiers["full"] / 100.0
    mw_path, _ = writers.write_loads(model, None, tmp_path, snet=snet)
    mw = np.loadtxt(mw_path, delimiter=",")
    np.testing.assert_allclose(
        mw, np.abs(model.scenarios.tiers["full"].real).T)


def test_write_loads_complete_tier_fallback(tiny_model, tmp_path):
    model = _bus_level(tiny_model)
    model.scenarios.tiers["complete"] = model.scenarios.tiers["full"] * 2
    mw_path, _ = writers.write_loads(model, None, tmp_path)
    mw = np.loadtxt(mw_path, delimiter=",")
    np.testing.assert_allclose(
        mw, np.abs(model.scenarios.tiers["complete"].real).T)
    del model.scenarios.tiers["complete"]
    with pytest.raises(ValueError, match="complete"):
        writers.write_loads(model, None, tmp_path)


# ---------------------------------------------------------------------------
# WriteSubstationData / bus tables
# ---------------------------------------------------------------------------

def test_write_substation_data(tiny_model, tmp_path):
    path = writers.write_substation_data(tiny_model, tmp_path)
    df = pd.read_csv(path, keep_default_na=False)
    assert list(df.columns) == ["Num", "Lat", "Lon", "Name", "Type",
                                "NumInt", "MW", "MVar", "MaxMW", "MinMW",
                                "GenFuelType", "GenUnitType"]
    assert df["NumInt"].tolist() == list(range(1, 7))       # 1-based parity
    assert df.loc[0, "GenFuelType"] == "Gas"                # first unit
    assert df.loc[3, "GenFuelType"] == "Coal"
    assert (df.loc[[1, 2, 4, 5], "GenFuelType"] == "").all()
    np.testing.assert_allclose(
        df["MaxMW"], tiny_model.substations["gen_max_mw"])


def test_write_bus_tables(tiny_model, tmp_path):
    paths = writers.write_bus_tables(tiny_model, tmp_path)
    assert [p.name for p in paths] == ["SubstationData.xlsx"]  # buses empty
    model = _bus_level(tiny_model)
    paths = writers.write_bus_tables(model, tmp_path)
    assert [p.name for p in paths] == ["SubstationData.xlsx",
                                       "BusData.xlsx"]


# ---------------------------------------------------------------------------
# WriteMetrics
# ---------------------------------------------------------------------------

def _fake_cycles():
    return {"exp_fit": {"a": 0.5, "b": -0.2, "r2": 0.95, "area": 1.1},
            "power_fit": {"a": 1.0, "b": -1.5, "c": 0.0, "r2": 0.9,
                          "area": 0.8}}


def _fake_metrics(scale=1.0):
    fits = {}
    for i, label in enumerate(writers.ND_LABELS):
        fits[label] = {"degrees": np.arange(1, 6),
                       "dist": np.geomspace(0.5, 0.01, 5),
                       "fit": {"slope": -0.4 * scale, "intercept": 0.0,
                               "r2": 0.9, "area": float(i) * scale}}
    return {"cycles": _fake_cycles(), "degree_fits": fits}


def _fake_actual():
    d = _fake_metrics(scale=2.0)
    d["degree_fits"]["bus_degree_in_service"] = \
        d["degree_fits"]["bus_degree"]
    return {"branch": d, "bus": _fake_metrics(scale=3.0)}


def test_write_metrics_workbook(config, tmp_path):
    targets = {"EI": _fake_actual(), "WECC": _fake_actual()}
    path = writers.write_metrics(_fake_metrics(), _fake_metrics(0.5),
                                 targets, config, tmp_path)
    assert path.name == "KCL-KVL SRTV Statistics.xlsx"
    sheets = pd.read_excel(path, sheet_name=None)
    assert list(sheets) == ["Branch", "Bus", "Length", "HistAreas",
                            "NDAreas"]
    ha = sheets["HistAreas"]
    assert ha["Label_CB"].tolist() == [
        "KCL/KVL", "KCL-Only", "EI All Branches", "EI Positive Impedances",
        "WECC All Branches", "WECC Positive Impedances"]
    assert list(ha.columns) == ["Label_CB", "Exp_RSq", "Area_Exp",
                                "Power_RSq", "Area_Power"]
    nd = sheets["NDAreas"]
    assert nd["Label_ND"].tolist() == list(writers.ND_LABELS)
    assert list(nd.columns) == [
        "Label_ND", "KVL_Area", "KVL_RSq", "KCL_Area", "KCL_RSq",
        "EIBranchArea", "EIBranchRSq", "EIBusArea", "EIBusRSq",
        "WECCBranchArea", "WECCBranchRSq", "WECCBusArea", "WECCBusRSq"]
    # label-keyed join: branch flavor scale=2, bus flavor scale=3
    assert nd.loc[1, "EIBranchArea"] == pytest.approx(2.0)
    assert nd.loc[1, "EIBusArea"] == pytest.approx(3.0)


def test_write_metrics_self_compare_zeros_kcl(config, tmp_path):
    path = writers.write_metrics(_fake_metrics(), None, {}, config, tmp_path)
    nd = pd.read_excel(path, sheet_name="NDAreas")
    assert (nd["KCL_Area"] == 0.0).all()
    assert (nd["KCL_RSq"] == 0.0).all()
    assert nd["EIBranchArea"].isna().all()      # no targets → NaN
    ha = pd.read_excel(path, sheet_name="HistAreas")
    assert ha.loc[1, "Label_CB"] == "KCL-Only"
    assert np.isnan(ha.loc[1, "Area_Exp"])


def test_write_acpf_and_electrical_stats(tmp_path):
    acpf = ACPFResult(losses=pd.DataFrame({"LossMW": [1.0]}),
                      stats=pd.DataFrame({"VMin": [0.97]}))
    p = writers.write_acpf_stats(acpf, tmp_path, "SRTV")
    sheets = pd.read_excel(p, sheet_name=None)
    assert list(sheets) == ["Losses", "Stats"]
    assert writers.write_acpf_stats(ACPFResult(), tmp_path, "SRTV") is None

    dc = pd.DataFrame({"Overloads": [0]})
    opf = pd.DataFrame({"Cost": [123.0]})
    paths = writers.write_electrical_stats(dc, opf, tmp_path, "SRTV")
    assert [p.name for p in paths] == [
        "Synthetic SRTV Electrical DC Power Flow Statistics.xlsx",
        "Synthetic SRTV Electrical DC OPF Statistics.xlsx"]
