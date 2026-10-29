"""Conductor leaves + TUNE Loop (against the REAL migrated
conductor catalog and transformer tables)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gridcrucible.config import CrucibleConfig
from gridcrucible.loops.tune import TUNELoop
from gridcrucible.network.conductors import (assign_conductors,
                                             load_conductor_catalog,
                                             resize_branches,
                                             transformer_impedances)

DATA_DIR = CrucibleConfig().io.data_dir
pytestmark = pytest.mark.skipif(
    not (DATA_DIR / "conductor_catalog.parquet").exists(),
    reason="run `python -m gridcrucible.io.matlab_import` first")


@pytest.fixture(scope="module")
def catalog():
    return load_conductor_catalog(DATA_DIR)


@pytest.fixture(scope="module")
def xf_tables():
    # The transformer tables are not committed; skip until migrated.
    if not (DATA_DIR / "xfdata.parquet").exists():
        pytest.skip("run the migration first")
    return (pd.read_parquet(DATA_DIR / "xfdata.parquet"),
            pd.read_parquet(DATA_DIR / "xfxr.parquet"))


def _tune_ready(model, catalog) -> None:
    """Give tiny_model's candidates the columns CreateOLMatrices provides."""
    c138 = catalog[catalog["kv"] == 138]
    cand = model.candidates
    cand["conductor_idx"] = float(c138.index[len(c138) // 2])
    cand["max_conductor_idx"] = float(c138.index.max())
    cand["min_conductor_idx"] = float(c138.index.min())
    cand["min_kv"] = 69.0
    cand["capacity_mva"] = float(catalog.at[int(cand["conductor_idx"].iloc[0]), "mva"])
    cand["resize"] = False
    cand["kcmil"] = 0.0
    cand["cond_per_phase"] = 1.0
    cand["max_flow"] = 50.0
    cand["max_dcpf"] = 50.0


class TestCatalog:
    def test_master_table_semantics(self, catalog):
        assert list(catalog.index) == list(range(1, len(catalog) + 1))
        assert catalog["mva"].is_monotonic_increasing
        assert catalog.at[1, "kv"] == 69          # smallest conductor is 69 kV
        assert set(catalog["kv"].unique()) == {69, 138, 230, 345, 500, 765}


class TestResizeBranches:
    def _cand(self, catalog, n=10) -> pd.DataFrame:
        c138 = catalog[catalog["kv"] == 138]
        return pd.DataFrame({
            "capacity_mva": 100.0, "conductor_idx": float(c138.index[3]),
            "max_conductor_idx": float(c138.index.max()),
            "min_conductor_idx": float(c138.index.min()),
            "keep": [False] * (n - 2) + [True, True],
        })

    def test_upsize_caps_and_flags(self, catalog):
        cand = self._cand(catalog)
        flows = np.full(10, 150.0)               # all overloaded (util 1.5)
        counts = resize_branches(
            cand, flows, percent_resize=0.2, percent_change=1.3,
            conductor_margin=1.0, downsize_threshold=0.5,
            upsize=True, downsize=False, downsize_keep=False,
            random_resize=True, rng=np.random.default_rng(3))
        assert counts["upsized"] == 2            # ceil(0.2*10)
        resized = cand[cand["resize"]]
        assert (resized["capacity_mva"] == np.ceil(100 * 1.3)).all()

    def test_at_max_conductor_flagged_not_resized(self, catalog):
        cand = self._cand(catalog)
        cand["conductor_idx"] = cand["max_conductor_idx"]
        counts = resize_branches(
            cand, np.full(10, 150.0), percent_resize=0.5, percent_change=1.3,
            conductor_margin=1.0, downsize_threshold=0.5,
            upsize=True, downsize=False, downsize_keep=False,
            random_resize=True, rng=np.random.default_rng(3))
        assert counts["upsized"] == 0
        assert cand["overloaded"].all()          # manual-review diagnostics

    def test_downsize_keep_only_touches_kept(self, catalog):
        cand = self._cand(catalog)
        counts = resize_branches(
            cand, np.full(10, 10.0),             # util 0.1 < threshold
            percent_resize=0.2, percent_change=1.3, conductor_margin=1.0,
            downsize_threshold=0.5, upsize=False, downsize=False,
            downsize_keep=True, random_resize=True,
            rng=np.random.default_rng(3))
        assert counts["downsized_keep"] == 2
        assert cand.loc[cand["resize"], "keep"].all()
        assert (cand.loc[cand["resize"], "capacity_mva"]
                == np.ceil(100 / 1.3)).all()


class TestAssignConductors:
    def test_snaps_to_exact_catalog_rating(self, tiny_model, catalog):
        _tune_ready(tiny_model, catalog)
        cand = tiny_model.candidates
        cand.loc[0, "capacity_mva"] = 137.0      # not a real conductor rating
        cand.loc[0, "resize"] = True
        assign_conductors(cand, catalog, change_voltages=True)
        ci = int(cand.loc[0, "conductor_idx"])
        assert cand.loc[0, "capacity_mva"] == catalog.at[ci, "mva"]
        assert cand.loc[0, "voltage_kv_from"] == catalog.at[ci, "kv"]
        assert cand.loc[0, "voltage_kv_to"] == catalog.at[ci, "kv"]
        assert not cand["resize"].any()

    def test_min_kv_floor(self, tiny_model, catalog):
        _tune_ready(tiny_model, catalog)
        cand = tiny_model.candidates
        cand.loc[0, "capacity_mva"] = catalog["mva"].min()  # tiny 69 kV size
        cand.loc[0, "min_kv"] = 230.0
        cand.loc[0, "resize"] = True
        assign_conductors(cand, catalog, change_voltages=True)
        assert cand.loc[0, "voltage_kv_from"] == 230.0

    def test_fixed_voltage_stays_in_class(self, tiny_model, catalog):
        _tune_ready(tiny_model, catalog)
        cand = tiny_model.candidates
        cand.loc[0, "capacity_mva"] = 5000.0     # would prefer 765 kV
        cand.loc[0, "resize"] = True
        assign_conductors(cand, catalog, change_voltages=False)
        assert cand.loc[0, "voltage_kv_from"] == 138.0


class TestTransformerImpedances:
    def test_impedance_from_real_tables(self, xf_tables):
        xfdata, xfxr = xf_tables
        br = pd.DataFrame({
            "tap": [1.0], "capacity_mva": [100.0],
            "voltage_kv_from": [70.0], "voltage_kv_to": [140.0],  # snap 69/138
            "n_circuits": [1.0], "resize": [True],
            "conductor_idx": [0.0], "reactance_x": [0.0], "r_pu": [0.0],
        })
        n = transformer_impedances(br, xfdata, xfxr)
        assert n == 1
        assert br.at[0, "voltage_kv_from"] == 69.0
        assert br.at[0, "voltage_kv_to"] == 138.0
        xr = float(np.interp(100.0, xfxr["MVA"], xfxr["XR"]))
        xhl = float(xfdata[(xfdata["Pri"] == 69) & (xfdata["Sec"] == 138)]
                    ["XHL"].iloc[0])
        z_pu = (100.0 / 100.0) * (xhl / 100.0)
        expected_x = np.sqrt(z_pu ** 2 / (1 + 1 / xr ** 2))
        assert br.at[0, "reactance_x"] == pytest.approx(expected_x, rel=1e-12)
        assert br.at[0, "r_pu"] == pytest.approx(expected_x / xr)
        assert br.at[0, "conductor_idx"] == 100.0   # per-circuit MVA overload
        assert not br.at[0, "resize"]


class TestTUNELoop:
    def _loop(self, config, catalog, **kw) -> TUNELoop:
        return TUNELoop(config, catalog, rng=np.random.default_rng(5), **kw)

    def _snet(self, model) -> np.ndarray:
        p = -model.substations["load_mw"].to_numpy(float)
        total = -p.sum()
        p[0] += 0.45 * total
        p[3] += 0.55 * total
        return np.tile(p[:, None], (1, 2))

    def test_dense_network_flags_only(self, config, tiny_model, catalog):
        _tune_ready(tiny_model, catalog)
        config.trim.bbr_high = 1.0               # density 1.5 > "high"
        report = self._loop(config, catalog).run_dc(
            tiny_model, self._snet(tiny_model))
        assert report.tier == "flags-only"
        assert report.n_upsized == 0

    def test_refine_tier_terminates(self, config, tiny_model, catalog):
        _tune_ready(tiny_model, catalog)
        with pytest.warns(UserWarning, match="impedance_fn"):
            report = self._loop(config, catalog).run_dc(
                tiny_model, self._snet(tiny_model))
        assert report.tier == "refine"           # density 1.5 < low (3)
        assert report.term_criteria_final >= 5
        assert (tiny_model.candidates["max_dcpf"] >= 0).all()
        assert not tiny_model.candidates["resize"].any()  # flags all cleared

    def test_impedance_fn_never_sees_transformer_rows(self, config,
                                                      tiny_model, catalog):
        """Regression (real-SRTV crash 2026-07-11): post-create_buses the
        candidates table carries XF rows whose conductor_idx holds MVA
        sizing values (e.g. 2000) — NOT catalog indices. The RXB recompute
        must receive LINE rows only (MATLAB hid the equivalent crash in
        CreateBuses step 8's try/catch)."""
        _tune_ready(tiny_model, catalog)
        cand = tiny_model.candidates
        xf = cand.iloc[[0]].copy()
        xf["tap"] = 1.0
        xf["conductor_idx"] = 2000.0             # MVA value, out of range
        tiny_model.candidates = pd.concat([cand, xf], ignore_index=True)
        max_row = int(catalog.index.max())

        def strict_rxb(df, cat):
            idx = df["conductor_idx"].to_numpy(float)
            if (idx > max_row).any():
                raise IndexError("positional indexers are out-of-bounds")
            df["r_pu"] = 0.01
            df["reactance_x"] = 0.05
            df["b_pu"] = 0.001

        loop = self._loop(config, catalog, impedance_fn=strict_rxb)
        loop._reassign_and_recompute(tiny_model)  # must NOT raise
        cand = tiny_model.candidates
        lines = cand["tap"].to_numpy(float) == 0
        assert (cand.loc[lines, "reactance_x"] == 0.05).all()
        assert float(cand.loc[~lines, "conductor_idx"].iloc[0]) == 2000.0
        assert (cand.loc[~lines, "reactance_x"] != 0.05).all()  # untouched

    def test_refine_upsizes_overloads(self, config, tiny_model, catalog):
        _tune_ready(tiny_model, catalog)
        cand = tiny_model.candidates
        c138 = catalog[catalog["kv"] == 138]
        cand["conductor_idx"] = float(c138.index[0])
        cand["capacity_mva"] = float(c138["mva"].iloc[0])  # smallest 138 kV
        cand["max_dcpf"] = 400.0                 # everything hugely overloaded
        loop = self._loop(config, catalog,
                          impedance_fn=lambda df, cat: None)  # ratings-only
        report = loop.run_dc(tiny_model, self._snet(tiny_model))
        assert report.n_upsized > 0
        assert (tiny_model.candidates["capacity_mva"]
                > float(c138["mva"].iloc[0])).sum() > 0

    def test_run_ac_envelope(self, config, tiny_model, catalog):
        _tune_ready(tiny_model, catalog)
        cand = tiny_model.candidates
        cand["x_pu"] = cand["reactance_x"]
        cand["r_pu"] = 0.1 * cand["reactance_x"]
        cand["b_pu"] = 0.002
        cand["capacity_mva"] = 400.0             # generous: no resizing needed
        snet = self._snet(tiny_model) * (1 + 0.2j)
        snet = np.column_stack([snet[:, 0], snet])   # 3 cols; col 0 dropped
        report = self._loop(config, catalog,
                            impedance_fn=lambda df, cat: None).run_ac(
            tiny_model, snet)
        assert report.tier == "ac-refine"
        assert report.refine_iters == 0
        assert (tiny_model.candidates["max_acpf"] > 0).any()
