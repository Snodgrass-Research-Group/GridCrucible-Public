"""migration tests — run against the REAL MATLAB sources in the repo
(fast parsers only; the full network-case conversion is exercised by running
`python -m gridcrucible.io.matlab_import`, not in the unit suite)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gridcrucible.config import MATLAB_ROOT
from gridcrucible.io import matlab_import as mi

pytestmark = pytest.mark.skipif(not MATLAB_ROOT.exists(),
                                reason="MATLAB source folder not present")


class TestConductorCatalog:
    def test_migrate(self, tmp_path):
        out = mi.migrate_conductor_tables(MATLAB_ROOT, tmp_path)
        cat = pd.read_parquet(out)
        assert list(cat.columns) == mi.CONDUCTOR_COLUMNS
        assert sorted(cat["kv"].unique()) == list(mi.KV_LEVELS)
        assert (cat["mva"] > 0).all()
        # cond_index is a 1-based index into a MASTER conductor list —
        # NON-consecutive within a kV class (e.g. 138 kV starts at 21 and
        # skips values). MaxCondIndex lookups
        # must resolve against this column, not against row position.
        assert (cat["cond_index"] >= 1).all()
        assert (cat["cond_index"] == cat["cond_index"].astype(int)).all()
        for kv, grp in cat.groupby("kv"):
            assert grp["cond_index"].is_monotonic_increasing


class TestGMDTables:
    def test_parse_srtv(self):
        gmd, xd = mi._parse_gmd_file(
            MATLAB_ROOT / "GMD Statistics and lookup tables" / "GMDDataSRTV.m")
        assert gmd["region"].iloc[0] == "SRTV"
        assert len(gmd) == 6000  # 6 kV levels x 1000 indices
        g69 = gmd[gmd["kv"] == 69].set_index("gmd_index")["gmd_ft"]
        # step function from the file: I1=[1;182;364;727;818;1000], GMD=[4;7;8;14;15]
        assert g69.loc[1] == 4 and g69.loc[181] == 4
        assert g69.loc[182] == 7 and g69.loc[363] == 7
        assert g69.loc[1000] == 15
        assert (gmd["gmd_ft"] > 0).all()
        assert len(xd) == 99 and xd["xd"].iloc[0] == 0.0

    def test_all_regions(self, tmp_path):
        out = mi.migrate_gmd_tables(MATLAB_ROOT, tmp_path)
        gmd = pd.read_parquet(out)
        assert gmd["region"].nunique() == 14
        assert len(gmd) == 14 * 6000


class TestMatpowerParser:
    def test_synthetic_case(self, tmp_path):
        case = tmp_path / "TinyCase.m"
        case.write_text(
            "% comment\nfunction mpc = TinyCase\n"
            "mpc.version = '2';\nmpc.baseMVA =  100.00;\n"
            "%% bus data\nmpc.bus = [\n"
            "1\t3\t0.0\t0.0\t0 0 1 1.0 0.0 138 1 1.1 0.9\n"
            "2\t1\t90.0\t30.0\t0 0 1 1.0 0.0 138 1 1.1 0.9\n"
            "];\n"
            "mpc.gen = [\n1\t100\t0\t80\t-50\t1.0\t100\t1\t150\t0\n];\n"
            "mpc.branch = [\n1\t2\t0.01\t0.05\t0.02\t120\t0\t0\t0\t0\t1\t-360\t360\n];\n"
            "mpc.bus_name = {\n'Alpha';\n'Beta'\n};\n")
        mpc = mi._parse_matpower_case(case)
        assert mpc["baseMVA"] == 100.0
        assert mpc["bus"].shape == (2, 13)
        assert mpc["gen"].shape == (1, 10)
        assert mpc["branch"][0, 3] == 0.05
        assert "bus_name" not in mpc  # cell block skipped

    def test_real_wecc_header_parses(self):
        """Parse the smaller real WECC island case end-to-end (arrays only)."""
        p = MATLAB_ROOT / "Default Input Data" / "Network Data" / "WECCLargeIslandSanitized.m"
        mpc = mi._parse_matpower_case(p)
        assert mpc["bus"].shape[1] >= 13
        assert mpc["branch"].shape[0] > 1000
        assert np.isfinite(mpc["branch"]).all()


class TestMasterLoad:
    def test_srtv_region(self, tmp_path):
        written = mi.migrate_master_load_data(MATLAB_ROOT, tmp_path, regions=["SRTV"])
        assert "SRTV" in written
        df = pd.read_parquet(tmp_path / "master_load" / "SRTV.parquet")
        al = df[(df["source"] == "Alabama") & (df["var"] == "AL")]
        assert len(al) == 8760  # hourly year
        assert (al["value"] > 0).all()


class TestEIStats:
    def test_targets(self, tmp_path):
        import json
        written = mi.migrate_default_inputs(MATLAB_ROOT, tmp_path)
        ei = json.loads((tmp_path / "ei_stats.json").read_text())
        assert ei["node_degree"]["2"] == 20537
        assert 2.0 < ei["node_degree_avg"] < 2.6
        assert abs(sum(ei["node_degree_dist"]) - 1.0) < 1e-12

    def test_exports_pickup_always_runs(self, tmp_path):
        # Empty data dir: every opaque export is recorded as still-needed.
        written = mi.ingest_matlab_exports(tmp_path)
        assert any("XFData" in k for k in written)  # opaque hand-off recorded
        # Drop in one export CSV; a re-run ingests it, a second re-run skips.
        exports = tmp_path / "matlab_exports"
        exports.mkdir()
        (exports / "XFData.csv").write_text("kv,mva\n138,100\n")
        written = mi.ingest_matlab_exports(tmp_path)
        assert (tmp_path / "xfdata.parquet").exists()
        assert "xfdata.parquet" in written["XFData.csv (via MATLAB export)"]
        written = mi.ingest_matlab_exports(tmp_path)
        assert written["XFData.csv (via MATLAB export)"].startswith("up to date")
