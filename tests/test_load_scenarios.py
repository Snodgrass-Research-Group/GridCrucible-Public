"""smooth_profile QP + build_load_scenarios routing."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gridcrucible.config import CrucibleConfig
from gridcrucible.data.load_profiles import make_load_profiles, smooth_profile
from gridcrucible.data.scenarios import build_load_scenarios

gp = pytest.importorskip("gurobipy")


class TestSmoothProfile:
    def test_short_input_passthrough(self):
        p = np.random.default_rng(0).random((30, 2))
        assert np.array_equal(smooth_profile(p), p)

    def test_positive_curvature_input_unchanged(self):
        """When midnight curvature is already positive (V-shape trough), the
        constraint holds at x=0 and NO correction is applied — the QP only
        acts on concave day boundaries."""
        h = np.arange(48, dtype=float)
        p = (np.abs(h - 24.0) + 50.0)[:, None]           # b1 = b2 = +2 > 0
        out = smooth_profile(p)
        # x=0 is optimal; Gurobi barrier residuals leave ~1e-5 (negligible on MW scale)
        assert np.abs(out - p).max() < 1e-4

    def test_kink_gets_positive_curvature(self):
        """A hard downward kink at midnight violates the curvature bound —
        the correction must enforce the constraint at hours 23..25."""
        p = np.concatenate([np.full(24, 100.0), np.full(24, 60.0)])[:, None]
        out = smooth_profile(p)
        seg = out[:, 0]
        d2_1 = seg[24] - 2 * seg[23] + seg[22]
        b1_orig = 60 - 2 * 100 + 100                     # -40, the old bound
        # constraint A x <= b tightens curvature vs the raw kink
        assert d2_1 >= b1_orig - 1e-6
        assert not np.allclose(out, p)                   # correction applied
        # correction confined to hours 20..24 / 25..29
        assert np.array_equal(out[:19, 0], p[:19, 0])
        assert np.array_equal(out[29:, 0], p[29:, 0])

    def test_multichannel(self):
        rng = np.random.default_rng(1)
        p = np.cumsum(rng.normal(0, 2, (72, 3)), axis=0) + 100
        out = smooth_profile(p)
        assert out.shape == (72, 3)
        assert np.isfinite(out).all()


class TestBuildLoadScenarios:
    def _bus_input(self) -> pd.DataFrame:
        return pd.DataFrame({
            "Region": ["SRTV"] * 8,
            "Type": ["Gen", "Load", "Load", "Gen", "Load", "Load", "Load", "Load"],
        })

    def test_gapped_bus_column_scatters(self, config):
        """Profiles carrying a gapped bus-number column land on the right
        rows of the full regional matrix."""
        bus_nums = np.array([2, 3, 5, 6, 7])             # gap at 4 -> 2 transitions
        prof = np.column_stack([bus_nums,
                                np.arange(1, 6)[:, None] * np.ones((5, 48))])
        fn = lambda region, data_dir: (prof, 0.3 * prof)
        p, q = build_load_scenarios(self._bus_input(), config, profile_fn=fn)
        assert p.shape == (8, 48)
        assert (p[0] == 0).all() and (p[3] == 0).all()   # unlisted buses zero
        assert (p[1] == 1).all()                          # bus 2 -> row idx 1
        assert (p[6] == 5).all()                          # bus 7 -> 5th profile
        assert q[4] == pytest.approx(0.9)                 # bus 5 -> 0.3 * 3rd profile

    def test_contiguous_bus_column_pads_gens(self, config):
        bus_nums = np.arange(1, 7)                       # strictly consecutive
        prof = np.column_stack([bus_nums, np.full((6, 24), 5.0)])
        fn = lambda region, data_dir: (prof, prof * 0.2)
        p, _ = build_load_scenarios(self._bus_input(), config, profile_fn=fn)
        assert p.shape == (8, 24)                        # 2 gen rows + 6 load rows
        assert (p[:2] == 0).all()
        assert (p[2:] == 5.0).all()

    def test_no_bus_column_pads_gens(self, config):
        prof = np.full((6, 24), 7.0)                     # col0 constant: no bus col
        fn = lambda region, data_dir: (prof, prof)
        p, _ = build_load_scenarios(self._bus_input(), config, profile_fn=fn)
        assert p.shape == (8, 24)
        assert (p[:2] == 0).all() and (p[2:] == 7.0).all()

    def test_model_subset_filter(self, config):
        prof = np.full((6, 24), 7.0)
        fn = lambda region, data_dir: (prof, prof)
        p, _ = build_load_scenarios(self._bus_input(), config,
                                    model_bus_external=np.array([1, 3, 5]),
                                    profile_fn=fn)
        assert p.shape == (3, 24)

    def test_missing_cache_message(self):
        cfg = CrucibleConfig()
        with pytest.raises(FileNotFoundError, match="make_load_profiles_from_loadsmith"):
            make_load_profiles("NOSUCHREGION", cfg.io.data_dir)


@pytest.mark.skipif(
    not (CrucibleConfig().io.data_dir / "load_profiles" / "SRTV.npz").exists(),
    reason="SRTV load-profile bridge export not run yet")
class TestRealSRTVProfiles:
    #: Known raw-data gaps (docs/algorithms/MakeLoadProfiles.md §Data quality):
    #: KY source NaN at 1-based hours 1587/7298, TN at hour 699. MATLAB's
    #: builder propagates them; hour selection skips them (min/max omit NaN).
    KNOWN_NAN_HOURS = {699, 1587, 7298}

    def test_shape_and_positivity(self, config):
        p, q = make_load_profiles("SRTV", config.io.data_dir)
        assert p.shape[1] in (8760, 8761)                # ± bus-number column
        assert p.shape == q.shape
        has_bus_col = p.shape[1] == 8761
        hours = p[:, 1:] if has_bus_col else p
        bad_cols = set(np.where(~np.isfinite(hours).all(axis=0))[0] + 1)
        assert bad_cols <= self.KNOWN_NAN_HOURS           # 1-based hours
        finite = hours[np.isfinite(hours)]
        # smoothing-QP corrections dip a few small night loads below zero
        # (real MATLAB output: ~800 of 39M cells, worst ≈ -6 MW — doc §Data
        # quality); anything beyond that band would be a new export problem
        neg = finite[finite < 0]
        assert neg.size < 2e-4 * finite.size
        assert neg.min() > -10 if neg.size else True

    def test_hour_selection_skips_nan_hours(self, config):
        from gridcrucible.data.scenarios import _select_scenario_hours
        p, q = make_load_profiles("SRTV", config.io.data_dir)
        scen = _select_scenario_hours(p[:, 1:], q[:, 1:])
        assert scen.shape[1] == 242                       # ±60 both windows
        assert np.isfinite(scen).all()
        # the real min/peak hours, not the first-NaN poisoning argmin gives
        totals = p[:, 1:].sum(axis=0)
        assert int(np.nanargmax(totals)) == 4187          # late-June peak
        assert int(np.nanargmin(totals)) == 7439          # November trough
