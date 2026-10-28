"""Tests for viz/plots.py.

Every function must be Agg-safe (matplotlib OO API, no pyplot/backend
dependence), return a Figure (or list), save a nonempty PNG when ``out``
is given, and tolerate degenerate metric dicts (None fits, empty dists,
JSON-cache lists instead of ndarrays).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from gridcrucible.metrics.realism import realism_metrics
from gridcrucible.viz import plots


@pytest.fixture
def varied_model(tiny_model):
    """tiny_model with an extra degree-3→ varied topology so degree fits
    are non-degenerate (the 6-ring + chords is 3-regular → empty fits)."""
    cand = tiny_model.candidates
    extra = cand.iloc[[0]].copy()
    extra["from_bus"], extra["to_bus"] = 0, 2
    extra["from_orig"], extra["to_orig"] = 0, 2
    tiny_model.candidates = pd.concat([cand, extra], ignore_index=True)
    return tiny_model


@pytest.fixture
def synthetic(varied_model, config):
    return realism_metrics(varied_model, config, table="candidates")


def _fake_cycles():
    x = np.arange(3.0, 21.0, 2.0)
    return {"lengths": np.repeat(x, 3), "mean_length": float(x.mean()),
            "hist": {"bin_mid": x, "counts": np.ones_like(x),
                     "norm": np.geomspace(0.4, 0.01, x.size)},
            "exp_fit": {"a": 0.5, "b": -0.2, "r2": 0.95, "area": 1.1},
            "power_fit": {"a": 1.0, "b": -1.5, "c": 0.0, "r2": 0.9,
                          "area": 0.8},
            "weibull_fit": {"a": 0.15, "b": 1.2, "r2": 0.92, "area": 0.9}}


def _fake_degree_fits(labels):
    fits = {}
    for i, label in enumerate(labels):
        d = np.arange(1, 7)
        dist = np.geomspace(0.5, 0.005, 6)
        fits[label] = {"degrees": d, "dist": dist,
                       "fit": {"slope": -0.4 - 0.02 * i, "intercept": -0.2,
                               "r2": 0.9, "area": -10.0}}
    return fits


def _fake_actual_flavor():
    return {"cycles": _fake_cycles(),
            "degree_fits": _fake_degree_fits(
                list(plots.DEGREE_VARIANTS) + ["bus_degree_in_service"])}


@pytest.fixture
def targets():
    return {"EI": {"branch": _fake_actual_flavor(),
                   "bus": _fake_actual_flavor()},
            "WECC": {"branch": _fake_actual_flavor(),
                     "bus": _fake_actual_flavor()}}


def _check(fig, out=None):
    assert isinstance(fig, Figure)
    if out is not None:
        assert out.exists() and out.stat().st_size > 0


# ---------------------------------------------------------------------------

def test_plot_network(varied_model, tmp_path):
    out = tmp_path / "net.png"
    fig = plots.plot_network(varied_model, out, iteration=3)
    _check(fig, out)
    # sparse network (10 < 5*6) → abstract-graph panel + geographic panel
    assert len(fig.axes) >= 2


def test_plot_network_dense_no_graph_panel(varied_model, tmp_path):
    big = pd.concat([varied_model.candidates] * 4, ignore_index=True)
    varied_model.candidates = big               # 40 >= 5*6 → geo only
    fig = plots.plot_network(varied_model)
    assert len(fig.axes) == 1


def test_plot_cycle_fits(synthetic, targets, tmp_path):
    out = tmp_path / "cb.png"
    fig = plots.plot_cycle_fits(synthetic, targets, out, region="SRTV")
    _check(fig, out)
    assert len(fig.axes) == 6                   # 2 synthetic + 2×2 compare


def test_plot_cycle_fits_no_targets(synthetic):
    fig = plots.plot_cycle_fits(synthetic)
    assert len(fig.axes) == 2


def test_plot_cycle_fits_degenerate():
    fig = plots.plot_cycle_fits({"cycles": None})
    _check(fig)


def test_plot_real_cycle_fits(targets, tmp_path):
    out = tmp_path / "real_cb.png"
    fig = plots.plot_real_cycle_fits(targets, out)
    _check(fig, out)
    assert len(fig.axes) == 2


def test_plot_degree_fit(synthetic, tmp_path):
    out = tmp_path / "nd.png"
    entry = synthetic["degree_fits"]["node_degree"]
    fig = plots.plot_degree_fit(entry, out, label="node_degree")
    _check(fig, out)


def test_plot_degree_fit_degenerate():
    fig = plots.plot_degree_fit({"degrees": np.empty(0),
                                 "dist": np.empty(0), "fit": None})
    _check(fig)


def test_plot_degree_fit_comparison(synthetic, targets, tmp_path):
    out = tmp_path / "ndc.png"
    fig = plots.plot_degree_fit_comparison(
        synthetic, _fake_actual_flavor(), targets, "node_degree", out)
    _check(fig, out)
    assert len(fig.axes) == 2                   # one panel per case


def test_plot_mva_histograms(synthetic, targets, tmp_path):
    synthetic["mva_hist"] = {
        138: {"bin_mid": np.linspace(100, 400, 10),
              "counts": np.ones(10), "norm": np.full(10, 0.1)}}
    targets["EI"]["form715"] = {"mva_hist": {
        138: {"bin_mid": np.linspace(120, 640, 20),
              "counts": np.ones(20), "norm": np.full(20, 0.05)}}}
    out = tmp_path / "mva.png"
    fig = plots.plot_mva_histograms(synthetic, targets, out)
    _check(fig, out)
    # both bars landed on the panel
    assert len(fig.axes[0].patches) == 30


def test_plot_node_degrees(synthetic, targets, tmp_path):
    out = tmp_path / "deg.png"
    fig = plots.plot_node_degrees(synthetic, targets, out)
    _check(fig, out)
    assert len(fig.axes) == 10                  # 5 variants × 2 cases


def test_plot_node_degrees_comparison(synthetic, targets, tmp_path):
    garver = {"degree_fits": _fake_degree_fits(list(plots.DEGREE_VARIANTS))}
    out = tmp_path / "degc.png"
    fig = plots.plot_node_degrees_comparison(synthetic, garver, targets, out)
    _check(fig, out)
    # KCL-Only line present in the first panel legend
    labels = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
    assert any("KCL-Only" in s for s in labels)
    assert any("KCL/KVL" in s for s in labels)


def test_plot_node_degrees_actual(targets, tmp_path):
    out = tmp_path / "dega.png"
    fig = plots.plot_node_degrees_actual(targets, out)
    _check(fig, out)
    assert len([a for a in fig.axes if a.get_visible()]) >= 6


def test_plot_node_degrees_actual_json_roundtrip(targets, tmp_path):
    """realism_metrics_actual results may come back from the JSON cache —
    lists instead of ndarrays must not break the plots."""
    def jsonable(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.integer, np.floating)):
            return o.item()
        raise TypeError
    rt = json.loads(json.dumps(targets, default=jsonable))
    fig = plots.plot_node_degrees_actual(rt)
    _check(fig)


def test_plot_generator_degrees(varied_model, tmp_path):
    out = tmp_path / "gen_deg.png"
    fig = plots.plot_generator_degrees(varied_model, out)
    _check(fig, out)
    assert len(fig.axes) == 2


# ---------------------------------------------------------------------------
# PlotILFun
# ---------------------------------------------------------------------------

@pytest.fixture
def sweep(varied_model):
    rng = np.random.default_rng(7)
    m = len(varied_model.candidates)
    gammas = np.logspace(-5, -4, 5)
    flows = np.abs(rng.normal(50, 30, size=(m, 5)))
    flows[rng.random((m, 5)) < 0.3] = 0.001     # some near-zero flows
    on = flows >= 1.0
    return varied_model, gammas, flows, on


def test_plot_inner_loop_diagnostics(sweep, tmp_path):
    model, gammas, flows, on = sweep
    out = tmp_path / "il.png"
    figs = plots.plot_inner_loop_diagnostics(
        gammas, flows, on, out,
        from_bus=model.candidates["from_bus"].to_numpy(),
        to_bus=model.candidates["to_bus"].to_numpy(),
        percent_low_flow=np.linspace(0.1, 0.5, 5))
    assert len(figs) == 3                       # hist, stats, dstats
    for f in figs:
        assert isinstance(f, Figure)
    for suffix in ("_hist", "_stats", "_dstats"):
        p = tmp_path / f"il{suffix}.png"
        assert p.exists() and p.stat().st_size > 0


def test_plot_inner_loop_no_graph_inputs(sweep):
    """MATLAB crashed when Graphs=0 but NodeDegree=1 (doc Open Q4) — the
    port degrades to flow-only stats."""
    _, gammas, flows, on = sweep
    figs = plots.plot_inner_loop_diagnostics(gammas, flows, on)
    assert len(figs) == 3


def test_plot_inner_loop_short_sweep(sweep):
    """< 4 gammas broke the MATLAB Lam(4) spacing test — port guards it."""
    _, gammas, flows, on = sweep
    figs = plots.plot_inner_loop_diagnostics(gammas[:2], flows[:, :2],
                                             on[:, :2])
    assert len(figs) == 3


def test_plot_inner_loop_single_gamma(sweep):
    _, gammas, flows, on = sweep
    figs = plots.plot_inner_loop_diagnostics(gammas[:1], flows[:, :1],
                                             on[:, :1])
    assert len(figs) == 2                       # no derivative panel
