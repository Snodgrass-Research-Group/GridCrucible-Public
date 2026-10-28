# RealismMetrics.m → `metrics/realism.py`

**Source:** RealismMetrics.m (611 loc — of which ~400 are figure code that
belongs to `viz/plots.py`; this module COMPUTES and returns).
Called near TRIM convergence, at termination, and
in the metrics stage.

## Computed metrics (the returned dict / model.meta['realism'])

1. **Cycle basis** (`cycles`): lengths of the graph's cycle basis, mean
   length, histogram on bins 2:2:80 (values at BIN MIDDLES, normalized by
   total count), plus `exp_fit` (a·e^{bx}) and `power_fit` (a·x^b+c) with
   coefficients, R², and trapezoid area — the numbers the dissertation
   compares against the EI/WECC fits. Skipped (None) when the network has
   no cycles.
2. **Node-degree distributions + log-linear fits** (`degree_fits`), FIVE
   variants matching MATLAB exactly:
   * `node_degree` — diag(A·Aᵀ) over ALL branches (incidence-based),
   * `sub_degree_lines` — substation endpoints (from_orig/to_orig when
     present), tap==0 rows only,
   * `sub_degree` — substation endpoints, all branches,
   * `bus_degree_lines` — internal endpoints, tap==0 only,
   * `bus_degree` — internal endpoints, all branches.
   Each: distribution over degrees 1..max, TRUNCATED AT THE FIRST ZERO
   entry (MATLAB rule), poly1 fit to log10(dist) vs degree → slope,
   intercept, R², trapz area. Degenerate cases (first entry zero — e.g.
   regular graphs; < 3 surviving points) return fit=None where MATLAB would
   crash; distributions are still reported.
   (Pre-CreateBuses the sub_/bus_ variants coincide — same in MATLAB.)
3. **MVA-limit histograms** (`mva_hist`): per voltage level with the FIXED
   MATLAB bin counts {69: 27, 138: 20, 230: 20, 345: 12, 500: 8, 765: 6},
   over rows whose BOTH end kVs equal the level; raw + normalized values at
   bin middles.
4. **Sogol spectral block** (`sogol`, opt-in `include_sogol=True` —
   codectrl.StatsSogol): dominant eigencentrality, Fiedler vector,
   normalized-Laplacian spectrum, B-matrix singular values, TIER values,
   deduped node degree (needs reactance + generator buses).

## Interface

```python
realism_metrics(model, config, table="candidates",
                include_sogol=False) -> dict   # also -> model.meta['realism']
```

## Open questions

1. MATLAB's `BusDegree` variant bins `histcounts(..., params.nodes-1)`
   (count-based) unlike the other four (edge-based) — inconsistency ported
   as edge-based for all five (flagged; affects nothing downstream, only
   the fit inputs marginally).
2. The Bus.MaxMW→slack coupling in Table2Matpower is irrelevant here (the
   Sogol leaves ignore gen placement except TIER's gen-bus std).
