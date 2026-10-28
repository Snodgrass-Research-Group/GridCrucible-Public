"""TRIM Loop — Topology Reduction via Informed Metrics (the outer loop).

MATLAB origin: OuterLoop.m (163 loc) + RemovePaths.m (86 loc).
Algorithm doc: docs/algorithms/OuterLoop.md — the scoring cascade, quantile
tiers, protection rules, and both removal guards are specified there; this
file ports them rule-for-rule (deviations are flagged inline).

Place in the flow: after OHM produces per-scenario flow matrices, TRIM scores
every candidate path, protects radial lines and degree tiers against the EI
Form-715 targets, then deletes ≤ percent_remove of the marked paths under
bus-stranding and connectivity guards.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import CrucibleConfig
from ..model import CrucibleModel
from ..power.power_flow import check_connected, SingularNetworkError


@dataclass
class TrimMarking:
    """score_and_mark report (strategy switches the orchestrator must see)."""

    n_keep_locked: int = 0
    n_radial: int = 0
    normalize_length: bool = True
    normalize_node_degree: bool = True
    inner_terminate_override: float | None = None   # 0.2 when mean ND < 3.5
    create_buses_signal: bool = False               # HV target reached


@dataclass
class TrimResult:
    """Outcome of one TRIM outer-loop iteration."""
    n_removed: int = 0                 # lines dropped this iteration
    n_protected: int = 0               # re-kept by guards
    went_sequential: bool = False      # RemoveSequential latched on


class TRIMLoop:
    """Outer-loop engine. Holds the adaptive normalization state MATLAB kept
    in ``options.Normalize`` (mutated by OuterLoop as the network thins).
    All randomness through ``rng`` — MATLAB used bare rand()."""

    def __init__(self, config: CrucibleConfig, rng: np.random.Generator) -> None:
        self.config = config
        self.rng = rng
        self.normalize_length = config.trim.normalize_length
        self.normalize_node_degree = config.trim.normalize_node_degree
        self.remove_sequential = config.trim.remove_sequential

    # ------------------------------------------------------------ phase 1

    def score_and_mark(self, model: CrucibleModel, flows: np.ndarray,
                       dcpf_flows: np.ndarray, cutoffs: np.ndarray,
                       snet_mw: np.ndarray) -> TrimMarking:
        """OuterLoop.m port. flows/dcpf_flows: (n_paths × n_scenarios) |MW|;
        cutoffs: per-scenario zero thresholds (MW); snet_mw: the scenario
        injection matrix used for SNet normalization.

        Appends scoring columns to model.candidates, applies protection and
        the L1-zeroing contract, SORTS candidates by mean_norm ascending
        (remove() consumes the first k marked rows of that order), and
        returns the strategy-switch report."""
        cfg = self.config.trim
        cand = model.candidates
        n = len(cand)
        n_scen = flows.shape[1]
        ei = model.meta.get("ei_stats", {})
        ei_dist = np.asarray(ei.get("node_degree_dist", [0.22, 0.45]), float)
        ei_avg = float(ei.get("node_degree_avg", 2.4))

        nd = model.node_degree("candidates")
        model.meta["node_degree"] = nd
        counts = np.bincount(nd[nd > 0])
        my_dist = counts[1:] / max(counts[1:].sum(), 1)

        fb = cand["from_bus"].to_numpy(np.int64)
        tb = cand["to_bus"].to_numpy(np.int64)

        # 1. per-path scenario statistics -----------------------------------
        low_flow = (flows <= cutoffs[None, :]).sum(axis=1)
        cand["low_flow_count"] = low_flow
        cand["max_flow"] = flows.max(axis=1)
        cand["max_dcpf"] = np.abs(dcpf_flows).max(axis=1)

        end_nd = np.stack([nd[fb], nd[tb]])
        line_nd = (end_nd.max(axis=0) if cfg.normalize_node_degree_max
                   else end_nd.mean(axis=0)).astype(float)

        # 2. normalization cascade (order matters; each step rescales F) ----
        F = flows.astype(float).copy()
        if cfg.normalize_rate_a:
            rate = cand["capacity_mva"].to_numpy(float)
            F = rate.mean() * F / rate[:, None]
        if cfg.normalize_snet:
            max_snet = np.abs(np.real(np.atleast_2d(snet_mw))).max(axis=1)
            line_snet = np.maximum(max_snet[fb], max_snet[tb])
            line_snet[line_snet == 0] = line_snet.mean() or 1.0
            F = float(model.meta.get("snet_scaling", 1.0)) * F / line_snet[:, None]
        if self.normalize_node_degree:
            F = line_nd.mean() * F / line_nd[:, None]
        length = cand["length_km"].to_numpy(float)
        if self.normalize_length:
            F = length.mean() * F / length[:, None]
        cand["mean_norm"] = F.mean(axis=1)
        cand["std_norm"] = F.std(axis=1, ddof=1) if n_scen > 1 else 0.0

        # 3. protection ------------------------------------------------------
        keep = cand["keep"].to_numpy(bool).copy()
        radial_buses = np.flatnonzero(nd < 2)
        is_radial = np.isin(fb, radial_buses) | np.isin(tb, radial_buses)
        keep |= is_radial
        cand["is_radial"] = cand["is_radial"].to_numpy(bool) | is_radial

        if len(my_dist) >= 1 and len(ei_dist) >= 1 \
                and my_dist[0] > 0.8 * ei_dist[0]:
            nd2 = np.flatnonzero(nd == 2)
            keep |= np.isin(fb, nd2) | np.isin(tb, nd2)
            if len(my_dist) >= 2 and len(ei_dist) >= 2 \
                    and my_dist[1] > 0.8 * ei_dist[1]:
                # MATLAB OuterLoop.m:89 re-tests ==2 where the comment says 3;
                # default implements the comment, legacy flag reproduces the bug
                tier3 = 2 if cfg.legacy_nd3_bug else 3
                nd3 = np.flatnonzero(nd == tier3)
                keep |= np.isin(fb, nd3) | np.isin(tb, nd3)

        # HV-length targets (miles; MATLAB compares NetworkStatistics'
        # cumulative HV mileage — computed here directly from candidates,
        # ASSUMPTION documented in the algorithm doc's open questions)
        kv_from = cand["voltage_kv_from"].to_numpy(float)
        hv_total_mi = float(length[kv_from >= 230].sum()) / 1.61
        target_hv = float(sum(v for k, v in
                              self.config.case.target_lengths_miles.items()
                              if 230 <= k <= 765))
        marking = TrimMarking()
        mean_nd = float(nd.mean())
        if mean_nd < 3 * ei_avg:
            self.normalize_length = hv_total_mi >= (mean_nd / ei_avg) * target_hv
            if mean_nd > ei_avg:
                self.normalize_node_degree = True
        if mean_nd < 3.5:
            marking.inner_terminate_override = 0.2
        # second toggle deliberately overrides the first (MATLAB order)
        self.normalize_length = hv_total_mi >= 1.25 * target_hv
        if hv_total_mi <= target_hv:
            keep |= kv_from >= 230
            marking.create_buses_signal = True

        cand["keep"] = keep
        cand.loc[keep, "l1_cost"] = 0.0     # the OHM/TRIM shared contract
        marking.n_keep_locked = int(keep.sum())
        marking.n_radial = int(is_radial.sum())
        marking.normalize_length = self.normalize_length
        marking.normalize_node_degree = self.normalize_node_degree

        # 4. removal probabilities (on the mean_norm-sorted table) ----------
        cand = cand.sort_values("mean_norm").reset_index(drop=True)
        low_flow = cand["low_flow_count"].to_numpy(float)
        mean_norm = cand["mean_norm"].to_numpy(float)
        std_norm = cand["std_norm"].to_numpy(float)
        keep = cand["keep"].to_numpy(bool)
        not_rad = ~cand["is_radial"].to_numpy(bool)

        prob = np.zeros(n)
        highest = low_flow >= np.quantile(low_flow, 0.95)
        prob[highest] = 0.25 * self.rng.random(int(highest.sum()))

        positive_lf = low_flow[low_flow > 0]
        lf_thresh = (max(float(np.quantile(positive_lf, 0.25)), 0.1 * n_scen)
                     if positive_lf.size else 0.1 * n_scen)
        mid = ((mean_norm <= np.median(mean_norm[not_rad]))
               & (std_norm <= np.quantile(std_norm[not_rad], 0.75))
               & ~keep & (low_flow > lf_thresh))
        prob[mid] = self.rng.random(int(mid.sum()))

        strict = ((mean_norm <= np.quantile(mean_norm[not_rad], 0.25))
                  & (std_norm <= std_norm[not_rad].mean())
                  & ~keep & (low_flow > lf_thresh))
        prob[strict] = 1.0
        prob[keep] = 0.0

        n_target = math.ceil(cfg.percent_remove * n)
        if cfg.stochastic_remove:
            if int((prob == 1.0).sum()) >= n_target:
                prob = (prob >= 1.0).astype(float)
            elif int((prob > 0).sum()) > n_target:
                prob = (prob >= self.rng.random()).astype(float)
        else:
            prob = (prob > 0).astype(float)
        cand["remove_prob"] = prob
        model.candidates = cand
        return marking

    # ------------------------------------------------------------ phase 2

    def remove(self, model: CrucibleModel) -> TrimResult:
        """RemovePaths.m port: delete ≤ ceil(percent_remove·n) marked paths
        under the bus-stranding pre-filter and an exact connectivity guard
        (replaces condest(BBus)≥1e16 — see the doc)."""
        cand = model.candidates
        n = len(cand)
        n_remove = math.ceil(self.config.trim.percent_remove * n)
        result = TrimResult()
        if n_remove <= 0:
            return result

        if self.remove_sequential:
            result.n_removed, result.n_protected = self._remove_one_by_one(
                model, n_remove)
            result.went_sequential = True
            model.meta["node_degree"] = model.node_degree("candidates")
            return result

        marked = np.flatnonzero(cand["remove_prob"].to_numpy(float) > 0)
        batch = marked[:n_remove]

        # bus-stranding pre-filter: no endpoint may drop below degree 1
        nd = model.node_degree("candidates")
        fb = cand["from_bus"].to_numpy(np.int64)
        tb = cand["to_bus"].to_numpy(np.int64)
        hits = np.bincount(np.concatenate([fb[batch], tb[batch]]),
                           minlength=len(nd))
        stranded = np.flatnonzero((nd - hits < 1) & (hits > 0))
        if stranded.size:
            touches = np.isin(fb, stranded) | np.isin(tb, stranded)
            cand.loc[touches, "keep"] = True
            cand.loc[touches, "remove_prob"] = 0.0
            result.n_protected += int(np.isin(batch, np.flatnonzero(touches)).sum())
            batch = batch[~np.isin(batch, np.flatnonzero(touches))]

        trial = cand.drop(index=cand.index[batch]).reset_index(drop=True)
        old = model.candidates
        model.candidates = trial
        try:
            check_connected(model.incidence("candidates"))
            result.n_removed = len(batch)
        except SingularNetworkError:
            model.candidates = old              # rollback → one-at-a-time
            if model.density < self.config.trim.bbr_mid:
                self.remove_sequential = True   # latch (MATLAB options mutation)
            removed, protected = self._remove_one_by_one(model, n_remove)
            result.n_removed = removed
            result.n_protected += protected
            result.went_sequential = True
        model.meta["node_degree"] = model.node_degree("candidates")
        return result

    def _remove_one_by_one(self, model: CrucibleModel, n_remove: int) -> tuple[int, int]:
        removed = protected = 0
        for _ in range(n_remove):
            cand = model.candidates
            marked = np.flatnonzero(cand["remove_prob"].to_numpy(float) > 0)
            if marked.size == 0:
                break
            i = int(marked[0])
            trial = cand.drop(index=cand.index[i]).reset_index(drop=True)
            old = model.candidates
            model.candidates = trial
            try:
                check_connected(model.incidence("candidates"))
                removed += 1
            except SingularNetworkError:
                model.candidates = old
                model.candidates.loc[model.candidates.index[i], "keep"] = True
                model.candidates.loc[model.candidates.index[i], "remove_prob"] = 0.0
                protected += 1
        return removed, protected

    # ------------------------------------------------------------ terminal

    def converged(self, model: CrucibleModel) -> bool:
        """MultiLevelProgram 355-385: mean node degree below the EI average
        (bus-level model) or 2.6 (substation-level)."""
        nd = model.node_degree("candidates")
        ei_avg = float(model.meta.get("ei_stats", {}).get("node_degree_avg", 2.4))
        bus_level = len(model.buses) > len(model.substations)
        threshold = ei_avg if bus_level else 2.6
        return float(nd.mean()) < threshold
