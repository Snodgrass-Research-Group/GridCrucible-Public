"""Regional hourly load-profile access + the day-boundary smoothing QP.

MATLAB origin: the feval-dispatched MakeLoadProfiles_<REGION>.m family
(15 files, ~8,350 loc) + smoothprofile.m (47 loc — ACTIVE; the call-tree
analyzer wrongly orphaned it).
Algorithm doc: docs/algorithms/MakeLoadProfiles.md — including the port
strategy: the builders' outputs are deterministic data products, BRIDGED for
now via a one-time MATLAB export (tools/export_opaque_tables.m v3 →
data/load_profiles/<REGION>.npz), while the pure-Python builder (NREL
aggregation with the hard-coded cluster groups, sector construction, BusDist
distribution) is a tracked follow-up. `smooth_profile` — the one genuine
algorithm in the chain — is ported here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def make_load_profiles(region: str, data_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return the region's (P_Load, Q_Load) EXACTLY as the MATLAB builder
    returned them — bus-number first column possibly included; downstream
    handling (stripping/scattering) belongs to scenarios.build_load_scenarios,
    mirroring the MATLAB call chain."""
    npz = Path(data_dir) / "load_profiles" / f"{region}.npz"
    if not npz.exists():
        raise FileNotFoundError(
            f"{npz} — build it from LoadSmith with "
            "`python tools/make_load_profiles_from_loadsmith.py --loadsmith "
            f"<LoadSmith clone> --region {region}` (or export it from MATLAB "
            "with tools/export_opaque_tables.m and re-run the migration)")
    d = np.load(npz)
    return d["p_load"], d["q_load"]


def smooth_profile(pin: np.ndarray, gurobi_env=None) -> np.ndarray:
    """smoothprofile.m port: patch non-smooth jumps at day boundaries.

    For each consecutive day pair, add a quadratic correction pivoting over
    hours 20..24 of day 1 (z = a·[0..4]²/16 per channel) and hours 1..5 of
    day 2 (z = b·[4..0]²/16), minimizing (Σa)² + (Σb)² subject to the
    corrected second derivative around midnight being ≥ the MATLAB bound
    (A·x ≤ b with A = (2−9/16)I − antidiagonal blocks). Sequential over
    days, each pair seeing prior corrections. Accepts (hours × channels);
    generalizes MATLAB's hard-coded 3 channels. Rows < 48 pass through."""
    import gurobipy as gp

    pin = np.atleast_2d(np.asarray(pin, float))
    if pin.shape[0] == 1:
        pin = pin.T
    if pin.shape[0] < 48:
        return pin
    n_ch = pin.shape[1]
    pout = pin.copy()
    n_days = pin.shape[0] // 24

    # constant problem structure (MATLAB H/A with n channels)
    A = (2 - 9 / 16) * np.eye(2 * n_ch)
    A[:n_ch, n_ch:] = -np.eye(n_ch)
    A[n_ch:, :n_ch] = -np.eye(n_ch)
    rise = np.arange(5, dtype=float) ** 2 / 16          # [0..4]²/16
    fall = np.arange(4, -1, -1, dtype=float) ** 2 / 16  # [4..0]²/16

    for day in range(n_days - 1):
        a1 = day * 24
        seg = pout[a1:a1 + 48, :]
        b1 = seg[24, :] - 2 * seg[23, :] + seg[22, :]   # 2nd-deriv estimates
        b2 = seg[25, :] - 2 * seg[24, :] + seg[23, :]
        rhs = np.concatenate([b1, b2])

        m = gp.Model("smooth", env=gurobi_env)
        m.Params.OutputFlag = 0
        x = m.addMVar(2 * n_ch, lb=-gp.GRB.INFINITY)
        m.addMConstr(A, x, "<", rhs)
        # (Σa)² + (Σb)² == xᵀ blkdiag(ones, ones) x (MATLAB H, no ½ factor
        # difference matters: argmin is identical)
        H = np.zeros((2 * n_ch, 2 * n_ch))
        H[:n_ch, :n_ch] = 1.0
        H[n_ch:, n_ch:] = 1.0
        m.setObjective(x @ H @ x, gp.GRB.MINIMIZE)
        m.optimize()
        if m.Status != gp.GRB.OPTIMAL:
            continue                                    # leave this pair as-is
        sol = x.X
        z = np.zeros_like(seg)
        z[19:24, :] = rise[:, None] * sol[:n_ch][None, :]
        z[24:29, :] = fall[:, None] * sol[n_ch:][None, :]
        pout[a1:a1 + 48, :] = seg + z
    return pout
