"""Graph-theoretic realism leaves (Sogol's helpers + cycle basis + fits).

MATLAB origin: Matlab Functions/Sogol Realism/*.m + cyclebasisport.m.
Algorithm doc: docs/algorithms/graph_metrics.md — notably: all metrics work
on the DEDUPLICATED simple graph (different degree semantics than
CrucibleModel.node_degree), and TIER's singular-pivot hack is replaced by
grounding (std is offset-invariant, so values match).
"""

from __future__ import annotations

import warnings

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh, splu


def dedup_edges(from_bus: np.ndarray, to_bus: np.ndarray
                ) -> tuple[np.ndarray, np.ndarray]:
    """Unique unordered bus pairs (MATLAB unique(A','rows','stable'))."""
    lo = np.minimum(from_bus, to_bus)
    hi = np.maximum(from_bus, to_bus)
    pairs = np.unique(np.column_stack([lo, hi]), axis=0)
    return pairs[:, 0], pairs[:, 1]


def _adjacency(f: np.ndarray, t: np.ndarray, n: int) -> sp.csr_matrix:
    data = np.ones(2 * len(f))
    return sp.csr_matrix((data, (np.concatenate([f, t]),
                                 np.concatenate([t, f]))), shape=(n, n))


def _simple_laplacian(f: np.ndarray, t: np.ndarray, n: int) -> sp.csr_matrix:
    a = _adjacency(f, t, n)
    return sp.diags(np.asarray(a.sum(axis=1)).ravel()) - a


def node_degree_simple(from_bus, to_bus, n_buses: int) -> np.ndarray:
    """nodedegre port: degree on the deduplicated simple graph."""
    f, t = dedup_edges(np.asarray(from_bus, int), np.asarray(to_bus, int))
    return np.bincount(np.concatenate([f, t]), minlength=n_buses)


def eigencentrality(from_bus, to_bus, n_buses: int) -> tuple[np.ndarray, float]:
    """eigencentrality port: dominant (Perron) eigenvector/value of the
    adjacency. Uses which='LA': for adjacency matrices the Perron root
    equals the largest magnitude, and this avoids MATLAB 'largestabs'
    returning −λ on bipartite graphs (symmetric spectrum)."""
    f, t = dedup_edges(np.asarray(from_bus, int), np.asarray(to_bus, int))
    vals, vecs = eigsh(_adjacency(f, t, n_buses), k=1, which="LA")
    vec = vecs[:, 0]
    if vec.sum() < 0:                      # canonical nonnegative Perron vector
        vec = -vec
    return vec, float(vals[0])


def fiedler_vector(from_bus, to_bus, n_buses: int) -> tuple[float, np.ndarray]:
    """fieldervector port: (λ₂, v₂) of the simple-graph Laplacian."""
    f, t = dedup_edges(np.asarray(from_bus, int), np.asarray(to_bus, int))
    L = _simple_laplacian(f, t, n_buses)
    if n_buses < 3000:
        vals, vecs = np.linalg.eigh(L.toarray())
        return float(vals[1]), vecs[:, 1]
    vals, vecs = eigsh(L.tocsc(), k=2, sigma=-1e-8, which="LM")
    order = np.argsort(vals)
    return float(vals[order[1]]), vecs[:, order[1]]


def normalized_laplacian_eigs(from_bus, to_bus, n_buses: int) -> np.ndarray:
    """normalizedlaplacian port: FULL spectrum of D^{-1/2}·L·D^{-1/2}
    (dense O(n³) — same as MATLAB's eigs(L,n); warn at scale)."""
    if n_buses >= 20000:
        warnings.warn(f"full normalized-Laplacian spectrum on {n_buses} "
                      "buses is O(n^3)", stacklevel=2)
    f, t = dedup_edges(np.asarray(from_bus, int), np.asarray(to_bus, int))
    L = _simple_laplacian(f, t, n_buses).toarray()
    d = np.sqrt(np.diag(L))
    d[d == 0] = 1.0
    Ln = L / np.outer(d, d)
    return np.linalg.eigvalsh(Ln)[::-1]          # descending like eigs


def b_matrix_singular_values(from_bus, to_bus, x_pu, n_buses: int) -> np.ndarray:
    """Singularvalues port: full SVD of B = A·diag(1/x)·Aᵀ (NO dedup —
    MATLAB makeBdc keeps parallel lines)."""
    from ..power.power_flow import b_matrix

    f = np.asarray(from_bus, int)
    t = np.asarray(to_bus, int)
    m = len(f)
    A = sp.csc_matrix((np.concatenate([np.ones(m), -np.ones(m)]),
                       (np.concatenate([f, t]), np.tile(np.arange(m), 2))),
                      shape=(n_buses, m))
    B = b_matrix(A, np.asarray(x_pu, float)).toarray()
    return np.linalg.svd(B, compute_uv=False)


def tier_efficiency(from_bus, to_bus, x_pu, gen_buses, n_buses: int,
                    chunk: int = 2000) -> tuple[np.ndarray, np.ndarray]:
    """TIERefficient port: per-branch importance = std over GENERATOR buses
    of the angle-sensitivity column λ = B⁻¹·(A·diag(1/x))[:, branch].

    MATLAB solved the SINGULAR B via an LU pivot patch (U(end,end)=1);
    std(·) is invariant to the constant nullspace offset, so we ground the
    last bus instead — identical TIER, well-posed solve. Returns
    (tier_sorted_desc, tier)."""
    f = np.asarray(from_bus, int)
    t = np.asarray(to_bus, int)
    m = len(f)
    A = sp.csc_matrix((np.concatenate([np.ones(m), -np.ones(m)]),
                       (np.concatenate([f, t]), np.tile(np.arange(m), 2))),
                      shape=(n_buses, m))
    b = sp.diags(1.0 / np.asarray(x_pu, float))
    B = (A @ b @ A.T).tocsc()
    F = (A @ b).tocsc()
    lu = splu(B[:-1, :-1])                        # ground the last bus
    gen = np.asarray(gen_buses, int)
    tier = np.zeros(m)
    for lo in range(0, m, chunk):
        hi = min(lo + chunk, m)
        lam_red = lu.solve(F[:-1, lo:hi].toarray())
        lam = np.vstack([lam_red, np.zeros((1, hi - lo))])
        tier[lo:hi] = lam[gen, :].std(axis=0, ddof=1)
    return np.sort(tier)[::-1], tier


def cycle_basis_lengths(from_bus, to_bus) -> np.ndarray:
    """cyclebasis(port): lengths of a cycle basis of the simple graph."""
    import networkx as nx

    f, t = dedup_edges(np.asarray(from_bus, int), np.asarray(to_bus, int))
    g = nx.Graph(zip(f.tolist(), t.tolist()))
    return np.array([len(c) for c in nx.cycle_basis(g)], dtype=int)


# ---------------------------------------------------------------------------
# Fit helpers (MATLAB fit('exp1'/'power2'/'poly1') + trapz replacements)
# ---------------------------------------------------------------------------

def _r_squared(y: np.ndarray, y_hat: np.ndarray) -> float:
    ss_res = float(((y - y_hat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0


def exp_fit(x: np.ndarray, y: np.ndarray) -> dict | None:
    """MATLAB fit(...,'exp1'): y = a·e^{bx}. None on degenerate input."""
    from scipy.optimize import curve_fit

    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3:
        return None
    try:
        p, _ = curve_fit(lambda xv, a, bb: a * np.exp(bb * xv), x, y,
                         p0=(max(y.max(), 1e-6), -0.3), maxfev=10000)
    except RuntimeError:
        return None
    y_hat = p[0] * np.exp(p[1] * x)
    return {"a": float(p[0]), "b": float(p[1]),
            "r2": _r_squared(y, y_hat), "area": float(np.trapezoid(y_hat, x))}


def power_fit(x: np.ndarray, y: np.ndarray) -> dict | None:
    """MATLAB fit(...,'power2'): y = a·x^b + c. None on degenerate input."""
    from scipy.optimize import curve_fit

    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 4 or (x <= 0).any():
        return None
    try:
        p, _ = curve_fit(lambda xv, a, bb, c: a * xv ** bb + c, x, y,
                         p0=(1.0, -1.0, 0.0), maxfev=20000)
    except RuntimeError:
        return None
    y_hat = p[0] * x ** p[1] + p[2]
    return {"a": float(p[0]), "b": float(p[1]), "c": float(p[2]),
            "r2": _r_squared(y, y_hat), "area": float(np.trapezoid(y_hat, x))}


def log_linear_fit(degrees: np.ndarray, dist: np.ndarray) -> dict | None:
    """MATLAB poly1 fit of log10(dist) vs degree, with the truncate-at-
    first-zero rule applied by the CALLER. None when < 3 points (MATLAB
    would crash on regular graphs where dist[0] == 0)."""
    degrees, dist = np.asarray(degrees, float), np.asarray(dist, float)
    if len(degrees) < 3 or (dist <= 0).any():
        return None
    logd = np.log10(dist)
    slope, intercept = np.polyfit(degrees, logd, 1)
    y_hat = slope * degrees + intercept
    return {"slope": float(slope), "intercept": float(intercept),
            "r2": _r_squared(logd, y_hat),
            "area": float(np.trapezoid(y_hat, degrees))}


def histogram_normalized(values: np.ndarray, bin_edges: np.ndarray) -> dict:
    """MATLAB histogram + bin-middle normalization used for cycle/MVA hists."""
    counts, edges = np.histogram(np.asarray(values, float), bins=bin_edges)
    mids = (edges[:-1] + edges[1:]) / 2.0
    total = counts.sum()
    return {"bin_mid": mids, "counts": counts,
            "norm": counts / total if total else counts.astype(float)}
