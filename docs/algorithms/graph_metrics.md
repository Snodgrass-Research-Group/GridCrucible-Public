# Sogol Realism leaves + cyclebasisport → `metrics/graph_metrics.py`

**Sources:** Sogol Realism/{eigencentrality (24), fieldervector (32),
normalizedlaplacian (34), nodedegre (25), Singularvalues (20),
TIERefficient (87)}.m + cyclebasisport.m (191 — itself a networkx
cycle_basis port, so Python "conversion" = networkx).

All operate on the UNDIRECTED SIMPLE graph — parallel lines are deduplicated
first (MATLAB `unique(A','rows','stable')`), which makes their degree
semantics DIFFERENT from `CrucibleModel.node_degree` (which counts
parallels, matching diag(A·Aᵀ)). Python functions take
`(from_bus, to_bus, n_buses)` arrays (0-based) — no MATPOWER/pandapower
struct needed (MATLAB went through Table2Matpower + ext2int only for
renumbering, which our 0-based ids already guarantee).

| MATLAB | Python | Method |
|---|---|---|
| nodedegre | `node_degree_simple` | dedup edges → bincount |
| eigencentrality | `eigencentrality` | `eigsh(adjacency, k=1, which='LM')` (dominant vec+val) |
| fieldervector | `fiedler_vector` | L = Ã·Ãᵀ (deduped); 2 smallest eigs → (λ₂, v₂); dense fallback for robustness on small n |
| normalizedlaplacian | `normalized_laplacian_eigs` | D^{-1/2}·L·D^{-1/2}, FULL spectrum → dense `eigvalsh` (MATLAB eigs(L,n) was full too; O(n³) — fine at SRTV scale, warn ≥ 20k) |
| Singularvalues | `b_matrix_singular_values` | B = A·diag(1/x)·Aᵀ (power_flow.b_matrix), full SVD (dense) |
| TIERefficient | `tier_efficiency` | λ = B⁻¹·(A·diag(1/x)) column block-solves (2000-col chunks, like MATLAB); TIER(branch) = std over GENERATOR buses (ddof=1). MATLAB patched the singular B with `U(end,end)=1`; std is invariant to the nullspace (constant) offset, so Python grounds the last bus instead — same TIER values, well-posed solve |
| cyclebasis(port) | `cycle_basis_lengths` | `networkx.cycle_basis` on the deduped graph → cycle lengths |

Shared fit helpers (used by RealismMetrics on cycle/degree histograms):
`exp_fit` (a·e^{bx}), `power_fit` (a·x^b + c) via `scipy.optimize.curve_fit`
with R² + trapezoid area (MATLAB `fit('exp1'/'power2')` + `trapz`), and
`log_linear_fit` (poly1 on log10). All return None on degenerate inputs
(< 3 points) instead of MATLAB's crash.
