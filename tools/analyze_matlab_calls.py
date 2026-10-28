"""MATLAB call-graph analyzer for the GridCrucible conversion project.

GridCrucible is the Python conversion of the KCL-KVL Compliant Grid Generator
(the ``main_SRTV.m`` QP + L1-regularizer pipeline). This tool parses every
``.m`` file in the dissertation code folders, builds a static call graph
rooted at the two mains (``main_SRTV.m`` and ``MultiLevelProgram.m``), and
reports:

* the call tree (who calls whom, to arbitrary depth),
* which functions are ACTIVE (reachable from a root) vs ORPHANED,
* which active functions are ALREADY OWNED by the parallel GridForge
  conversion effort (anything in gridforge/docs/call_graph.json's active set;
  those are converted in GridForge, NOT here),
* dynamic ``feval`` dispatch families (``GMDData<Region>``,
  ``MakeLoadProfiles_<Region>``) and ``loadcase('...')`` string calls that a
  naive static scan would miss,
* which external dependencies each file touches (CPLEX, Optimization Toolbox,
  MATPOWER, MEX files, parallel pool, statistics toolbox).

Outputs (written to docs/ unless --out is given):
    call_tree.txt   - indented text tree + flat adjacency list (machine-parseable)
    call_graph.json - full structured dump
    CALL_TREE.md    - clustered Mermaid diagram (renders on GitHub / VS Code)

Usage:  python tools/analyze_matlab_calls.py --matlab-parent <folder>
            [--gridforge-graph <GridForge>/docs/call_graph.json] [--out docs]
where <folder> holds "KCL-KVL Compliant Grid Generator" and "KCL-Compliant
Grid Generator". Reads the MATLAB sources; writes only the three outputs.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date
from pathlib import Path

# Set from the command line in main().
REPO = Path(".")
KVL = REPO / "KCL-KVL Compliant Grid Generator"
KCL = REPO / "KCL-Compliant Grid Generator"
DOCS = Path(__file__).resolve().parents[1] / "docs"
GRIDFORGE_GRAPH = Path("gridforge-call-graph-not-given.json")

ROOTS = ["main_SRTV", "MultiLevelProgram"]

# Folders inside the KVL tree that contain callable .m code, in precedence
# order (first hit wins for duplicate stems). "Default Input Data" ships
# duplicate GMD tables + MATPOWER case scripts, so it comes last.
KVL_CODE_DIRS = [
    KVL / "Matlab Functions",          # includes Sogol Realism/ subfolder
    KVL / "SRTV",                      # main_SRTV.m + User Input Data/InitOptions.m
    KVL / "GMD Statistics and lookup tables",
    KVL / "Master Load Data",          # MakeLoadProfiles_<REGION>.m per region
    KVL / "Default Input Data",        # loadcase() MATPOWER case scripts
]

# Dynamic feval() dispatch: caller code pattern -> function-name prefix.
# Every universe file whose stem starts with the prefix becomes a callee,
# collapsed under one synthetic "family" node so the tree stays readable.
DYNAMIC_FAMILIES = {
    "GMDData": re.compile(r"feval\(\s*\[\s*'GMDData'"),
    "MakeLoadProfiles_": re.compile(r"feval\(\s*\[\s*'MakeLoadProfiles_'"),
}

# loadcase('CaseName') pulls in a MATPOWER case script by string.
LOADCASE_RE = re.compile(r"loadcase\(\s*'(\w+)'\s*\)")

# Functional clusters: drive the Mermaid subgraphs AND mirror the planned
# gridcrucible Python module layout (see docs/ARCHITECTURE.md §3).
CLUSTERS = {
    "Driver (GridCrucible)": ["main_SRTV", "InitOptions", "MultiLevelProgram"],
    "OHM Loop (inner)": ["DCInnerLoop", "ACInnerLoopOptim"],
    "TRIM Loop (outer)": ["OuterLoop", "RemovePaths"],
    "TUNE Loop (middle)": ["MiddleLoop", "MiddleLoopAC", "AssignConductors",
                           "ResizeBranches", "TransformerImpedances"],
    "Network construction": ["CreateBuses", "CreateBranch", "CreateOLMatrices",
                             "ReactivePowerPlacement", "PVPQSwitching",
                             "TrimQNCALines"],
    "Power flow / OPF": ["DCPowerFlowSimple", "DCPowerFlowFull",
                         "ACPowerFlowSimple", "ACPowerFlowFinal", "YBusCalc",
                         "YBusCalcDC", "MatpowerDCOPF", "Table2Matpower"],
    "Data / dispatch": ["CreateSNet", "CreateLoadScenarios",
                        "MakeLoadProfiles_<Region>", "CreateGeneratorDataNational",
                        "UCED", "UCCD", "GMDData<Region>", "RXB",
                        "OLPathDistances", "EIBranchSanitized",
                        "EILargeIslandSanitized", "WECCBranchSanitized",
                        "WECCLargeIslandSanitized"],
    "Metrics / comparison": ["RealismMetrics", "RealismMetricsActualEI",
                             "RealismMetricsActualWECC", "RealismMetricsComparison",
                             "NetworkGraphComparison", "ElectricalComparison",
                             "NetworkStatistics", "Singularvalues", "TIERefficient",
                             "cyclebasisport", "eigencentrality", "fieldervector",
                             "nodedegre", "normalizedlaplacian"],
    "Plots": ["PlotNetwork", "PlotNodeDegrees", "PlotNodeDegreesActual",
              "PlotNodeDegreesComparison", "PlotILFun"],
    "Writers / IO": ["SaveResults", "WriteLoads", "WriteMetrics",
                     "WritePowerWorld", "WriteSubstationData", "cell2csv"],
}

# External symbols worth tracking (not part of the dissertation codebase).
EXTERNALS = {
    "cplexmilp": "CPLEX MILP solver   -> gurobipy",
    "cplexlp": "CPLEX LP solver     -> gurobipy",
    "cplexqp": "CPLEX QP solver     -> gurobipy",
    "linprog": "Opt. Toolbox LP     -> gurobipy",
    "quadprog": "Opt. Toolbox QP     -> gurobipy",
    "intlinprog": "Opt. Toolbox MILP   -> gurobipy",
    "fmincon": "Opt. Toolbox NLP    -> gurobipy / scipy.optimize",
    "runpf": "MATPOWER AC PF      -> pandapower.runpp",
    "rundcpf": "MATPOWER DC PF      -> pandapower.rundcpp",
    "rundcopf": "MATPOWER DC OPF     -> pandapower.rundcopp",
    "runopf": "MATPOWER AC OPF     -> pandapower.runopp",
    "mpoption": "MATPOWER options    -> pandapower kwargs",
    "loadcase": "MATPOWER case load  -> pandapower converter / parquet",
    "VChooseK": "C MEX combinations  -> np.triu_indices / itertools",
    "parpool": "Parallel Toolbox    -> concurrent.futures / joblib",
    "parfeval": "Parallel Toolbox    -> concurrent.futures",
    "eigs": "sparse eigensolver  -> scipy.sparse.linalg.eigsh",
    "graphconncomp": "Bioinfo Toolbox     -> scipy.sparse.csgraph / networkx",
    "graphshortestpath": "Bioinfo Toolbox     -> networkx / scipy csgraph",
    "distance": "Mapping Toolbox     -> haversine (numpy)",
    "deg2km": "Mapping Toolbox     -> numpy constant",
    "fitdist": "Statistics Toolbox  -> scipy.stats",
    "ksdensity": "Statistics Toolbox  -> scipy.stats.gaussian_kde",
    "histfit": "Statistics Toolbox  -> matplotlib + scipy.stats",
    "xlswrite": "Excel write         -> pandas/openpyxl",
    "xlsread": "Excel read          -> pandas/openpyxl",
}


def matlab_files(folder: Path) -> list[Path]:
    """All .m files under folder, skipping OLD/backup subdirectories."""
    out = []
    if not folder.exists():
        return out
    for p in sorted(folder.rglob("*.m")):
        if "OLD" in p.parts or "Backup" in p.parts:
            continue
        out.append(p)
    return out


def strip_comments(src: str) -> str:
    """Remove %{...%} blocks and % line comments (string-aware, best effort)."""
    src = re.sub(r"^\s*%\{.*?^\s*%\}", "", src, flags=re.M | re.S)
    lines = []
    for line in src.splitlines():
        out, in_str = [], False
        for c in line:
            if c == "'":
                in_str = not in_str
                out.append(c)
            elif c == "%" and not in_str:
                break
            else:
                out.append(c)
        lines.append("".join(out))
    return "\n".join(lines)


def find_calls(code: str, names: set[str], self_name: str) -> set[str]:
    """Names from `names` referenced in code as calls (paren or bare-script)."""
    found = set()
    for name in names:
        if name == self_name:
            continue
        if re.search(rf"\b{re.escape(name)}\s*\(", code):
            found.add(name)
        elif re.search(rf"(?m)^\s*{re.escape(name)}\s*;?\s*$", code):
            found.add(name)
    return found


def main(argv: list[str] | None = None) -> None:
    global REPO, KVL, KCL, DOCS, GRIDFORGE_GRAPH
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--matlab-parent", required=True, type=Path,
                    help="folder holding the two MATLAB code folders")
    ap.add_argument("--gridforge-graph", type=Path, default=GRIDFORGE_GRAPH,
                    help="GridForge's docs/call_graph.json (marks GridForge-owned functions)")
    ap.add_argument("--out", type=Path, default=DOCS,
                    help="output folder (default: this repository's docs/)")
    args = ap.parse_args(argv)
    REPO = args.matlab_parent
    KVL = REPO / "KCL-KVL Compliant Grid Generator"
    KCL = REPO / "KCL-Compliant Grid Generator"
    DOCS, GRIDFORGE_GRAPH = args.out, args.gridforge_graph
    if not KCL.is_dir() or not KVL.is_dir():
        sys.exit(f"{REPO} must hold both MATLAB folders; nothing written")
    # ---- collect the function universe -------------------------------------
    # KVL copy is authoritative for GridCrucible; record where each stem lives.
    kvl_paths: dict[str, Path] = {}
    for folder in KVL_CODE_DIRS:
        for p in matlab_files(folder):
            kvl_paths.setdefault(p.stem, p)
    kcl_paths = {p.stem: p for p in matlab_files(KCL)}

    universe: dict[str, dict] = {}
    for name, p in kvl_paths.items():
        universe[name] = {"kvl": str(p.relative_to(REPO)), "kcl": None}
    for name, p in kcl_paths.items():
        universe.setdefault(name, {"kvl": None, "kcl": None})
        universe[name]["kcl"] = str(p.relative_to(REPO))

    # ---- gridforge ownership ------------------------------------------------
    # Anything the parallel GridForge effort already tracks as active is
    # converted on the gridforge branch; GridCrucible only *calls* it.
    gridforge_owned: set[str] = set()
    if GRIDFORGE_GRAPH.exists():
        gf = json.loads(GRIDFORGE_GRAPH.read_text())
        gridforge_owned = set(gf.get("active", {}))

    # Synthetic family nodes for feval-dispatch (one node per family).
    family_members = {
        fam: sorted(n for n in universe if n.startswith(fam) and n != fam)
        for fam in DYNAMIC_FAMILIES
    }

    names = set(universe) | set(EXTERNALS)

    # ---- parse every file, record edges ------------------------------------
    edges: dict[str, list[str]] = {}
    ext_use: dict[str, list[str]] = {}
    loc: dict[str, int] = {}
    for name, info in universe.items():
        path = REPO / (info["kvl"] or info["kcl"])  # KVL authoritative
        try:
            src = path.read_text(errors="replace")
        except OSError:
            continue
        loc[name] = src.count("\n") + 1
        code = strip_comments(src)
        called = find_calls(code, names, name)
        # feval dynamic dispatch -> family pseudo-node
        for fam, pat in DYNAMIC_FAMILIES.items():
            if pat.search(code):
                called.add(f"{fam}<Region>")
        # loadcase('X') string call -> case script X
        for case in LOADCASE_RE.findall(code):
            if case in universe:
                called.add(case)
        edges[name] = sorted(c for c in called
                             if c in universe or c.endswith("<Region>"))
        ext_use[name] = sorted(c for c in called if c in EXTERNALS)

    # Family pseudo-nodes call nothing; record members' loc as aggregate.
    for fam, members in family_members.items():
        node = f"{fam}<Region>"
        edges[node] = []
        ext_use[node] = []
        loc[node] = sum(loc.get(m, 0) for m in members)

    # ---- BFS from roots -> active set --------------------------------------
    active: dict[str, int] = {}
    queue = [(r, 0) for r in ROOTS]
    while queue:
        cur, d = queue.pop(0)
        if cur in active and active[cur] <= d:
            continue
        active[cur] = d
        for callee in edges.get(cur, []):
            if callee not in active or active[callee] > d + 1:
                queue.append((callee, d + 1))

    # Orphans: KVL-folder .m files unreachable from both mains AND not owned
    # by GridForge. (Catalog only — the author deletes them himself.)
    orphans = sorted(
        n for n, info in universe.items()
        if n not in active and info["kvl"] is not None
        and n not in gridforge_owned
        and not any(n.startswith(f) for f in DYNAMIC_FAMILIES)
    )

    def tags_for(node: str) -> list[str]:
        info = universe.get(node, {})
        tags = []
        if node in gridforge_owned:
            tags.append("GRIDFORGE-OWNED")
        if info.get("kcl") and info.get("kvl"):
            tags.append("BOTH-FOLDERS")
        elif info.get("kcl") and not info.get("kvl"):
            tags.append("KCL-ONLY")
        if node.endswith("<Region>"):
            fam = node.removesuffix("<Region>")
            tags.append(f"family:{len(family_members[fam])} regional files")
        if ext_use.get(node):
            tags.append("ext:" + ",".join(ext_use[node]))
        return tags

    # ---- text tree ----------------------------------------------------------
    def tree_lines(node: str, depth: int, seen: frozenset) -> list[str]:
        pad = "    " * depth
        tag = tags_for(node)
        tag_s = ("  [" + "; ".join(tag) + "]") if tag else ""
        if node in seen:
            return [f"{pad}{node} (^ repeat){tag_s}"]
        lines = [f"{pad}{node} ({loc.get(node, 0)} loc){tag_s}"]
        for callee in edges.get(node, []):
            lines += tree_lines(callee, depth + 1, seen | {node})
        return lines

    txt = [
        "# GridCrucible MATLAB call tree  (generated " + date.today().isoformat() + ")",
        "# Roots: main_SRTV.m (driver) + MultiLevelProgram.m (OHM/TUNE/TRIM core).",
        "# Format: indentation = call depth; (N loc) = lines of code;",
        "#   [GRIDFORGE-OWNED] = converted by the parallel gridforge effort - do NOT convert here;",
        "#   [BOTH-FOLDERS] = exists in KCL and KCL-KVL folders (KVL copy authoritative);",
        "#   [family:N] = feval-dispatched regional family collapsed to one node;",
        "#   (^ repeat) = subtree already expanded above; ext: = external deps.",
        "",
    ]
    for root in ROOTS:
        txt += tree_lines(root, 0, frozenset()) + [""]

    txt += ["# --- FLAT ADJACENCY (caller -> callees), BFS order ---"]
    for name in sorted(active, key=lambda n: (active[n], n)):
        txt.append(f"{name} [depth {active[name]}] -> "
                   + (", ".join(edges.get(name, [])) or "(leaf)"))
    txt += ["", "# --- ORPHANED (in KVL folder, unreachable from mains, not gridforge-owned) ---"]
    txt += [f"{n} ({loc.get(n, 0)} loc)  {universe[n]['kvl']}" for n in orphans]

    DOCS.mkdir(parents=True, exist_ok=True)
    (DOCS / "call_tree.txt").write_text("\n".join(txt), encoding="utf-8")

    # ---- JSON dump ----------------------------------------------------------
    dump = {
        "generated": date.today().isoformat(),
        "roots": ROOTS,
        "active": {n: {"depth": d, "loc": loc.get(n, 0),
                       "calls": edges.get(n, []),
                       "externals": ext_use.get(n, []),
                       "gridforge_owned": n in gridforge_owned,
                       "kvl_path": universe.get(n, {}).get("kvl"),
                       "kcl_path": universe.get(n, {}).get("kcl")}
                   for n, d in sorted(active.items())},
        "families": family_members,
        "orphans": {n: {"loc": loc.get(n, 0), "kvl_path": universe[n]["kvl"]}
                    for n in orphans},
    }
    (DOCS / "call_graph.json").write_text(json.dumps(dump, indent=1), encoding="utf-8")

    # ---- Mermaid (clustered) -------------------------------------------------
    def mm_id(name: str) -> str:
        """Mermaid-safe node id (no <> chars)."""
        return name.replace("<Region>", "_Region")

    mm = [
        "# GridCrucible MATLAB Call Tree",
        "",
        f"Generated {date.today().isoformat()} by `tools/analyze_matlab_calls.py`.",
        "Subgraphs = functional clusters (mirror the planned Python modules).",
        "Green = GridForge-owned (converted by the parallel effort, GridCrucible",
        "only calls them); orange = the two conversion roots; `<Region>` nodes",
        "collapse feval-dispatched regional file families.",
        "",
        "```mermaid",
        "flowchart TD",
    ]
    clustered = {n for members in CLUSTERS.values() for n in members}
    for cluster, members in CLUSTERS.items():
        live = [n for n in members if n in active]
        if not live:
            continue
        mm.append(f'    subgraph "{cluster}"')
        for n in live:
            label = n if not n.endswith("<Region>") else n.replace("<", "&lt;").replace(">", "&gt;")
            mm.append(f'        {mm_id(n)}["{label}"]')
        mm.append("    end")
    strays = [n for n in active if n not in clustered]
    for n in strays:
        mm.append(f'    {mm_id(n)}["{n} (uncategorized)"]')
    shown = set()
    for caller in sorted(active, key=lambda n: active[n]):
        for callee in edges.get(caller, []):
            edge = f"    {mm_id(caller)} --> {mm_id(callee)}"
            if edge not in shown:
                shown.add(edge)
                mm.append(edge)
    for n in sorted(active):
        if n in gridforge_owned:
            mm.append(f"    class {mm_id(n)} gridforge;")
    for r in ROOTS:
        mm.append(f"    class {mm_id(r)} root;")
    mm += [
        "    classDef gridforge fill:#c8e6c9,stroke:#2e7d32;",
        "    classDef root fill:#ffe0b2,stroke:#e65100,stroke-width:3px;",
        "```",
        "",
    ]
    (DOCS / "CALL_TREE.md").write_text("\n".join(mm), encoding="utf-8")
    if strays:
        print(f"WARNING: uncategorized active nodes (add to CLUSTERS): {strays}")

    n_owned = sum(1 for n in active if n in gridforge_owned)
    print(f"Active: {len(active)}  (gridforge-owned: {n_owned}, "
          f"GridCrucible: {len(active) - n_owned})   Orphans (KVL): {len(orphans)}")
    print("Wrote docs/call_tree.txt, docs/call_graph.json, docs/CALL_TREE.md")


if __name__ == "__main__":
    main()
