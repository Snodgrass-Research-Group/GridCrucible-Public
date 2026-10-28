"""CrucibleConfig: TOML-backed run configuration.

Replaces THREE MATLAB mechanisms (ARCHITECTURE.md §2.5, algorithms/GridCrucible.md):
* the ``codectrl`` flag block hand-edited at the top of main_SRTV.m
  (ReRun/Resume/Add/Analyze/Load + ~25 feature flags),
* the ``options`` struct defaults from SRTV/User Input Data/InitOptions.m,
* the mid-run "Section to change variables" overrides.

Defaults below ARE the SRTV 4600-bus case (InitOptions.m). A run is fully
described by one TOML file; ``stages`` selects what executes; resume comes
from the checkpoint registry, not flags.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

#: Repo root, anchored on this file (src/gridcrucible/config.py) so paths
#: work regardless of the current working directory.
REPO_ROOT = Path(__file__).resolve().parents[2]

#: The KCL-KVL MATLAB tree is not part of this repository; the one-time
#: migration finds it through GRIDCRUCIBLE_MATLAB_ROOT, else next to the clone.
MATLAB_ROOT = Path(os.environ.get(
    "GRIDCRUCIBLE_MATLAB_ROOT",
    REPO_ROOT.parent / "KCL-KVL Compliant Grid Generator"))

#: Pipeline stage names, in execution order (algorithms/GridCrucible.md table).
STAGES = (
    "load_inputs", "setup_substations", "setup_generators", "build_scenarios",
    "build_candidates", "trim_iterations", "create_buses",
    "shunt_compensation", "ac_refinement", "final_acpf", "metrics",
    "write_outputs",
)


@dataclass
class CaseConfig:
    """[case] — what system to build (InitOptions 'unique to a case')."""

    regions: list[str] = field(default_factory=lambda: ["SRTV"])
    garver_regions: list[str] = field(default_factory=lambda: ["SRTV"])
    import_buses_ext: list[int] = field(default_factory=list)
    voltage_levels: list[int] = field(default_factory=lambda: [69, 138, 230, 500])
    max_length_miles: list[float] = field(default_factory=lambda: [25, 65, 90, 140])
    # Form 715 / DOE total-mileage targets per kV (InitOptions TargetLengths)
    target_lengths_miles: dict[int, float] = field(default_factory=lambda: {
        69: 1290, 138: 12720, 230: 4013, 345: 640, 500: 806, 765: 142, 999: 15724})
    target_gva_miles: dict[int, float] = field(default_factory=lambda: {
        69: 82, 138: 2878, 230: 1900, 345: 436, 500: 904, 765: 234, 999: 6436})
    length_modifier: float = 1.0
    routing_factor: float = 1.0
    s_base: float = 100.0
    rng_seed: int = 715  # seeds GMD index, TRIM draws, dispatch noise


@dataclass
class OHMConfig:
    """[ohm] — inner loop (algorithms/DCInnerLoop.md)."""

    formulation: str = "dc"            # 'dc' | 'ac'  (equal priority, §2.7)
    solver: str = "gurobi"             # 'gurobi' | 'newton' (MATLAB default IS gurobi)
    use_qp_with_newton: bool = False   # MATLAB options.UseOpti
    gamma_grid: tuple[float, float, int] = (-5.0, -4.0, 5)  # logspace args (PC preset)
    gamma_grid_large: tuple[float, float, int] = (-7.0, -5.0, 6)  # Linux/large preset
    inner_loop_terminate: float = 0.3  # zero-flow fraction to stop the γ sweep
    qp_zero_cutoff: float = 1e-5       # options.OptiCutoff
    newton_zero_cutoff: float = 0.0    # options.NewtonCutoff (0 = fixed)
    newton_recalc_cutoff: bool = False # options.RecalcCutoff (25% DC-flow quantile)
    theta_max_deg: float = 45.0
    base_snet_scaling: float = 1.0     # options.BaseSNetScaling


@dataclass
class TRIMConfig:
    """[trim] — outer loop (algorithms/OuterLoop.md)."""

    percent_remove: float = 0.05
    stochastic_remove: bool = True
    remove_sequential: bool = False    # latched on by connectivity failures
    normalize_rate_a: bool = False
    normalize_snet: bool = True
    normalize_node_degree: bool = True
    normalize_node_degree_max: bool = True   # max vs mean of endpoint degrees
    normalize_length: bool = True
    min_branch_bus: float = 1.6        # options.MinBranchBus (round-1 stop)
    legacy_nd3_bug: bool = False       # reproduce OuterLoop.m:89 '==2' retest
    # BBR density thresholds → scenario tier (MultiLevelProgram)
    bbr_high: float = 20.0
    bbr_mid: float = 12.0
    bbr_midlow: float = 6.0
    bbr_low: float = 3.0
    # Safety cap on TRIM iterations (MATLAB's while-loop was unbounded;
    # ~5%/iter removal from a complete graph needs a few hundred at most)
    max_iterations: int = 500


@dataclass
class TUNEConfig:
    """[tune] — middle loop (algorithms/MiddleLoop.md, conductors.md)."""

    percent_resize: float = 0.05
    percent_change: float = 1.3
    random_resize: bool = True         # resize largest overloads vs random pick
    change_voltages: bool = True
    conductor_margin: float = 1.0
    downsize_threshold: float = 0.5
    l1_existing_lines: bool = True     # reduce L1 weight on existing-line paths
    gen_to_gen_penalty: bool = True
    l1_weight: float = 1.0


@dataclass
class DispatchConfig:
    """[dispatch] — scenario generation (algorithms/CreateSNet.md, UCCD.md)."""

    mode: str = "Full"                 # 'Full' | 'Regional'
    load_data: str = "EPIGRIDS"        # 'EPIGRIDS' | 'scenarios' | 'sine'
    buses: str = "EPIGRIDS"            # 'EPIGRIDS' | 'given'
    load_scenarios: int = 24
    uc_scenarios: int = 2              # 2 = ED + complementary dispatch (CD)
    use_ed: bool = True
    use_cd: bool = True
    reserve: float = 0.1
    gen_n_minus_1: bool = False
    gen_contingency_min_mw: float = 2500.0


@dataclass
class ParallelConfig:
    """[parallel] — replaces the BBR-tier parpool resizing (ARCHITECTURE.md §5)."""

    workers: int = 2                   # options.parallelPC default; -1 = all cores
    ac_parallel: bool = False
    parallel_scenarios: bool = True


@dataclass
class IOConfig:
    """[io] — data locations, checkpoints, writers, plots."""

    data_dir: Path = REPO_ROOT / "data"   # migrated parquet/npz/pp-JSON
    matlab_root: Path = MATLAB_ROOT
    checkpoint_dir: Path = REPO_ROOT / "runs"
    output_dir: Path = REPO_ROOT / "output"
    write_final_results: bool = True
    write_generators: bool = True
    write_buses: bool = True
    write_metrics: bool = True
    plot_synthetic_metrics: bool = True
    plot_real_metrics: bool = False
    plot_inner_loop: bool = False
    plot_one_lines: bool = False
    reset_gmd_index: bool = False      # codectrl.ResetGMDIndex


@dataclass
class CrucibleConfig:
    """Top-level config: one TOML file == one reproducible run."""

    stages: list[str] = field(default_factory=lambda: list(STAGES))
    case: CaseConfig = field(default_factory=CaseConfig)
    ohm: OHMConfig = field(default_factory=OHMConfig)
    trim: TRIMConfig = field(default_factory=TRIMConfig)
    tune: TUNEConfig = field(default_factory=TUNEConfig)
    dispatch: DispatchConfig = field(default_factory=DispatchConfig)
    parallel: ParallelConfig = field(default_factory=ParallelConfig)
    io: IOConfig = field(default_factory=IOConfig)
    verbose: bool = False              # OutputFlag for Gurobi, pp verbosity

    @classmethod
    def from_toml(cls, path: str | Path) -> "CrucibleConfig":
        """Load a TOML run file over the defaults. Unknown sections/keys are
        ERRORS (typo protection); Path-typed fields are coerced."""
        import tomllib

        data = tomllib.loads(Path(path).read_text(encoding="utf-8"))
        cfg = cls()
        for section, values in data.items():
            if section == "stages":
                cfg.stages = list(values)
                continue
            if section == "verbose":
                cfg.verbose = bool(values)
                continue
            if not hasattr(cfg, section) or not isinstance(values, dict):
                raise ValueError(f"unknown config section [{section}]")
            target = getattr(cfg, section)
            for key, val in values.items():
                if not hasattr(target, key):
                    raise ValueError(f"unknown key {key!r} in [{section}]")
                if isinstance(getattr(target, key), Path):
                    val = Path(val)
                setattr(target, key, val)
        problems = cfg.validate()
        if problems:
            raise ValueError("invalid config: " + "; ".join(problems))
        return cfg

    def validate(self) -> list[str]:
        """Cross-field checks; returns a list of problems (empty = ok)."""
        v: list[str] = []
        if len(self.case.max_length_miles) != len(self.case.voltage_levels):
            v.append("case: each voltage level needs a max_length_miles entry"
                     " (InitOptions guard)")
        unknown = [s for s in self.stages if s not in STAGES]
        if unknown:
            v.append(f"stages: unknown {unknown}")
        order = [STAGES.index(s) for s in self.stages if s in STAGES]
        if order != sorted(order):
            v.append("stages: must follow canonical order "
                     f"{list(STAGES)}")
        if self.ohm.formulation not in ("dc", "ac"):
            v.append("ohm.formulation must be 'dc' or 'ac'")
        if not 0 < self.trim.percent_remove < 1:
            v.append("trim.percent_remove must be in (0, 1)")
        return v
