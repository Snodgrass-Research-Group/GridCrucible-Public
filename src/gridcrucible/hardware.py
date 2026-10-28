"""Runtime hardware detection + machine-tuned Gurobi environments.

Requested by the author (2026-07-11): detect how much system memory and how
many CPU cores the machine has and set Gurobi options to maximize
performance without running out of memory. Stdlib-only (no psutil).

Tuning rationale for this pipeline's solves (continuous QPs via barrier):
* ``Threads`` — Gurobi defaults to all logical cores; when scenario-level
  process pools run N solver workers side by side (ARCHITECTURE.md §5), each env
  gets ``cores // n_workers`` so the machine isn't oversubscribed.
* ``MemLimit`` — a hard ceiling per env (Gurobi raises OUT_OF_MEMORY
  instead of driving the box into swap death). Sized as a fraction of
  TOTAL system memory divided across workers, floored at 1 GB. The
  measured full-scale OHM QP (5M vars) peaks well under 10 GB, so the
  default 80% budget is pure headroom, not a constraint.
"""

from __future__ import annotations

import os

#: Fraction of total system memory Gurobi environments may claim in total.
MEM_BUDGET_FRACTION = 0.8


def cpu_count() -> int:
    """Logical CPU cores (≥ 1)."""
    return os.cpu_count() or 1


def total_memory_gb() -> float:
    """Total physical system memory in GB (best effort; 0.0 if unknown)."""
    try:
        if os.name == "nt":
            import ctypes

            class _MemStatus(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong),
                            ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_uint64),
                            ("ullAvailPhys", ctypes.c_uint64),
                            ("ullTotalPageFile", ctypes.c_uint64),
                            ("ullAvailPageFile", ctypes.c_uint64),
                            ("ullTotalVirtual", ctypes.c_uint64),
                            ("ullAvailVirtual", ctypes.c_uint64),
                            ("ullAvailExtendedVirtual", ctypes.c_uint64)]

            stat = _MemStatus()
            stat.dwLength = ctypes.sizeof(stat)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                return stat.ullTotalPhys / 1024 ** 3
            return 0.0
        page = os.sysconf("SC_PAGE_SIZE")
        pages = os.sysconf("SC_PHYS_PAGES")
        return page * pages / 1024 ** 3
    except Exception:
        return 0.0


def gurobi_params(n_workers: int = 1) -> dict:
    """Machine-tuned Gurobi parameters for one of ``n_workers`` concurrent
    solver processes/environments."""
    n_workers = max(1, int(n_workers))
    params: dict = {"Threads": max(1, cpu_count() // n_workers)}
    total = total_memory_gb()
    if total > 0:
        params["MemLimit"] = max(1.0, MEM_BUDGET_FRACTION * total / n_workers)
    return params


def make_gurobi_env(n_workers: int = 1, verbose: bool = False):
    """A gurobipy Env with gurobi_params applied (None if gurobipy or its
    license is unavailable — callers fall back to Gurobi defaults)."""
    try:
        import gurobipy as gp

        params = {"OutputFlag": 1 if verbose else 0, **gurobi_params(n_workers)}
        return gp.Env(params=params)
    except Exception:
        return None
