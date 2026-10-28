"""Build GridCrucible's regional load-profile cache from LoadSmith.

GridCrucible reads ``data/load_profiles/<REGION>.npz`` (arrays ``p_load`` and
``q_load``: national bus number in column 0, then 8,760 hourly values per load
bus), the shape the MATLAB ``MakeLoadProfiles_<REGION>`` functions returned.
LoadSmith is the validated Python port of those builders, so this script runs
it with the MATLAB function's settings (no peak scaling; GridCrucible applies
its own load scaling) and writes the cache. The files are ~300 MB per region,
too large to commit.

Usage (from the repository root, with LoadSmith installed, e.g.
``pip install -e ../LoadSmith-Public``):

    python tools/make_load_profiles_from_loadsmith.py --loadsmith ../LoadSmith-Public
    python tools/make_load_profiles_from_loadsmith.py --loadsmith ../LoadSmith-Public --region SRTV
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--loadsmith", type=Path, required=True,
                    help="LoadSmith clone (its data/ store is read)")
    ap.add_argument("--region", default="SRTV")
    ap.add_argument("--data-dir", type=Path,
                    default=Path(__file__).resolve().parents[1] / "data")
    args = ap.parse_args(argv)

    from loadsmith import LoadSmith

    result = LoadSmith(store=str(args.loadsmith / "data")).run(
        args.region.upper(), formats=(), peak_scaling="none")
    p_load, q_load = result.as_matrices()
    out = args.data_dir / "load_profiles" / f"{args.region.upper()}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, p_load=p_load, q_load=q_load)
    print(f"wrote {out}: {p_load.shape[0]} load buses x {p_load.shape[1] - 1} h")


if __name__ == "__main__":
    main()
