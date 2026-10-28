"""Checkpoint registry: stage + BBR-milestone snapshots with auto-resume.

MATLAB origin: SaveResults.m + the ad-hoc QNCAResults*/PFData*/BBR*.mat
save/load cascades — replaced wholesale (ARCHITECTURE.md §2.5).

Bundle layout (data-format policy: parquet for tables, npz for arrays,
JSON for scalars — no pickle, checkpoints stay readable across versions):

    <checkpoint_dir>/<run_id>/<label>/
        substations.parquet / buses.parquet / branches.parquet /
        candidates.parquet / generators.parquet     (only non-empty tables)
        scenarios.npz      (tier arrays + slack/gen/load lists + v_setpoints)
        meta.json          (JSON-able meta entries + schema version)
        meta_arrays.npz    (ndarray meta entries, e.g. node_degree)
    <checkpoint_dir>/<run_id>/index.json             (label -> timestamp log)

Meta entries that are neither JSON-able nor ndarrays (e.g. the nested
realism dict) are DROPPED with a warning — they are recomputable.
"""

from __future__ import annotations

import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import CrucibleConfig
from ..model import ACPFResult, CrucibleModel, ScenarioSet

_SCHEMA_VERSION = 1
_TABLES = ("substations", "buses", "branches", "candidates", "generators")


class CheckpointRegistry:
    """One per run. Labels: pipeline stage names plus TRIM milestones
    ('bbr_<tier>', 'nd2', 'trim_iter_<n%5>' rolling, 'trim_final')."""

    def __init__(self, config: CrucibleConfig, run_id: str | None = None) -> None:
        self.config = config
        if run_id is None:
            run_id = datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S") \
                + f"_seed{config.case.rng_seed}"
        self.run_id = run_id
        self.root = Path(config.io.checkpoint_dir) / run_id
        self.root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ save

    def save(self, model: CrucibleModel, label: str) -> Path:
        """Write a labeled checkpoint and register it in the run index.

        Non-empty model tables go to parquet; the scenario set and any ndarray
        ``meta`` entries go to npz; JSON-serializable ``meta`` goes to
        ``meta.json`` (non-serializable, recomputable entries are dropped with
        a warning). Returns the checkpoint directory.
        """
        out = self.root / label
        out.mkdir(parents=True, exist_ok=True)
        for name in _TABLES:
            df: pd.DataFrame = getattr(model, name)
            if len(df):
                df.to_parquet(out / f"{name}.parquet", index=False)
        s = model.scenarios
        np.savez_compressed(
            out / "scenarios.npz",
            slack_bus=s.slack_bus, gen_buses=s.gen_buses,
            load_buses=s.load_buses, v_setpoints=s.v_setpoints,
            **{f"tier_{k}": v for k, v in s.tiers.items()})
        meta_json: dict = {"_schema": _SCHEMA_VERSION}
        meta_arrays: dict = {}
        for k, v in model.meta.items():
            if isinstance(v, np.ndarray):
                meta_arrays[k] = v
            else:
                try:
                    json.dumps(v)
                    meta_json[k] = v
                except TypeError:
                    warnings.warn(f"checkpoint '{label}': dropping "
                                  f"non-serializable meta[{k!r}] "
                                  "(recomputable)", stacklevel=2)
        (out / "meta.json").write_text(json.dumps(meta_json, indent=1))
        if meta_arrays:
            np.savez_compressed(out / "meta_arrays.npz", **meta_arrays)
        self._index_append(label)
        return out

    def _index_append(self, label: str) -> None:
        idx_path = self.root / "index.json"
        idx = json.loads(idx_path.read_text()) if idx_path.exists() else []
        idx.append({"label": label,
                    "time": datetime.now(timezone.utc).isoformat()})
        idx_path.write_text(json.dumps(idx, indent=1))

    # ------------------------------------------------------------------ load

    def load(self, label: str) -> CrucibleModel:
        """Reconstruct a :class:`CrucibleModel` from a labeled checkpoint.

        Inverse of :meth:`save`. Raises ``FileNotFoundError`` if no checkpoint
        with ``label`` exists under this run.
        """
        src = self.root / label
        if not (src / "meta.json").exists():
            raise FileNotFoundError(f"no checkpoint '{label}' under {self.root}")
        model = CrucibleModel()
        for name in _TABLES:
            p = src / f"{name}.parquet"
            if p.exists():
                setattr(model, name, pd.read_parquet(p))
        z = np.load(src / "scenarios.npz")
        model.scenarios = ScenarioSet(
            tiers={k.removeprefix("tier_"): z[k]
                   for k in z.files if k.startswith("tier_")},
            slack_bus=int(z["slack_bus"]),
            gen_buses=z["gen_buses"], load_buses=z["load_buses"],
            v_setpoints=z["v_setpoints"])
        meta = json.loads((src / "meta.json").read_text())
        meta.pop("_schema", None)
        arr_path = src / "meta_arrays.npz"
        if arr_path.exists():
            arrs = np.load(arr_path)
            meta.update({k: arrs[k] for k in arrs.files})
        model.meta = meta
        return model

    def labels(self) -> list[str]:
        """Saved labels in save order (from the index)."""
        idx_path = self.root / "index.json"
        if not idx_path.exists():
            return []
        return [e["label"] for e in json.loads(idx_path.read_text())]

    def latest(self, among: list[str] | None = None
               ) -> tuple[str, CrucibleModel] | None:
        """Newest checkpoint, optionally restricted to given labels — the
        auto-resume entry point used by GridCrucible.run()."""
        labels = self.labels()
        if among is not None:
            labels = [lb for lb in labels if lb in among]
        if not labels:
            return None
        label = labels[-1]
        return label, self.load(label)
