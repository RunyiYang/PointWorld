#!/usr/bin/env python3
"""Validate RoboTwin-style HDF5 episodes and enumerate PointWorld-style clip windows.

This mirrors the PointWorld data branch structure:

  robotwin_integrity_check.py -> make_robotwin_wds_manifest.py -> convert_robotwin_to_wds.py

It does not modify the source data. The output JSON contains a deterministic list
of valid clips as [h5_path, clip_key], where clip_key is START:END and END is
exclusive.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np

# Allow running from a PointWorld repository root without installing this patch.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.robotwin2g.convert_robotwin_to_wds import (  # noqa: E402
    MANIFEST_SCHEMA_VERSION,
    ROBOTWIN_DOMAIN,
    canonical_episode_uuid,
    h5_has,
    h5_len,
    valid_clip_starts,
)


@dataclass
class CheckResult:
    path: str
    rel_path: str
    task: str
    episode_uuid: str
    ok: bool
    length: int = 0
    num_clips: int = 0
    valid_clips: List[List[str]] | None = None
    warnings: List[str] | None = None
    error: Optional[str] = None


def _discover(input_dir: Path) -> List[Path]:
    if input_dir.is_file():
        return [input_dir]
    paths = sorted(input_dir.rglob("*.hdf5")) + sorted(input_dir.rglob("*.h5"))
    return [p for p in paths if p.is_file()]


def _clip_key(start: int, end: int) -> str:
    return f"{start:06d}:{end:06d}"


def _shape_or_none(f: h5py.File, key: str) -> Optional[Tuple[int, ...]]:
    key = key.strip("/")
    if key not in f:
        return None
    return tuple(int(x) for x in f[key].shape)


def _check_one(payload: Tuple[str, Dict[str, Any]]) -> CheckResult:
    path_str, cfg = payload
    path = Path(path_str)
    input_dir = Path(cfg["input_dir"])
    try:
        rel = path.resolve().relative_to(input_dir.resolve()).as_posix()
    except ValueError:
        rel = path.name
    task = rel.split("/", 1)[0] if "/" in rel else path.parent.name or "task"
    episode_uuid = canonical_episode_uuid(path, input_dir)
    warnings: List[str] = []

    try:
        with h5py.File(path, "r") as f:
            if not h5_has(f, cfg["action_key"]):
                raise KeyError(f"missing action key {cfg['action_key']}")
            action_shape = _shape_or_none(f, cfg["action_key"])
            assert action_shape is not None
            if len(action_shape) != 2:
                raise ValueError(f"action key {cfg['action_key']} must be rank-2, got shape={action_shape}")
            if cfg["action_dim"] > 0 and action_shape[1] != cfg["action_dim"]:
                raise ValueError(
                    f"action dim mismatch: expected {cfg['action_dim']}, got {action_shape[1]} at {cfg['action_key']}"
                )

            pc_shape = _shape_or_none(f, cfg["pointcloud_key"])
            if pc_shape is None:
                for cam in cfg["camera_names"]:
                    pc_shape = _shape_or_none(f, f"/observation/{cam}/pcd")
                    if pc_shape is not None:
                        warnings.append(f"using /observation/{cam}/pcd fallback")
                        break
            if pc_shape is None:
                raise KeyError(f"missing pointcloud key {cfg['pointcloud_key']} and no camera pcd fallback")
            if len(pc_shape) != 3 or pc_shape[-1] < 3:
                raise ValueError(f"pointcloud must be (T,N,C>=3), got shape={pc_shape}")

            # RGB and depth are not required; converter can synthesize missing depth and black missing RGB.
            for cam in cfg["camera_names"]:
                if not h5_has(f, f"/observation/{cam}/rgb"):
                    warnings.append(f"missing /observation/{cam}/rgb; converter will write black image")
                if not h5_has(f, f"/observation/{cam}/depth"):
                    warnings.append(f"missing /observation/{cam}/depth; converter will synthesize depth")

            length = h5_len(f, [cfg["action_key"], cfg["pointcloud_key"], "/pointcloud"])
            length = int(min(length, action_shape[0], pc_shape[0]))
            starts = valid_clip_starts(length, action_shape[0], cfg["clip_horizon"], cfg["clip_stride"])
            if cfg["max_clips_per_episode"] > 0:
                starts = starts[: cfg["max_clips_per_episode"]]
            valid = [[str(path.resolve()), _clip_key(s, s + cfg["clip_horizon"])] for s in starts]
            if not valid:
                raise ValueError(
                    f"no valid clips for length={length}, horizon={cfg['clip_horizon']}, stride={cfg['clip_stride']}"
                )
            return CheckResult(
                path=str(path.resolve()),
                rel_path=rel,
                task=task,
                episode_uuid=episode_uuid,
                ok=True,
                length=length,
                num_clips=len(valid),
                valid_clips=valid,
                warnings=warnings,
            )
    except Exception as exc:  # noqa: BLE001
        return CheckResult(
            path=str(path.resolve()),
            rel_path=rel,
            task=task,
            episode_uuid=episode_uuid,
            ok=False,
            warnings=warnings,
            error=repr(exc),
        )


def run(args: argparse.Namespace) -> Dict[str, Any]:
    input_dir = Path(args.input_dir)
    paths = _discover(input_dir)
    if not paths:
        raise RuntimeError(f"No .h5/.hdf5 files found under {input_dir}")

    cfg = {
        "input_dir": str(input_dir),
        "action_key": args.action_key,
        "action_dim": args.action_dim,
        "pointcloud_key": args.pointcloud_key,
        "camera_names": args.camera_names,
        "clip_horizon": args.clip_horizon,
        "clip_stride": args.clip_stride,
        "max_clips_per_episode": args.max_clips_per_episode,
    }

    work = [(str(p), cfg) for p in paths]
    if args.num_mp_workers and args.num_mp_workers > 1:
        with mp.Pool(processes=args.num_mp_workers) as pool:
            results = list(pool.imap_unordered(_check_one, work, chunksize=max(1, len(work) // (args.num_mp_workers * 8))))
    else:
        results = [_check_one(x) for x in work]
    results.sort(key=lambda r: r.rel_path)

    valid_clips: List[List[str]] = []
    episodes: List[Dict[str, Any]] = []
    bad: List[Dict[str, Any]] = []
    task_stats: Dict[str, Dict[str, int]] = {}

    for r in results:
        task_stats.setdefault(r.task, {"episodes": 0, "ok_episodes": 0, "bad_episodes": 0, "clips": 0})
        task_stats[r.task]["episodes"] += 1
        if r.ok:
            task_stats[r.task]["ok_episodes"] += 1
            task_stats[r.task]["clips"] += int(r.num_clips)
            valid_clips.extend(r.valid_clips or [])
            episodes.append(asdict(r))
        else:
            task_stats[r.task]["bad_episodes"] += 1
            bad.append(asdict(r))

    out = {
        "schema_version": "robotwin_integrity.v1",
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "domain": ROBOTWIN_DOMAIN,
        "created_at_unix": int(time.time()),
        "input_dir": str(input_dir.resolve()),
        "config": cfg,
        "stats": {
            "num_files": len(paths),
            "num_ok_episodes": len(episodes),
            "num_bad_episodes": len(bad),
            "num_valid_clips": len(valid_clips),
            "tasks": task_stats,
        },
        "valid_clips": valid_clips,
        "episodes": episodes,
        "bad_episodes": bad,
    }
    return out


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", required=True, help="RoboTwin data root or one HDF5 file")
    p.add_argument("--output", default=None, help="Output JSON; default: <input-dir>/integrity_check.json")
    p.add_argument("--num-mp-workers", type=int, default=0, help="Local multiprocessing workers; 0/1 = serial")
    p.add_argument("--clip-horizon", type=int, default=11)
    p.add_argument("--clip-stride", type=int, default=5)
    p.add_argument("--max-clips-per-episode", type=int, default=-1, help="Debug cap; <=0 disables")
    p.add_argument("--action-key", default="/joint_action/vector")
    p.add_argument("--action-dim", type=int, default=14, help="Expected action dim; <=0 disables check")
    p.add_argument("--pointcloud-key", default="/pointcloud")
    p.add_argument("--camera-names", nargs="+", default=["cam0", "cam1"])
    return p


def main() -> None:
    args = build_argparser().parse_args()
    if args.clip_horizon < 2:
        raise ValueError("--clip-horizon must be >= 2")
    if args.clip_stride < 1:
        raise ValueError("--clip-stride must be >= 1")

    out = run(args)
    output = Path(args.output) if args.output else Path(args.input_dir) / "integrity_check.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as fp:
        json.dump(out, fp, indent=2, sort_keys=True)
    print(
        f"Wrote {output} | ok_episodes={out['stats']['num_ok_episodes']} "
        f"bad_episodes={out['stats']['num_bad_episodes']} valid_clips={out['stats']['num_valid_clips']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
