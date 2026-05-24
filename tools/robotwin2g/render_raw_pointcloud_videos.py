#!/usr/bin/env python3
"""Render raw RoboTwin observed point-cloud videos for WDS test samples.

These videos are not PointWorld model predictions. They show the original
per-frame observed point cloud from the source HDF5 referenced by each WDS
sample. Since RoboTwin point clouds are per-frame observations, by-index lines
are diagnostic only unless the source export guarantees stable point identity.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import tarfile
from pathlib import Path
from typing import Any, Dict, Iterator, List

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import h5py
import numpy as np

from tools.robotwin2g.render_flow_videos import _render_video


def _iter_metadata(wds_root: Path, split: str) -> Iterator[Dict[str, Any]]:
    for shard in sorted((wds_root / split).glob("*.tar")):
        with tarfile.open(shard, "r") as tar:
            for member in tar:
                name = Path(member.name).name
                if not name.endswith(".metadata.json"):
                    continue
                fp = tar.extractfile(member)
                if fp is None:
                    continue
                yield json.load(io.TextIOWrapper(fp, encoding="utf-8"))


def _read_raw_pointcloud(meta: Dict[str, Any], key: str) -> np.ndarray:
    source = Path(meta["source_path"])
    start = int(meta["source_start"])
    end = int(meta["source_end"])
    with h5py.File(source, "r") as f:
        if key in f:
            arr = np.asarray(f[key], dtype=np.float32)
        elif key.startswith("/") and key[1:] in f:
            arr = np.asarray(f[key[1:]], dtype=np.float32)
        else:
            raise KeyError(f"{source} does not contain pointcloud key {key}")
    return arr[start:end]


def _subsample_time_consistent(pcd: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    if pcd.shape[1] <= max_points:
        return pcd
    rng = np.random.default_rng(seed)
    idx = rng.choice(np.arange(pcd.shape[1]), size=max_points, replace=False)
    idx.sort()
    return pcd[:, idx]


def main() -> None:
    args = build_argparser().parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "wds_root": str(args.wds_root),
        "split": args.split,
        "note": "Raw observed RoboTwin pointcloud sequence. By-index motion is diagnostic, not guaranteed ground-truth scene flow.",
        "samples": [],
    }

    for i, meta in enumerate(_iter_metadata(Path(args.wds_root), args.split)):
        if i >= args.num_samples:
            break
        key = str(meta.get("sample_key") or f"sample_{i:03d}")
        pcd = _read_raw_pointcloud(meta, args.pointcloud_key)
        pcd = _subsample_time_consistent(pcd, args.max_points, args.seed + i)
        xyz = pcd[:, :, :3].astype(np.float32)
        scene_exists = np.ones(xyz.shape[:2], dtype=bool)
        robot = np.empty((xyz.shape[0], 0, 3), dtype=np.float32)
        robot_exists = np.zeros((xyz.shape[0], 0), dtype=bool)
        out_path = out_dir / f"{i:03d}_{key}.mp4"
        print(f"[render raw] {i + 1}/{args.num_samples}: {key} source={meta.get('source_path')} frames={meta.get('source_start')}:{meta.get('source_end')}", flush=True)
        item = _render_video(
            out_path,
            key=f"{key} raw observed",
            pred=xyz,
            target=np.repeat(xyz[:1], xyz.shape[0], axis=0),
            scene_exists=scene_exists,
            robot=robot,
            robot_exists=robot_exists,
            max_points=args.max_points,
            max_robot_points=0,
            fps=args.fps,
            hold_final=args.hold_final,
            seed=args.seed + i,
            elev=args.elev,
            azim=args.azim,
            left_title="Raw frame 0",
            left_subtitle="static first-frame reference",
            middle_title="Raw observed PC",
            middle_subtitle="per-frame source HDF5 point cloud",
            right_title="Raw by-index motion",
            right_subtitle="diagnostic only, not guaranteed GT flow",
            top_note="Color: by-index displacement from frame 0. This is observed point-cloud motion, not model output.",
        )
        item.update(
            {
                "source_path": str(meta.get("source_path")),
                "source_task": meta.get("source_task"),
                "source_episode": meta.get("source_episode"),
                "source_start": meta.get("source_start"),
                "source_end": meta.get("source_end"),
            }
        )
        summary["samples"].append(item)

    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as fp:
        json.dump(summary, fp, indent=2)
    print(f"[done] wrote {len(summary['samples'])} raw videos to {out_dir}", flush=True)
    print(f"[done] summary: {summary_path}", flush=True)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--wds-root", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--split", default="test", choices=["train", "test"])
    p.add_argument("--num-samples", type=int, default=6)
    p.add_argument("--max-points", type=int, default=1024)
    p.add_argument("--fps", type=float, default=2.0)
    p.add_argument("--hold-final", type=int, default=4)
    p.add_argument("--elev", type=float, default=25.0)
    p.add_argument("--azim", type=float, default=-55.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pointcloud-key", default="/pointcloud")
    return p


if __name__ == "__main__":
    main()
