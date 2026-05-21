#!/usr/bin/env python3
"""Convert validated RoboTwin two-gripper generated-H5 clips into WDS shards.

The input is the H5 format produced by convert_robotwin_to_pointworld_h5.py:
fixed-horizon clip groups named "start:end" with PointWorld-like scene/robot
fields plus RoboTwin action labels.  This script preserves action_state,
action_target, and action_mask, which the official PointWorld data converter does
not know about.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.robotwin2g.convert_robotwin_to_wds import TarShardWriter, json_bytes, npy_bytes, stable_int  # noqa: E402

DIRECT_NPY_KEYS = [
    "scene_flows",
    "scene_colors",
    "scene_normals",
    "scene_visibility",
    "scene_depth_valid_mask",
    "robot_flows",
    "robot_colors",
    "robot_normals",
    "left_gripper_pose",
    "right_gripper_pose",
    "left_gripper_open",
    "right_gripper_open",
    "action_state",
    "action_target",
    "action_mask",
]

CAMERA_DATA_KEYS = ["initial_depth", "intrinsic", "extrinsic"]


def _read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fp:
        return json.load(fp)


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(obj, fp, indent=2, sort_keys=True)


def _decode_vlen_bytes(value: Any) -> bytes:
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value)
    if isinstance(value, np.ndarray):
        return value.astype(np.uint8, copy=False).tobytes()
    if hasattr(value, "tobytes"):
        return value.tobytes()
    return bytes(value)


def _clip_payload(f: h5py.File, h5_path: Path, clip_key: str, split: str, sample_key: str) -> Tuple[Dict[str, bytes], np.ndarray]:
    if clip_key not in f:
        raise KeyError(f"Missing clip group {clip_key!r} in {h5_path}")
    group = f[clip_key]
    payload: Dict[str, bytes] = {}

    for key in DIRECT_NPY_KEYS:
        if key not in group:
            raise KeyError(f"{h5_path}::{clip_key} missing {key}")
        payload[f"{key}.npy"] = npy_bytes(np.asarray(group[key][()]))

    camera_keys = sorted(k for k in group.keys() if k.startswith("camera_") and isinstance(group[k], h5py.Group))
    for cam_idx, cam_key in enumerate(camera_keys):
        cam = group[cam_key]
        prefix = f"cam{cam_idx}"
        if "initial_rgb" not in cam:
            raise KeyError(f"{h5_path}::{clip_key}/{cam_key} missing initial_rgb")
        payload[f"{prefix}_initial_rgb.jpg"] = _decode_vlen_bytes(cam["initial_rgb"][0])
        for data_key in CAMERA_DATA_KEYS:
            if data_key not in cam:
                raise KeyError(f"{h5_path}::{clip_key}/{cam_key} missing {data_key}")
            arr = np.asarray(cam[data_key][()])
            if data_key == "initial_depth" and arr.dtype == np.uint16:
                arr = arr.astype(np.float32) / 1000.0
            payload[f"{prefix}_{data_key}.npy"] = npy_bytes(arr.astype(np.float32) if arr.dtype.kind in {"u", "i"} else arr)

    meta = {
        "sample_key": sample_key,
        "split": split,
        "h5_path": str(h5_path),
        "clip_key": clip_key,
        "source_path": str(group.attrs.get("source_path", "")),
        "source_task": str(group.attrs.get("source_task", "")),
        "source_episode": str(group.attrs.get("source_episode", "")),
        "source_start": int(group.attrs.get("source_start", -1)),
        "source_end": int(group.attrs.get("source_end", -1)),
        "scene_flow_mode": str(group.attrs.get("scene_flow_mode", "")),
        "policy_robot_input": bool(group.attrs.get("policy_robot_input", False)),
    }
    payload["metadata.json"] = json_bytes(meta)
    return payload, np.asarray(group["action_target"][()], dtype=np.float32)


def _load_manifest(manifest_path: Path) -> Dict[str, List[Dict[str, Any]]]:
    manifest = _read_json(manifest_path)
    rows = manifest.get("clips", [])
    by_split: Dict[str, List[Dict[str, Any]]] = {"train": [], "test": []}
    for row in rows:
        split = row.get("split", "train")
        if split not in by_split:
            by_split[split] = []
        by_split[split].append(row)
    return by_split


def _sample_key(split: str, row: Dict[str, Any], index: int) -> str:
    raw = f"{row.get('clip_id','')}|{row.get('h5_path','')}|{row.get('clip_key','')}|{index}"
    return f"{split}-{stable_int(raw):016x}"


def _compute_action_stats(sum_vec: np.ndarray | None, sumsq_vec: np.ndarray | None, count: int) -> Dict[str, Any]:
    if sum_vec is None or sumsq_vec is None or count <= 0:
        return {"count": 0, "mean": [], "std": []}
    mean = sum_vec / float(count)
    var = np.maximum(sumsq_vec / float(count) - mean * mean, 1e-8)
    std = np.sqrt(var)
    return {"count": int(count), "mean": mean.astype(float).tolist(), "std": std.astype(float).tolist()}


def convert(args: argparse.Namespace) -> None:
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    manifest_path = Path(args.manifest)
    if output_dir.exists() and args.overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    by_split = _load_manifest(manifest_path)
    writers = {
        split: TarShardWriter(output_dir, split, max_samples_per_shard=args.max_samples_per_shard)
        for split in ["train", "test"]
    }
    processed: Dict[str, int] = {"train": 0, "test": 0}
    errors: List[Dict[str, str]] = []
    source_rows: Dict[str, List[str]] = {"train": [], "test": []}

    train_sum: np.ndarray | None = None
    train_sumsq: np.ndarray | None = None
    train_count = 0

    try:
        for split in ["train", "test"]:
            rows = list(by_split.get(split, []))
            for i, row in enumerate(rows):
                h5_path = Path(row["h5_path"])
                if not h5_path.is_absolute():
                    h5_path = input_dir / h5_path
                clip_key = str(row["clip_key"])
                key = _sample_key(split, row, i)
                try:
                    with h5py.File(h5_path, "r") as f:
                        payload, action_target = _clip_payload(f, h5_path, clip_key, split, key)
                    writers[split].add(key, payload)
                    processed[split] += 1
                    source_rows[split].append(f"{h5_path}::{clip_key}")
                    if split == "train":
                        flat = action_target.reshape(-1, action_target.shape[-1]).astype(np.float64)
                        valid = np.isfinite(flat).all(axis=1)
                        flat = flat[valid]
                        if flat.size > 0:
                            if train_sum is None:
                                train_sum = np.zeros(flat.shape[-1], dtype=np.float64)
                                train_sumsq = np.zeros(flat.shape[-1], dtype=np.float64)
                            train_sum += flat.sum(axis=0)
                            assert train_sumsq is not None
                            train_sumsq += (flat * flat).sum(axis=0)
                            train_count += int(flat.shape[0])
                except Exception as exc:
                    errors.append({"split": split, "h5_path": str(h5_path), "clip_key": clip_key, "error": repr(exc)})
                    if not args.keep_going:
                        raise
                if (i + 1) % 100 == 0 or i == len(rows) - 1:
                    print(f"{split}: processed {i + 1}/{len(rows)} clips", flush=True)
    finally:
        for writer in writers.values():
            writer.close()

    action_stats = _compute_action_stats(train_sum, train_sumsq, train_count)
    _write_json(output_dir / "action_stats.json", action_stats)
    for split in ["train", "test"]:
        split_dir = output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / f"{split}_sources.txt", "w", encoding="utf-8") as fp:
            for row in source_rows[split]:
                fp.write(row + "\n")

    meta = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "manifest": str(manifest_path),
        "domain": "robotwin2g",
        "train": {"processed_count": processed["train"], "shards": [str(p) for p in writers["train"].shard_paths]},
        "test": {"processed_count": processed["test"], "shards": [str(p) for p in writers["test"].shard_paths]},
        "action_stats": action_stats,
        "errors": errors,
        "config": vars(args),
    }
    _write_json(output_dir / "metadata_rank0.json", meta)
    print(json.dumps({"train": processed["train"], "test": processed["test"], "errors": len(errors)}, indent=2), flush=True)
    print(f"wrote WDS root: {output_dir}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", required=True, help="Generated-H5 flows root")
    p.add_argument("--output-dir", required=True, help="Output WDS root")
    p.add_argument("--manifest", required=True, help="Manifest from make_robotwin_wds_manifest.py")
    p.add_argument("--max-samples-per-shard", type=int, default=512)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--keep-going", action="store_true")
    args = p.parse_args()
    convert(args)


if __name__ == "__main__":
    main()
