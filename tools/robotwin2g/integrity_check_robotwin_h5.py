#!/usr/bin/env python3
"""Integrity check for RoboTwin two-gripper PointWorld-style generated H5 clips.

This mirrors the PointWorld data-branch workflow: generated H5 files are checked
first, then a manifest is built from the valid clip list, then WDS shards are
created.  It validates the direct arrays used by the custom action trainer and
the BEHAVIOR-like camera/mesh fields used to stay close to PointWorld's data
contract.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import h5py
import numpy as np

DIRECT_ARRAY_NDIMS = {
    "scene_flows": 3,
    "scene_colors": 3,
    "scene_normals": 3,
    "scene_visibility": 2,
    "scene_depth_valid_mask": 2,
    "robot_flows": 3,
    "robot_colors": 3,
    "robot_normals": 3,
    "left_gripper_pose": 2,
    "right_gripper_pose": 2,
    "left_gripper_open": 2,
    "right_gripper_open": 2,
    "action_state": 2,
    "action_target": 2,
    "action_mask": 1,
}

REQUIRED_CAMERA_DATASETS = ["initial_rgb", "initial_depth", "intrinsic", "extrinsic"]


def _clip_sort_key(key: str) -> Tuple[int, int, str]:
    try:
        a, b = key.split(":", 1)
        return int(a), int(b), key
    except Exception:
        return (10**12, 10**12, key)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _iter_h5_files(input_dir: Path) -> List[Path]:
    if input_dir.is_file():
        return [input_dir]
    return sorted(input_dir.rglob("*.hdf5")) + sorted(input_dir.rglob("*.h5"))


def _check_finite(name: str, arr: np.ndarray, errors: List[str]) -> None:
    if arr.dtype.kind in {"f", "c"} and not np.isfinite(arr).all():
        errors.append(f"{name}: contains non-finite values")


def _require_dataset(group: h5py.Group, key: str, ndim: int, errors: List[str]) -> np.ndarray | None:
    if key not in group:
        errors.append(f"missing dataset {key}")
        return None
    obj = group[key]
    if not isinstance(obj, h5py.Dataset):
        errors.append(f"{key}: expected dataset, got {type(obj).__name__}")
        return None
    if obj.ndim != ndim:
        errors.append(f"{key}: expected ndim={ndim}, got shape={obj.shape}")
        return None
    arr = obj[()]
    _check_finite(key, np.asarray(arr), errors)
    return np.asarray(arr)


def _check_camera_group(cam: h5py.Group, cam_name: str, T: int, N: int, errors: List[str]) -> None:
    for key in REQUIRED_CAMERA_DATASETS:
        if key not in cam:
            errors.append(f"{cam_name}: missing {key}")

    if "initial_rgb" in cam:
        rgb = cam["initial_rgb"]
        if not isinstance(rgb, h5py.Dataset) or rgb.shape[:1] != (1,):
            errors.append(f"{cam_name}/initial_rgb: expected one JPEG byte payload, got {getattr(rgb, 'shape', None)}")

    if "initial_depth" in cam:
        depth = np.asarray(cam["initial_depth"])
        if depth.ndim != 2:
            errors.append(f"{cam_name}/initial_depth: expected 2-D, got {depth.shape}")
        _check_finite(f"{cam_name}/initial_depth", depth, errors)

    if "intrinsic" in cam:
        intr = np.asarray(cam["intrinsic"])
        if intr.shape != (3, 3):
            errors.append(f"{cam_name}/intrinsic: expected (3,3), got {intr.shape}")
        _check_finite(f"{cam_name}/intrinsic", intr, errors)

    if "extrinsic" in cam:
        extr = np.asarray(cam["extrinsic"])
        if extr.shape != (4, 4):
            errors.append(f"{cam_name}/extrinsic: expected (4,4), got {extr.shape}")
        _check_finite(f"{cam_name}/extrinsic", extr, errors)

    if "extrinsic_trajectory" in cam:
        ext_t = np.asarray(cam["extrinsic_trajectory"])
        if ext_t.shape != (T, 4, 4):
            errors.append(f"{cam_name}/extrinsic_trajectory: expected ({T},4,4), got {ext_t.shape}")
        _check_finite(f"{cam_name}/extrinsic_trajectory", ext_t, errors)

    # BEHAVIOR-like local mesh groups are required for compatibility/debugging,
    # although the custom WDS converter reads the direct scene_* arrays above.
    for group_key in ["local_scene_points", "local_scene_colors", "local_scene_normals", "scene_mesh_trajectories"]:
        if group_key not in cam or not isinstance(cam[group_key], h5py.Group):
            errors.append(f"{cam_name}: missing group {group_key}")

    if "local_scene_points" in cam and isinstance(cam["local_scene_points"], h5py.Group):
        names = sorted(cam["local_scene_points"].keys())
        if not names:
            errors.append(f"{cam_name}/local_scene_points: empty")
        for mesh in names[:4]:
            pts = np.asarray(cam["local_scene_points"][mesh])
            if pts.ndim != 2 or pts.shape[-1] != 3:
                errors.append(f"{cam_name}/local_scene_points/{mesh}: expected (N,3), got {pts.shape}")
            _check_finite(f"{cam_name}/local_scene_points/{mesh}", pts, errors)
            if "scene_mesh_trajectories" in cam and mesh in cam["scene_mesh_trajectories"]:
                traj = np.asarray(cam["scene_mesh_trajectories"][mesh])
                if traj.shape != (T, 7):
                    errors.append(f"{cam_name}/scene_mesh_trajectories/{mesh}: expected ({T},7), got {traj.shape}")
                _check_finite(f"{cam_name}/scene_mesh_trajectories/{mesh}", traj, errors)


def check_clip(h5_path: Path, clip_key: str, group: h5py.Group, require_cameras: bool = True) -> Tuple[bool, Dict[str, Any]]:
    errors: List[str] = []
    arrays: Dict[str, np.ndarray] = {}

    for key, ndim in DIRECT_ARRAY_NDIMS.items():
        arr = _require_dataset(group, key, ndim, errors)
        if arr is not None:
            arrays[key] = arr

    if "scene_flows" in arrays:
        T, N, C = arrays["scene_flows"].shape
        if C != 3:
            errors.append(f"scene_flows: last dim must be 3, got {C}")
    else:
        T, N = -1, -1

    if T > 0:
        shape_expectations = {
            "scene_colors": (T, N, 3),
            "scene_normals": (T, N, 3),
            "scene_visibility": (T, N),
            "scene_depth_valid_mask": (T, N),
            "action_state": (T, arrays.get("action_state", np.empty((T, 0))).shape[-1]),
            "action_target": (T - 1, arrays.get("action_target", np.empty((T - 1, 0))).shape[-1]),
            "action_mask": (T - 1,),
        }
        for key, expected in shape_expectations.items():
            if key in arrays and arrays[key].shape != expected:
                errors.append(f"{key}: expected shape {expected}, got {arrays[key].shape}")
        if "robot_flows" in arrays:
            rt = arrays["robot_flows"].shape[0]
            rn = arrays["robot_flows"].shape[1]
            if arrays["robot_flows"].shape[-1] != 3 or rt != T:
                errors.append(f"robot_flows: expected ({T},Nr,3), got {arrays['robot_flows'].shape}")
            for key in ["robot_colors", "robot_normals"]:
                if key in arrays and arrays[key].shape[:2] != (T, rn):
                    errors.append(f"{key}: expected first dims ({T},{rn}), got {arrays[key].shape}")

    camera_keys = sorted(k for k in group.keys() if k.startswith("camera_") and isinstance(group[k], h5py.Group))
    if require_cameras and not camera_keys:
        errors.append("no camera_* groups found")
    for cam_key in camera_keys:
        _check_camera_group(group[cam_key], cam_key, T, N, errors)

    info = {
        "h5_path": str(h5_path),
        "clip_key": clip_key,
        "ok": not errors,
        "errors": errors,
        "num_frames": int(T) if T > 0 else None,
        "num_scene_points": int(N) if N > 0 else None,
        "num_cameras": len(camera_keys),
        "source_task": str(group.attrs.get("source_task", "")),
        "source_episode": str(group.attrs.get("source_episode", "")),
        "source_start": int(group.attrs.get("source_start", -1)),
        "source_end": int(group.attrs.get("source_end", -1)),
    }
    return not errors, info


def check_file(path: Path, require_cameras: bool = True) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    valid: List[Dict[str, Any]] = []
    invalid: List[Dict[str, Any]] = []
    try:
        with h5py.File(path, "r") as f:
            clip_keys = sorted([k for k in f.keys() if isinstance(f[k], h5py.Group) and ":" in k], key=_clip_sort_key)
            if not clip_keys:
                invalid.append({"h5_path": str(path), "clip_key": None, "ok": False, "errors": ["no clip groups named start:end"]})
                return valid, invalid
            for clip_key in clip_keys:
                ok, info = check_clip(path, clip_key, f[clip_key], require_cameras=require_cameras)
                if ok:
                    valid.append(info)
                else:
                    invalid.append(info)
    except Exception as exc:
        invalid.append({"h5_path": str(path), "clip_key": None, "ok": False, "errors": [repr(exc)]})
    return valid, invalid


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", required=True, help="Generated-H5 flows root or one .hdf5 file")
    p.add_argument("--output", default=None, help="Output JSON path; default <input-dir>/integrity_check.json")
    p.add_argument("--no-require-cameras", action="store_true")
    args = p.parse_args()

    input_dir = Path(args.input_dir)
    output = Path(args.output) if args.output else (input_dir / "integrity_check.json")
    valid: List[Dict[str, Any]] = []
    invalid: List[Dict[str, Any]] = []
    files = _iter_h5_files(input_dir)

    for i, path in enumerate(files):
        v, inv = check_file(path, require_cameras=not args.no_require_cameras)
        valid.extend(v)
        invalid.extend(inv)
        if (i + 1) % 20 == 0 or i == len(files) - 1:
            print(f"checked files {i + 1}/{len(files)} | valid clips={len(valid)} invalid={len(invalid)}", flush=True)

    result = {
        "input_dir": str(input_dir),
        "num_files": len(files),
        "num_valid_clips": len(valid),
        "num_invalid_items": len(invalid),
        "valid_clips": valid,
        "invalid_items": invalid,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as fp:
        json.dump(result, fp, indent=2, sort_keys=True, default=_json_default)
    print(f"wrote {output}", flush=True)
    print(f"valid clips: {len(valid)}", flush=True)
    print(f"invalid items: {len(invalid)}", flush=True)
    if len(valid) == 0:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
