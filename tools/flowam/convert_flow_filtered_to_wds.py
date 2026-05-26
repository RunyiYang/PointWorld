#!/usr/bin/env python3
"""Convert FloWAM filtered flow episodes into direct PointWorld-style WDS clips.

The input files are full filtered episodes named ``*_flow_filtered.hdf5``.  Each
file stores tracked scene points at every sampled frame plus precomputed robot
point trajectories under ``/extra_flows``.  This converter slices those episodes
into fixed-horizon clips and writes WebDataset tar shards with the direct fields
used by the local RoboTwin/PointWorld WDS reader:

  scene_flows, scene_colors, scene_normals, scene_visibility,
  scene_depth_valid_mask, robot_flows, robot_colors, robot_normals,
  left/right_gripper_pose, left/right_gripper_open, action_state,
  action_target, action_mask, and camera payloads.

The filtered flow files do not store policy labels directly.  For dexterous
FlowAM training, this script resolves the matching original HDF5 episode and
builds a 54-D action vector in this order:

  left_arm(7), right_arm(7), left_hand(20), right_hand(20)
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.robotwin2g.convert_robotwin_to_wds import (  # noqa: E402
    TarShardWriter,
    encode_jpeg_rgb,
    infer_camera_from_points,
    json_bytes,
    npy_bytes,
    rasterize_depth,
    split_episodes,
)


SCENE_GROUP_CANDIDATES = ("extra_flows/scene", "")
ROBOT_GROUP_KEYS = {
    "arm": "extra_flows/arm",
    "eef": "extra_flows/eef",
    "hand": "extra_flows/hand",
}
SIDE_KEYS = {
    "arm": "arm_sample_side",
    "eef": "eef_sample_side",
    "hand": "point_side",
}


@dataclass(frozen=True)
class FlowEpisode:
    path: Path
    task: str
    episode: str
    length: int


def _print(msg: str) -> None:
    print(msg, flush=True)


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(obj, fp, indent=2, sort_keys=True)


def _split_csv(value: str) -> List[str]:
    return [token.strip() for token in value.split(",") if token.strip()]


def _task_from_path(path: Path) -> str:
    name = path.parent.name
    return name[:-5] if name.endswith("_flow") else name


def _episode_from_path(path: Path) -> str:
    name = path.stem
    return name[: -len("_flow_filtered")] if name.endswith("_flow_filtered") else name


def _find_scene_group(f: h5py.File) -> h5py.Group:
    for group_key in SCENE_GROUP_CANDIDATES:
        group = f[group_key] if group_key else f
        if "pointcloud" in group:
            return group
    raise KeyError("No scene pointcloud found at /extra_flows/scene/pointcloud or /pointcloud")


def discover_episodes(input_root: Path) -> List[FlowEpisode]:
    if input_root.is_file():
        paths = [input_root]
    else:
        paths = sorted(input_root.rglob("*_flow_filtered.hdf5"))
        if not paths:
            paths = sorted(input_root.rglob("*.hdf5")) + sorted(input_root.rglob("*.h5"))
    episodes: List[FlowEpisode] = []
    for path in paths:
        try:
            with h5py.File(path, "r") as f:
                length = int(_find_scene_group(f)["pointcloud"].shape[0])
        except Exception as exc:
            _print(f"[skip] {path}: {exc}")
            continue
        episodes.append(
            FlowEpisode(
                path=path,
                task=_task_from_path(path),
                episode=_episode_from_path(path),
                length=length,
            )
        )
    return episodes


def _default_action_root(input_root: Path) -> Path:
    if input_root.is_file():
        if input_root.parent.name.endswith("_flow"):
            return input_root.parent.parent.parent / "origin"
        return input_root.parent.parent / "origin"
    return input_root.parent / "origin"


def _resolve_action_root(args: argparse.Namespace, input_root: Path) -> Path:
    if args.action_root:
        return Path(args.action_root).expanduser().resolve()
    return _default_action_root(input_root).resolve()


def _resolve_action_path(ep: FlowEpisode, action_root: Path) -> Path:
    return action_root / ep.task / f"{ep.episode}.hdf5"


def valid_clip_starts(length: int, horizon: int, stride: int) -> List[int]:
    last_start = int(length) - int(horizon)
    if last_start < 0:
        return []
    return list(range(0, last_start + 1, int(stride)))


def _dataset_or_default(group: h5py.Group, key: str, frames: np.ndarray, shape: Tuple[int, ...], dtype: Any, fill: Any) -> np.ndarray:
    if key not in group:
        return np.full(shape, fill, dtype=dtype)
    arr = np.asarray(group[key][frames], dtype=dtype)
    return arr


def _unit_z_normals(shape: Tuple[int, int, int]) -> np.ndarray:
    normals = np.zeros(shape, dtype=np.float32)
    normals[..., 2] = 1.0
    return normals


def _read_scene_arrays(f: h5py.File, frames: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    group = _find_scene_group(f)
    group_name = group.name
    points = np.asarray(group["pointcloud"][frames], dtype=np.float32)
    t, n = points.shape[:2]
    colors = _dataset_or_default(group, "point_rgb", frames, (t, n, 3), np.uint8, 0)
    normals = _unit_z_normals((t, n, 3))

    visibility = np.ones((t, n), dtype=np.bool_)
    for key in ("visibs", "point_valid"):
        if key in group:
            visibility &= np.asarray(group[key][frames], dtype=np.bool_)

    depth_valid = np.ones((t, n), dtype=np.bool_)
    for key in ("flow_valid", "point_valid"):
        if key in group:
            depth_valid &= np.asarray(group[key][frames], dtype=np.bool_)

    return points, colors, normals, visibility, depth_valid, group_name


def _read_robot_group(
    f: h5py.File,
    group_name: str,
    frames: np.ndarray,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], str]]:
    h5_key = ROBOT_GROUP_KEYS[group_name]
    if h5_key not in f:
        return None
    group = f[h5_key]
    if "pointcloud" not in group:
        return None
    points = np.asarray(group["pointcloud"][frames], dtype=np.float32)
    t, n = points.shape[:2]
    colors = _dataset_or_default(group, "point_rgb", frames, (t, n, 3), np.uint8, 160)
    normals = _unit_z_normals((t, n, 3))
    side = None
    side_key = SIDE_KEYS[group_name]
    if side_key in group and group[side_key].shape[0] == n:
        side = np.asarray(group[side_key][()], dtype=np.int8)
    return points, colors, normals, side, h5_key


def _concat_robot_arrays(
    f: h5py.File,
    frames: np.ndarray,
    group_names: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], List[str]]:
    point_parts: List[np.ndarray] = []
    color_parts: List[np.ndarray] = []
    normal_parts: List[np.ndarray] = []
    side_parts: List[np.ndarray] = []
    used_groups: List[str] = []

    for group_name in group_names:
        payload = _read_robot_group(f, group_name, frames)
        if payload is None:
            continue
        points, colors, normals, side, h5_key = payload
        point_parts.append(points)
        color_parts.append(colors)
        normal_parts.append(normals)
        used_groups.append(h5_key)
        if side is not None:
            side_parts.append(side)
        else:
            n = points.shape[1]
            fallback_side = np.zeros((n,), dtype=np.int8)
            fallback_side[n // 2 :] = 1
            side_parts.append(fallback_side)

    if not point_parts:
        raise KeyError(f"No robot pointcloud groups found from requested groups: {list(group_names)}")

    robot_points = np.concatenate(point_parts, axis=1).astype(np.float32)
    robot_colors = np.concatenate(color_parts, axis=1).astype(np.uint8)
    robot_normals = np.concatenate(normal_parts, axis=1).astype(np.float32)
    robot_side = np.concatenate(side_parts, axis=0).astype(np.int8) if side_parts else None
    return robot_points, robot_colors, robot_normals, robot_side, used_groups


def _pose_from_points(points: np.ndarray, side: Optional[np.ndarray], wanted_side: int) -> np.ndarray:
    t = points.shape[0]
    pose = np.zeros((t, 7), dtype=np.float32)
    pose[:, 6] = 1.0
    if side is None or side.shape[0] != points.shape[1]:
        mask = np.arange(points.shape[1]) < (points.shape[1] // 2)
        if wanted_side == 1:
            mask = ~mask
    else:
        mask = side == wanted_side
    if not np.any(mask):
        pose[:, :3] = np.mean(points, axis=1)
    else:
        pose[:, :3] = np.mean(points[:, mask], axis=1)
    return pose


def _resize_rgb(rgb: np.ndarray, width: int, height: int) -> np.ndarray:
    if rgb.shape[:2] == (height, width):
        return rgb.astype(np.uint8, copy=False)
    return cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA).astype(np.uint8)


def _read_video_frame(path: Path, frame_index: int, width: int, height: int) -> Optional[np.ndarray]:
    if not path.exists():
        return None
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            return None
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame_bgr = cap.read()
        if not ok or frame_bgr is None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame_bgr = cap.read()
            if not ok or frame_bgr is None:
                return None
        rgb = frame_bgr[..., ::-1].copy()
        return _resize_rgb(rgb, width, height)
    finally:
        cap.release()


def _rasterize_rgb(points: np.ndarray, colors: np.ndarray, intr: np.ndarray, extr: np.ndarray, width: int, height: int) -> np.ndarray:
    img = np.zeros((height, width, 3), dtype=np.uint8)
    pts_h = np.concatenate([points, np.ones((points.shape[0], 1), dtype=np.float32)], axis=1)
    cam = (extr @ pts_h.T).T[:, :3]
    z = cam[:, 2]
    valid = np.isfinite(cam).all(axis=1) & (z > 1e-4)
    if not np.any(valid):
        return img
    pix = (intr @ cam[valid].T).T
    u = np.round(pix[:, 0] / np.clip(pix[:, 2], 1e-6, None)).astype(np.int64)
    v = np.round(pix[:, 1] / np.clip(pix[:, 2], 1e-6, None)).astype(np.int64)
    in_img = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    if np.any(in_img):
        img[v[in_img], u[in_img]] = colors[valid][in_img]
    return img


def _video_candidates(h5_path: Path) -> List[Path]:
    stem = h5_path.stem
    return [
        h5_path.with_name(f"{stem}_scene.mp4"),
        h5_path.with_suffix(".mp4"),
    ]


def _camera_payloads(
    h5_path: Path,
    start: int,
    frame_indices: np.ndarray,
    scene_points0: np.ndarray,
    scene_colors0: np.ndarray,
    args: argparse.Namespace,
) -> Tuple[Dict[str, bytes], str]:
    intr, extr = infer_camera_from_points(scene_points0, args.image_width, args.image_height)
    depth = rasterize_depth(scene_points0, intr, extr, args.image_width, args.image_height)
    rgb: Optional[np.ndarray] = None
    rgb_source = "rasterized_pointcloud"
    # The sidecar MP4s are rendered from the filtered sequence, so their frame
    # index matches the filtered HDF5 index, not the original raw frame id.
    video_frame = int(start)
    for video_path in _video_candidates(h5_path):
        rgb = _read_video_frame(video_path, video_frame, args.image_width, args.image_height)
        if rgb is not None:
            rgb_source = str(video_path)
            break
    if rgb is None:
        rgb = _rasterize_rgb(scene_points0, scene_colors0, intr, extr, args.image_width, args.image_height)

    payload: Dict[str, bytes] = {}
    for cam_idx in range(args.num_cameras):
        prefix = f"cam{cam_idx}"
        payload[f"{prefix}_initial_rgb.jpg"] = encode_jpeg_rgb(rgb, quality=args.jpeg_quality)
        payload[f"{prefix}_initial_depth.npy"] = npy_bytes(depth.astype(np.float32))
        payload[f"{prefix}_intrinsic.npy"] = npy_bytes(intr.astype(np.float32))
        payload[f"{prefix}_extrinsic.npy"] = npy_bytes(extr.astype(np.float32))
    return payload, rgb_source


def _read_frame_indices(f: h5py.File, length: int) -> np.ndarray:
    for key in ("extra_flows/scene/frame_indices", "frame_indices"):
        if key in f:
            arr = np.asarray(f[key][()], dtype=np.int64)
            if arr.shape[0] == length:
                return arr
    return np.arange(length, dtype=np.int64)


def _read_first_dataset(f: h5py.File, keys: Sequence[str]) -> Tuple[np.ndarray, str]:
    for key in keys:
        h5_key = key.strip("/")
        if h5_key in f:
            return np.asarray(f[h5_key][()], dtype=np.float32), "/" + h5_key
    raise KeyError(f"None of the requested datasets exist: {list(keys)}")


def _read_dexterous_action_sequence(action_path: Path, args: argparse.Namespace) -> Tuple[np.ndarray, Dict[str, Any]]:
    with h5py.File(action_path, "r") as f:
        arm, arm_key = _read_first_dataset(f, [args.arm_action_key])
        hand, hand_key = _read_first_dataset(f, args.hand_action_keys)
    if arm.ndim != 2 or arm.shape[1] < 14:
        raise ValueError(f"{action_path}:{arm_key} must be (T, >=14), got {arm.shape}")
    if hand.ndim != 2 or hand.shape[1] < 40:
        raise ValueError(f"{action_path}:{hand_key} must be (T, >=40), got {hand.shape}")
    length = min(int(arm.shape[0]), int(hand.shape[0]))
    if length <= 0:
        raise ValueError(f"{action_path} has empty action arrays")
    action = np.concatenate(
        [
            arm[:length, :7],
            arm[:length, 7:14],
            hand[:length, :20],
            hand[:length, 20:40],
        ],
        axis=-1,
    ).astype(np.float32)
    meta = {
        "action_source_path": str(action_path),
        "arm_action_key": arm_key,
        "hand_action_key": hand_key,
        "action_order": ["left_arm", "right_arm", "left_hand", "right_hand"],
        "action_dims": {
            "left_arm": 7,
            "right_arm": 7,
            "left_hand": 20,
            "right_hand": 20,
        },
    }
    return action, meta


def _build_clip_payload(
    f: h5py.File,
    ep: FlowEpisode,
    split: str,
    start: int,
    args: argparse.Namespace,
    action_sequence: np.ndarray,
    action_meta: Dict[str, Any],
) -> Tuple[str, Dict[str, bytes], np.ndarray, np.ndarray]:
    frames = np.arange(start, start + args.clip_horizon, dtype=np.int64)
    scene_flows, scene_colors, scene_normals, scene_visibility, depth_valid, scene_group = _read_scene_arrays(f, frames)
    robot_flows, robot_colors, robot_normals, robot_side, robot_groups = _concat_robot_arrays(
        f,
        frames,
        args.robot_groups,
    )

    left_pose = _pose_from_points(robot_flows, robot_side, wanted_side=0)
    right_pose = _pose_from_points(robot_flows, robot_side, wanted_side=1)
    left_open = np.zeros((args.clip_horizon, 1), dtype=np.float32)
    right_open = np.zeros((args.clip_horizon, 1), dtype=np.float32)

    frame_indices = _read_frame_indices(f, ep.length)
    source_frames = frame_indices[frames].astype(np.int64)
    max_frame = int(source_frames.max(initial=-1))
    if max_frame >= action_sequence.shape[0]:
        raise IndexError(
            f"Action sequence too short for {ep.path}: max source frame {max_frame}, "
            f"action length {action_sequence.shape[0]}"
        )
    action_state = action_sequence[source_frames].astype(np.float32)
    action_target = action_sequence[source_frames[1:]].astype(np.float32)
    action_mask = np.ones((args.clip_horizon - 1,), dtype=np.bool_)

    camera_payload, rgb_source = _camera_payloads(
        ep.path,
        start,
        frame_indices,
        scene_flows[0],
        scene_colors[0],
        args,
    )

    key = f"{ep.task}-{ep.episode}-{start:06d}-{start + args.clip_horizon:06d}"
    flow_delta = scene_flows[1:] - scene_flows[:-1]
    payload: Dict[str, bytes] = {
        "scene_flows.npy": npy_bytes(scene_flows.astype(np.float32)),
        "scene_colors.npy": npy_bytes(scene_colors.astype(np.uint8)),
        "scene_normals.npy": npy_bytes(scene_normals.astype(np.float32)),
        "scene_visibility.npy": npy_bytes(scene_visibility.astype(np.bool_)),
        "scene_depth_valid_mask.npy": npy_bytes(depth_valid.astype(np.bool_)),
        "robot_flows.npy": npy_bytes(robot_flows.astype(np.float32)),
        "robot_colors.npy": npy_bytes(robot_colors.astype(np.uint8)),
        "robot_normals.npy": npy_bytes(robot_normals.astype(np.float32)),
        "left_gripper_pose.npy": npy_bytes(left_pose),
        "right_gripper_pose.npy": npy_bytes(right_pose),
        "left_gripper_open.npy": npy_bytes(left_open),
        "right_gripper_open.npy": npy_bytes(right_open),
        "action_state.npy": npy_bytes(action_state),
        "action_target.npy": npy_bytes(action_target),
        "action_mask.npy": npy_bytes(action_mask),
        "metadata.json": json_bytes(
            {
                "sample_key": key,
                "split": split,
                "task": ep.task,
                "episode": ep.episode,
                "source_path": str(ep.path),
                "start": int(start),
                "end": int(start + args.clip_horizon),
                "frame_indices": frame_indices[frames].astype(int).tolist(),
                "scene_group": scene_group,
                "robot_groups": robot_groups,
                "rgb_source": rgb_source,
                "format": "flowam_filtered_direct_wds_v1",
                **action_meta,
            }
        ),
    }
    payload.update(camera_payload)
    return key, payload, action_target, flow_delta.astype(np.float32)


def _update_stats(
    sum_vec: Optional[np.ndarray],
    sumsq_vec: Optional[np.ndarray],
    count: int,
    values: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, int]:
    flat = values.reshape(-1, values.shape[-1]).astype(np.float64)
    valid = np.isfinite(flat).all(axis=1)
    flat = flat[valid]
    if sum_vec is None:
        sum_vec = np.zeros((flat.shape[-1],), dtype=np.float64)
        sumsq_vec = np.zeros((flat.shape[-1],), dtype=np.float64)
    if flat.size:
        sum_vec += flat.sum(axis=0)
        assert sumsq_vec is not None
        sumsq_vec += (flat * flat).sum(axis=0)
        count += int(flat.shape[0])
    return sum_vec, sumsq_vec, count


def _stats_dict(sum_vec: Optional[np.ndarray], sumsq_vec: Optional[np.ndarray], count: int) -> Dict[str, Any]:
    if sum_vec is None or sumsq_vec is None or count <= 0:
        return {"count": 0, "mean": [], "std": []}
    mean = sum_vec / float(count)
    var = np.maximum(sumsq_vec / float(count) - mean * mean, 1e-8)
    return {
        "count": int(count),
        "mean": mean.astype(float).tolist(),
        "std": np.sqrt(var).astype(float).tolist(),
    }


def convert(args: argparse.Namespace) -> None:
    input_root = Path(args.input_root).resolve()
    output_root = Path(args.output_root).resolve()
    action_root = _resolve_action_root(args, input_root)
    if input_root == output_root:
        raise ValueError("input_root and output_root must be different")
    if output_root.exists() and args.overwrite:
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    episodes = discover_episodes(input_root)
    if not episodes:
        raise RuntimeError(f"No readable filtered flow HDF5 files found under {input_root}")
    split_assignment = split_episodes(episodes, args.test_ratio, args.seed, args.split_scope)

    writers = {
        "train": TarShardWriter(output_root, "train", args.max_samples_per_shard),
        "test": TarShardWriter(output_root, "test", args.max_samples_per_shard),
    }
    processed = {"train": 0, "test": 0}
    source_rows = {"train": [], "test": []}
    errors: List[Dict[str, str]] = []
    action_sum: Optional[np.ndarray] = None
    action_sumsq: Optional[np.ndarray] = None
    action_count = 0
    flow_sum: Optional[np.ndarray] = None
    flow_sumsq: Optional[np.ndarray] = None
    flow_count = 0

    manifest_path = output_root / "manifest.jsonl"
    try:
        with open(manifest_path, "w", encoding="utf-8") as manifest_fp:
            for ep_idx, ep in enumerate(episodes):
                split = split_assignment[ep.path]
                starts = valid_clip_starts(ep.length, args.clip_horizon, args.clip_stride)
                if args.max_clips_per_episode > 0:
                    starts = starts[: args.max_clips_per_episode]
                try:
                    action_path = _resolve_action_path(ep, action_root)
                    if not action_path.exists():
                        raise FileNotFoundError(f"Missing action source for {ep.path}: {action_path}")
                    action_sequence, action_meta = _read_dexterous_action_sequence(action_path, args)
                    with h5py.File(ep.path, "r") as f:
                        for start in starts:
                            key, payload, action_target, flow_delta = _build_clip_payload(
                                f,
                                ep,
                                split,
                                start,
                                args,
                                action_sequence=action_sequence,
                                action_meta=action_meta,
                            )
                            writers[split].add(key, payload)
                            processed[split] += 1
                            source_rows[split].append(f"{ep.path}::{start}:{start + args.clip_horizon}")
                            manifest_fp.write(
                                json.dumps(
                                    {
                                        "key": key,
                                        "split": split,
                                        "task": ep.task,
                                        "episode": ep.episode,
                                        "source_path": str(ep.path),
                                        "start": int(start),
                                        "end": int(start + args.clip_horizon),
                                    },
                                    sort_keys=True,
                                )
                                + "\n"
                            )
                            if split == "train":
                                action_sum, action_sumsq, action_count = _update_stats(
                                    action_sum,
                                    action_sumsq,
                                    action_count,
                                    action_target,
                                )
                                flow_sum, flow_sumsq, flow_count = _update_stats(
                                    flow_sum,
                                    flow_sumsq,
                                    flow_count,
                                    flow_delta,
                                )
                except Exception as exc:
                    errors.append({"path": str(ep.path), "error": repr(exc)})
                    _print(f"[bad] {ep.path}: {exc}")
                    if not args.keep_going:
                        raise

                if (ep_idx + 1) % 10 == 0 or ep_idx == len(episodes) - 1:
                    _print(
                        f"processed episodes {ep_idx + 1}/{len(episodes)} | "
                        f"train={processed['train']} test={processed['test']} errors={len(errors)}"
                    )
    finally:
        for writer in writers.values():
            writer.close()

    for split in ("train", "test"):
        with open(output_root / f"{split}_sources.txt", "w", encoding="utf-8") as fp:
            for row in source_rows[split]:
                fp.write(row + "\n")

    flow_stats = _stats_dict(flow_sum, flow_sumsq, flow_count)
    action_stats = _stats_dict(action_sum, action_sumsq, action_count)
    if action_stats["count"] == 0:
        dim = int(args.action_dim_fallback)
        action_stats = {
            "count": 0,
            "mean": np.zeros((dim,), dtype=float).tolist(),
            "std": np.ones((dim,), dtype=float).tolist(),
        }
    action_stats.update(
        {
            "source": "train split dexterous action_target",
            "action_order": ["left_arm", "right_arm", "left_hand", "right_hand"],
            "action_dims": {
                "left_arm": 7,
                "right_arm": 7,
                "left_hand": 20,
                "right_hand": 20,
            },
        }
    )
    _write_json(output_root / "flow_delta_stats.json", flow_stats)
    _write_json(output_root / "action_stats.json", action_stats)
    _write_json(
        output_root / "metadata_rank0.json",
        {
            "input_root": str(input_root),
            "action_root": str(action_root),
            "output_root": str(output_root),
            "domain": "flowam_filtered_direct",
            "train": {
                "processed_count": processed["train"],
                "shards": [str(p) for p in writers["train"].shard_paths],
            },
            "test": {
                "processed_count": processed["test"],
                "shards": [str(p) for p in writers["test"].shard_paths],
            },
            "errors": errors,
            "action_stats": action_stats,
            "flow_delta_stats": flow_stats,
            "config": vars(args),
        },
    )
    _write_json(
        output_root / "dataset_summary.json",
        {
            "num_source_episodes": len(episodes),
            "num_train_clips": processed["train"],
            "num_test_clips": processed["test"],
            "num_errors": len(errors),
            "action_dim": len(action_stats["mean"]),
            "action_order": ["left_arm", "right_arm", "left_hand", "right_hand"],
            "clip_horizon": args.clip_horizon,
            "clip_stride": args.clip_stride,
            "robot_groups": list(args.robot_groups),
        },
    )
    _print(json.dumps({"train": processed["train"], "test": processed["test"], "errors": len(errors)}, indent=2))
    _print(f"wrote WDS root: {output_root}")


def _parse_robot_groups(value: str) -> List[str]:
    out = [token.strip() for token in value.split(",") if token.strip()]
    unknown = sorted(set(out) - set(ROBOT_GROUP_KEYS))
    if unknown:
        raise argparse.ArgumentTypeError(f"Unknown robot groups: {unknown}; valid={sorted(ROBOT_GROUP_KEYS)}")
    return out


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--action-root", default=None, help="Root with original action HDF5s; defaults to sibling 'origin'.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-ratio", type=float, default=0.02)
    parser.add_argument("--split-scope", choices=["task", "global"], default="task")
    parser.add_argument("--clip-horizon", type=int, default=11)
    parser.add_argument("--clip-stride", type=int, default=5)
    parser.add_argument("--max-clips-per-episode", type=int, default=-1)
    parser.add_argument("--max-samples-per-shard", type=int, default=512)
    parser.add_argument("--arm-action-key", default="/joint_action/vector")
    parser.add_argument(
        "--hand-action-keys",
        type=_split_csv,
        default=_split_csv("/joint_action/hand_qpos,/joint_state/hand_qpos"),
        help="Comma-separated hand qpos keys to try, in order.",
    )
    parser.add_argument("--action-dim-fallback", type=int, default=54)
    parser.add_argument("--robot-groups", type=_parse_robot_groups, default=_parse_robot_groups("arm,eef,hand"))
    parser.add_argument("--num-cameras", type=int, default=2)
    parser.add_argument("--image-width", type=int, default=320)
    parser.add_argument("--image-height", type=int, default=180)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    if args.clip_horizon < 2:
        raise ValueError("--clip-horizon must be >= 2")
    if args.clip_stride < 1:
        raise ValueError("--clip-stride must be >= 1")
    if args.num_cameras < 1:
        raise ValueError("--num-cameras must be >= 1")
    convert(args)


if __name__ == "__main__":
    main()
