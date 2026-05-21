#!/usr/bin/env python3
"""Convert RoboTwin two-gripper demonstrations into PointWorld-style generated H5 clips.

This follows the PointWorld data-branch pattern more closely than writing WDS
from raw episodes directly:

  raw RoboTwin HDF5 episodes -> generated clip H5 files -> integrity JSON ->
  split manifest -> WDS shards.

The generated H5 is intentionally "behavior-like": each top-level group is a
fixed-horizon clip named "start:end", each camera group is named camera_*, and
basic BEHAVIOR clip attributes are present.  Extra RoboTwin action fields are
stored per clip and preserved by tools/robotwin2g/convert_robotwin_h5_to_wds.py.

RoboTwin point clouds are per-frame observations.  Unless your simulator exports
stable point identities, keep --scene-flow-mode repeat_t0; this creates a stable
PointWorld input contract for action fine-tuning without pretending to supervise
true scene flow.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.robotwin2g.convert_robotwin_to_wds import (  # noqa: E402
    EpisodeInfo,
    decode_jpeg_bytes,
    derive_gripper_open,
    discover_episodes,
    encode_jpeg_rgb,
    fibonacci_sphere,
    h5_has,
    h5_len,
    h5_read,
    infer_camera_from_points,
    normalize_pose_xyzw,
    parse_slice,
    pointcloud_to_scene_arrays,
    rasterize_depth,
    read_action_arrays,
    read_pointcloud,
    resize_depth_m,
    resize_rgb,
    transform_local_points,
    valid_clip_starts,
)

try:
    from shared.data_contract import BEHAVIOR_CLIP_ATTRIBUTE_KEYS
except Exception:  # main branch does not contain shared/data_contract.py
    BEHAVIOR_CLIP_ATTRIBUTE_KEYS = [
        "clip_key",
        "num_frames",
        "num_scene_points",
        "has_transition",
        "any_object_moving",
        "gripper_moving",
        "has_gripper_state_change",
        "robot_nonbase_moving",
        "has_trunk_arm_collision",
        "has_left_gripper_finger_collision",
        "has_right_gripper_finger_collision",
        "max_object_pos_movement",
        "max_object_rot_movement",
        "max_gripper_pos_movement",
        "max_gripper_rot_movement",
        "max_joint_movement",
        "left_min_distance_to_moving_objects",
        "left_min_distance_to_all_objects",
        "right_min_distance_to_moving_objects",
        "right_min_distance_to_all_objects",
        "clip_complete",
    ]


UINT8_VLEN = h5py.vlen_dtype(np.dtype("uint8"))


def _print(msg: str) -> None:
    print(msg, flush=True)


def _safe_task_name(name: str) -> str:
    out = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(name))
    return out or "task"


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(obj, fp, indent=2, sort_keys=True)


def _create_dataset(group: h5py.Group, name: str, value: np.ndarray, *, compression: bool = True) -> None:
    value = np.asarray(value)
    kwargs: Dict[str, Any] = {}
    if compression and value.size > 0 and value.dtype.kind not in {"O", "S", "U"}:
        kwargs.update(compression="gzip", compression_opts=3, shuffle=True)
    group.create_dataset(name, data=value, **kwargs)


def _write_vlen_jpeg(group: h5py.Group, name: str, jpeg_bytes: bytes) -> None:
    ds = group.create_dataset(name, shape=(1,), dtype=UINT8_VLEN)
    ds[0] = np.frombuffer(jpeg_bytes, dtype=np.uint8)


def _read_h5_frame_dataset(f: h5py.File, key: str, frame: int, default: Any = None) -> Any:
    key = key.strip("/")
    if key not in f:
        return default
    ds = f[key]
    if not isinstance(ds, h5py.Dataset):
        return default
    if ds.shape and ds.shape[0] > frame:
        return ds[frame]
    return default


def _maybe_read_matrix(f: h5py.File, keys: Sequence[str], shape: Tuple[int, ...]) -> Optional[np.ndarray]:
    for key in keys:
        key = key.strip("/")
        if key in f:
            arr = np.asarray(f[key][()], dtype=np.float32)
            if arr.shape == shape:
                return arr
            if arr.ndim == len(shape) + 1 and tuple(arr.shape[1:]) == shape:
                return arr[0].astype(np.float32)
    return None


def _to_depth_uint16_mm(depth_m: np.ndarray) -> np.ndarray:
    depth_m = np.asarray(depth_m, dtype=np.float32)
    depth_m = np.nan_to_num(depth_m, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(np.round(depth_m * 1000.0), 0, 65535).astype(np.uint16)


def _quantize_normals(normals: np.ndarray) -> np.ndarray:
    normals = np.asarray(normals, dtype=np.float32)
    return np.clip(np.round(normals * 127.0), -127, 127).astype(np.int8)


def _subsample_scene_arrays(
    scene_flows: np.ndarray,
    scene_colors: np.ndarray,
    scene_normals: np.ndarray,
    scene_visibility: np.ndarray,
    scene_depth_valid_mask: np.ndarray,
    *,
    max_scene_points: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = int(scene_flows.shape[1])
    if max_scene_points <= 0 or n <= max_scene_points:
        return scene_flows, scene_colors, scene_normals, scene_visibility, scene_depth_valid_mask
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(n, size=max_scene_points, replace=False))
    return scene_flows[:, idx], scene_colors[:, idx], scene_normals[:, idx], scene_visibility[:, idx], scene_depth_valid_mask[:, idx]


def _robot_surrogate_arrays(
    endpose: Optional[np.ndarray],
    frames: np.ndarray,
    args: argparse.Namespace,
    hand_qpos: Optional[np.ndarray],
    total_len: int,
) -> Dict[str, np.ndarray]:
    T = len(frames)
    left_slice = parse_slice(args.left_pose_slice)
    right_slice = parse_slice(args.right_pose_slice)
    if endpose is None:
        base = np.zeros((total_len, 14), dtype=np.float32)
        base[:, 6] = 1.0
        base[:, 13] = 1.0
    else:
        base = np.asarray(endpose, dtype=np.float32)
    left_pose = normalize_pose_xyzw(base[:, left_slice])[frames]
    right_pose = normalize_pose_xyzw(base[:, right_slice])[frames]

    if args.policy_robot_input:
        # Strict policy setting: do not inject future endpose trajectory.  Repeat t0 robot geometry.
        left_pose_for_points = np.repeat(left_pose[:1], T, axis=0)
        right_pose_for_points = np.repeat(right_pose[:1], T, axis=0)
    else:
        # PointWorld-style robot flow input: full robot/gripper trajectory across the clip.
        left_pose_for_points = left_pose
        right_pose_for_points = right_pose

    local_pts, local_normals = fibonacci_sphere(args.robot_points_per_gripper, args.robot_point_radius)
    left_pts, left_normals = transform_local_points(left_pose_for_points, local_pts, local_normals)
    right_pts, right_normals = transform_local_points(right_pose_for_points, local_pts, local_normals)

    # Keep right first so feature construction matches PointWorld bimanual conventions.
    robot_flows = np.concatenate([right_pts, left_pts], axis=1).astype(np.float32)
    robot_normals = np.concatenate([right_normals, left_normals], axis=1).astype(np.float32)
    right_color = np.tile(np.array([255, 0, 255], dtype=np.uint8), (T, args.robot_points_per_gripper, 1))
    left_color = np.tile(np.array([0, 255, 255], dtype=np.uint8), (T, args.robot_points_per_gripper, 1))
    robot_colors = np.concatenate([right_color, left_color], axis=1)

    left_open, right_open = derive_gripper_open(hand_qpos, frames, total_len)
    return {
        "robot_flows": robot_flows,
        "robot_normals": robot_normals,
        "robot_colors": robot_colors,
        "left_gripper_pose": left_pose.astype(np.float32),
        "right_gripper_pose": right_pose.astype(np.float32),
        "left_gripper_open": left_open.astype(np.float32),
        "right_gripper_open": right_open.astype(np.float32),
    }


def _write_camera_group(
    clip_group: h5py.Group,
    f: h5py.File,
    camera_name: str,
    camera_index: int,
    start: int,
    scene_points0: np.ndarray,
    scene_colors0: np.ndarray,
    scene_normals0: np.ndarray,
    args: argparse.Namespace,
) -> None:
    cam_group = clip_group.create_group(f"camera_{camera_name}")

    rgb_raw = _read_h5_frame_dataset(f, f"/observation/{camera_name}/rgb", start, default=None)
    if rgb_raw is not None:
        rgb = resize_rgb(decode_jpeg_bytes(rgb_raw), args.image_width, args.image_height)
    else:
        rgb = np.zeros((args.image_height, args.image_width, 3), dtype=np.uint8)
    _write_vlen_jpeg(cam_group, "initial_rgb", encode_jpeg_rgb(rgb, quality=args.jpeg_quality))

    intrinsic = _maybe_read_matrix(
        f,
        [
            f"/observation/{camera_name}/intrinsic",
            f"/observation/{camera_name}/intrinsics",
            f"/camera/{camera_name}/intrinsic",
            f"/camera/{camera_name}/intrinsics",
        ],
        (3, 3),
    )
    extrinsic = _maybe_read_matrix(
        f,
        [
            f"/observation/{camera_name}/extrinsic",
            f"/observation/{camera_name}/extrinsics",
            f"/camera/{camera_name}/extrinsic",
            f"/camera/{camera_name}/extrinsics",
        ],
        (4, 4),
    )
    if intrinsic is None or extrinsic is None:
        fallback_intr, fallback_extr = infer_camera_from_points(scene_points0, args.image_width, args.image_height)
        intrinsic = fallback_intr if intrinsic is None else intrinsic
        extrinsic = fallback_extr if extrinsic is None else extrinsic

    depth_raw = _read_h5_frame_dataset(f, f"/observation/{camera_name}/depth", start, default=None)
    if depth_raw is not None:
        depth_m = resize_depth_m(depth_raw, args.image_width, args.image_height)
    else:
        depth_m = rasterize_depth(scene_points0, intrinsic, extrinsic, args.image_width, args.image_height)
    depth_uint16 = _to_depth_uint16_mm(depth_m)

    _create_dataset(cam_group, "initial_depth", depth_uint16, compression=True)
    _create_dataset(cam_group, "intrinsic", intrinsic.astype(np.float32), compression=False)
    _create_dataset(cam_group, "extrinsic", extrinsic.astype(np.float32), compression=False)
    extr_traj = np.repeat(extrinsic.astype(np.float32)[None], args.clip_horizon, axis=0)
    _create_dataset(cam_group, "extrinsic_trajectory", extr_traj, compression=True)

    # Behavior-like local mesh fields.  For default repeat_t0, identity trajectory reconstructs the direct scene arrays.
    lp = cam_group.create_group("local_scene_points")
    lc = cam_group.create_group("local_scene_colors")
    ln = cam_group.create_group("local_scene_normals")
    mt = cam_group.create_group("scene_mesh_trajectories")
    mesh_name = "robotwin_observed"
    _create_dataset(lp, mesh_name, scene_points0.astype(np.float16), compression=True)
    _create_dataset(lc, mesh_name, scene_colors0.astype(np.uint8), compression=True)
    _create_dataset(ln, mesh_name, _quantize_normals(scene_normals0), compression=True)
    traj = np.zeros((args.clip_horizon, 7), dtype=np.float32)
    traj[:, 6] = 1.0
    _create_dataset(mt, mesh_name, traj, compression=True)


def _default_clip_attrs(clip_key: str, num_frames: int, num_scene_points: int) -> Dict[str, Any]:
    values: Dict[str, Any] = {
        "clip_key": clip_key,
        "num_frames": int(num_frames),
        "num_scene_points": int(num_scene_points),
        "has_transition": True,
        "any_object_moving": False,
        "gripper_moving": True,
        "has_gripper_state_change": False,
        "robot_nonbase_moving": True,
        "has_trunk_arm_collision": False,
        "has_left_gripper_finger_collision": False,
        "has_right_gripper_finger_collision": False,
        "max_object_pos_movement": 0.0,
        "max_object_rot_movement": 0.0,
        "max_gripper_pos_movement": 0.0,
        "max_gripper_rot_movement": 0.0,
        "max_joint_movement": 0.0,
        "left_min_distance_to_moving_objects": -1.0,
        "left_min_distance_to_all_objects": -1.0,
        "right_min_distance_to_moving_objects": -1.0,
        "right_min_distance_to_all_objects": -1.0,
        "clip_complete": True,
    }
    # Keep exactly the contract keys when available.
    return {k: values.get(k, False) for k in BEHAVIOR_CLIP_ATTRIBUTE_KEYS}


def _write_clip_group(
    out_h5: h5py.File,
    in_h5: h5py.File,
    ep: EpisodeInfo,
    start: int,
    args: argparse.Namespace,
) -> Tuple[str, np.ndarray]:
    horizon = int(args.clip_horizon)
    frames = np.arange(start, start + horizon, dtype=np.int64)
    clip_key = f"{start}:{start + horizon}"
    clip_group = out_h5.create_group(clip_key)

    action, state, hand_qpos = read_action_arrays(in_h5, args)
    total_len = int(min(ep.length, action.shape[0], state.shape[0]))
    pcd_clip = read_pointcloud(in_h5, args, frames)
    scene_flows, scene_colors, scene_normals, visibility, depth_valid = pointcloud_to_scene_arrays(
        pcd_clip, args.scene_flow_mode
    )
    scene_flows, scene_colors, scene_normals, visibility, depth_valid = _subsample_scene_arrays(
        scene_flows,
        scene_colors,
        scene_normals,
        visibility,
        depth_valid,
        max_scene_points=args.max_scene_points,
        seed=args.seed + int(start),
    )

    action_state = state[frames].astype(np.float32)
    action_target = action[frames[1:]].astype(np.float32)
    action_mask = np.ones((horizon - 1,), dtype=np.bool_)

    endpose = h5_read(in_h5, args.endpose_key, default=None) if args.endpose_key else None
    robot_arrays = _robot_surrogate_arrays(endpose, frames, args, hand_qpos, total_len)

    # Direct fields preserved by the custom WDS converter and action trainer.
    for name, value in {
        "scene_flows": scene_flows.astype(np.float32),
        "scene_colors": scene_colors.astype(np.uint8),
        "scene_normals": scene_normals.astype(np.float32),
        "scene_visibility": visibility.astype(np.bool_),
        "scene_depth_valid_mask": depth_valid.astype(np.bool_),
        "action_state": action_state,
        "action_target": action_target,
        "action_mask": action_mask,
        **robot_arrays,
    }.items():
        _create_dataset(clip_group, name, value, compression=True)

    # Behavior-like robot state fields.
    base_pose = np.zeros((horizon, 7), dtype=np.float32)
    base_pose[:, 6] = 1.0
    _create_dataset(clip_group, "base_pose", base_pose, compression=True)
    _create_dataset(clip_group, "world_to_robot", np.eye(4, dtype=np.float32), compression=False)
    joint_names = np.asarray([f"robotwin2g_joint_{i:02d}" for i in range(action_state.shape[-1])], dtype=h5py.string_dtype("utf-8"))
    clip_group.create_dataset("joint_names", data=joint_names)
    _create_dataset(clip_group, "joint_positions", action_state, compression=True)
    _create_dataset(clip_group, "left_is_grasping", np.zeros((horizon, 1), dtype=np.bool_), compression=True)
    _create_dataset(clip_group, "right_is_grasping", np.zeros((horizon, 1), dtype=np.bool_), compression=True)

    for cam_idx, cam in enumerate(args.camera_names):
        _write_camera_group(
            clip_group,
            in_h5,
            camera_name=cam,
            camera_index=cam_idx,
            start=start,
            scene_points0=scene_flows[0],
            scene_colors0=scene_colors[0],
            scene_normals0=scene_normals[0],
            args=args,
        )

    clip_group.attrs.update(_default_clip_attrs(clip_key, horizon, int(scene_flows.shape[1])))
    clip_group.attrs["source_start"] = int(start)
    clip_group.attrs["source_end"] = int(start + horizon)
    clip_group.attrs["source_path"] = str(ep.path)
    clip_group.attrs["source_task"] = str(ep.task)
    clip_group.attrs["source_episode"] = str(ep.episode)
    clip_group.attrs["scene_flow_mode"] = str(args.scene_flow_mode)
    clip_group.attrs["policy_robot_input"] = bool(args.policy_robot_input)
    return clip_key, action_target


def _episode_output_path(output_root: Path, task_id: str, episode_index: int) -> Path:
    return output_root / "flows" / task_id / f"episode_{episode_index:08d}.hdf5"


def convert(args: argparse.Namespace) -> None:
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    flows_root = output_root / "flows"
    flows_root.mkdir(parents=True, exist_ok=True)

    preferred_len_keys = [args.action_key, args.pointcloud_key, "/pointcloud"]
    episodes = discover_episodes(input_root, preferred_len_keys)
    if not episodes:
        raise RuntimeError(f"No readable HDF5 episodes found under {input_root}")

    task_names = sorted({ep.task for ep in episodes})
    task_to_id = {task: f"task-{idx:04d}" for idx, task in enumerate(task_names)}
    _write_json(output_root / "task_map.json", task_to_id)

    per_task_counter: Dict[str, int] = {task: 0 for task in task_names}
    manifest_rows: List[Dict[str, Any]] = []
    bad: List[Dict[str, str]] = []
    total_clips = 0

    for ep_i, ep in enumerate(episodes):
        task_id = task_to_id[ep.task]
        episode_index = per_task_counter[ep.task]
        per_task_counter[ep.task] += 1
        out_path = _episode_output_path(output_root, task_id, episode_index)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.exists() and args.overwrite:
            out_path.unlink()
        elif out_path.exists() and not args.overwrite:
            _print(f"[skip-existing] {out_path}")
            continue

        try:
            with h5py.File(ep.path, "r") as in_h5:
                action = np.asarray(h5_read(in_h5, args.action_key), dtype=np.float32)
                starts = valid_clip_starts(ep.length, action.shape[0], args.clip_horizon, args.clip_stride)
                if args.max_clips_per_episode > 0:
                    starts = starts[: args.max_clips_per_episode]
                if len(starts) == 0:
                    bad.append({"path": str(ep.path), "error": "no valid clip starts"})
                    continue
                with h5py.File(out_path, "w") as out_h5:
                    out_h5.attrs["domain"] = "robotwin2g"
                    out_h5.attrs["format"] = "pointworld_generated_h5_like"
                    out_h5.attrs["source_path"] = str(ep.path)
                    out_h5.attrs["source_task"] = str(ep.task)
                    out_h5.attrs["source_episode"] = str(ep.episode)
                    out_h5.attrs["task_id"] = task_id
                    out_h5.attrs["episode_index"] = int(episode_index)
                    out_h5.attrs["clip_horizon"] = int(args.clip_horizon)
                    out_h5.attrs["clip_stride"] = int(args.clip_stride)
                    out_h5.attrs["action_dim"] = int(action.shape[-1])
                    for start in starts:
                        clip_key, _ = _write_clip_group(out_h5, in_h5, ep, start, args)
                        manifest_rows.append(
                            {
                                "h5_path": str(out_path),
                                "clip_key": clip_key,
                                "source_path": str(ep.path),
                                "source_task": str(ep.task),
                                "source_episode": str(ep.episode),
                                "task_id": task_id,
                                "episode_index": episode_index,
                                "start": int(start),
                                "end": int(start + args.clip_horizon),
                            }
                        )
                        total_clips += 1
        except Exception as exc:
            bad.append({"path": str(ep.path), "error": repr(exc)})
            _print(f"[bad] {ep.path}: {exc}")

        if (ep_i + 1) % 10 == 0 or ep_i == len(episodes) - 1:
            _print(f"processed episodes {ep_i + 1}/{len(episodes)} | generated clips={total_clips} bad={len(bad)}")

    with open(output_root / "generated_h5_manifest.jsonl", "w", encoding="utf-8") as fp:
        for row in manifest_rows:
            fp.write(json.dumps(row, sort_keys=True) + "\n")
    _write_json(
        output_root / "metadata.json",
        {
            "input_root": str(input_root),
            "output_root": str(output_root),
            "num_source_episodes": len(episodes),
            "num_generated_clips": total_clips,
            "bad_episodes": bad,
            "config": vars(args),
        },
    )
    _print("done")
    _print(f"generated_h5_root: {output_root}")
    _print(f"flows_root: {flows_root}")
    _print(f"clips: {total_clips}")
    _print(f"bad episodes: {len(bad)}")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-root", required=True, help="RoboTwin data root or one .hdf5 file")
    p.add_argument("--output-root", required=True, help="Output generated-H5 root; creates <root>/flows/task-*/episode_*.hdf5")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--clip-horizon", type=int, default=11)
    p.add_argument("--clip-stride", type=int, default=5)
    p.add_argument("--max-clips-per-episode", type=int, default=-1, help="Debug cap; <=0 disables")
    p.add_argument("--max-scene-points", type=int, default=2048, help="Subsample scene points per clip; <=0 keeps all")

    p.add_argument("--action-key", default="/joint_action/vector")
    p.add_argument("--state-key", default=None, help="Defaults to --action-key")
    p.add_argument("--hand-qpos-key", default="/joint_action/hand_qpos")
    p.add_argument("--endpose-key", default="/endpose")
    p.add_argument("--pointcloud-key", default="/pointcloud")
    p.add_argument("--camera-names", nargs="+", default=["cam0", "cam1"])

    p.add_argument("--left-pose-slice", default="0:7", help="Slice into endpose for left gripper pose [x,y,z,qx,qy,qz,qw]")
    p.add_argument("--right-pose-slice", default="7:14", help="Slice into endpose for right gripper pose [x,y,z,qx,qy,qz,qw]")
    p.add_argument("--robot-points-per-gripper", type=int, default=64)
    p.add_argument("--robot-point-radius", type=float, default=0.025)
    p.add_argument(
        "--policy-robot-input",
        action="store_true",
        help="Repeat t=0 robot geometry instead of using future endpose trajectory; safer for closed-loop policy training.",
    )

    p.add_argument("--scene-flow-mode", choices=["repeat_t0", "by_index"], default="repeat_t0")
    p.add_argument("--image-width", type=int, default=320)
    p.add_argument("--image-height", type=int, default=180)
    p.add_argument("--jpeg-quality", type=int, default=95)
    return p


def main() -> None:
    args = build_argparser().parse_args()
    if args.clip_horizon < 2:
        raise ValueError("--clip-horizon must be >= 2")
    if args.clip_stride < 1:
        raise ValueError("--clip-stride must be >= 1")
    if (args.image_width, args.image_height) != (320, 180):
        _print("[warn] PointWorld release camera contract expects initial_depth shape 180x320; non-default image size is for custom use only.")
    convert(args)


if __name__ == "__main__":
    main()
