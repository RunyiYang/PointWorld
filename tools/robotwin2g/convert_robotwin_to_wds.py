#!/usr/bin/env python3
"""Convert RoboTwin-style two-gripper HDF5 episodes into PointWorld-like WDS clips.

This converter is intentionally self-contained. It writes WebDataset-compatible tar
shards using Python's standard tarfile module, so it can run before the PointWorld
training environment has webdataset installed.

The output is designed for tools/robotwin2g/train_robotwin_action.py. It also
keeps the tensor names close to PointWorld's release pipeline:

  scene_flows, scene_colors, scene_normals, scene_visibility,
  scene_depth_valid_mask, robot_flows, robot_colors, robot_normals,
  left/right_gripper_pose, left/right_gripper_open, cam*_initial_rgb/depth,
  cam*_intrinsic/extrinsic, action_state, action_target.

RoboTwin point clouds are observed independently per frame. Unless your simulator
exports stable point identities, use the default --scene-flow-mode repeat_t0 for
action-only fine-tuning. Use --scene-flow-mode by_index only if point n is a
persistent physical point across the clip.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import random
import tarfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import cv2
import h5py
import numpy as np


def _print(msg: str) -> None:
    print(msg, flush=True)


def stable_int(*parts: object) -> int:
    h = hashlib.sha256()
    for p in parts:
        h.update(str(p).encode("utf-8"))
        h.update(b"\0")
    return int.from_bytes(h.digest()[:8], "little", signed=False)


def parse_slice(spec: str) -> slice:
    """Parse CLI slice syntax like '0:7'."""
    if ":" not in spec:
        idx = int(spec)
        return slice(idx, idx + 1)
    parts = spec.split(":")
    if len(parts) > 3:
        raise ValueError(f"Invalid slice spec {spec!r}")
    start = int(parts[0]) if parts[0] else None
    stop = int(parts[1]) if len(parts) > 1 and parts[1] else None
    step = int(parts[2]) if len(parts) > 2 and parts[2] else None
    return slice(start, stop, step)


def h5_has(f: h5py.File, key: str) -> bool:
    key = key.strip("/")
    return key in f


def h5_read(f: h5py.File, key: str, default: Any = None) -> Any:
    key = key.strip("/")
    if key not in f:
        return default
    return f[key][()]


def h5_len(f: h5py.File, preferred_keys: Sequence[str]) -> int:
    for key in preferred_keys:
        key = key.strip("/")
        if key in f:
            return int(f[key].shape[0])
    raise KeyError(f"Could not infer episode length from keys: {preferred_keys}")


def decode_jpeg_bytes(raw: Any) -> np.ndarray:
    if isinstance(raw, np.ndarray):
        raw = raw.tobytes()
    elif hasattr(raw, "tobytes") and not isinstance(raw, (bytes, bytearray, memoryview)):
        raw = raw.tobytes()
    raw = bytes(raw)
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError("cv2.imdecode failed for JPEG payload")
    return img[..., ::-1].copy()  # BGR -> RGB


def encode_jpeg_rgb(img_rgb: np.ndarray, quality: int = 95) -> bytes:
    if img_rgb.dtype != np.uint8:
        img_rgb = np.clip(img_rgb, 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", img_rgb[..., ::-1], [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return buf.tobytes()


def resize_rgb(img: np.ndarray, width: int, height: int) -> np.ndarray:
    if img.shape[1] == width and img.shape[0] == height:
        return img.astype(np.uint8, copy=False)
    return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA).astype(np.uint8)


def resize_depth_m(depth: np.ndarray, width: int, height: int) -> np.ndarray:
    depth = np.asarray(depth)
    if depth.dtype == np.uint16 or depth.dtype == np.uint32:
        depth = depth.astype(np.float32) / 1000.0
    else:
        depth = depth.astype(np.float32)
    if depth.shape[1] == width and depth.shape[0] == height:
        return depth
    return cv2.resize(depth, (width, height), interpolation=cv2.INTER_NEAREST).astype(np.float32)


def npy_bytes(array: np.ndarray) -> bytes:
    bio = io.BytesIO()
    np.save(bio, array, allow_pickle=False)
    return bio.getvalue()


def json_bytes(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")


def quat_xyzw_to_mat(quat: np.ndarray) -> np.ndarray:
    """Convert (...,4) xyzw quaternion to (...,3,3), normalizing robustly."""
    q = np.asarray(quat, dtype=np.float32)
    out_shape = q.shape[:-1] + (3, 3)
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    bad = (~np.isfinite(norm)) | (norm < 1e-6)
    q = np.where(bad, np.array([0, 0, 0, 1], dtype=np.float32), q / np.clip(norm, 1e-6, None))
    x, y, z, w = [q[..., i] for i in range(4)]
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    R = np.empty(out_shape, dtype=np.float32)
    R[..., 0, 0] = 1 - 2 * (yy + zz)
    R[..., 0, 1] = 2 * (xy - wz)
    R[..., 0, 2] = 2 * (xz + wy)
    R[..., 1, 0] = 2 * (xy + wz)
    R[..., 1, 1] = 1 - 2 * (xx + zz)
    R[..., 1, 2] = 2 * (yz - wx)
    R[..., 2, 0] = 2 * (xz - wy)
    R[..., 2, 1] = 2 * (yz + wx)
    R[..., 2, 2] = 1 - 2 * (xx + yy)
    return R


def normalize_pose_xyzw(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float32)
    if pose.shape[-1] < 7:
        padded = np.zeros(pose.shape[:-1] + (7,), dtype=np.float32)
        padded[..., : pose.shape[-1]] = pose
        padded[..., 6] = 1.0
        pose = padded
    pose = pose[..., :7].copy()
    q = pose[..., 3:7]
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    good = np.isfinite(n) & (n > 1e-6)
    q_norm = np.where(good, q / np.clip(n, 1e-6, None), np.array([0, 0, 0, 1], dtype=np.float32))
    pose[..., 3:7] = q_norm
    return pose.astype(np.float32)


def fibonacci_sphere(n: int, radius: float) -> Tuple[np.ndarray, np.ndarray]:
    """Return local points and outward normals on a deterministic sphere."""
    n = max(1, int(n))
    pts = np.zeros((n, 3), dtype=np.float32)
    if n == 1:
        pts[0] = 0.0
    else:
        golden = math.pi * (3.0 - math.sqrt(5.0))
        for i in range(n):
            y = 1.0 - (i / float(n - 1)) * 2.0
            r = math.sqrt(max(0.0, 1.0 - y * y))
            theta = golden * i
            pts[i] = [math.cos(theta) * r, y, math.sin(theta) * r]
    normals = pts.copy()
    norm = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = np.where(norm > 1e-6, normals / np.clip(norm, 1e-6, None), np.array([0, 0, 1], dtype=np.float32))
    return (pts * float(radius)).astype(np.float32), normals.astype(np.float32)


def transform_local_points(pose_t7: np.ndarray, local_points: np.ndarray, local_normals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    pose_t7 = normalize_pose_xyzw(pose_t7)
    R = quat_xyzw_to_mat(pose_t7[:, 3:7])  # (T,3,3)
    t = pose_t7[:, :3]
    pts = np.einsum("tij,nj->tni", R, local_points) + t[:, None, :]
    nrm = np.einsum("tij,nj->tni", R, local_normals)
    return pts.astype(np.float32), nrm.astype(np.float32)


def infer_camera_from_points(points: np.ndarray, width: int, height: int) -> Tuple[np.ndarray, np.ndarray]:
    """Fit a simple perspective camera so most initial points project into view.

    The returned extrinsic is world-to-camera. It adds a z translation if needed
    to make points positive-depth, then chooses focal length from projected x/z
    and y/z quantiles. This is a fallback for RoboTwin samples that do not store
    calibration. Use real calibration if available.
    """
    pts = np.asarray(points, dtype=np.float32)
    finite = np.isfinite(pts).all(axis=1)
    pts = pts[finite]
    if pts.size == 0:
        pts = np.zeros((1, 3), dtype=np.float32)
    z_min = float(np.nanpercentile(pts[:, 2], 2))
    z_offset = 0.25 - z_min if z_min < 0.25 else 0.0
    extr = np.eye(4, dtype=np.float32)
    extr[2, 3] = z_offset
    z = np.clip(pts[:, 2] + z_offset, 1e-3, None)
    x_over_z = pts[:, 0] / z
    y_over_z = pts[:, 1] / z
    sx = float(np.nanpercentile(np.abs(x_over_z), 95))
    sy = float(np.nanpercentile(np.abs(y_over_z), 95))
    fx = 0.45 * width / max(sx, 1e-3)
    fy = 0.45 * height / max(sy, 1e-3)
    f = float(np.clip(min(fx, fy), 20.0, 5000.0))
    intr = np.array([[f, 0.0, width / 2.0], [0.0, f, height / 2.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    return intr, extr


def rasterize_depth(points: np.ndarray, intr: np.ndarray, extr: np.ndarray, width: int, height: int) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    depth = np.full((height, width), np.inf, dtype=np.float32)
    pts_h = np.concatenate([pts, np.ones((pts.shape[0], 1), dtype=np.float32)], axis=1)
    cam = (extr @ pts_h.T).T[:, :3]
    z = cam[:, 2]
    valid = np.isfinite(cam).all(axis=1) & (z > 1e-4)
    if valid.any():
        pix = (intr @ cam[valid].T).T
        u = np.round(pix[:, 0] / np.clip(pix[:, 2], 1e-6, None)).astype(np.int64)
        v = np.round(pix[:, 1] / np.clip(pix[:, 2], 1e-6, None)).astype(np.int64)
        zz = z[valid]
        in_img = (u >= 0) & (u < width) & (v >= 0) & (v < height)
        if in_img.any():
            flat = v[in_img] * width + u[in_img]
            np.minimum.at(depth.reshape(-1), flat, zz[in_img])
    finite = np.isfinite(depth)
    fill = float(np.nanmedian(depth[finite])) if finite.any() else 1.0
    depth[~finite] = fill
    return depth.astype(np.float32)


@dataclass
class EpisodeInfo:
    path: Path
    task: str
    episode: str
    length: int


class TarShardWriter:
    def __init__(self, out_dir: Path, split: str, max_samples_per_shard: int = 512):
        self.out_dir = out_dir / split
        self.split = split
        self.max_samples_per_shard = int(max_samples_per_shard)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.shard_index = -1
        self.samples_in_shard = 0
        self.tar: Optional[tarfile.TarFile] = None
        self.shard_paths: List[Path] = []
        self.count = 0

    def _open_next(self) -> None:
        self.close()
        self.shard_index += 1
        path = self.out_dir / f"{self.split}-{self.shard_index:06d}.tar"
        self.tar = tarfile.open(path, mode="w")
        self.shard_paths.append(path)
        self.samples_in_shard = 0

    def add(self, key: str, payload: Dict[str, bytes]) -> None:
        if self.tar is None or self.samples_in_shard >= self.max_samples_per_shard:
            self._open_next()
        assert self.tar is not None
        for field, data in payload.items():
            name = f"{key}.{field}"
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            info.mtime = int(time.time())
            self.tar.addfile(info, io.BytesIO(data))
        self.samples_in_shard += 1
        self.count += 1

    def close(self) -> None:
        if self.tar is not None:
            self.tar.close()
            self.tar = None


def discover_episodes(
    input_root: Path,
    preferred_len_keys: Sequence[str],
    tasks: Optional[Sequence[str]] = None,
) -> List[EpisodeInfo]:
    if input_root.is_file():
        paths = [input_root]
    else:
        task_filter = [task for task in (tasks or []) if task]
        if task_filter:
            paths = []
            for task in task_filter:
                task_dir = input_root / task
                if not task_dir.exists():
                    _print(f"[skip] missing task directory: {task_dir}")
                    continue
                paths.extend(sorted(task_dir.rglob("*.hdf5")))
                paths.extend(sorted(task_dir.rglob("*.h5")))
        else:
            paths = sorted(input_root.rglob("*.hdf5")) + sorted(input_root.rglob("*.h5"))
    out: List[EpisodeInfo] = []
    for path in paths:
        try:
            with h5py.File(path, "r") as f:
                length = h5_len(f, preferred_len_keys)
        except Exception as exc:
            _print(f"[skip] {path}: failed to read length: {exc}")
            continue
        if input_root.is_dir():
            try:
                rel = path.relative_to(input_root)
                task = rel.parts[0] if len(rel.parts) > 1 else path.parent.name
            except ValueError:
                task = path.parent.name if path.parent.name else "task"
        else:
            task = path.parent.name if path.parent.name else "task"
        episode = path.stem
        out.append(EpisodeInfo(path=path, task=task, episode=episode, length=length))
    return out


def split_episodes(episodes: List[EpisodeInfo], test_ratio: float, seed: int, scope: str) -> Dict[Path, str]:
    if not 0.0 <= test_ratio < 1.0:
        raise ValueError("test_ratio must be in [0, 1)")
    rng = random.Random(seed)
    assignment: Dict[Path, str] = {}

    def assign_group(group: List[EpisodeInfo]) -> None:
        group = list(group)
        rng.shuffle(group)
        if len(group) <= 1 or test_ratio == 0.0:
            n_test = 0
        else:
            n_test = max(1, int(round(len(group) * test_ratio)))
            n_test = min(n_test, len(group) - 1)
        test_set = {ep.path for ep in group[:n_test]}
        for ep in group:
            assignment[ep.path] = "test" if ep.path in test_set else "train"

    if scope == "global":
        assign_group(episodes)
    elif scope == "task":
        by_task: Dict[str, List[EpisodeInfo]] = {}
        for ep in episodes:
            by_task.setdefault(ep.task, []).append(ep)
        for task in sorted(by_task):
            assign_group(by_task[task])
    else:
        raise ValueError(f"Unsupported split scope {scope!r}")
    return assignment


def read_action_arrays(f: h5py.File, args: argparse.Namespace) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    action = h5_read(f, args.action_key)
    if action is None:
        raise KeyError(f"Missing action key {args.action_key}")
    action = np.asarray(action, dtype=np.float32)
    state_key = args.state_key or args.action_key
    state = h5_read(f, state_key)
    if state is None:
        state = action
    state = np.asarray(state, dtype=np.float32)
    hand = h5_read(f, args.hand_qpos_key, default=None) if args.hand_qpos_key else None
    if hand is not None:
        hand = np.asarray(hand, dtype=np.float32)
    return action, state, hand


def read_pointcloud(f: h5py.File, args: argparse.Namespace, frames: np.ndarray) -> np.ndarray:
    pcd = h5_read(f, args.pointcloud_key, default=None)
    if pcd is None:
        # Fall back to the first camera pcd if the shared pointcloud does not exist.
        for cam in args.camera_names:
            pcd = h5_read(f, f"/observation/{cam}/pcd", default=None)
            if pcd is not None:
                break
    if pcd is None:
        raise KeyError(f"Missing point cloud key {args.pointcloud_key} and no observation/<cam>/pcd fallback")
    pcd = np.asarray(pcd, dtype=np.float32)
    return pcd[frames]


def pointcloud_to_scene_arrays(pcd_clip: np.ndarray, scene_flow_mode: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xyz = pcd_clip[:, :, :3].astype(np.float32)
    if scene_flow_mode == "repeat_t0":
        xyz = np.repeat(xyz[:1], repeats=pcd_clip.shape[0], axis=0)
    elif scene_flow_mode == "by_index":
        pass
    else:
        raise ValueError(f"Unsupported scene flow mode {scene_flow_mode}")

    if pcd_clip.shape[-1] >= 6:
        colors = pcd_clip[:, :, 3:6].copy()
        if scene_flow_mode == "repeat_t0":
            colors = np.repeat(colors[:1], repeats=pcd_clip.shape[0], axis=0)
    else:
        colors = np.zeros_like(xyz)
    if np.nanmax(colors) <= 1.5:
        colors = colors * 255.0
    colors = np.clip(colors, 0, 255).astype(np.uint8)

    normals = np.zeros_like(xyz, dtype=np.float32)
    normals[..., 2] = 1.0
    visibility = np.ones(xyz.shape[:2], dtype=np.bool_)
    depth_valid = np.ones(xyz.shape[:2], dtype=np.bool_)
    return xyz, colors, normals, visibility, depth_valid


def derive_gripper_open(hand_qpos: Optional[np.ndarray], frames: np.ndarray, total_len: int) -> Tuple[np.ndarray, np.ndarray]:
    if hand_qpos is None or hand_qpos.ndim != 2 or hand_qpos.shape[1] < 2:
        z = np.zeros((len(frames), 1), dtype=np.float32)
        return z.copy(), z.copy()
    hand = hand_qpos.astype(np.float32)
    split = hand.shape[1] // 2
    left = np.mean(np.abs(hand[:, :split]), axis=1, keepdims=True)
    right = np.mean(np.abs(hand[:, split:]), axis=1, keepdims=True)
    # Normalize per episode into [0,1]. This is a magnitude proxy, not a semantic open/closed guarantee.
    denom = max(float(np.nanmax([left.max(initial=0.0), right.max(initial=0.0)])), 1e-6)
    left = np.clip(left / denom, 0.0, 1.0)
    right = np.clip(right / denom, 0.0, 1.0)
    return left[frames].astype(np.float32), right[frames].astype(np.float32)


def build_robot_surrogate(
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

    local_pts, local_normals = fibonacci_sphere(args.robot_points_per_gripper, args.robot_point_radius)
    left_pts, left_normals = transform_local_points(left_pose, local_pts, local_normals)
    right_pts, right_normals = transform_local_points(right_pose, local_pts, local_normals)

    robot_flows = np.concatenate([right_pts, left_pts], axis=1).astype(np.float32)  # right first matches PointWorld bimanual slots
    robot_normals = np.concatenate([right_normals, left_normals], axis=1).astype(np.float32)
    right_color = np.tile(np.array([255, 0, 255], dtype=np.uint8), (T, args.robot_points_per_gripper, 1))
    left_color = np.tile(np.array([0, 255, 255], dtype=np.uint8), (T, args.robot_points_per_gripper, 1))
    robot_colors = np.concatenate([right_color, left_color], axis=1)

    left_open, right_open = derive_gripper_open(hand_qpos, frames, total_len)
    return {
        "robot_flows.npy": npy_bytes(robot_flows),
        "robot_normals.npy": npy_bytes(robot_normals),
        "robot_colors.npy": npy_bytes(robot_colors),
        "left_gripper_pose.npy": npy_bytes(left_pose.astype(np.float32)),
        "right_gripper_pose.npy": npy_bytes(right_pose.astype(np.float32)),
        "left_gripper_open.npy": npy_bytes(left_open.astype(np.float32)),
        "right_gripper_open.npy": npy_bytes(right_open.astype(np.float32)),
    }


def build_camera_payloads(
    f: h5py.File,
    pcd0_xyz: np.ndarray,
    start: int,
    args: argparse.Namespace,
) -> Dict[str, bytes]:
    payload: Dict[str, bytes] = {}
    intr, extr = infer_camera_from_points(pcd0_xyz, args.image_width, args.image_height)
    synthetic_depth = rasterize_depth(pcd0_xyz, intr, extr, args.image_width, args.image_height)

    for cam_idx, cam in enumerate(args.camera_names):
        rgb_key = f"/observation/{cam}/rgb"
        if h5_has(f, rgb_key):
            rgb = decode_jpeg_bytes(f[rgb_key][start])
            rgb = resize_rgb(rgb, args.image_width, args.image_height)
        else:
            rgb = np.zeros((args.image_height, args.image_width, 3), dtype=np.uint8)

        depth_key = f"/observation/{cam}/depth"
        if h5_has(f, depth_key):
            depth = resize_depth_m(f[depth_key][start], args.image_width, args.image_height)
        else:
            depth = synthetic_depth

        prefix = f"cam{cam_idx}"
        payload[f"{prefix}_initial_rgb.jpg"] = encode_jpeg_rgb(rgb, quality=args.jpeg_quality)
        payload[f"{prefix}_initial_depth.npy"] = npy_bytes(depth.astype(np.float32))
        payload[f"{prefix}_intrinsic.npy"] = npy_bytes(intr.astype(np.float32))
        payload[f"{prefix}_extrinsic.npy"] = npy_bytes(extr.astype(np.float32))
    return payload


def build_clip_payload(
    f: h5py.File,
    ep: EpisodeInfo,
    split: str,
    start: int,
    args: argparse.Namespace,
) -> Tuple[str, Dict[str, bytes], np.ndarray]:
    horizon = args.clip_horizon
    frames = np.arange(start, start + horizon, dtype=np.int64)
    action, state, hand_qpos = read_action_arrays(f, args)
    pcd_clip = read_pointcloud(f, args, frames)
    scene_flows, scene_colors, scene_normals, visibility, depth_valid = pointcloud_to_scene_arrays(
        pcd_clip, args.scene_flow_mode
    )

    # Target is the next action/state for each future step in the clip.
    # Shape: (horizon - 1, action_dim), matching PointWorld's 1-context + 10-prediction setup.
    action_target = action[frames[1:]].astype(np.float32)
    action_state = state[frames].astype(np.float32)
    action_mask = np.ones((horizon - 1,), dtype=np.bool_)

    endpose = h5_read(f, args.endpose_key, default=None) if args.endpose_key else None
    total_len = int(min(ep.length, action.shape[0], state.shape[0]))

    key = f"{ep.task}-{ep.episode}-{start:06d}-{start + horizon:06d}"
    payload: Dict[str, bytes] = {
        "scene_flows.npy": npy_bytes(scene_flows.astype(np.float32)),
        "scene_colors.npy": npy_bytes(scene_colors),
        "scene_normals.npy": npy_bytes(scene_normals.astype(np.float32)),
        "scene_visibility.npy": npy_bytes(visibility),
        "scene_depth_valid_mask.npy": npy_bytes(depth_valid),
        "action_state.npy": npy_bytes(action_state),
        "action_target.npy": npy_bytes(action_target),
        "action_mask.npy": npy_bytes(action_mask),
        "metadata.json": json_bytes(
            {
                "task": ep.task,
                "episode": ep.episode,
                "source_path": str(ep.path),
                "split": split,
                "start": int(start),
                "end": int(start + horizon),
                "frames": frames.tolist(),
                "scene_flow_mode": args.scene_flow_mode,
                "action_key": args.action_key,
                "state_key": args.state_key or args.action_key,
            }
        ),
    }
    payload.update(build_robot_surrogate(endpose, frames, args, hand_qpos, total_len))
    payload.update(build_camera_payloads(f, scene_flows[0], start, args))
    return key, payload, action_target


def valid_clip_starts(length: int, action_len: int, horizon: int, stride: int) -> List[int]:
    # Need frames[start : start+horizon] and targets frames[start+1 : start+horizon].
    max_len = min(length, action_len)
    last_start = max_len - horizon
    if last_start < 0:
        return []
    return list(range(0, last_start + 1, stride))


def convert(args: argparse.Namespace) -> None:
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    split_dir = out_root / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)

    preferred_len_keys = [args.action_key, args.pointcloud_key, "/pointcloud"]
    episodes = discover_episodes(Path(args.input_root), preferred_len_keys, args.tasks)
    if not episodes:
        raise RuntimeError(f"No readable HDF5 episodes found under {args.input_root}")
    assignment = split_episodes(episodes, args.test_ratio, args.seed, args.split_scope)

    for split in ["train", "test"]:
        with open(split_dir / f"{split}_episodes.txt", "w") as fp:
            for ep in episodes:
                if assignment[ep.path] == split:
                    fp.write(f"{ep.task}/{ep.path.name}\t{ep.path}\n")

    writers = {
        "train": TarShardWriter(out_root, "train", args.max_samples_per_shard),
        "test": TarShardWriter(out_root, "test", args.max_samples_per_shard),
    }
    manifest_path = out_root / "manifest.jsonl"
    action_sum = None
    action_sq_sum = None
    action_count = 0
    bad: List[Dict[str, str]] = []

    with open(manifest_path, "w") as manifest_fp:
        for ep_i, ep in enumerate(episodes):
            split = assignment[ep.path]
            try:
                with h5py.File(ep.path, "r") as f:
                    action = np.asarray(h5_read(f, args.action_key), dtype=np.float32)
                    starts = valid_clip_starts(ep.length, action.shape[0], args.clip_horizon, args.clip_stride)
                    if args.max_clips_per_episode > 0:
                        starts = starts[: args.max_clips_per_episode]
                    for start in starts:
                        key, payload, action_target = build_clip_payload(f, ep, split, start, args)
                        writers[split].add(key, payload)
                        manifest_fp.write(
                            json.dumps(
                                {
                                    "key": key,
                                    "split": split,
                                    "task": ep.task,
                                    "episode": ep.episode,
                                    "path": str(ep.path),
                                    "start": start,
                                    "end": start + args.clip_horizon,
                                },
                                sort_keys=True,
                            )
                            + "\n"
                        )
                        if split == "train":
                            arr = action_target.reshape(-1, action_target.shape[-1]).astype(np.float64)
                            if action_sum is None:
                                action_sum = arr.sum(axis=0)
                                action_sq_sum = (arr * arr).sum(axis=0)
                            else:
                                action_sum += arr.sum(axis=0)
                                action_sq_sum += (arr * arr).sum(axis=0)
                            action_count += arr.shape[0]
            except Exception as exc:
                bad.append({"path": str(ep.path), "error": repr(exc)})
                _print(f"[bad] {ep.path}: {exc}")
            if (ep_i + 1) % 25 == 0 or ep_i == len(episodes) - 1:
                _print(
                    f"processed episodes {ep_i + 1}/{len(episodes)} | "
                    f"train clips={writers['train'].count} test clips={writers['test'].count} bad={len(bad)}"
                )

    for w in writers.values():
        w.close()

    if action_count > 0 and action_sum is not None and action_sq_sum is not None:
        mean = action_sum / action_count
        var = np.maximum(action_sq_sum / action_count - mean * mean, 1e-12)
        std = np.sqrt(var)
    else:
        # Fallback for tiny runs where all clips fell into test.
        first_dim = int(args.action_dim_fallback)
        mean = np.zeros((first_dim,), dtype=np.float64)
        std = np.ones((first_dim,), dtype=np.float64)

    action_stats = {
        "mean": mean.astype(float).tolist(),
        "std": std.astype(float).tolist(),
        "count": int(action_count),
        "source": "train split action_target",
    }
    with open(out_root / "action_stats.json", "w") as fp:
        json.dump(action_stats, fp, indent=2, sort_keys=True)

    metadata = {
        "train": {"processed_count": writers["train"].count, "num_shards": len(writers["train"].shard_paths)},
        "test": {"processed_count": writers["test"].count, "num_shards": len(writers["test"].shard_paths)},
        "config": vars(args),
        "bad_episodes": bad,
    }
    with open(out_root / "metadata_rank0.json", "w") as fp:
        json.dump(metadata, fp, indent=2, sort_keys=True)

    _print("done")
    _print(f"output_root: {out_root}")
    _print(f"train clips: {writers['train'].count}")
    _print(f"test clips:  {writers['test'].count}")
    _print(f"bad episodes: {len(bad)}")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-root", required=True, help="RoboTwin data root or one .hdf5 file")
    p.add_argument("--output-root", required=True, help="Output WDS root")
    p.add_argument("--tasks", nargs="+", default=None, help="Optional task directory names to include under --input-root")
    p.add_argument("--test-ratio", type=float, default=0.02)
    p.add_argument("--split-scope", choices=["task", "global"], default="task")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--clip-horizon", type=int, default=11)
    p.add_argument("--clip-stride", type=int, default=5)
    p.add_argument("--max-clips-per-episode", type=int, default=-1, help="Debug cap; <=0 disables")
    p.add_argument("--max-samples-per-shard", type=int, default=512)

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

    p.add_argument("--scene-flow-mode", choices=["repeat_t0", "by_index"], default="repeat_t0")
    p.add_argument("--image-width", type=int, default=320)
    p.add_argument("--image-height", type=int, default=180)
    p.add_argument("--jpeg-quality", type=int, default=95)
    p.add_argument("--action-dim-fallback", type=int, default=14)
    return p


def main() -> None:
    args = build_argparser().parse_args()
    if args.clip_horizon < 2:
        raise ValueError("--clip-horizon must be >= 2")
    if args.clip_stride < 1:
        raise ValueError("--clip-stride must be >= 1")
    convert(args)


if __name__ == "__main__":
    main()
