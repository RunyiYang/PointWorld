#!/usr/bin/env python3
"""Visualize raw PointWorld-BEHAVIOR HDF5 scene flow with Viser.

This reads restored BEHAVIOR HDF5 data, reconstructs tracked 3D points from
local mesh samples and per-frame mesh pose trajectories, and displays either one
clip or a merged full HDF5 episode sequence. It can also visualize RoboTwin-style
HDF5 files by animating `/pointcloud` point indices over the full episode.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import h5py
import numpy as np
import viser


DEFAULT_HDF5 = (
    "/work/runyi_yang/FloWAM/data/pointworld_behavior_restored/"
    "behavior/flows/task-0000/episode_00000010.hdf5"
)
DEFAULT_CLIP = "175:186"
DEFAULT_CAMERA = "camera_head"
DEFAULT_MESH = "World__scene_0__radio_89__base_link__visuals"


def _parse_slice(text: str) -> slice:
    parts = text.split(":")
    if len(parts) > 3:
        raise ValueError(f"Invalid slice {text!r}")
    values = [int(p) if p else None for p in parts]
    while len(values) < 3:
        values.append(None)
    return slice(*values)


def _point_indices(count: int, point_slice: str) -> np.ndarray:
    if point_slice.strip().lower() == "all":
        return np.arange(count, dtype=np.int64)
    return np.arange(count, dtype=np.int64)[_parse_slice(point_slice)]


def _normalize_colors(colors: np.ndarray) -> np.ndarray:
    colors = np.asarray(colors)
    if colors.dtype == np.uint8:
        return colors
    colors_f = colors.astype(np.float32)
    if np.nanmax(colors_f) <= 1.0:
        colors_f = colors_f * 255.0
    return np.nan_to_num(colors_f, nan=0.0).clip(0, 255).astype(np.uint8)


def _clip_sort_key(key: str) -> tuple[int, int]:
    start, end = key.split(":")
    return int(start), int(end)


def _quat_xyzw_to_matrix(quat: np.ndarray) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float64)
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    norm = np.maximum(norm, 1e-12)
    x, y, z, w = np.moveaxis(q / norm, -1, 0)

    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    rot = np.empty(q.shape[:-1] + (3, 3), dtype=np.float32)
    rot[..., 0, 0] = 1.0 - 2.0 * (yy + zz)
    rot[..., 0, 1] = 2.0 * (xy - wz)
    rot[..., 0, 2] = 2.0 * (xz + wy)
    rot[..., 1, 0] = 2.0 * (xy + wz)
    rot[..., 1, 1] = 1.0 - 2.0 * (xx + zz)
    rot[..., 1, 2] = 2.0 * (yz - wx)
    rot[..., 2, 0] = 2.0 * (xz - wy)
    rot[..., 2, 1] = 2.0 * (yz + wx)
    rot[..., 2, 2] = 1.0 - 2.0 * (xx + yy)
    return rot


def _transform_points(local_points: np.ndarray, pose_xyzw: np.ndarray) -> np.ndarray:
    points = np.asarray(local_points, dtype=np.float32)
    pose = np.asarray(pose_xyzw, dtype=np.float32)
    trans = pose[:, :3]
    rot = _quat_xyzw_to_matrix(pose[:, 3:7])
    return np.einsum("tij,nj->tni", rot, points) + trans[:, None, :]


def _build_segments(
    tracks: np.ndarray,
    track_colors: np.ndarray,
    *,
    line_point_stride: int,
) -> tuple[np.ndarray, np.ndarray]:
    line_point_stride = max(int(line_point_stride), 1)
    tracks = tracks[:, ::line_point_stride, :]
    track_colors = track_colors[::line_point_stride]
    t_count, n_count, _ = tracks.shape
    if t_count < 2 or n_count == 0:
        return (
            np.empty((0, 2, 3), dtype=np.float32),
            np.empty((0, 2, 3), dtype=np.uint8),
        )

    segment_grid = np.stack([tracks[:-1], tracks[1:]], axis=2)  # (T-1,N,2,3)
    valid = np.isfinite(segment_grid).all(axis=(2, 3))
    if not np.any(valid):
        return (
            np.empty((0, 2, 3), dtype=np.float32),
            np.empty((0, 2, 3), dtype=np.uint8),
        )

    if track_colors.ndim == 3:
        base_colors = _first_finite_points(tracks, track_colors)[1]
        if base_colors.shape[0] != n_count:
            base_colors = _normalize_colors(track_colors[0])
    else:
        base_colors = _normalize_colors(track_colors)
    base_colors = np.maximum(base_colors.astype(np.float32), 40.0)
    brightness = np.linspace(0.35, 1.0, t_count - 1, dtype=np.float32)[:, None, None]
    color_grid = np.broadcast_to(base_colors[None, :, :], (t_count - 1, n_count, 3))
    color_grid = np.clip(color_grid * brightness, 0, 255).astype(np.uint8)
    color_grid = np.stack([color_grid, color_grid], axis=2)
    return segment_grid[valid].astype(np.float32), color_grid[valid].astype(np.uint8)


def _build_flow_window_from_tracks(
    tracks: np.ndarray,
    track_colors: np.ndarray,
    frame: int,
    *,
    flow_window: int,
    line_point_stride: int,
) -> tuple[np.ndarray, np.ndarray]:
    frame = int(np.clip(frame, 0, tracks.shape[0] - 1))
    flow_window = int(flow_window)
    line_point_stride = max(int(line_point_stride), 1)
    if flow_window <= 0:
        return (
            np.empty((0, 2, 3), dtype=np.float32),
            np.empty((0, 2, 3), dtype=np.uint8),
        )
    start = max(0, frame - flow_window)
    end = min(tracks.shape[0] - 1, frame + flow_window)
    if end <= start:
        return (
            np.empty((0, 2, 3), dtype=np.float32),
            np.empty((0, 2, 3), dtype=np.uint8),
        )

    tracks_view = tracks[:, ::line_point_stride, :]
    if track_colors.ndim == 3:
        colors_view = track_colors[:, ::line_point_stride, :]
    else:
        colors_view = track_colors[::line_point_stride]

    segments = []
    colors = []
    for t in range(start, end):
        seg = np.stack([tracks_view[t], tracks_view[t + 1]], axis=1)
        valid = np.isfinite(seg).all(axis=(1, 2))
        if not np.any(valid):
            continue
        if colors_view.ndim == 3:
            color = _normalize_colors(colors_view[t])
        else:
            color = _normalize_colors(colors_view)
        color = np.stack([color, color], axis=1)
        segments.append(seg[valid])
        colors.append(color[valid])

    if not segments:
        return (
            np.empty((0, 2, 3), dtype=np.float32),
            np.empty((0, 2, 3), dtype=np.uint8),
        )
    return np.concatenate(segments, axis=0).astype(np.float32), np.concatenate(colors, axis=0).astype(np.uint8)


def _selected_camera_names(clip: h5py.Group, camera: str) -> list[str]:
    if camera == "all":
        names = sorted(name for name in clip.keys() if name.startswith("camera_"))
    else:
        names = [camera]
    missing = [name for name in names if name not in clip]
    if missing:
        raise KeyError(f"Camera(s) not found: {missing}")
    return names


def _selected_mesh_names(cam: h5py.Group, mesh: str) -> list[str]:
    all_meshes = sorted(cam["local_scene_points"].keys())
    if mesh == "all":
        return all_meshes
    if mesh == "first":
        return all_meshes[:1]
    if mesh not in cam["local_scene_points"]:
        raise KeyError(f"Mesh {mesh!r} not found. Available: {all_meshes[:10]}")
    return [mesh]


def _downsample_background(points: np.ndarray, colors: np.ndarray, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    if points.shape[0] > max_points:
        stride = int(np.ceil(points.shape[0] / max_points))
        points = points[::stride]
        colors = colors[::stride]
    colors = _normalize_colors(colors)
    return points.astype(np.float32), (colors.astype(np.float32) * 0.35).astype(np.uint8)


def _first_finite_points(tracks: np.ndarray, colors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    finite = np.isfinite(tracks).all(axis=-1)
    valid = finite.any(axis=0)
    if not np.any(valid):
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)
    first = finite.argmax(axis=0)
    point_ids = np.nonzero(valid)[0]
    if colors.ndim == 3:
        point_colors = colors[first[valid], point_ids]
    else:
        point_colors = colors[valid]
    return tracks[first[valid], point_ids].astype(np.float32), _normalize_colors(point_colors)


def _frame_points(tracks: np.ndarray, colors: np.ndarray, frame: int) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(tracks[frame]).all(axis=-1)
    if colors.ndim == 3:
        frame_colors = colors[frame, mask]
    else:
        frame_colors = colors[mask]
    return tracks[frame, mask].astype(np.float32), _normalize_colors(frame_colors)


def _track_displacements(tracks: np.ndarray) -> np.ndarray:
    finite = np.isfinite(tracks).all(axis=-1)
    valid = finite.any(axis=0)
    if not np.any(valid):
        return np.empty((0,), dtype=np.float32)
    first = finite.argmax(axis=0)
    last = tracks.shape[0] - 1 - finite[::-1].argmax(axis=0)
    point_ids = np.nonzero(valid)[0]
    disp = tracks[last[valid], point_ids] - tracks[first[valid], point_ids]
    return np.linalg.norm(disp, axis=-1)


def _load_clip_flow(
    hdf5_path: Path,
    clip_key: str,
    camera: str,
    mesh: str,
    point_slice: str,
    background_max_points: int,
) -> dict:
    with h5py.File(hdf5_path, "r") as f:
        if clip_key == "first":
            clip_key = sorted(f.keys(), key=_clip_sort_key)[0]
        clip = f[clip_key]
        flow_chunks = []
        color_chunks = []
        item_labels = []
        for cam_name in _selected_camera_names(clip, camera):
            cam = clip[cam_name]
            for mesh_name in _selected_mesh_names(cam, mesh):
                local = cam["local_scene_points"][mesh_name][:].astype(np.float32)
                traj = cam["scene_mesh_trajectories"][mesh_name][:].astype(np.float32)
                colors = cam["local_scene_colors"][mesh_name][:].astype(np.uint8)
                flow_chunks.append(_transform_points(local, traj))
                color_chunks.append(colors)
                item_labels.append(f"{cam_name}/{mesh_name}")

        all_flow = np.concatenate(flow_chunks, axis=1).astype(np.float32)
        all_colors = np.concatenate(color_chunks, axis=0).astype(np.uint8)
        point_indices = _point_indices(all_flow.shape[1], point_slice)
        tracks = all_flow[:, point_indices, :]
        track_colors = all_colors[point_indices]
        bg_points, bg_colors = _downsample_background(all_flow[0], all_colors, background_max_points)

    return {
        "sequence_mode": "clip",
        "clip_key": clip_key,
        "camera": camera,
        "mesh": mesh,
        "tracks": tracks.astype(np.float32),
        "track_colors": track_colors,
        "point_indices": point_indices,
        "background_points": bg_points.astype(np.float32),
        "background_colors": bg_colors,
        "item_count": len(item_labels),
        "frame_start": _clip_sort_key(clip_key)[0],
        "frame_end": _clip_sort_key(clip_key)[1],
    }


def _load_full_sequence_flow(
    hdf5_path: Path,
    camera: str,
    mesh: str,
    point_slice: str,
    background_max_points: int,
) -> dict:
    with h5py.File(hdf5_path, "r") as f:
        clips = [(start, end, key) for key in f.keys() for start, end in [_clip_sort_key(key)]]
        clips.sort()
        frame_start = clips[0][0]
        frame_end = max(end for _, end, _ in clips)
        global_t = frame_end - frame_start
        item_data: dict[tuple[str, str], dict[str, np.ndarray]] = {}

        for start, _end, clip_key in clips:
            clip = f[clip_key]
            for cam_name in _selected_camera_names(clip, camera):
                cam = clip[cam_name]
                for mesh_name in _selected_mesh_names(cam, mesh):
                    item_key = (cam_name, mesh_name)
                    traj = cam["scene_mesh_trajectories"][mesh_name][:].astype(np.float32)
                    if item_key not in item_data:
                        item_data[item_key] = {
                            "local": cam["local_scene_points"][mesh_name][:].astype(np.float32),
                            "colors": cam["local_scene_colors"][mesh_name][:].astype(np.uint8),
                            "poses": np.full((global_t, 7), np.nan, dtype=np.float32),
                        }
                    poses = item_data[item_key]["poses"]
                    for i in range(traj.shape[0]):
                        global_i = start + i - frame_start
                        if 0 <= global_i < global_t:
                            poses[global_i] = traj[i]

        flow_chunks = []
        color_chunks = []
        for data in item_data.values():
            local = data["local"]
            poses = data["poses"]
            flow = np.full((global_t, local.shape[0], 3), np.nan, dtype=np.float32)
            valid = np.isfinite(poses).all(axis=-1)
            if np.any(valid):
                flow[valid] = _transform_points(local, poses[valid])
            flow_chunks.append(flow)
            color_chunks.append(data["colors"])

        all_flow = np.concatenate(flow_chunks, axis=1).astype(np.float32)
        all_colors = np.concatenate(color_chunks, axis=0).astype(np.uint8)
        point_indices = _point_indices(all_flow.shape[1], point_slice)
        tracks = all_flow[:, point_indices, :]
        track_colors = all_colors[point_indices]
        bg_points, bg_colors = _first_finite_points(tracks, track_colors)
        bg_points, bg_colors = _downsample_background(bg_points, bg_colors, background_max_points)

    return {
        "sequence_mode": "full",
        "clip_key": f"{frame_start}:{frame_end}",
        "camera": camera,
        "mesh": mesh,
        "tracks": tracks.astype(np.float32),
        "track_colors": track_colors,
        "point_indices": point_indices,
        "background_points": bg_points.astype(np.float32),
        "background_colors": bg_colors,
        "item_count": len(item_data),
        "frame_start": frame_start,
        "frame_end": frame_end,
    }


def _load_flow(
    hdf5_path: Path,
    sequence: str,
    clip_key: str,
    camera: str,
    mesh: str,
    point_slice: str,
    background_max_points: int,
) -> dict:
    if sequence == "clip":
        return _load_clip_flow(hdf5_path, clip_key, camera, mesh, point_slice, background_max_points)
    if sequence == "full":
        return _load_full_sequence_flow(hdf5_path, camera, mesh, point_slice, background_max_points)
    raise ValueError(f"Unknown sequence mode {sequence!r}")


def _detect_format(hdf5_path: Path) -> str:
    with h5py.File(hdf5_path, "r") as f:
        if "pointcloud" in f and getattr(f["pointcloud"], "ndim", 0) == 3:
            return "robotwin"
        return "behavior"


def _load_robotwin_pointcloud(
    hdf5_path: Path,
    point_slice: str,
    background_max_points: int,
) -> dict:
    with h5py.File(hdf5_path, "r") as f:
        pointcloud = f["pointcloud"]
        point_indices = _point_indices(pointcloud.shape[1], point_slice)
        first = pointcloud[0, point_indices, :].astype(np.float32)
        last = pointcloud[pointcloud.shape[0] - 1, point_indices, :].astype(np.float32)
        frame_count = int(pointcloud.shape[0])
    first_points = first[..., :3].astype(np.float32)
    first_colors = _normalize_colors(first[..., 3:6])
    displacement = np.linalg.norm(last[..., :3] - first[..., :3], axis=-1)
    bg_points, bg_colors = first_points, first_colors
    bg_points, bg_colors = _downsample_background(bg_points, bg_colors, background_max_points)
    return {
        "sequence_mode": "robotwin_full",
        "clip_key": f"0:{frame_count}",
        "camera": "pointcloud",
        "mesh": "point-index tracks",
        "lazy_robotwin": True,
        "hdf5_path": str(hdf5_path),
        "dataset_key": "pointcloud",
        "frame_count": frame_count,
        "point_count": int(point_indices.shape[0]),
        "initial_points": first_points,
        "initial_colors": first_colors,
        "displacement": displacement.astype(np.float32),
        "point_indices": point_indices,
        "background_points": bg_points,
        "background_colors": bg_colors,
        "item_count": 1,
        "frame_start": 0,
        "frame_end": frame_count,
    }


def _read_robotwin_frame(h5_file: h5py.File, sample: dict, frame: int) -> tuple[np.ndarray, np.ndarray]:
    frame = int(np.clip(frame, 0, sample["frame_count"] - 1))
    pc = h5_file[sample["dataset_key"]][frame, sample["point_indices"], :].astype(np.float32)
    return pc[..., :3].astype(np.float32), _normalize_colors(pc[..., 3:6])


def _read_robotwin_flow_window(
    h5_file: h5py.File,
    sample: dict,
    frame: int,
    *,
    flow_window: int,
    line_point_stride: int,
) -> tuple[np.ndarray, np.ndarray]:
    frame = int(np.clip(frame, 0, sample["frame_count"] - 1))
    flow_window = int(flow_window)
    line_point_stride = max(int(line_point_stride), 1)
    if flow_window <= 0:
        return (
            np.empty((0, 2, 3), dtype=np.float32),
            np.empty((0, 2, 3), dtype=np.uint8),
        )
    start = max(0, frame - flow_window)
    end = min(sample["frame_count"] - 1, frame + flow_window)
    if end <= start:
        return (
            np.empty((0, 2, 3), dtype=np.float32),
            np.empty((0, 2, 3), dtype=np.uint8),
        )

    idx = sample["point_indices"][::line_point_stride]
    segments = []
    colors = []
    for t in range(start, end):
        pc0 = h5_file[sample["dataset_key"]][t, idx, :].astype(np.float32)
        pc1 = h5_file[sample["dataset_key"]][t + 1, idx, :].astype(np.float32)
        seg = np.stack([pc0[..., :3], pc1[..., :3]], axis=1)
        valid = np.isfinite(seg).all(axis=(1, 2))
        if not np.any(valid):
            continue
        color = _normalize_colors(pc0[..., 3:6])
        color = np.stack([color, color], axis=1)
        segments.append(seg[valid])
        colors.append(color[valid])
    if not segments:
        return (
            np.empty((0, 2, 3), dtype=np.float32),
            np.empty((0, 2, 3), dtype=np.uint8),
        )
    return np.concatenate(segments, axis=0).astype(np.float32), np.concatenate(colors, axis=0).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdf5", type=Path, default=Path(DEFAULT_HDF5))
    parser.add_argument("--format", choices=("auto", "behavior", "robotwin"), default="auto")
    parser.add_argument("--sequence", choices=("clip", "full"), default="clip")
    parser.add_argument("--clip", default=DEFAULT_CLIP)
    parser.add_argument("--camera", default=DEFAULT_CAMERA, help="Camera name, or 'all'.")
    parser.add_argument("--mesh", default=DEFAULT_MESH, help="Mesh name, 'first', or 'all'.")
    parser.add_argument("--point-slice", default="0:100:10")
    parser.add_argument("--background-max-points", type=int, default=12000)
    parser.add_argument("--line-point-stride", type=int, default=1)
    parser.add_argument(
        "--flow-window",
        type=int,
        default=1,
        help="For RoboTwin pointcloud mode, draw frame-to-frame flow within +/- this many frames.",
    )
    parser.add_argument("--init-only", action="store_true", help="Render only the initial frame with no flow lines.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    data_format = _detect_format(args.hdf5) if args.format == "auto" else args.format
    if data_format == "robotwin":
        sample = _load_robotwin_pointcloud(args.hdf5, args.point_slice, args.background_max_points)
    else:
        sample = _load_flow(
            args.hdf5,
            args.sequence,
            args.clip,
            args.camera,
            args.mesh,
            args.point_slice,
            args.background_max_points,
        )
    tracks = sample.get("tracks")
    displacement = sample["displacement"] if sample.get("lazy_robotwin") else _track_displacements(tracks)

    print(f"hdf5: {args.hdf5}", flush=True)
    print(f"format: {data_format}", flush=True)
    print(f"sequence/camera/mesh: {sample['sequence_mode']} / {sample['camera']} / {sample['mesh']}", flush=True)
    frame_count = sample["frame_count"] if sample.get("lazy_robotwin") else tracks.shape[0]
    point_count = sample["point_count"] if sample.get("lazy_robotwin") else tracks.shape[1]
    print(f"frame range: {sample['frame_start']}:{sample['frame_end']} ({frame_count} frames)", flush=True)
    print(f"tracked item count: {sample['item_count']}", flush=True)
    if sample.get("lazy_robotwin"):
        print(f"tracked tensor shape: ({frame_count}, {point_count}, 3) [streamed from HDF5]", flush=True)
    else:
        print(f"tracked tensor shape: {tracks.shape}", flush=True)
    if sample["point_indices"].shape[0] <= 50:
        print(f"tracked point indices: {sample['point_indices'].tolist()}", flush=True)
    else:
        print(
            "tracked point indices: "
            f"{sample['point_indices'][0]}..{sample['point_indices'][-1]} "
            f"({sample['point_indices'].shape[0]} total)",
            flush=True,
        )
    print(
        "tracked displacement: "
        f"min={displacement.min():.5f} mean={displacement.mean():.5f} max={displacement.max():.5f}",
        flush=True,
    )
    if args.dry_run:
        return

    server = viser.ViserServer(host=args.host, port=args.port)
    server.gui.add_markdown(
        "\n".join(
            [
                "### PointWorld Raw HDF5 Flow",
                f"`{args.hdf5}`",
                f"{sample['sequence_mode']} frames `{sample['clip_key']}`, camera `{sample['camera']}`",
                f"mesh `{sample['mesh']}`",
                (
                    f"track tensor `({frame_count}, {point_count}, 3)` streamed from HDF5"
                    if sample.get("lazy_robotwin")
                    else f"track tensor `{tuple(tracks.shape)}` from point slice `{args.point_slice}`"
                ),
            ]
        )
    )

    server.scene.world_axes.visible = True
    server.scene.add_point_cloud(
        "/background/t0_all_camera_points",
        points=sample["background_points"],
        colors=sample["background_colors"],
        point_size=0.004,
        point_shape="rounded",
        precision="float32",
    )
    robotwin_h5 = None
    flow_window = 0 if args.init_only else args.flow_window

    if sample.get("lazy_robotwin"):
        robotwin_h5 = h5py.File(sample["hdf5_path"], "r")
        current_points, current_colors = _read_robotwin_frame(robotwin_h5, sample, 0)
        segments, segment_colors = _read_robotwin_flow_window(
            robotwin_h5,
            sample,
            0,
            flow_window=flow_window,
            line_point_stride=args.line_point_stride,
        )
    else:
        current_points, current_colors = _frame_points(tracks, sample["track_colors"], 0)
        segments, segment_colors = _build_flow_window_from_tracks(
            tracks,
            sample["track_colors"],
            0,
            flow_window=flow_window,
            line_point_stride=args.line_point_stride,
        )
    flow_handle = server.scene.add_line_segments(
        "/tracked_points/flow_lines",
        points=segments,
        colors=segment_colors,
        line_width=6.0,
    )
    track_handle = server.scene.add_point_cloud(
        "/tracked_points/current",
        points=current_points,
        colors=current_colors,
        point_size=0.01 if point_count > 1000 else 0.03,
        point_shape="sparkle",
        precision="float32",
    )

    if not args.init_only:
        slider = server.gui.add_slider("Frame", min=0, max=frame_count - 1, step=1, initial_value=0)

        def _on_frame(event) -> None:
            frame = int(event.target.value)
            if sample.get("lazy_robotwin"):
                assert robotwin_h5 is not None
                current_points, current_colors = _read_robotwin_frame(robotwin_h5, sample, frame)
                segments, segment_colors = _read_robotwin_flow_window(
                    robotwin_h5,
                    sample,
                    frame,
                    flow_window=flow_window,
                    line_point_stride=args.line_point_stride,
                )
                flow_handle.points = segments
                flow_handle.colors = segment_colors
            else:
                current_points, current_colors = _frame_points(tracks, sample["track_colors"], frame)
                segments, segment_colors = _build_flow_window_from_tracks(
                    tracks,
                    sample["track_colors"],
                    frame,
                    flow_window=flow_window,
                    line_point_stride=args.line_point_stride,
                )
                flow_handle.points = segments
                flow_handle.colors = segment_colors
            track_handle.points = current_points
            track_handle.colors = current_colors

        slider.on_update(_on_frame)
    print(f"Viser running at http://{args.host}:{args.port}", flush=True)
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
