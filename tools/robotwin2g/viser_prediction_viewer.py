#!/usr/bin/env python3
"""Interactive Viser viewer for RoboTwin2G PointWorld predictions."""

from __future__ import annotations

import argparse
import time
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import h5py
import numpy as np
import torch
import viser
from torch.utils.data import DataLoader

from tools.robotwin2g.render_flow_videos import _load_model
from tools.robotwin2g.train_robotwin_action import to_device
from tools.robotwin2g.wds_dataset import RobotWinWDSDataset, robotwin_action_collate


def _as_rgb_u8(colors: np.ndarray) -> np.ndarray:
    colors = np.asarray(colors)
    if colors.dtype == np.uint8:
        return colors[..., :3]
    colors_f = colors.astype(np.float32)
    if colors_f.size and np.nanmax(colors_f) <= 1.5:
        colors_f = colors_f * 255.0
    return np.nan_to_num(colors_f, nan=0.0).clip(0, 255).astype(np.uint8)[..., :3]


def _turbo(values: np.ndarray, vmax: Optional[float] = None) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if vmax is None:
        vmax = float(np.percentile(values, 98)) if values.size else 1.0
    vmax = max(float(vmax), 1e-6)
    lut = np.round(np.clip(values / vmax, 0.0, 1.0) * 255).astype(np.uint8).reshape(-1, 1)
    bgr = cv2.applyColorMap(lut, cv2.COLORMAP_TURBO).reshape(-1, 3)
    return bgr[:, ::-1].copy()


def _take_sample(loader: DataLoader, sample_index: int) -> Dict[str, Any]:
    for i, batch in enumerate(loader):
        if i == sample_index:
            return batch
    raise IndexError(f"Sample index {sample_index} is out of range")


def _safe_key(batch: Dict[str, Any], sample_index: int) -> str:
    key = batch.get("__key__", [f"sample_{sample_index:03d}"])
    if isinstance(key, (list, tuple)):
        key = key[0]
    if isinstance(key, bytes):
        key = key.decode("utf-8", errors="replace")
    return str(key)


def _load_raw_observed(meta: Dict[str, Any], shift_amount: np.ndarray, point_count: int) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    source = meta.get("source_path")
    if not source:
        return None
    path = Path(source)
    if not path.is_file():
        return None
    start = int(meta.get("source_start", 0))
    end = int(meta.get("source_end", start + 1))
    with h5py.File(path, "r") as f:
        if "pointcloud" not in f:
            return None
        raw = np.asarray(f["pointcloud"][start:end], dtype=np.float32)
    if raw.shape[1] > point_count:
        raw = raw[:, :point_count]
    raw_xyz = raw[..., :3] + shift_amount.reshape(1, 1, 3)
    raw_rgb = _as_rgb_u8(raw[..., 3:6])
    return raw_xyz.astype(np.float32), raw_rgb


def _line_segments(source: np.ndarray, dest: np.ndarray, idx: np.ndarray) -> np.ndarray:
    return np.stack([source[idx], dest[idx]], axis=1).astype(np.float32)


def main() -> None:
    args = build_argparser().parse_args()
    model, device, feature_dims = _load_model(args)

    ds = RobotWinWDSDataset(
        args.wds_root,
        args.split,
        shuffle=False,
        seed=args.seed,
        target_scene_features_dim=feature_dims.scene_features_dim,
        target_robot_features_dim=feature_dims.robot_features_dim,
        shuffle_buffer=0,
    )
    loader = DataLoader(ds, batch_size=1, num_workers=0, collate_fn=robotwin_action_collate)
    batch_cpu = _take_sample(loader, args.sample_index)
    key = _safe_key(batch_cpu, args.sample_index)
    meta = batch_cpu.get("metadata", [{}])[0] if isinstance(batch_cpu.get("metadata"), list) else {}

    batch = to_device(batch_cpu, device)
    with torch.no_grad(), torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        enabled=args.amp and device.type == "cuda",
    ):
        outputs = model.world_model(batch, training=False)

    pred = outputs["scene_flows"][0].detach().float().cpu().numpy()
    wds = batch_cpu["scene_flows"][0].float().numpy()
    scene_exists = batch_cpu["scene_exists"][0].numpy().astype(bool)
    colors = _as_rgb_u8(batch_cpu["scene_colors"][0].numpy())
    robot = batch_cpu["robot_flows"][0].float().numpy()
    robot_exists = batch_cpu["robot_exists"][0].numpy().astype(bool)
    shift = batch_cpu.get("__shift_amount__", torch.zeros((1, 3)))[0].float().numpy()
    raw = _load_raw_observed(meta, shift, wds.shape[1]) if args.include_raw else None

    valid = np.flatnonzero(scene_exists[0])
    if valid.shape[0] > args.max_points:
        valid = valid[:: int(np.ceil(valid.shape[0] / args.max_points))]
    line_idx = valid[:: max(1, args.line_point_stride)]
    T = int(pred.shape[0])

    disp = np.linalg.norm(pred - wds[:1], axis=-1)
    disp_vmax = float(np.percentile(disp[:, valid], 98)) if valid.size else 1.0
    pred_colors = [_turbo(disp[t, valid], disp_vmax) for t in range(T)]

    print(f"sample: {key}", flush=True)
    print(f"metadata scene_flow_mode: {meta.get('scene_flow_mode')}", flush=True)
    print(f"source: {meta.get('source_path')} frames {meta.get('source_start')}:{meta.get('source_end')}", flush=True)
    print(f"WDS input displacement from t0: mean={np.linalg.norm(wds - wds[:1], axis=-1).mean():.6f}", flush=True)
    print(f"prediction displacement: mean={disp.mean():.6f} max={disp.max():.6f}", flush=True)
    if raw is not None:
        raw_xyz, _raw_rgb = raw
        raw_disp = np.linalg.norm(raw_xyz - raw_xyz[:1], axis=-1)
        print(f"raw observed displacement by index: mean={raw_disp.mean():.6f} max={raw_disp.max():.6f}", flush=True)

    server = viser.ViserServer(host=args.host, port=args.port)
    server.scene.world_axes.visible = True
    server.gui.add_markdown(
        "\n".join(
            [
                "### RoboTwin2G PointWorld Prediction",
                f"sample `{key}`",
                f"WDS scene_flow_mode `{meta.get('scene_flow_mode')}`",
                "`WDS input/static` is not GT when mode is `repeat_t0`.",
                "`PointWorld prediction` and `Predicted flow` are model output.",
                "`Raw observed` is source HDF5 pointcloud, by-index only; not guaranteed GT flow.",
            ]
        )
    )

    server.scene.add_point_cloud(
        "/wds/input_static_not_gt",
        points=wds[0, valid],
        colors=(colors[0, valid].astype(np.float32) * 0.35).astype(np.uint8),
        point_size=args.point_size * 0.8,
        point_shape="rounded",
        precision="float32",
    )
    pred_handle = server.scene.add_point_cloud(
        "/pointworld/prediction_current",
        points=pred[0, valid],
        colors=pred_colors[0],
        point_size=args.point_size,
        point_shape="sparkle",
        precision="float32",
    )
    seg0 = _line_segments(wds[0], pred[0], line_idx)
    seg_color0 = np.stack([_turbo(disp[0, line_idx], disp_vmax), _turbo(disp[0, line_idx], disp_vmax)], axis=1)
    flow_handle = server.scene.add_line_segments(
        "/pointworld/predicted_flow_from_t0",
        points=seg0,
        colors=seg_color0,
        line_width=args.line_width,
    )

    robot_idx = np.flatnonzero(robot_exists[0])
    robot_handle = server.scene.add_point_cloud(
        "/robot/current",
        points=robot[0, robot_idx],
        colors=np.tile(np.array([[230, 230, 230]], dtype=np.uint8), (robot_idx.shape[0], 1)),
        point_size=args.point_size * 0.9,
        point_shape="rounded",
        precision="float32",
    )

    raw_handle = None
    if raw is not None:
        raw_xyz, raw_rgb = raw
        raw_valid = valid[valid < raw_xyz.shape[1]]
        raw_handle = server.scene.add_point_cloud(
            "/raw_observed/source_pointcloud_by_index",
            points=raw_xyz[0, raw_valid],
            colors=raw_rgb[0, raw_valid],
            point_size=args.point_size * 0.75,
            point_shape="rounded",
            precision="float32",
        )

    slider = server.gui.add_slider("Frame", min=0, max=T - 1, step=1, initial_value=0)

    def _on_frame(event) -> None:
        frame = int(event.target.value)
        pred_handle.points = pred[frame, valid]
        pred_handle.colors = pred_colors[frame]
        seg = _line_segments(wds[0], pred[frame], line_idx)
        seg_color = _turbo(disp[frame, line_idx], disp_vmax)
        flow_handle.points = seg
        flow_handle.colors = np.stack([seg_color, seg_color], axis=1)
        rmask = robot_exists[frame, robot_idx]
        robot_handle.points = robot[frame, robot_idx[rmask]]
        robot_handle.colors = np.tile(np.array([[230, 230, 230]], dtype=np.uint8), (int(rmask.sum()), 1))
        if raw_handle is not None and raw is not None:
            raw_xyz, raw_rgb = raw
            raw_frame = min(frame, raw_xyz.shape[0] - 1)
            raw_valid = valid[valid < raw_xyz.shape[1]]
            raw_handle.points = raw_xyz[raw_frame, raw_valid]
            raw_handle.colors = raw_rgb[raw_frame, raw_valid]

    slider.on_update(_on_frame)
    print(f"Viser prediction viewer running at http://{args.host}:{args.port}", flush=True)
    while True:
        time.sleep(1.0)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--wds-root", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--pretrained-checkpoint", default="pretrained_checkpoints/large-droid+behavior/model-best.pt")
    p.add_argument("--split", default="test", choices=["train", "test"])
    p.add_argument("--sample-index", type=int, default=0)
    p.add_argument("--max-points", type=int, default=2048)
    p.add_argument("--line-point-stride", type=int, default=8)
    p.add_argument("--point-size", type=float, default=0.01)
    p.add_argument("--line-width", type=float, default=3.0)
    p.add_argument("--include-raw", action="store_true")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8091)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    p.add_argument("--amp", action="store_true")

    p.add_argument("--norm-stats-path", default="stats/droid_behavior")
    p.add_argument("--ptv3-size", default="base")
    p.add_argument("--predictor-dim", type=int, default=256)
    p.add_argument("--grid-size", type=float, default=0.015)
    p.add_argument("--depth-threshold", type=float, default=0.003)
    p.add_argument("--clip-horizon", type=int, default=11)
    p.add_argument("--decoder-hidden-dim", type=int, default=512)
    p.add_argument("--decoder-layers", type=int, default=3)
    return p


if __name__ == "__main__":
    main()
