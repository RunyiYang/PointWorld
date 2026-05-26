#!/usr/bin/env python3
"""Render RoboTwin2G PointWorld scene-flow prediction videos.

This is a headless visualization path for Slurm jobs. It renders input/static
RoboTwin scene trajectories, PointWorld-predicted scene trajectories, and an
overlay with pointwise predicted motion from the first frame.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from tools.robotwin2g.action_model import (
    PointWorldActionModel,
    build_pointworld_args,
    infer_feature_dims_from_checkpoint,
)
from tools.robotwin2g.train_robotwin_action import load_action_stats, load_torch, to_device
from tools.robotwin2g.wds_dataset import RobotWinWDSDataset, robotwin_action_collate


def _safe_name(value: Any, fallback: str) -> str:
    if isinstance(value, (list, tuple)) and value:
        value = value[0]
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    text = str(value) if value else fallback
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")
    return text or fallback


def _load_model(args: argparse.Namespace) -> Tuple[PointWorldActionModel, torch.device, Any]:
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    payload = load_torch(checkpoint_path, map_location="cpu")
    pretrained_payload = None
    if args.pretrained_checkpoint:
        pretrained_path = Path(args.pretrained_checkpoint)
        if not pretrained_path.is_file():
            raise FileNotFoundError(f"Missing pretrained checkpoint: {pretrained_path}")
        pretrained_payload = load_torch(pretrained_path, map_location="cpu")

    dims_source = pretrained_payload if pretrained_payload is not None else payload
    feature_dims = infer_feature_dims_from_checkpoint(dims_source)
    ckpt_args = payload.get("args", {})

    if "action_mean" in payload:
        action_mean = torch.as_tensor(payload["action_mean"], dtype=torch.float32)
        action_std = torch.as_tensor(payload["action_std"], dtype=torch.float32)
    else:
        action_mean, action_std = load_action_stats(args.wds_root, torch.device("cpu"))
    action_dim = int(action_mean.numel())
    clip_horizon = int(ckpt_args.get("clip_horizon", args.clip_horizon))

    pw_args = build_pointworld_args(
        checkpoint=pretrained_payload,
        device=str(device),
        norm_stats_path=str(ckpt_args.get("norm_stats_path", args.norm_stats_path)),
        ptv3_size=str(ckpt_args.get("ptv3_size", args.ptv3_size)),
        predictor_dim=int(ckpt_args.get("predictor_dim", args.predictor_dim)),
        disable_compile=bool(ckpt_args.get("disable_compile", True)),
        grid_size=float(ckpt_args.get("grid_size", args.grid_size)),
        depth_threshold=float(ckpt_args.get("depth_threshold", args.depth_threshold)),
    )
    data_info = {
        "scene_features_dim": feature_dims.scene_features_dim,
        "robot_features_dim": feature_dims.robot_features_dim,
    }
    model = PointWorldActionModel(
        pw_args,
        data_info,
        action_dim=action_dim,
        action_horizon=clip_horizon - 1,
        state_dim=action_dim,
        decoder_hidden_dim=int(ckpt_args.get("decoder_hidden_dim", args.decoder_hidden_dim)),
        decoder_layers=int(ckpt_args.get("decoder_layers", args.decoder_layers)),
    )

    if int(ckpt_args.get("lora_rank", 0)) > 0:
        names = model.enable_lora(
            rank=int(ckpt_args.get("lora_rank", 0)),
            alpha=float(ckpt_args.get("lora_alpha", 16.0)),
            dropout=float(ckpt_args.get("lora_dropout", 0.0)),
            target_patterns=[p for p in str(ckpt_args.get("lora_targets", "")).split(",") if p],
            exclude_patterns=[p for p in str(ckpt_args.get("lora_exclude", "")).split(",") if p],
            include_dinov3=bool(ckpt_args.get("lora_include_dinov3", False)),
        )
        print(f"[lora] injected {len(names)} adapters for checkpoint load", flush=True)

    if pretrained_payload is not None:
        missing, unexpected = model.load_pointworld_checkpoint(pretrained_payload, strict=False)
        print(f"[pretrained] loaded PointWorld: missing={len(missing)} unexpected={len(unexpected)}", flush=True)

    state = payload.get("model", payload)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[checkpoint] missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    print(
        f"[checkpoint] step={payload.get('step', 'unknown')} best_metric={payload.get('best_metric', 'unknown')}",
        flush=True,
    )

    model.to(device)
    model.eval()
    return model, device, feature_dims


def _rotation(elev_deg: float, azim_deg: float) -> np.ndarray:
    elev = math.radians(elev_deg)
    azim = math.radians(azim_deg)
    rz = np.array(
        [
            [math.cos(azim), -math.sin(azim), 0.0],
            [math.sin(azim), math.cos(azim), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    rx = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, math.cos(elev), -math.sin(elev)],
            [0.0, math.sin(elev), math.cos(elev)],
        ],
        dtype=np.float32,
    )
    return rx @ rz


def _project(points: np.ndarray, rot: np.ndarray, center: np.ndarray, scale: float, width: int, height: int) -> Tuple[np.ndarray, np.ndarray]:
    rotated = points @ rot.T
    xy = rotated[:, :2]
    pix = np.empty((points.shape[0], 2), dtype=np.int32)
    pix[:, 0] = np.round(width * 0.5 + (xy[:, 0] - center[0]) * scale).astype(np.int32)
    pix[:, 1] = np.round(height * 0.5 - (xy[:, 1] - center[1]) * scale).astype(np.int32)
    return pix, rotated[:, 2]


def _value_colors(values: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin + 1e-8:
        vmax = float(np.nanmax(values)) if values.size else 1.0
        vmin = float(np.nanmin(values)) if values.size else 0.0
        if vmax <= vmin + 1e-8:
            vmax = vmin + 1.0
    norm = np.clip((values - vmin) / (vmax - vmin), 0.0, 1.0)
    lut = np.round(norm * 255).astype(np.uint8).reshape(-1, 1)
    return cv2.applyColorMap(lut, cv2.COLORMAP_TURBO).reshape(-1, 3)


def _draw_axes(img: np.ndarray, rot: np.ndarray, origin: Tuple[int, int]) -> None:
    axes = [
        (np.array([[0.0, 0.0, 0.0], [0.12, 0.0, 0.0]], dtype=np.float32), (0, 0, 255), "X"),
        (np.array([[0.0, 0.0, 0.0], [0.0, 0.12, 0.0]], dtype=np.float32), (0, 210, 0), "Y"),
        (np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.12]], dtype=np.float32), (255, 90, 0), "Z"),
    ]
    base = np.array(origin, dtype=np.int32)
    for points, color, label in axes:
        rotated = points @ rot.T
        delta = np.round(rotated[1, :2] * np.array([230.0, -230.0], dtype=np.float32)).astype(np.int32)
        end = tuple((base + delta).tolist())
        cv2.arrowedLine(img, tuple(base.tolist()), end, color, 2, tipLength=0.25)
        cv2.putText(img, label, (end[0] + 4, end[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def _draw_points(
    img: np.ndarray,
    points: np.ndarray,
    values: np.ndarray,
    *,
    rot: np.ndarray,
    center: np.ndarray,
    scale: float,
    color_range: Tuple[float, float],
    radius: int,
    alpha: float = 1.0,
) -> np.ndarray:
    if points.size == 0:
        return np.empty((0, 2), dtype=np.int32)
    pix, depth = _project(points, rot, center, scale, img.shape[1], img.shape[0])
    colors = _value_colors(values, color_range[0], color_range[1])
    order = np.argsort(depth)
    for idx in order:
        x, y = pix[idx]
        if x < 0 or y < 0 or x >= img.shape[1] or y >= img.shape[0]:
            continue
        color = tuple(int(c) for c in colors[idx])
        if alpha >= 0.999:
            cv2.circle(img, (int(x), int(y)), radius, color, -1, cv2.LINE_AA)
        else:
            overlay = img.copy()
            cv2.circle(overlay, (int(x), int(y)), radius, color, -1, cv2.LINE_AA)
            cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0.0, img)
    return pix


def _draw_robot(
    img: np.ndarray,
    points: np.ndarray,
    *,
    rot: np.ndarray,
    center: np.ndarray,
    scale: float,
) -> None:
    if points.size == 0:
        return
    pix, depth = _project(points, rot, center, scale, img.shape[1], img.shape[0])
    order = np.argsort(depth)
    for idx in order:
        x, y = pix[idx]
        if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
            cv2.circle(img, (int(x), int(y)), 2, (235, 235, 235), -1, cv2.LINE_AA)
            cv2.circle(img, (int(x), int(y)), 2, (40, 40, 40), 1, cv2.LINE_AA)


def _panel(title: str, subtitle: str, width: int, height: int) -> np.ndarray:
    img = np.full((height, width, 3), 18, dtype=np.uint8)
    cv2.putText(img, title, (18, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (245, 245, 245), 2, cv2.LINE_AA)
    cv2.putText(img, subtitle, (18, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (190, 190, 190), 1, cv2.LINE_AA)
    return img


def _scene_bounds(*arrays: np.ndarray, rot: np.ndarray, panel_w: int, panel_h: int) -> Tuple[np.ndarray, float]:
    pts = [a.reshape(-1, 3) for a in arrays if a.size]
    if not pts:
        return np.zeros(2, dtype=np.float32), 1.0
    all_points = np.concatenate(pts, axis=0)
    all_points = all_points[np.isfinite(all_points).all(axis=1)]
    if all_points.size == 0:
        return np.zeros(2, dtype=np.float32), 1.0
    rotated = all_points @ rot.T
    xy = rotated[:, :2]
    mins = xy.min(axis=0)
    maxs = xy.max(axis=0)
    center = (mins + maxs) * 0.5
    span = float(np.max(maxs - mins))
    span = max(span, 1e-3)
    scale = 0.82 * float(min(panel_w, panel_h)) / span
    return center.astype(np.float32), scale


def _render_video(
    out_path: Path,
    *,
    key: str,
    pred: np.ndarray,
    target: np.ndarray,
    scene_exists: np.ndarray,
    robot: np.ndarray,
    robot_exists: np.ndarray,
    max_points: int,
    max_robot_points: int,
    fps: float,
    hold_final: int,
    seed: int,
    elev: float,
    azim: float,
    left_title: str = "WDS input, not GT",
    left_subtitle: str = "repeat-t0 scene tensor",
    middle_title: str = "PointWorld prediction",
    middle_subtitle: str = "predicted scene trajectory",
    right_title: str = "Predicted flow",
    right_subtitle: str = "gray t0 to colored predicted t",
    top_note: str = "Color: displacement from frame 0. Robot points are white/gray.",
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    valid_scene = np.flatnonzero(scene_exists[0].astype(bool))
    if valid_scene.size == 0:
        raise ValueError(f"Sample {key} has no valid scene points")
    if valid_scene.size > max_points:
        scene_idx = rng.choice(valid_scene, size=max_points, replace=False)
        scene_idx.sort()
    else:
        scene_idx = valid_scene

    valid_robot = np.flatnonzero(robot_exists[0].astype(bool))
    if valid_robot.size > max_robot_points:
        robot_idx = rng.choice(valid_robot, size=max_robot_points, replace=False)
        robot_idx.sort()
    else:
        robot_idx = valid_robot

    pred_s = np.nan_to_num(pred[:, scene_idx], nan=0.0, posinf=0.0, neginf=0.0)
    target_s = np.nan_to_num(target[:, scene_idx], nan=0.0, posinf=0.0, neginf=0.0)
    robot_s = np.nan_to_num(robot[:, robot_idx], nan=0.0, posinf=0.0, neginf=0.0) if robot_idx.size else np.empty((pred.shape[0], 0, 3), dtype=np.float32)
    robot_mask_s = robot_exists[:, robot_idx].astype(bool) if robot_idx.size else np.zeros((pred.shape[0], 0), dtype=bool)

    rot = _rotation(elev, azim)
    panel_w, panel_h = 520, 520
    margin, header = 22, 76
    frame_w = panel_w * 3 + margin * 4
    frame_h = panel_h + header + margin
    center, scale = _scene_bounds(pred_s, target_s, robot_s, rot=rot, panel_w=panel_w, panel_h=panel_h)

    pred_motion = np.linalg.norm(pred_s - target_s[:1], axis=-1)
    target_motion = np.linalg.norm(target_s - target_s[:1], axis=-1)
    color_max = float(max(np.percentile(pred_motion, 98), np.percentile(target_motion, 98), 1e-4))
    color_range = (0.0, color_max)

    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (frame_w, frame_h),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {out_path}")

    line_subset = np.linspace(0, scene_idx.size - 1, min(scene_idx.size, 256), dtype=np.int64)
    initial_points = target_s[0]
    first_pred_l2 = np.linalg.norm(pred_s - target_s, axis=-1)

    try:
        for t in range(pred_s.shape[0] + max(0, hold_final)):
            ti = min(t, pred_s.shape[0] - 1)
            canvas = np.full((frame_h, frame_w, 3), 10, dtype=np.uint8)
            cv2.putText(
                canvas,
                f"{key} | frame {ti:02d}/{pred_s.shape[0] - 1:02d}",
                (margin, 38),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.78,
                (245, 245, 245),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                top_note,
                (margin, 64),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (178, 178, 178),
                1,
                cv2.LINE_AA,
            )

            panels = [
                _panel(left_title, left_subtitle, panel_w, panel_h),
                _panel(middle_title, middle_subtitle, panel_w, panel_h),
                _panel(right_title, right_subtitle, panel_w, panel_h),
            ]

            _draw_points(
                panels[0],
                target_s[ti],
                target_motion[ti],
                rot=rot,
                center=center,
                scale=scale,
                color_range=color_range,
                radius=2,
            )
            _draw_robot(panels[0], robot_s[ti, robot_mask_s[ti]], rot=rot, center=center, scale=scale)

            _draw_points(
                panels[1],
                pred_s[ti],
                pred_motion[ti],
                rot=rot,
                center=center,
                scale=scale,
                color_range=color_range,
                radius=2,
            )
            _draw_robot(panels[1], robot_s[ti, robot_mask_s[ti]], rot=rot, center=center, scale=scale)

            init_pix, _ = _project(initial_points, rot, center, scale, panel_w, panel_h)
            pred_pix, _ = _project(pred_s[ti], rot, center, scale, panel_w, panel_h)
            for idx in line_subset:
                a = init_pix[idx]
                b = pred_pix[idx]
                if (
                    0 <= a[0] < panel_w
                    and 0 <= a[1] < panel_h
                    and 0 <= b[0] < panel_w
                    and 0 <= b[1] < panel_h
                ):
                    cv2.line(panels[2], tuple(a.tolist()), tuple(b.tolist()), (95, 95, 95), 1, cv2.LINE_AA)
            _draw_points(
                panels[2],
                initial_points,
                np.zeros(initial_points.shape[0], dtype=np.float32),
                rot=rot,
                center=center,
                scale=scale,
                color_range=color_range,
                radius=1,
                alpha=0.35,
            )
            _draw_points(
                panels[2],
                pred_s[ti],
                pred_motion[ti],
                rot=rot,
                center=center,
                scale=scale,
                color_range=color_range,
                radius=2,
            )
            _draw_robot(panels[2], robot_s[ti, robot_mask_s[ti]], rot=rot, center=center, scale=scale)

            for panel in panels:
                _draw_axes(panel, rot, (panel_w - 88, panel_h - 68))

            for i, panel in enumerate(panels):
                x0 = margin + i * (panel_w + margin)
                y0 = header
                canvas[y0 : y0 + panel_h, x0 : x0 + panel_w] = panel

            writer.write(canvas)
    finally:
        writer.release()

    final_png = out_path.with_suffix(".png")
    if out_path.exists():
        # Save a lightweight contact image: the last encoded frame before writer release.
        cv2.imwrite(str(final_png), canvas)

    return {
        "key": key,
        "video": str(out_path),
        "preview": str(final_png),
        "num_scene_points_total": int(valid_scene.size),
        "num_scene_points_rendered": int(scene_idx.size),
        "num_robot_points_rendered": int(robot_idx.size),
        "mean_l2_pred_vs_target_rendered": float(first_pred_l2.mean()),
        "p95_l2_pred_vs_target_rendered": float(np.percentile(first_pred_l2, 95)),
        "max_pred_displacement_rendered": float(pred_motion.max()),
    }


def _iter_loader(args: argparse.Namespace, feature_dims: Any) -> DataLoader:
    ds = RobotWinWDSDataset(
        args.wds_root,
        args.split,
        shuffle=False,
        seed=args.seed,
        target_scene_features_dim=feature_dims.scene_features_dim,
        target_robot_features_dim=feature_dims.robot_features_dim,
        shuffle_buffer=0,
    )
    return DataLoader(
        ds,
        batch_size=1,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=robotwin_action_collate,
        persistent_workers=args.num_workers > 0,
    )


def main() -> None:
    args = build_argparser().parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model, device, feature_dims = _load_model(args)
    loader = _iter_loader(args, feature_dims)

    summary: Dict[str, Any] = {
        "wds_root": str(args.wds_root),
        "split": args.split,
        "checkpoint": str(args.checkpoint),
        "pretrained_checkpoint": str(args.pretrained_checkpoint),
        "note": "RoboTwin WDS target trajectories are whatever was written by preprocessing; current default uses repeat_t0 static scene points.",
        "samples": [],
    }

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= args.num_samples:
                break
            key = _safe_name(batch.get("__key__", None), f"sample_{i:03d}")
            print(f"[render] {i + 1}/{args.num_samples}: {key}", flush=True)
            batch = to_device(batch, device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
                enabled=args.amp and device.type == "cuda",
            ):
                outputs = model.world_model(batch, training=False)

            pred = outputs["scene_flows"][0].detach().float().cpu().numpy()
            target_tensor = batch.get("gt_scene_flows", batch["scene_flows"])
            target = target_tensor[0].detach().float().cpu().numpy()
            scene_exists = batch["scene_exists"][0].detach().cpu().numpy().astype(bool)
            robot = batch["robot_flows"][0].detach().float().cpu().numpy()
            robot_exists = batch["robot_exists"][0].detach().cpu().numpy().astype(bool)

            video_path = out_dir / f"{i:03d}_{key}.mp4"
            sample_summary = _render_video(
                video_path,
                key=key,
                pred=pred,
                target=target,
                scene_exists=scene_exists,
                robot=robot,
                robot_exists=robot_exists,
                max_points=args.max_points,
                max_robot_points=args.max_robot_points,
                fps=args.fps,
                hold_final=args.hold_final,
                seed=args.seed + i,
                elev=args.elev,
                azim=args.azim,
            )
            summary["samples"].append(sample_summary)

            if device.type == "cuda":
                torch.cuda.empty_cache()

    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as fp:
        json.dump(summary, fp, indent=2)
    print(f"[done] wrote {len(summary['samples'])} videos to {out_dir}", flush=True)
    print(f"[done] summary: {summary_path}", flush=True)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--wds-root", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--pretrained-checkpoint", default="pretrained_checkpoints/large-droid+behavior/model-best.pt")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--split", default="test", choices=["train", "test"])
    p.add_argument("--num-samples", type=int, default=6)
    p.add_argument("--max-points", type=int, default=1024)
    p.add_argument("--max-robot-points", type=int, default=512)
    p.add_argument("--fps", type=float, default=2.0)
    p.add_argument("--hold-final", type=int, default=4)
    p.add_argument("--elev", type=float, default=25.0)
    p.add_argument("--azim", type=float, default=-55.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=0)
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
