#!/usr/bin/env python3
"""Two-stage action-space fine-tuning for RoboTwin two-gripper WDS clips.

Stage 1:
  --stage action_decoder  -> freezes PointWorld and trains only the new action head.

Stage 2:
  --stage lora --resume <stage1 ckpt> -> trains LoRA adapters in PointWorld plus
  the 54-D action head. Scene loss can be enabled to adapt the dynamics head.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

# Ensure repo root is importable when called from tools/robotwin2g.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from tools.robotwin2g.action_model import (
    PointWorldActionModel,
    build_pointworld_args,
    infer_feature_dims_from_checkpoint,
)
from tools.robotwin2g.wds_dataset import RobotWinWDSDataset, read_metadata_count, robotwin_action_collate


def to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def load_torch(path: str | Path, map_location="cpu") -> dict:
    return torch.load(str(path), map_location=map_location)


def load_action_stats(wds_root: str | Path, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    stats_path = Path(wds_root) / "action_stats.json"
    if not stats_path.exists():
        raise FileNotFoundError(f"Missing action stats: {stats_path}. Run convert_robotwin_to_wds.py first.")
    with open(stats_path, "r") as fp:
        stats = json.load(fp)
    mean = torch.tensor(stats["mean"], dtype=torch.float32, device=device)
    std = torch.tensor(stats["std"], dtype=torch.float32, device=device).clamp(min=1e-6)
    return mean, std


def masked_action_loss(
    pred_norm: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    loss_type: str,
) -> torch.Tensor:
    target_norm = (target - mean.view(1, 1, -1)) / std.view(1, 1, -1)
    mask = mask.bool()
    while mask.ndim < pred_norm.ndim:
        mask = mask.unsqueeze(-1)
    if loss_type == "mse":
        per = (pred_norm - target_norm).pow(2)
    elif loss_type == "l1":
        per = (pred_norm - target_norm).abs()
    elif loss_type == "smooth_l1":
        per = F.smooth_l1_loss(pred_norm, target_norm, reduction="none")
    else:
        raise ValueError(f"Unsupported loss type {loss_type}")
    per = per * mask.to(per.dtype)
    denom = mask.to(per.dtype).sum().clamp(min=1.0) * pred_norm.shape[-1]
    return per.sum() / denom


def _masked_scene_loss(scene_outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor], loss_type: str) -> torch.Tensor:
    pred = scene_outputs["scene_flows"].float()[:, 1:]
    target = batch["scene_flows"].float()[:, 1:]
    mask = batch["scene_exists"].bool()[:, 1:]
    if "scene_visibility" in batch:
        mask = mask & batch["scene_visibility"].bool()[:, 1:]
    if "scene_depth_valid_mask" in batch:
        mask = mask & batch["scene_depth_valid_mask"].bool()[:, 1:]
    while mask.ndim < pred.ndim:
        mask = mask.unsqueeze(-1)
    if loss_type == "mse":
        per = (pred - target).pow(2)
    elif loss_type == "l1":
        per = (pred - target).abs()
    elif loss_type == "smooth_l1":
        per = F.smooth_l1_loss(pred, target, reduction="none")
    else:
        raise ValueError(f"Unsupported scene loss type {loss_type}")
    per = per * mask.to(per.dtype)
    denom = mask.to(per.dtype).sum().clamp(min=1.0) * pred.shape[-1]
    return per.sum() / denom


def _point_rmse_cm(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> Tuple[float, float]:
    diff2 = (pred - target).pow(2).sum(dim=-1)
    diff2 = diff2 * mask.to(diff2.dtype)
    count = mask.to(diff2.dtype).sum().clamp(min=1.0)
    rmse_m = torch.sqrt(diff2.sum() / count)
    return float((rmse_m * 100.0).detach().cpu()), float(count.detach().cpu())


def _point_rmse_cm_per_scene(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> Tuple[float, int]:
    total = 0.0
    count = 0
    diff2 = (pred - target).pow(2).sum(dim=-1) * mask.to(pred.dtype)
    valid = mask.to(pred.dtype).sum(dim=tuple(range(1, mask.ndim)))
    err = diff2.sum(dim=tuple(range(1, diff2.ndim)))
    for b in range(pred.shape[0]):
        if valid[b] <= 0:
            continue
        rmse_m = torch.sqrt(err[b] / valid[b].clamp(min=1.0))
        total += float((rmse_m * 100.0).detach().cpu())
        count += 1
    return total, count


def _select_valid_points(points: torch.Tensor, mask: torch.Tensor, max_points: int) -> torch.Tensor:
    pts = points[mask]
    if pts.numel() == 0:
        return pts.reshape(0, points.shape[-1])
    if max_points > 0 and pts.shape[0] > max_points:
        idx = torch.linspace(0, pts.shape[0] - 1, steps=max_points, device=pts.device).long()
        pts = pts[idx]
    return pts.float()


def _chamfer_points_cm(
    pred_points: torch.Tensor,
    target_points: torch.Tensor,
    pred_mask: torch.Tensor,
    target_mask: torch.Tensor,
    *,
    max_points: int,
) -> Optional[float]:
    pred_pts = _select_valid_points(pred_points, pred_mask, max_points)
    tgt_pts = _select_valid_points(target_points, target_mask, max_points)
    if pred_pts.shape[0] == 0 or tgt_pts.shape[0] == 0:
        return None
    d = torch.cdist(pred_pts.unsqueeze(0), tgt_pts.unsqueeze(0), p=2).squeeze(0)
    cd_m = 0.5 * (d.min(dim=1).values.mean() + d.min(dim=0).values.mean())
    return float((cd_m * 100.0).detach().cpu())


def _chamfer_cm(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    max_points: int,
) -> Tuple[float, int]:
    total = 0.0
    count = 0
    B, T = pred.shape[:2]
    for b in range(B):
        for t in range(T):
            cd_cm = _chamfer_points_cm(pred[b, t], target[b, t], mask[b, t], mask[b, t], max_points=max_points)
            if cd_cm is None:
                continue
            total += cd_cm
            count += 1
    return total, count


def _chamfer_cm_per_scene(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    max_points: int,
) -> Tuple[float, int]:
    total = 0.0
    count = 0
    B, T = pred.shape[:2]
    for b in range(B):
        frame_vals = []
        for t in range(T):
            cd_cm = _chamfer_points_cm(pred[b, t], target[b, t], mask[b, t], mask[b, t], max_points=max_points)
            if cd_cm is not None:
                frame_vals.append(cd_cm)
        if frame_vals:
            total += sum(frame_vals) / len(frame_vals)
            count += 1
    return total, count


def _match_last_dim(x: torch.Tensor, dim: int) -> torch.Tensor:
    if x.shape[-1] > dim:
        return x[..., :dim]
    if x.shape[-1] < dim:
        return F.pad(x, (0, dim - x.shape[-1]))
    return x


def _normalize_action(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    dim = x.shape[-1]
    mean = _match_last_dim(mean, dim).to(device=x.device, dtype=x.dtype)
    std = _match_last_dim(std, dim).to(device=x.device, dtype=x.dtype).clamp(min=1e-6)
    return (x - mean.view(*([1] * (x.ndim - 1)), dim)) / std.view(*([1] * (x.ndim - 1)), dim)


def _action_robot_flow_nn_metrics_cm(
    pred: torch.Tensor,
    target: torch.Tensor,
    action_state: torch.Tensor,
    robot_flows: torch.Tensor,
    robot_exists: torch.Tensor,
    mask: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    *,
    max_points: int,
) -> Tuple[float, int, float, int]:
    pred_norm = _normalize_action(pred.float(), mean, std)
    target_norm = _normalize_action(target.float(), mean, std)
    state_norm = _normalize_action(_match_last_dim(action_state.float(), pred.shape[-1]), mean, std)

    rmse_total = 0.0
    cd_total = 0.0
    rmse_count = 0
    cd_count = 0
    B, H = pred.shape[:2]
    for b in range(B):
        rmse_vals = []
        cd_vals = []
        state_b = state_norm[b]
        for h in range(H):
            if not bool(mask[b, h].detach().cpu()):
                continue
            pred_dist = (state_b - pred_norm[b, h].view(1, -1)).pow(2).mean(dim=-1)
            target_dist = (state_b - target_norm[b, h].view(1, -1)).pow(2).mean(dim=-1)
            pred_idx = int(pred_dist.argmin().detach().cpu())
            target_idx = int(target_dist.argmin().detach().cpu())
            pred_mask = robot_exists[b, pred_idx].bool()
            target_mask = robot_exists[b, target_idx].bool()
            common_mask = pred_mask & target_mask
            if common_mask.any():
                rmse_cm, _ = _point_rmse_cm(
                    robot_flows[b, pred_idx].unsqueeze(0).unsqueeze(0),
                    robot_flows[b, target_idx].unsqueeze(0).unsqueeze(0),
                    common_mask.unsqueeze(0).unsqueeze(0),
                )
                rmse_vals.append(rmse_cm)
            cd_cm = _chamfer_points_cm(
                robot_flows[b, pred_idx],
                robot_flows[b, target_idx],
                pred_mask,
                target_mask,
                max_points=max_points,
            )
            if cd_cm is not None:
                cd_vals.append(cd_cm)
        if rmse_vals:
            rmse_total += sum(rmse_vals) / len(rmse_vals)
            rmse_count += 1
        if cd_vals:
            cd_total += sum(cd_vals) / len(cd_vals)
            cd_count += 1
    return rmse_total, rmse_count, cd_total, cd_count


def _action_point_metrics_cm(
    pred: torch.Tensor,
    target: torch.Tensor,
    batch: Dict[str, torch.Tensor],
    mask: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    *,
    layout: str,
    max_points: int,
) -> Tuple[float, int, float, int]:
    if layout == "none":
        return 0.0, 0, 0.0, 0
    if layout == "robot_flow_nn":
        return _action_robot_flow_nn_metrics_cm(
            pred,
            target,
            batch["action_state"],
            batch["robot_flows"].float(),
            batch["robot_exists"].bool(),
            mask,
            mean,
            std,
            max_points=max_points,
        )
    if layout == "two_gripper_xyz":
        if pred.shape[-1] < 10:
            return 0.0, 0, 0.0, 0
        pred_pts = pred[..., [0, 1, 2, 7, 8, 9]].reshape(pred.shape[0], pred.shape[1], 2, 3)
        target_pts = target[..., [0, 1, 2, 7, 8, 9]].reshape(target.shape[0], target.shape[1], 2, 3)
        point_mask = mask.bool().unsqueeze(-1).expand(pred.shape[0], pred.shape[1], 2)
        rmse_sum, rmse_count = _point_rmse_cm_per_scene(pred_pts, target_pts, point_mask)
        cd_sum, cd_count = _chamfer_cm_per_scene(pred_pts, target_pts, point_mask, max_points=max_points)
        return rmse_sum, rmse_count, cd_sum, cd_count
    if layout != "xyz_points":
        raise ValueError(f"Unsupported action metric layout {layout}")
    if pred.shape[-1] % 3 != 0:
        return 0.0, 0, 0.0, 0
    num_points = pred.shape[-1] // 3
    pred_pts = pred.reshape(pred.shape[0], pred.shape[1], num_points, 3)
    target_pts = target.reshape(target.shape[0], target.shape[1], num_points, 3)
    point_mask = mask.bool().unsqueeze(-1).expand(pred.shape[0], pred.shape[1], num_points)
    rmse_sum, rmse_count = _point_rmse_cm_per_scene(pred_pts, target_pts, point_mask)
    cd_sum, cd_count = _chamfer_cm_per_scene(pred_pts, target_pts, point_mask, max_points=max_points)
    return rmse_sum, rmse_count, cd_sum, cd_count


@torch.no_grad()
def evaluate(
    model: PointWorldActionModel,
    loader: DataLoader,
    device: torch.device,
    mean: torch.Tensor,
    std: torch.Tensor,
    max_batches: int,
    amp: bool,
    scene_metrics: bool = True,
    scene_cd_max_points: int = 1024,
    action_metric_layout: str = "none",
    action_cd_max_points: int = 1024,
) -> Dict[str, float]:
    model.eval()
    total_mse = 0.0
    total_mae = 0.0
    total_count = 0.0
    action_rmse_cm_sum = 0.0
    action_rmse_cm_count = 0
    action_cd_cm_sum = 0.0
    action_cd_cm_count = 0
    scene_rmse_sum = 0.0
    scene_rmse_count = 0
    scene_cd_sum = 0.0
    scene_cd_count = 0
    for i, batch in enumerate(loader):
        if max_batches > 0 and i >= max_batches:
            break
        batch = to_device(batch, device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16 if device.type == "cuda" else torch.float32, enabled=amp and device.type == "cuda"):
            if scene_metrics:
                pred_norm, scene_outputs = model.forward_action_and_scene(batch, training=False)
            else:
                pred_norm = model(batch)
                scene_outputs = None
        pred = pred_norm.float() * std.view(1, 1, -1) + mean.view(1, 1, -1)
        target = batch["action_target"].float()
        action_mask = batch["action_mask"].bool()
        action_feature_mask = action_mask
        while action_feature_mask.ndim < pred.ndim:
            action_feature_mask = action_feature_mask.unsqueeze(-1)
        diff = (pred - target) * action_feature_mask.to(pred.dtype)
        denom = (action_feature_mask.to(pred.dtype).sum() * pred.shape[-1]).clamp(min=1.0)
        total_mse += float(diff.pow(2).sum().item())
        total_mae += float(diff.abs().sum().item())
        total_count += float(denom.item())
        action_rmse_sum, action_rmse_count, action_cd_sum, action_cd_count = _action_point_metrics_cm(
            pred,
            target,
            batch,
            action_mask,
            mean,
            std,
            layout=action_metric_layout,
            max_points=action_cd_max_points,
        )
        action_rmse_cm_sum += action_rmse_sum
        action_rmse_cm_count += action_rmse_count
        action_cd_cm_sum += action_cd_sum
        action_cd_cm_count += action_cd_count
        if scene_outputs is not None:
            scene_pred = scene_outputs["scene_flows"].float()[:, 1:]
            scene_target = batch["scene_flows"].float()[:, 1:]
            scene_mask = batch["scene_exists"].bool()[:, 1:]
            if "scene_visibility" in batch:
                scene_mask = scene_mask & batch["scene_visibility"].bool()[:, 1:]
            if "scene_depth_valid_mask" in batch:
                scene_mask = scene_mask & batch["scene_depth_valid_mask"].bool()[:, 1:]
            rmse_sum, rmse_count = _point_rmse_cm_per_scene(scene_pred, scene_target, scene_mask)
            scene_rmse_sum += rmse_sum
            scene_rmse_count += rmse_count
            cd_sum, cd_count = _chamfer_cm_per_scene(scene_pred, scene_target, scene_mask, max_points=scene_cd_max_points)
            scene_cd_sum += cd_sum
            scene_cd_count += cd_count
    if total_count <= 0:
        out = {"mse": float("nan"), "mae": float("nan"), "rmse": float("nan")}
    else:
        mse = total_mse / total_count
        out = {"mse": mse, "mae": total_mae / total_count, "rmse": math.sqrt(max(mse, 0.0))}
    out["action_vector_rmse"] = out["rmse"]
    out["action_vector_mae"] = out["mae"]
    out["action_rmse_cm"] = action_rmse_cm_sum / action_rmse_cm_count if action_rmse_cm_count > 0 else float("nan")
    out["action_cd_cm"] = action_cd_cm_sum / action_cd_cm_count if action_cd_cm_count > 0 else float("nan")
    out["scene_rmse_cm"] = scene_rmse_sum / scene_rmse_count if scene_rmse_count > 0 else float("nan")
    out["scene_cd_cm"] = scene_cd_sum / scene_cd_count if scene_cd_count > 0 else float("nan")
    out["action_rmse"] = out["action_rmse_cm"]
    out["action_cd"] = out["action_cd_cm"]
    out["scene_rmse"] = out["scene_rmse_cm"]
    out["scene_cd"] = out["scene_cd_cm"]
    out["actuator_rmse"] = out["action_vector_rmse"]
    out["actuator_cd_cm"] = out["action_cd_cm"]
    return out


def _fmt_metric(value: float, suffix: str = "") -> str:
    if value is None or not math.isfinite(float(value)):
        return "N/A"
    return f"{float(value):.6g}{suffix}"


def _action_metric_note(layout: str) -> str:
    if layout == "robot_flow_nn":
        return "Action RMSE/CD are computed per scene on robot pointclouds by selecting the nearest observed robot frame in normalized action space, then converting point distances to centimeters."
    if layout == "xyz_points":
        return "Action RMSE/CD are computed per scene after reshaping action vectors into XYZ triples in meters and converting to centimeters."
    if layout == "two_gripper_xyz":
        return "Action RMSE/CD are computed from the left/right gripper XYZ action slots [0:3] and [7:10] in meters, converted to centimeters."
    return "Action RMSE/CD in centimeters are not computed because the action vector is not declared as spatial XYZ triples."


def write_report(path: Path, *, step: int, stage: str, metrics: Dict[str, float], action_metric_note: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": int(step),
        "stage": stage,
        "metrics": metrics,
        "action_metric_note": action_metric_note,
    }
    with open(path.with_suffix(".json"), "w") as fp:
        json.dump(payload, fp, indent=2, sort_keys=True)
    lines = [
        f"# PointWorld FlowAM Report",
        "",
        f"stage: `{stage}`",
        f"step: `{step}`",
        "",
        "| Target | RMSE ↓ | CD (cm) ↓ |",
        "|---|---:|---:|",
        f"| Action | {_fmt_metric(metrics.get('action_rmse_cm', float('nan')), ' cm')} | {_fmt_metric(metrics.get('action_cd_cm', float('nan')))} |",
        f"| Scene | {_fmt_metric(metrics.get('scene_rmse_cm', float('nan')), ' cm')} | {_fmt_metric(metrics.get('scene_cd_cm', float('nan')))} |",
        "",
        f"Action metric note: {action_metric_note}",
        f"Raw 54-D action vector RMSE: {_fmt_metric(metrics.get('action_vector_rmse', float('nan')))}",
    ]
    path.write_text("\n".join(lines) + "\n")


def save_checkpoint(
    path: Path,
    model: PointWorldActionModel,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    step: int,
    epoch: int,
    best_metric: float,
    action_mean: torch.Tensor,
    action_std: torch.Tensor,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": int(step),
            "epoch": int(epoch),
            "best_metric": float(best_metric),
            "args": vars(args),
            "action_mean": action_mean.detach().cpu(),
            "action_std": action_std.detach().cpu(),
        },
        path,
    )


def load_resume(
    path: str | Path,
    model: PointWorldActionModel,
    optimizer: Optional[torch.optim.Optimizer] = None,
    strict: bool = True,
) -> Tuple[int, int, float]:
    payload = load_torch(path, map_location="cpu")
    state = payload.get("model", payload)
    missing, unexpected = model.load_state_dict(state, strict=strict)
    if missing or unexpected:
        print(f"[resume] missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    if optimizer is not None and "optimizer" in payload:
        try:
            optimizer.load_state_dict(payload["optimizer"])
        except Exception as exc:
            print(f"[resume] optimizer state not loaded: {exc}", flush=True)
    return int(payload.get("step", 0)), int(payload.get("epoch", 0)), float(payload.get("best_metric", float("inf")))


def make_loaders(args: argparse.Namespace, feature_dims) -> Tuple[DataLoader, Optional[DataLoader], int, int]:
    train_ds = RobotWinWDSDataset(
        args.wds_root,
        "train",
        shuffle=True,
        seed=args.seed,
        target_scene_features_dim=feature_dims.scene_features_dim,
        target_robot_features_dim=feature_dims.robot_features_dim,
        shuffle_buffer=args.shuffle_buffer,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=robotwin_action_collate,
        persistent_workers=args.num_workers > 0,
    )

    train_count = read_metadata_count(args.wds_root, "train")
    test_count = read_metadata_count(args.wds_root, "test")
    test_loader: Optional[DataLoader]
    try:
        test_ds = RobotWinWDSDataset(
            args.wds_root,
            "test",
            shuffle=False,
            seed=args.seed,
            target_scene_features_dim=feature_dims.scene_features_dim,
            target_robot_features_dim=feature_dims.robot_features_dim,
            shuffle_buffer=0,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=args.batch_size,
            num_workers=max(0, min(args.num_workers, args.eval_num_workers)),
            pin_memory=True,
            collate_fn=robotwin_action_collate,
            persistent_workers=(args.num_workers > 0 and args.eval_num_workers > 0),
        )
    except FileNotFoundError:
        if not args.allow_empty_test:
            raise
        test_loader = None
        test_count = 0
    return train_loader, test_loader, train_count, test_count


def train(args: argparse.Namespace) -> None:
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if args.resume:
        ckpt_for_dims = load_torch(args.resume, map_location="cpu")
        # Action checkpoints do not contain raw PointWorld key names at top level, so fall back to pretrained for dims if given.
        if args.pretrained_checkpoint and Path(args.pretrained_checkpoint).exists():
            ckpt_for_dims = load_torch(args.pretrained_checkpoint, map_location="cpu")
    else:
        if not args.pretrained_checkpoint:
            raise ValueError("--pretrained-checkpoint is required unless --resume points to a complete action checkpoint")
        ckpt_for_dims = load_torch(args.pretrained_checkpoint, map_location="cpu")
    feature_dims = infer_feature_dims_from_checkpoint(ckpt_for_dims)

    action_mean, action_std = load_action_stats(args.wds_root, device)
    action_dim = int(action_mean.numel())
    if action_dim <= 0:
        raise ValueError(
            f"Invalid action stats in {Path(args.wds_root) / 'action_stats.json'}: "
            "mean/std must contain at least one action dimension."
        )
    if args.action_dim > 0 and action_dim != args.action_dim:
        raise ValueError(
            f"WDS action_dim={action_dim} does not match requested --action-dim={args.action_dim}. "
            "Regenerate the data or pass the matching action dimension."
        )
    action_horizon = int(args.clip_horizon - 1)
    state_dim = action_dim

    pw_args = build_pointworld_args(
        checkpoint=ckpt_for_dims if args.pretrained_checkpoint else None,
        device=str(device),
        norm_stats_path=args.norm_stats_path,
        ptv3_size=args.ptv3_size,
        predictor_dim=args.predictor_dim,
        disable_compile=args.disable_compile,
        grid_size=args.grid_size,
        depth_threshold=args.depth_threshold,
    )
    data_info = {
        "scene_features_dim": feature_dims.scene_features_dim,
        "robot_features_dim": feature_dims.robot_features_dim,
    }
    model = PointWorldActionModel(
        pw_args,
        data_info,
        action_dim=action_dim,
        action_horizon=action_horizon,
        state_dim=state_dim,
        decoder_hidden_dim=args.decoder_hidden_dim,
        decoder_layers=args.decoder_layers,
    )

    if args.stage == "lora" or args.lora_rank > 0:
        if args.lora_rank <= 0:
            raise ValueError("--lora-rank must be > 0 for stage lora")
        lora_names = model.enable_lora(
            rank=args.lora_rank,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
            target_patterns=[p for p in args.lora_targets.split(",") if p],
            exclude_patterns=[p for p in args.lora_exclude.split(",") if p],
            include_dinov3=args.lora_include_dinov3,
        )
        print(f"[lora] injected {len(lora_names)} Linear adapters", flush=True)
        if lora_names:
            print(f"[lora] first adapters: {lora_names[:8]}", flush=True)

    if args.pretrained_checkpoint:
        pretrained = load_torch(args.pretrained_checkpoint, map_location="cpu")
        missing, unexpected = model.load_pointworld_checkpoint(pretrained, strict=False)
        print(f"[pretrained] loaded PointWorld: missing={len(missing)} unexpected={len(unexpected)}", flush=True)

    model.set_train_stage(args.stage, unfreeze_dinov3=args.unfreeze_dinov3)
    model.to(device)

    # Optimizer parameter groups: action head gets --lr; PointWorld/LoRA gets --world-lr.
    if args.stage == "all":
        world_params = [p for p in model.world_model.parameters() if p.requires_grad]
        head_params = [p for p in model.action_decoder.parameters() if p.requires_grad]
        param_groups = [
            {"params": head_params, "lr": args.lr},
            {"params": world_params, "lr": args.world_lr if args.world_lr > 0 else args.lr},
        ]
    elif args.stage == "lora":
        head_params = [p for p in model.action_decoder.parameters() if p.requires_grad]
        lora_params = list(model.lora_parameters())
        if not lora_params:
            raise RuntimeError("stage lora selected but no trainable LoRA parameters were found")
        param_groups = [
            {"params": head_params, "lr": args.lr},
            {"params": lora_params, "lr": args.world_lr if args.world_lr > 0 else args.lr},
        ]
    else:
        param_groups = [{"params": [p for p in model.parameters() if p.requires_grad], "lr": args.lr}]
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)

    start_step = 0
    start_epoch = 0
    best_metric = float("inf")
    if args.resume:
        start_step, start_epoch, best_metric = load_resume(args.resume, model, optimizer=None if args.reset_optimizer else optimizer, strict=False)
        if args.reset_progress:
            print(
                f"[resume] reset progress counters from loaded step={start_step} epoch={start_epoch} "
                "for a fresh training stage",
                flush=True,
            )
            start_step = 0
            start_epoch = 0
            best_metric = float("inf")
        # Apply the requested training stage after loading, because resume state may come from a different stage.
        model.set_train_stage(args.stage, unfreeze_dinov3=args.unfreeze_dinov3)
        print(f"[resume] step={start_step} epoch={start_epoch} best_metric={best_metric}", flush=True)

    train_loader, test_loader, train_count, test_count = make_loaders(args, feature_dims)
    print(
        f"dataset train={train_count} test={test_count} | action_dim={action_dim} horizon={action_horizon} | "
        f"feature_dims scene={feature_dims.scene_features_dim} robot={feature_dims.robot_features_dim}",
        flush=True,
    )
    print(f"stage={args.stage} trainable_params={sum(p.numel() for p in model.parameters() if p.requires_grad):,}", flush=True)

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")
    step = start_step
    last_eval = -args.eval_every
    epoch_start = start_epoch

    for epoch in range(epoch_start, args.num_epochs):
        model.train(True)
        pbar = tqdm(train_loader, desc=f"epoch {epoch+1}/{args.num_epochs}")
        for batch in pbar:
            step += 1
            batch = to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16 if device.type == "cuda" else torch.float32, enabled=args.amp and device.type == "cuda"):
                if args.scene_loss_weight > 0:
                    pred_norm, scene_outputs = model.forward_action_and_scene(batch, training=True)
                else:
                    pred_norm = model(batch)
                    scene_outputs = None
                loss = masked_action_loss(
                    pred_norm,
                    batch["action_target"].float(),
                    batch["action_mask"].bool(),
                    action_mean,
                    action_std,
                    args.loss,
                )
                action_loss = loss
                scene_loss = torch.zeros((), dtype=loss.dtype, device=loss.device)
                if scene_outputs is not None:
                    scene_loss = _masked_scene_loss(scene_outputs, batch, args.scene_loss)
                    loss = loss + float(args.scene_loss_weight) * scene_loss
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            pbar.set_postfix(
                {
                    "loss": f"{float(loss.detach().cpu()):.4g}",
                    "act": f"{float(action_loss.detach().cpu()):.4g}",
                    "scene": f"{float(scene_loss.detach().cpu()):.4g}",
                    "step": step,
                }
            )

            should_eval = args.eval_every > 0 and (step - last_eval >= args.eval_every)
            if should_eval:
                last_eval = step
                if test_loader is not None and test_count != 0:
                    metrics = evaluate(
                        model,
                        test_loader,
                        device,
                        action_mean,
                        action_std,
                        args.eval_batches,
                        args.amp,
                        scene_metrics=not args.skip_scene_metrics,
                        scene_cd_max_points=args.scene_cd_max_points,
                        action_metric_layout=args.action_metric_layout,
                        action_cd_max_points=args.action_cd_max_points,
                    )
                    metric = metrics["mae"]
                    print(
                        f"[eval] step={step} "
                        f"action_rmse_cm={_fmt_metric(metrics['action_rmse_cm'])} "
                        f"action_cd_cm={_fmt_metric(metrics['action_cd_cm'])} "
                        f"scene_rmse_cm={_fmt_metric(metrics['scene_rmse_cm'])} "
                        f"scene_cd_cm={_fmt_metric(metrics['scene_cd_cm'])} "
                        f"action_vector_rmse={metrics['action_vector_rmse']:.6g} "
                        f"action_vector_mae={metrics['action_vector_mae']:.6g}",
                        flush=True,
                    )
                    write_report(
                        Path(args.output_dir) / "report-latest.md",
                        step=step,
                        stage=args.stage,
                        metrics=metrics,
                        action_metric_note=_action_metric_note(args.action_metric_layout),
                    )
                else:
                    metric = float(loss.detach().cpu())
                    print(f"[eval skipped] step={step} no test shards; using train loss={metric:.6g} for checkpoint selection", flush=True)
                save_checkpoint(
                    Path(args.output_dir) / "checkpoint-last.pt",
                    model,
                    optimizer,
                    args,
                    step,
                    epoch,
                    best_metric,
                    action_mean,
                    action_std,
                )
                if metric < best_metric:
                    best_metric = metric
                    save_checkpoint(
                        Path(args.output_dir) / "checkpoint-best.pt",
                        model,
                        optimizer,
                        args,
                        step,
                        epoch,
                        best_metric,
                        action_mean,
                        action_std,
                    )
            if args.save_every > 0 and step % args.save_every == 0:
                save_checkpoint(
                    Path(args.output_dir) / f"checkpoint-step{step}.pt",
                    model,
                    optimizer,
                    args,
                    step,
                    epoch,
                    best_metric,
                    action_mean,
                    action_std,
                )
            if args.max_steps > 0 and step >= args.max_steps:
                save_checkpoint(
                    Path(args.output_dir) / "checkpoint-last.pt",
                    model,
                    optimizer,
                    args,
                    step,
                    epoch,
                    best_metric,
                    action_mean,
                    action_std,
                )
                return

    if test_loader is not None and test_count != 0:
        metrics = evaluate(
            model,
            test_loader,
            device,
            action_mean,
            action_std,
            args.eval_batches,
            args.amp,
            scene_metrics=not args.skip_scene_metrics,
            scene_cd_max_points=args.scene_cd_max_points,
            action_metric_layout=args.action_metric_layout,
            action_cd_max_points=args.action_cd_max_points,
        )
        print(
            f"[final eval] action_rmse_cm={_fmt_metric(metrics['action_rmse_cm'])} "
            f"action_cd_cm={_fmt_metric(metrics['action_cd_cm'])} "
            f"scene_rmse_cm={_fmt_metric(metrics['scene_rmse_cm'])} "
            f"scene_cd_cm={_fmt_metric(metrics['scene_cd_cm'])} "
            f"action_vector_rmse={metrics['action_vector_rmse']:.6g} "
            f"action_vector_mae={metrics['action_vector_mae']:.6g}",
            flush=True,
        )
        write_report(
            Path(args.output_dir) / "report-final.md",
            step=step,
            stage=args.stage,
            metrics=metrics,
            action_metric_note=_action_metric_note(args.action_metric_layout),
        )
    else:
        print("[final eval skipped] no test shards", flush=True)
    save_checkpoint(Path(args.output_dir) / "checkpoint-last.pt", model, optimizer, args, step, args.num_epochs, best_metric, action_mean, action_std)
    if not (Path(args.output_dir) / "checkpoint-best.pt").exists():
        save_checkpoint(Path(args.output_dir) / "checkpoint-best.pt", model, optimizer, args, step, args.num_epochs, best_metric, action_mean, action_std)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--wds-root", required=True)
    p.add_argument("--pretrained-checkpoint", default="pretrained_checkpoints/large-droid+behavior/model-best.pt")
    p.add_argument("--resume", default=None, help="Resume from a robotwin2g action checkpoint")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--stage", choices=["action_decoder", "lora", "all"], default="action_decoder")
    p.add_argument("--reset-optimizer", action="store_true")
    p.add_argument("--reset-progress", action="store_true", help="Load resume weights but restart step/epoch/best-metric counters; useful when starting a new training stage.")

    p.add_argument("--device", default="auto")
    p.add_argument("--norm-stats-path", default="stats/droid_behavior")
    p.add_argument("--ptv3-size", default="base")
    p.add_argument("--predictor-dim", type=int, default=256)
    p.add_argument("--disable-compile", action="store_true", default=True)
    p.add_argument("--grid-size", type=float, default=0.015)
    p.add_argument("--depth-threshold", type=float, default=0.003)
    p.add_argument("--unfreeze-dinov3", action="store_true")

    p.add_argument("--clip-horizon", type=int, default=11)
    p.add_argument("--action-dim", type=int, default=-1, help="Optional expected action dimension; use 54 for FlowAM dexterous data.")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--eval-num-workers", type=int, default=2)
    p.add_argument("--shuffle-buffer", type=int, default=128)
    p.add_argument("--num-epochs", type=int, default=20)
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--eval-batches", type=int, default=20)
    p.add_argument("--skip-scene-metrics", action="store_true")
    p.add_argument("--scene-cd-max-points", type=int, default=1024)
    p.add_argument(
        "--action-metric-layout",
        choices=["none", "xyz_points", "robot_flow_nn", "two_gripper_xyz"],
        default="none",
        help="How to compute action_rmse_cm/action_cd_cm. Use two_gripper_xyz for 14-D [left7,right7] actions, xyz_points for XYZ triples, or robot_flow_nn as a FlowAM proxy.",
    )
    p.add_argument("--action-cd-max-points", type=int, default=1024)
    p.add_argument("--allow-empty-test", dest="allow_empty_test", action="store_true", default=True, help="Do not crash when the episode-level split has no test shards; useful for one-episode smoke tests.")
    p.add_argument("--require-test", dest="allow_empty_test", action="store_false", help="Crash if the test split has no shards.")
    p.add_argument("--save-every", type=int, default=2000)

    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--world-lr", type=float, default=2e-5)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--grad-clip", type=float, default=5.0)
    p.add_argument("--loss", choices=["mse", "l1", "smooth_l1"], default="smooth_l1")
    p.add_argument("--scene-loss", choices=["mse", "l1", "smooth_l1"], default="smooth_l1")
    p.add_argument("--scene-loss-weight", type=float, default=0.0)
    p.add_argument("--decoder-hidden-dim", type=int, default=512)
    p.add_argument("--decoder-layers", type=int, default=3)
    p.add_argument("--lora-rank", type=int, default=0)
    p.add_argument("--lora-alpha", type=float, default=16.0)
    p.add_argument("--lora-dropout", type=float, default=0.0)
    p.add_argument(
        "--lora-targets",
        default="",
        help="Comma-separated substrings for Linear modules to adapt. Empty means all PointWorld Linear modules.",
    )
    p.add_argument("--lora-exclude", default="", help="Comma-separated substrings to exclude from LoRA.")
    p.add_argument("--lora-include-dinov3", action="store_true")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p


def main() -> None:
    args = build_argparser().parse_args()
    torch.manual_seed(args.seed)
    train(args)


if __name__ == "__main__":
    main()
