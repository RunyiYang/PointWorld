#!/usr/bin/env python3
"""Two-stage action-space fine-tuning for RoboTwin two-gripper WDS clips.

Stage 1:
  --stage action_decoder  -> freezes PointWorld and trains only the new action head.

Stage 2:
  --stage all --resume <stage1 ckpt>  -> unfreezes PointWorld non-DINO modules
  and continues action-loss fine-tuning with the same data.

This script intentionally trains on action loss only. RoboTwin's observed
pointcloud[t] is not guaranteed to contain persistent point identities, so scene
flow loss is not enabled by default.
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


@torch.no_grad()
def evaluate(
    model: PointWorldActionModel,
    loader: DataLoader,
    device: torch.device,
    mean: torch.Tensor,
    std: torch.Tensor,
    max_batches: int,
    amp: bool,
) -> Dict[str, float]:
    model.eval()
    total_mse = 0.0
    total_mae = 0.0
    total_count = 0.0
    for i, batch in enumerate(loader):
        if max_batches > 0 and i >= max_batches:
            break
        batch = to_device(batch, device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16 if device.type == "cuda" else torch.float32, enabled=amp and device.type == "cuda"):
            pred_norm = model(batch)
        pred = pred_norm.float() * std.view(1, 1, -1) + mean.view(1, 1, -1)
        target = batch["action_target"].float()
        mask = batch["action_mask"].bool()
        while mask.ndim < pred.ndim:
            mask = mask.unsqueeze(-1)
        diff = (pred - target) * mask.to(pred.dtype)
        denom = (mask.to(pred.dtype).sum() * pred.shape[-1]).clamp(min=1.0)
        total_mse += float(diff.pow(2).sum().item())
        total_mae += float(diff.abs().sum().item())
        total_count += float(denom.item())
    if total_count <= 0:
        return {"mse": float("nan"), "mae": float("nan")}
    return {"mse": total_mse / total_count, "mae": total_mae / total_count}


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


def make_loaders(args: argparse.Namespace, feature_dims) -> Tuple[DataLoader, DataLoader, int, int]:
    train_ds = RobotWinWDSDataset(
        args.wds_root,
        "train",
        shuffle=True,
        seed=args.seed,
        target_scene_features_dim=feature_dims.scene_features_dim,
        target_robot_features_dim=feature_dims.robot_features_dim,
        shuffle_buffer=args.shuffle_buffer,
    )
    test_ds = RobotWinWDSDataset(
        args.wds_root,
        "test",
        shuffle=False,
        seed=args.seed,
        target_scene_features_dim=feature_dims.scene_features_dim,
        target_robot_features_dim=feature_dims.robot_features_dim,
        shuffle_buffer=0,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=robotwin_action_collate,
        persistent_workers=args.num_workers > 0,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        num_workers=max(0, min(args.num_workers, args.eval_num_workers)),
        pin_memory=True,
        collate_fn=robotwin_action_collate,
        persistent_workers=(args.num_workers > 0 and args.eval_num_workers > 0),
    )
    train_count = read_metadata_count(args.wds_root, "train")
    test_count = read_metadata_count(args.wds_root, "test")
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

    if args.pretrained_checkpoint:
        pretrained = load_torch(args.pretrained_checkpoint, map_location="cpu")
        missing, unexpected = model.load_pointworld_checkpoint(pretrained, strict=False)
        print(f"[pretrained] loaded PointWorld: missing={len(missing)} unexpected={len(unexpected)}", flush=True)

    model.set_train_stage(args.stage, unfreeze_dinov3=args.unfreeze_dinov3)
    model.to(device)

    # Optimizer parameter groups: action head gets --lr; PointWorld gets --world-lr in stage all.
    if args.stage == "all":
        world_params = [p for p in model.world_model.parameters() if p.requires_grad]
        head_params = [p for p in model.action_decoder.parameters() if p.requires_grad]
        param_groups = [
            {"params": head_params, "lr": args.lr},
            {"params": world_params, "lr": args.world_lr if args.world_lr > 0 else args.lr},
        ]
    else:
        param_groups = [{"params": [p for p in model.parameters() if p.requires_grad], "lr": args.lr}]
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)

    start_step = 0
    start_epoch = 0
    best_metric = float("inf")
    if args.resume:
        start_step, start_epoch, best_metric = load_resume(args.resume, model, optimizer=None if args.reset_optimizer else optimizer, strict=False)
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
                pred_norm = model(batch)
                loss = masked_action_loss(
                    pred_norm,
                    batch["action_target"].float(),
                    batch["action_mask"].bool(),
                    action_mean,
                    action_std,
                    args.loss,
                )
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            pbar.set_postfix({"loss": f"{float(loss.detach().cpu()):.4g}", "step": step})

            should_eval = args.eval_every > 0 and (step - last_eval >= args.eval_every)
            if should_eval:
                metrics = evaluate(model, test_loader, device, action_mean, action_std, args.eval_batches, args.amp)
                last_eval = step
                metric = metrics["mae"]
                print(f"[eval] step={step} mse={metrics['mse']:.6g} mae={metrics['mae']:.6g}", flush=True)
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

    metrics = evaluate(model, test_loader, device, action_mean, action_std, args.eval_batches, args.amp)
    print(f"[final eval] mse={metrics['mse']:.6g} mae={metrics['mae']:.6g}", flush=True)
    save_checkpoint(Path(args.output_dir) / "checkpoint-last.pt", model, optimizer, args, step, args.num_epochs, best_metric, action_mean, action_std)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--wds-root", required=True)
    p.add_argument("--pretrained-checkpoint", default="pretrained_checkpoints/large-droid+behavior/model-best.pt")
    p.add_argument("--resume", default=None, help="Resume from a robotwin2g action checkpoint")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--stage", choices=["action_decoder", "all"], default="action_decoder")
    p.add_argument("--reset-optimizer", action="store_true")

    p.add_argument("--device", default="auto")
    p.add_argument("--norm-stats-path", default="stats/droid_behavior")
    p.add_argument("--ptv3-size", default="base")
    p.add_argument("--predictor-dim", type=int, default=256)
    p.add_argument("--disable-compile", action="store_true", default=True)
    p.add_argument("--grid-size", type=float, default=0.015)
    p.add_argument("--depth-threshold", type=float, default=0.003)
    p.add_argument("--unfreeze-dinov3", action="store_true")

    p.add_argument("--clip-horizon", type=int, default=11)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--eval-num-workers", type=int, default=2)
    p.add_argument("--shuffle-buffer", type=int, default=128)
    p.add_argument("--num-epochs", type=int, default=20)
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--eval-batches", type=int, default=20)
    p.add_argument("--save-every", type=int, default=2000)

    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--world-lr", type=float, default=2e-5)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--grad-clip", type=float, default=5.0)
    p.add_argument("--loss", choices=["mse", "l1", "smooth_l1"], default="smooth_l1")
    p.add_argument("--decoder-hidden-dim", type=int, default=512)
    p.add_argument("--decoder-layers", type=int, default=3)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p


def main() -> None:
    args = build_argparser().parse_args()
    torch.manual_seed(args.seed)
    train(args)


if __name__ == "__main__":
    main()
