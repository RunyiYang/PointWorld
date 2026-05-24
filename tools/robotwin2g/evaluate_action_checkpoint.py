#!/usr/bin/env python3
"""Evaluate a RoboTwin2G action checkpoint on a WDS split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from torch.utils.data import DataLoader

from tools.robotwin2g.render_flow_videos import _load_model
from tools.robotwin2g.train_robotwin_action import evaluate, load_action_stats, load_torch
from tools.robotwin2g.wds_dataset import RobotWinWDSDataset, read_metadata_count, robotwin_action_collate


def _evaluate_one(args: argparse.Namespace, checkpoint: str) -> Dict[str, Any]:
    local_args = argparse.Namespace(**vars(args))
    local_args.checkpoint = checkpoint
    model, device, feature_dims = _load_model(local_args)

    payload = load_torch(checkpoint, map_location="cpu")
    if "action_mean" in payload:
        action_mean = torch.as_tensor(payload["action_mean"], dtype=torch.float32, device=device)
        action_std = torch.as_tensor(payload["action_std"], dtype=torch.float32, device=device).clamp(min=1e-6)
    else:
        action_mean, action_std = load_action_stats(args.wds_root, device)

    ds = RobotWinWDSDataset(
        args.wds_root,
        args.split,
        shuffle=False,
        seed=args.seed,
        target_scene_features_dim=feature_dims.scene_features_dim,
        target_robot_features_dim=feature_dims.robot_features_dim,
        shuffle_buffer=0,
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=robotwin_action_collate,
        persistent_workers=args.num_workers > 0,
    )
    metrics = evaluate(model, loader, device, action_mean, action_std, args.max_batches, args.amp)
    count = read_metadata_count(args.wds_root, args.split)
    return {
        "checkpoint": str(checkpoint),
        "step": int(payload.get("step", -1)),
        "epoch": int(payload.get("epoch", -1)),
        "best_metric_in_checkpoint": float(payload.get("best_metric", float("nan"))),
        "split": args.split,
        "metadata_count": int(count),
        "max_batches": int(args.max_batches),
        "batch_size": int(args.batch_size),
        "mse": float(metrics["mse"]),
        "mae": float(metrics["mae"]),
    }


def main() -> None:
    args = build_argparser().parse_args()
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    results: List[Dict[str, Any]] = []
    for checkpoint in args.checkpoints:
        print(f"[eval] {checkpoint}", flush=True)
        results.append(_evaluate_one(args, checkpoint))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    payload = {
        "wds_root": str(args.wds_root),
        "split": args.split,
        "results": results,
    }
    with open(args.output_json, "w") as fp:
        json.dump(payload, fp, indent=2)
    for item in results:
        print(
            f"[result] step={item['step']} mae={item['mae']:.8f} mse={item['mse']:.8f} "
            f"checkpoint={item['checkpoint']}",
            flush=True,
        )
    print(f"[done] wrote {args.output_json}", flush=True)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--wds-root", required=True)
    p.add_argument("--checkpoints", nargs="+", required=True)
    p.add_argument("--output-json", required=True)
    p.add_argument("--split", default="test", choices=["train", "test"])
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--max-batches", type=int, default=0, help="0 means full split")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    p.add_argument("--amp", action="store_true")

    p.add_argument("--pretrained-checkpoint", default="pretrained_checkpoints/large-droid+behavior/model-best.pt")
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
