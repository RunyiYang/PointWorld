#!/usr/bin/env python3
"""Smoke-test a converted RoboTwin two-gripper WDS split."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from torch.utils.data import DataLoader

from tools.robotwin2g.wds_dataset import RobotWinWDSDataset, robotwin_action_collate


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--wds-root", required=True)
    p.add_argument("--split", default="train")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--scene-features-dim", type=int, default=33)
    p.add_argument("--robot-features-dim", type=int, default=14)
    args = p.parse_args()

    ds = RobotWinWDSDataset(
        args.wds_root,
        args.split,
        target_scene_features_dim=args.scene_features_dim,
        target_robot_features_dim=args.robot_features_dim,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, num_workers=0, collate_fn=robotwin_action_collate)
    batch = next(iter(loader))
    print("keys:", sorted(batch.keys()))
    for k in [
        "scene_flows",
        "scene_features",
        "scene_exists",
        "robot_flows",
        "robot_features",
        "robot_exists",
        "action_state",
        "action_target",
        "cam0_initial_rgb",
        "cam0_initial_depth",
    ]:
        if k in batch:
            print(k, tuple(batch[k].shape), batch[k].dtype)


if __name__ == "__main__":
    main()
