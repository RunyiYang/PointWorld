#!/usr/bin/env python3
"""Create a train/test WDS manifest from valid generated-H5 RoboTwin clips.

The split is episode-level, not clip-level, to prevent leakage between adjacent
sliding windows from the same demonstration.  This follows the PointWorld data
branch pattern of separating integrity checking, manifest construction, and WDS
conversion.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def _read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fp:
        return json.load(fp)


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(obj, fp, indent=2, sort_keys=True)


def _clip_key(input_dir: Path, h5_path: str, clip_key: str) -> str:
    path = Path(h5_path)
    try:
        rel = path.resolve().relative_to(input_dir.resolve())
    except Exception:
        rel = path
    return f"{rel.as_posix()}::{clip_key}"


def _episode_key(input_dir: Path, h5_path: str) -> str:
    path = Path(h5_path)
    try:
        rel = path.resolve().relative_to(input_dir.resolve())
    except Exception:
        rel = path
    return rel.with_suffix("").as_posix()


def _task_key(input_dir: Path, h5_path: str, source_task: str | None = None, split_scope: str = "task") -> str:
    if split_scope == "global":
        return "__global__"
    if source_task:
        return str(source_task)
    path = Path(h5_path)
    try:
        rel = path.resolve().relative_to(input_dir.resolve())
        if len(rel.parts) >= 2:
            return rel.parts[0]
    except Exception:
        pass
    return path.parent.name or "task"


def _load_valid_clips(input_dir: Path, integrity_check_file: Path | None) -> List[Dict[str, Any]]:
    if integrity_check_file is not None:
        data = _read_json(integrity_check_file)
        clips = [c for c in data.get("valid_clips", []) if c.get("ok", True)]
        return clips

    # Fallback: use generated_h5_manifest.jsonl from the H5 conversion step.
    manifest_path = input_dir.parent / "generated_h5_manifest.jsonl"
    if not manifest_path.exists():
        manifest_path = input_dir / "generated_h5_manifest.jsonl"
    clips: List[Dict[str, Any]] = []
    with open(manifest_path, "r", encoding="utf-8") as fp:
        for line in fp:
            row = json.loads(line)
            clips.append({
                "h5_path": row["h5_path"],
                "clip_key": row["clip_key"],
                "source_task": row.get("source_task", ""),
                "source_episode": row.get("source_episode", ""),
                "source_start": row.get("start", -1),
                "source_end": row.get("end", -1),
            })
    return clips


def _split_episodes(episodes_by_task: Dict[str, List[str]], test_ratio: float, seed: int) -> Tuple[set[str], set[str]]:
    if not 0.0 <= test_ratio < 1.0:
        raise ValueError("--test-ratio must be in [0, 1)")
    rng = random.Random(seed)
    train_episodes: set[str] = set()
    test_episodes: set[str] = set()
    for task in sorted(episodes_by_task):
        episodes = sorted(set(episodes_by_task[task]))
        rng.shuffle(episodes)
        if len(episodes) <= 1 or test_ratio == 0.0:
            n_test = 0
        else:
            n_test = max(1, int(round(len(episodes) * test_ratio)))
            n_test = min(n_test, len(episodes) - 1)
        task_test = set(episodes[:n_test])
        for ep in episodes:
            if ep in task_test:
                test_episodes.add(ep)
            else:
                train_episodes.add(ep)
    return train_episodes, test_episodes


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", required=True, help="Generated-H5 flows root, e.g. <generated_root>/flows")
    p.add_argument("--integrity-check-file", default=None, help="JSON produced by integrity_check_robotwin_h5.py")
    p.add_argument("--output-manifest", required=True)
    p.add_argument("--test-ratio", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split-scope", choices=["task", "global"], default="task")
    args = p.parse_args()

    input_dir = Path(args.input_dir)
    integrity_path = Path(args.integrity_check_file) if args.integrity_check_file else None
    clips = _load_valid_clips(input_dir, integrity_path)
    if not clips:
        raise RuntimeError("No valid clips found. Run integrity_check_robotwin_h5.py first.")

    rows: List[Dict[str, Any]] = []
    episodes_by_task: Dict[str, List[str]] = defaultdict(list)
    for c in clips:
        h5_path = str(c["h5_path"])
        ck = str(c["clip_key"])
        ep_key = _episode_key(input_dir, h5_path)
        task = _task_key(input_dir, h5_path, c.get("source_task"), args.split_scope)
        clip_id = _clip_key(input_dir, h5_path, ck)
        row = {
            "clip_id": clip_id,
            "h5_path": h5_path,
            "clip_key": ck,
            "episode_key": ep_key,
            "task_key": task,
            "source_task": c.get("source_task", ""),
            "source_episode": c.get("source_episode", ""),
            "source_start": c.get("source_start", -1),
            "source_end": c.get("source_end", -1),
        }
        rows.append(row)
        episodes_by_task[task].append(ep_key)

    train_eps, test_eps = _split_episodes(episodes_by_task, args.test_ratio, args.seed)
    train_rows: List[Dict[str, Any]] = []
    test_rows: List[Dict[str, Any]] = []
    for row in rows:
        if row["episode_key"] in test_eps:
            row["split"] = "test"
            test_rows.append(row)
        else:
            row["split"] = "train"
            train_rows.append(row)

    manifest = {
        "schema_version": "robotwin2g_pointworld_manifest_v1",
        "domain": "robotwin2g",
        "input_dir": str(input_dir),
        "seed": args.seed,
        "test_ratio": args.test_ratio,
        "split_scope": args.split_scope,
        "train_clip_keys": [r["clip_id"] for r in train_rows],
        "test_clip_keys": [r["clip_id"] for r in test_rows],
        "train_episode_keys": sorted(train_eps),
        "test_episode_keys": sorted(test_eps),
        "clips": rows,
        "stats": {
            "num_clips": len(rows),
            "num_train_clips": len(train_rows),
            "num_test_clips": len(test_rows),
            "num_train_episodes": len(train_eps),
            "num_test_episodes": len(test_eps),
            "num_tasks": len(episodes_by_task),
            "tasks": {task: len(set(eps)) for task, eps in sorted(episodes_by_task.items())},
        },
    }
    _write_json(Path(args.output_manifest), manifest)
    print(f"wrote {args.output_manifest}", flush=True)
    print(json.dumps(manifest["stats"], indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
