#!/usr/bin/env python3
"""Create a WDS manifest with explicit numeric source-episode ranges.

This is for per-scene RoboTwin finetuning where the split must be exactly
episode numbers 0-49 for train and 50-59 for evaluation, instead of a random
episode-level test ratio.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


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


def _load_valid_clips(input_dir: Path, integrity_check_file: Path) -> List[Dict[str, Any]]:
    data = _read_json(integrity_check_file)
    clips = [c for c in data.get("valid_clips", []) if c.get("ok", True)]
    if not clips:
        raise RuntimeError(f"No valid clips found in {integrity_check_file}")
    return clips


def _parse_range(spec: str) -> Tuple[int, int]:
    if ":" not in spec:
        value = int(spec)
        return value, value + 1
    left, right = spec.split(":", 1)
    start = int(left)
    end = int(right)
    if end <= start:
        raise ValueError(f"Invalid range {spec!r}: end must be greater than start")
    return start, end


def _episode_number(value: Any) -> Optional[int]:
    match = re.search(r"(\d+)$", str(value))
    if not match:
        return None
    return int(match.group(1))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", required=True, help="Generated-H5 flows root, e.g. <generated_root>/flows")
    p.add_argument("--integrity-check-file", required=True, help="JSON produced by integrity_check_robotwin_h5.py")
    p.add_argument("--output-manifest", required=True)
    p.add_argument("--train-range", default="0:50", help="Inclusive-exclusive episode number range, default 0:50")
    p.add_argument("--test-range", default="50:60", help="Inclusive-exclusive episode number range, default 50:60")
    p.add_argument("--task-key", default=None, help="Optional stable task/scene name to write into manifest rows")
    args = p.parse_args()

    input_dir = Path(args.input_dir)
    train_start, train_end = _parse_range(args.train_range)
    test_start, test_end = _parse_range(args.test_range)
    clips = _load_valid_clips(input_dir, Path(args.integrity_check_file))

    rows: List[Dict[str, Any]] = []
    train_rows: List[Dict[str, Any]] = []
    test_rows: List[Dict[str, Any]] = []
    ignored_rows = 0
    missing_episode_number = 0

    for c in clips:
        ep_num = _episode_number(c.get("source_episode", ""))
        if ep_num is None:
            missing_episode_number += 1
            ignored_rows += 1
            continue
        if train_start <= ep_num < train_end:
            split = "train"
        elif test_start <= ep_num < test_end:
            split = "test"
        else:
            ignored_rows += 1
            continue

        h5_path = str(c["h5_path"])
        ck = str(c["clip_key"])
        task = args.task_key or str(c.get("source_task", ""))
        row = {
            "clip_id": _clip_key(input_dir, h5_path, ck),
            "h5_path": h5_path,
            "clip_key": ck,
            "episode_key": _episode_key(input_dir, h5_path),
            "episode_number": int(ep_num),
            "task_key": task,
            "source_task": c.get("source_task", ""),
            "source_episode": c.get("source_episode", ""),
            "source_start": c.get("source_start", -1),
            "source_end": c.get("source_end", -1),
            "split": split,
        }
        rows.append(row)
        if split == "train":
            train_rows.append(row)
        else:
            test_rows.append(row)

    if not train_rows:
        raise RuntimeError(f"No train clips selected by --train-range {args.train_range}")
    if not test_rows:
        raise RuntimeError(f"No test clips selected by --test-range {args.test_range}")

    train_eps = sorted({r["episode_key"] for r in train_rows})
    test_eps = sorted({r["episode_key"] for r in test_rows})
    manifest = {
        "schema_version": "robotwin2g_pointworld_manifest_explicit_episode_ranges_v1",
        "domain": "robotwin2g",
        "input_dir": str(input_dir),
        "train_range": args.train_range,
        "test_range": args.test_range,
        "train_clip_keys": [r["clip_id"] for r in train_rows],
        "test_clip_keys": [r["clip_id"] for r in test_rows],
        "train_episode_keys": train_eps,
        "test_episode_keys": test_eps,
        "clips": rows,
        "stats": {
            "num_clips": len(rows),
            "num_train_clips": len(train_rows),
            "num_test_clips": len(test_rows),
            "num_train_episodes": len(train_eps),
            "num_test_episodes": len(test_eps),
            "ignored_clips": ignored_rows,
            "clips_missing_episode_number": missing_episode_number,
            "task_key": args.task_key,
            "train_episode_numbers": sorted({r["episode_number"] for r in train_rows}),
            "test_episode_numbers": sorted({r["episode_number"] for r in test_rows}),
        },
    }
    _write_json(Path(args.output_manifest), manifest)
    print(f"wrote {args.output_manifest}", flush=True)
    print(json.dumps(manifest["stats"], indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
