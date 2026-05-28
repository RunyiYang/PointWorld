#!/usr/bin/env python3
"""Collect per-scene PointWorld action finetune evaluation JSONs into tables."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


DEFAULT_TASKS = [
    "place_bread_skillet",
    "move_stapler_pad",
    "pick_diverse_bottles",
    "place_phone_stand",
    "stamp_seal",
    "rotate_qrcode",
    "adjust_bottle",
    "beat_block_hammer",
    "click_bell",
    "lift_pot",
    "place_bread_basket",
]


FIELDNAMES = [
    "scene",
    "stage",
    "checkpoint_kind",
    "step",
    "metadata_count",
    "train_episodes",
    "test_episodes",
    "missing_requested_episodes",
    "actuator_rmse_cm",
    "actuator_cd_cm",
    "scene_rmse_cm",
    "scene_cd_cm",
    "action_vector_rmse_raw",
    "action_vector_mae_raw",
    "checkpoint",
]


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as fp:
        return json.load(fp)


def _fmt(value: Any) -> str:
    try:
        f = float(value)
    except Exception:
        return ""
    if not math.isfinite(f):
        return ""
    return f"{f:.6f}"


def _stage_from_checkpoint(path: str) -> str:
    if "_action_decoder" in path or "action_decoder" in path:
        return "stage1_action_decoder"
    if "_all" in path or "/all" in path:
        return "stage2_all"
    if "_lora" in path or "lora" in path:
        return "stage2_lora"
    return "unknown"


def _kind_from_checkpoint(path: str) -> str:
    name = Path(path).name
    if name == "checkpoint-best.pt":
        return "best"
    if name == "checkpoint-last.pt":
        return "last"
    return name.replace(".pt", "")


def _manifest_stats(wds_root: Path) -> Dict[str, Any]:
    meta = _read_json(wds_root / "metadata_rank0.json") or {}
    manifest_path = meta.get("manifest")
    manifest = _read_json(Path(manifest_path)) if manifest_path else None
    if manifest:
        return {
            "train_episodes": len(manifest.get("train_episode_keys", [])),
            "test_episodes": len(manifest.get("test_episode_keys", [])),
        }
    return {
        "train_episodes": "",
        "test_episodes": "",
    }


def _selection_stats(scene_root: Path) -> Dict[str, Any]:
    meta = _read_json(scene_root / "selection_metadata.json") or {}
    missing = meta.get("missing_requested_episodes", [])
    return {"missing_requested_episodes": ",".join(str(x) for x in missing)}


def collect_rows(tasks: Iterable[str], out_root: Path, eval_root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for scene in tasks:
        scene_root = out_root / scene
        eval_json = _read_json(eval_root / f"{scene}.json")
        if not eval_json:
            continue
        stats = {}
        stats.update(_manifest_stats(scene_root / "wds"))
        stats.update(_selection_stats(scene_root))
        for result in eval_json.get("results", []):
            ckpt = str(result.get("checkpoint", ""))
            rows.append(
                {
                    "scene": scene,
                    "stage": _stage_from_checkpoint(ckpt),
                    "checkpoint_kind": _kind_from_checkpoint(ckpt),
                    "step": result.get("step", ""),
                    "metadata_count": result.get("metadata_count", ""),
                    "train_episodes": stats.get("train_episodes", ""),
                    "test_episodes": stats.get("test_episodes", ""),
                    "missing_requested_episodes": stats.get("missing_requested_episodes", ""),
                    "actuator_rmse_cm": result.get("action_rmse_cm", ""),
                    "actuator_cd_cm": result.get("action_cd_cm", ""),
                    "scene_rmse_cm": result.get("scene_rmse_cm", ""),
                    "scene_cd_cm": result.get("scene_cd_cm", ""),
                    "action_vector_rmse_raw": result.get("action_vector_rmse", result.get("rmse", "")),
                    "action_vector_mae_raw": result.get("action_vector_mae", result.get("mae", "")),
                    "checkpoint": ckpt,
                }
            )
    return rows


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in FIELDNAMES})


def write_md(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    final_rows = [r for r in rows if r.get("stage") == "stage2_all" and r.get("checkpoint_kind") == "best"]
    if not final_rows:
        final_rows = rows
    lines = [
        "| Scene | Actuator RMSE (cm) | Actuator CD (cm) | Scene RMSE (cm) | Scene CD (cm) | Train eps | Eval eps | Missing requested eps |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in final_rows:
        lines.append(
            "| {scene} | {armse} | {acd} | {srmse} | {scd} | {train_eps} | {test_eps} | {missing} |".format(
                scene=row.get("scene", ""),
                armse=_fmt(row.get("actuator_rmse_cm")),
                acd=_fmt(row.get("actuator_cd_cm")),
                srmse=_fmt(row.get("scene_rmse_cm")),
                scd=_fmt(row.get("scene_cd_cm")),
                train_eps=row.get("train_episodes", ""),
                test_eps=row.get("test_episodes", ""),
                missing=row.get("missing_requested_episodes", ""),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-root", default="/work/runyi_yang/FloWAM/data/FloWAM/FloWAM_PointWorld_PerScene")
    p.add_argument("--eval-root", default="/work/runyi_yang/FloWAM/outputs/robotwin2g_perscene_eval")
    p.add_argument("--output-csv", default="/work/runyi_yang/FloWAM/outputs/robotwin2g_perscene_eval/per_scene_scores.csv")
    p.add_argument("--output-md", default="/work/runyi_yang/FloWAM/outputs/robotwin2g_perscene_eval/per_scene_scores.md")
    p.add_argument("--tasks", nargs="*", default=DEFAULT_TASKS)
    args = p.parse_args()

    rows = collect_rows(args.tasks, Path(args.out_root), Path(args.eval_root))
    write_csv(Path(args.output_csv), rows)
    write_md(Path(args.output_md), rows)
    print(f"wrote {args.output_csv}", flush=True)
    print(f"wrote {args.output_md}", flush=True)
    print(f"rows: {len(rows)}", flush=True)


if __name__ == "__main__":
    main()
