"""Minimal WDS reader/collator for RoboTwin two-gripper PointWorld action fine-tuning."""

from __future__ import annotations

import io
import json
import random
import tarfile
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from dataset_components.constants import RELEASE_CONTEXT_HORIZON
from dataset_components.robot import gather_features
from dataset_components.transforms import center_shift, normalize_colors


NP_FLOAT_DTYPES = (np.float16, np.float32, np.float64)


def _npy_load(raw: bytes) -> np.ndarray:
    return np.load(io.BytesIO(raw), allow_pickle=False)


def _jpeg_load(raw: bytes) -> np.ndarray:
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError("Failed to decode JPEG in WDS sample")
    return img[..., ::-1].copy()  # BGR -> RGB


def _pad_feature_dim(array: np.ndarray, target_dim: Optional[int]) -> np.ndarray:
    if target_dim is None:
        return array
    dim = array.shape[-1]
    if dim == target_dim:
        return array
    if dim > target_dim:
        return array[..., :target_dim]
    pad_width = [(0, 0)] * array.ndim
    pad_width[-1] = (0, target_dim - dim)
    return np.pad(array, pad_width, mode="constant")


def decode_raw_tar_sample(raw: Dict[str, bytes]) -> Dict[str, Any]:
    decoded: Dict[str, Any] = {"__key__": raw.get("__key__", "")}
    for field, value in raw.items():
        if field == "__key__":
            continue
        if field.endswith(".npy"):
            decoded[field[:-4]] = _npy_load(value)
        elif field.endswith(".jpg"):
            decoded[field[:-4]] = _jpeg_load(value)
        elif field.endswith(".json"):
            decoded[field[:-5]] = json.loads(value.decode("utf-8"))
        elif field.endswith(".txt"):
            decoded[field[:-4]] = value.decode("utf-8")
        else:
            decoded[field] = value
    return decoded


def prepare_pointworld_action_sample(
    sample: Dict[str, Any],
    *,
    domain: str = "behavior",
    target_scene_features_dim: Optional[int] = None,
    target_robot_features_dim: Optional[int] = None,
) -> Dict[str, Any]:
    """Prepare one decoded WDS sample for PointWorld BaseModel.forward.

    The converter already writes direct scene/robot flows. This function applies
    the lightweight PointWorld-compatible transformations needed by the model:
    mean-centering, color normalization, bimanual feature construction, and
    feature-dimension padding/truncation when a checkpoint expects a fixed dim.
    """
    required = [
        "scene_flows",
        "scene_colors",
        "scene_normals",
        "scene_visibility",
        "scene_depth_valid_mask",
        "robot_flows",
        "robot_colors",
        "robot_normals",
        "right_gripper_pose",
        "left_gripper_pose",
        "right_gripper_open",
        "left_gripper_open",
        "action_state",
        "action_target",
        "action_mask",
    ]
    missing = [k for k in required if k not in sample]
    if missing:
        raise KeyError(f"Sample {sample.get('__key__')} missing required keys: {missing}")

    # Ensure canonical dtypes/shapes.
    for key in ["scene_flows", "scene_normals", "robot_flows", "robot_normals", "action_state", "action_target"]:
        sample[key] = np.asarray(sample[key], dtype=np.float32)
    for key in ["right_gripper_pose", "left_gripper_pose", "right_gripper_open", "left_gripper_open"]:
        sample[key] = np.asarray(sample[key], dtype=np.float32)
    sample["scene_colors"] = np.asarray(sample["scene_colors"], dtype=np.uint8)
    sample["robot_colors"] = np.asarray(sample["robot_colors"], dtype=np.uint8)
    sample["scene_visibility"] = np.asarray(sample["scene_visibility"], dtype=np.bool_)
    sample["scene_depth_valid_mask"] = np.asarray(sample["scene_depth_valid_mask"], dtype=np.bool_)
    sample["action_mask"] = np.asarray(sample["action_mask"], dtype=np.bool_)

    # PointWorld transform helpers update camera extrinsics and gripper poses when centering.
    sample = center_shift(sample)
    sample = normalize_colors(sample)

    sample = gather_features(
        sample,
        has_bimanual_robot=True,
        domain=domain,
        context_horizon=RELEASE_CONTEXT_HORIZON,
    )
    sample["scene_features"] = _pad_feature_dim(sample["scene_features"], target_scene_features_dim).astype(np.float32)
    sample["robot_features"] = _pad_feature_dim(sample["robot_features"], target_robot_features_dim).astype(np.float32)

    sample["__domain__"] = domain
    sample.setdefault("__out_of_bounds__", False)
    sample.setdefault("__scene_exceeds_max__", False)
    return sample


class RobotWinWDSDataset(IterableDataset):
    def __init__(
        self,
        wds_root: str | Path,
        split: str,
        *,
        domain: str = "behavior",
        shuffle: bool = False,
        seed: int = 42,
        target_scene_features_dim: Optional[int] = None,
        target_robot_features_dim: Optional[int] = None,
        shuffle_buffer: int = 128,
    ):
        self.wds_root = Path(wds_root)
        self.split = split
        self.domain = domain
        self.shuffle = shuffle
        self.seed = int(seed)
        self.target_scene_features_dim = target_scene_features_dim
        self.target_robot_features_dim = target_robot_features_dim
        self.shuffle_buffer = int(shuffle_buffer)
        split_dir = self.wds_root / split
        self.shards = sorted(split_dir.glob("*.tar"))
        if not self.shards:
            raise FileNotFoundError(f"No .tar shards found in {split_dir}")

    def _iter_shard(self, shard: Path) -> Iterator[Dict[str, Any]]:
        current_key: Optional[str] = None
        current: Dict[str, bytes] = {}
        with tarfile.open(shard, mode="r") as tar:
            for member in tar:
                if not member.isfile():
                    continue
                name = Path(member.name).name
                if "." not in name:
                    continue
                key, field = name.split(".", 1)
                if current_key is None:
                    current_key = key
                if key != current_key:
                    current["__key__"] = current_key.encode("utf-8")
                    yield decode_raw_tar_sample(current)
                    current_key = key
                    current = {}
                f = tar.extractfile(member)
                if f is None:
                    continue
                current[field] = f.read()
            if current_key is not None:
                current["__key__"] = current_key.encode("utf-8")
                yield decode_raw_tar_sample(current)

    def _iter_samples(self) -> Iterator[Dict[str, Any]]:
        shards = list(self.shards)
        worker = get_worker_info()
        if worker is not None:
            shards = shards[worker.id :: worker.num_workers]
            rng = random.Random(self.seed + worker.id)
        else:
            rng = random.Random(self.seed)
        if self.shuffle:
            rng.shuffle(shards)
        for shard in shards:
            for sample in self._iter_shard(shard):
                yield prepare_pointworld_action_sample(
                    sample,
                    domain=self.domain,
                    target_scene_features_dim=self.target_scene_features_dim,
                    target_robot_features_dim=self.target_robot_features_dim,
                )

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        if not self.shuffle or self.shuffle_buffer <= 1:
            yield from self._iter_samples()
            return
        rng = random.Random(self.seed)
        buf: List[Dict[str, Any]] = []
        for sample in self._iter_samples():
            buf.append(sample)
            if len(buf) >= self.shuffle_buffer:
                idx = rng.randrange(len(buf))
                yield buf.pop(idx)
        while buf:
            idx = rng.randrange(len(buf))
            yield buf.pop(idx)


def _to_tensor(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        try:
            tensor = torch.from_numpy(value)
        except ValueError:
            tensor = torch.from_numpy(value.copy())
        if value.dtype in NP_FLOAT_DTYPES:
            tensor = tensor.float()
        return tensor
    return value


def _pad_time_n(tensor: torch.Tensor, n_max: int) -> torch.Tensor:
    T, N = tensor.shape[:2]
    out_shape = (T, n_max) + tuple(tensor.shape[2:])
    out = torch.zeros(out_shape, dtype=tensor.dtype)
    out[:, :N, ...] = tensor
    return out


def robotwin_action_collate(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not batch:
        raise ValueError("Empty batch")
    batch_t = [{k: _to_tensor(v) for k, v in s.items()} for s in batch]
    n_scene_max = max(int(s["scene_flows"].shape[1]) for s in batch_t)
    n_robot_max = max(int(s["robot_flows"].shape[1]) for s in batch_t)
    T = int(batch_t[0]["scene_flows"].shape[0])

    keys = sorted(set().union(*(s.keys() for s in batch_t)))
    out: Dict[str, Any] = {}
    meta_keys = {"__key__", "__domain__", "metadata", "__out_of_bounds__", "__scene_exceeds_max__"}
    scene_exists: List[torch.Tensor] = []
    robot_exists: List[torch.Tensor] = []

    for sample in batch_t:
        ns = int(sample["scene_flows"].shape[1])
        nr = int(sample["robot_flows"].shape[1])
        sm = torch.zeros((T, n_scene_max), dtype=torch.bool)
        sm[:, :ns] = True
        rm = torch.zeros((T, n_robot_max), dtype=torch.bool)
        rm[:, :nr] = True
        scene_exists.append(sm)
        robot_exists.append(rm)

    for key in keys:
        vals = [s.get(key, None) for s in batch_t]
        if any(v is None for v in vals):
            continue
        if key in meta_keys:
            out[key] = vals
            continue
        first = vals[0]
        if not isinstance(first, torch.Tensor):
            out[key] = vals
            continue

        padded_vals: List[torch.Tensor] = []
        if key.startswith("scene_") or key in {"gt_scene_flows", "scene_features"}:
            for v in vals:
                assert isinstance(v, torch.Tensor)
                if v.ndim >= 2 and int(v.shape[1]) != n_scene_max:
                    padded_vals.append(_pad_time_n(v, n_scene_max))
                else:
                    padded_vals.append(v)
        elif key.startswith("robot_"):
            for v in vals:
                assert isinstance(v, torch.Tensor)
                if v.ndim >= 2 and int(v.shape[1]) != n_robot_max:
                    padded_vals.append(_pad_time_n(v, n_robot_max))
                else:
                    padded_vals.append(v)
        else:
            padded_vals = [v for v in vals if isinstance(v, torch.Tensor)]
        out[key] = torch.stack(padded_vals, dim=0)

    out["scene_exists"] = torch.stack(scene_exists, dim=0)
    out["robot_exists"] = torch.stack(robot_exists, dim=0)
    return out


def read_metadata_count(wds_root: str | Path, split: str) -> int:
    meta_path = Path(wds_root) / "metadata_rank0.json"
    if not meta_path.exists():
        return -1
    with open(meta_path, "r") as fp:
        meta = json.load(fp)
    return int(meta.get(split, {}).get("processed_count", -1))
