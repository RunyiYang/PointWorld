"""Action decoder wrapper around the released PointWorld BaseModel."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import torch
import torch.nn as nn

from arguments import parse_args
from pointworld.base import BaseModel

try:
    from pointworld.checkpoint_contract import apply_model_contract_to_args, read_checkpoint_contract
except Exception:  # pragma: no cover - older checkout fallback
    apply_model_contract_to_args = None
    read_checkpoint_contract = None


@dataclass
class FeatureDims:
    scene_features_dim: int
    robot_features_dim: int


def infer_feature_dims_from_checkpoint(checkpoint: dict, fallback_scene: int = 33, fallback_robot: int = 14) -> FeatureDims:
    state = checkpoint.get("model", checkpoint)
    scene_key = "scene_feature_encoder.scene_raw_feat_proj.weight"
    robot_key = "robot_proj.fc1.weight"
    scene_dim = fallback_scene
    robot_dim = fallback_robot
    if scene_key in state:
        scene_dim = int(state[scene_key].shape[1])
    if robot_key in state:
        robot_dim = int(state[robot_key].shape[1])
    return FeatureDims(scene_features_dim=scene_dim, robot_features_dim=robot_dim)


def build_pointworld_args(
    *,
    checkpoint: Optional[dict],
    device: str,
    norm_stats_path: str,
    ptv3_size: str,
    predictor_dim: int,
    disable_compile: bool,
    grid_size: float,
    depth_threshold: float,
) -> object:
    """Build an argparse Namespace compatible with BaseModel."""
    args = parse_args(skip_command_line=True)
    args.device = device
    args.distributed = False
    args.domains = ["behavior"]
    args.data_dirs = []
    args.norm_stats_path = norm_stats_path
    args.ptv3_size = ptv3_size
    args.predictor_dim = predictor_dim
    args.disable_compile = disable_compile
    args.grid_size = grid_size
    args.depth_threshold = depth_threshold
    args._explicit_cli_dests = set()
    if checkpoint is not None and read_checkpoint_contract is not None and apply_model_contract_to_args is not None:
        try:
            contract, _ = read_checkpoint_contract(checkpoint, context="robotwin2g action finetune checkpoint")
            apply_model_contract_to_args(args, contract, context="robotwin2g action finetune", explicit_cli_dests=set())
        except Exception as exc:
            warnings.warn(f"Could not apply PointWorld checkpoint contract; using CLI/default architecture: {exc}")
    return args


def masked_mean(x: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    mask = mask.to(dtype=x.dtype)
    while mask.ndim < x.ndim:
        mask = mask.unsqueeze(-1)
    numer = (x * mask).sum(dim=dim)
    denom = mask.sum(dim=dim).clamp(min=1.0)
    return numer / denom


class MLPActionDecoder(nn.Module):
    def __init__(self, in_dim: int, action_dim: int, action_horizon: int, hidden_dim: int = 512, layers: int = 3):
        super().__init__()
        if layers < 1:
            raise ValueError("layers must be >= 1")
        modules = [nn.LayerNorm(in_dim)]
        last = in_dim
        for _ in range(max(0, layers - 1)):
            modules += [nn.Linear(last, hidden_dim), nn.GELU()]
            last = hidden_dim
        modules.append(nn.Linear(last, action_dim * action_horizon))
        self.net = nn.Sequential(*modules)
        self.action_dim = int(action_dim)
        self.action_horizon = int(action_horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.net(x)
        return y.view(x.shape[0], self.action_horizon, self.action_dim)


class PointWorldActionModel(nn.Module):
    """PointWorld encoder + small action decoder.

    The wrapper uses the pretrained scene encoder, robot projection, temporal
    embeddings, and normalization stats. The action head predicts a future action
    sequence with shape (B, action_horizon, action_dim), normalized by the caller.
    """

    def __init__(
        self,
        pointworld_args: object,
        data_info_dict: Dict[str, int],
        *,
        action_dim: int,
        action_horizon: int,
        state_dim: int,
        decoder_hidden_dim: int = 512,
        decoder_layers: int = 3,
        rank: int = 0,
    ):
        super().__init__()
        self.world_model = BaseModel(pointworld_args, data_info_dict, rank=rank, cpu_pg=None)
        self.channels = int(pointworld_args.predictor_dim)
        self.action_dim = int(action_dim)
        self.action_horizon = int(action_horizon)
        self.state_dim = int(state_dim)
        decoder_in = self.channels * 2 + self.state_dim
        self.action_decoder = MLPActionDecoder(
            decoder_in,
            action_dim=self.action_dim,
            action_horizon=self.action_horizon,
            hidden_dim=decoder_hidden_dim,
            layers=decoder_layers,
        )

    def load_pointworld_checkpoint(self, checkpoint: dict, strict: bool = False) -> Tuple[list, list]:
        state = checkpoint.get("model", checkpoint)
        missing, unexpected = self.world_model.load_state_dict(state, strict=strict)
        return list(missing), list(unexpected)

    def set_train_stage(self, stage: str, *, unfreeze_dinov3: bool = False) -> None:
        if stage not in {"action_decoder", "all"}:
            raise ValueError("stage must be 'action_decoder' or 'all'")
        train_world = stage == "all"
        for p in self.world_model.parameters():
            p.requires_grad_(train_world)
        for p in self.action_decoder.parameters():
            p.requires_grad_(True)

        # The released SceneEncoder2D freezes DINOv3. Keep that default unless explicitly requested.
        scene_encoder = getattr(getattr(self.world_model, "scene_feature_encoder", None), "scene_encoder", None)
        dinov3 = getattr(scene_encoder, "dinov3", None)
        if dinov3 is not None and not unfreeze_dinov3:
            for p in dinov3.parameters():
                p.requires_grad_(False)
            dinov3.eval()

    def train(self, mode: bool = True):  # type: ignore[override]
        super().train(mode)
        # Keep frozen DINO in eval mode even when the wrapper trains.
        scene_encoder = getattr(getattr(self.world_model, "scene_feature_encoder", None), "scene_encoder", None)
        dinov3 = getattr(scene_encoder, "dinov3", None)
        if dinov3 is not None and not any(p.requires_grad for p in dinov3.parameters()):
            dinov3.eval()
        return self

    def _set_domain_indices(self, batch: Dict[str, object]) -> None:
        domains = batch.get("__domain__", ["behavior"] * int(batch["scene_flows"].shape[0]))
        if isinstance(domains, str):
            domains = [domains] * int(batch["scene_flows"].shape[0])
        idx = torch.tensor(
            [self.world_model._domain_to_index[str(d)] for d in domains],
            dtype=torch.long,
            device=batch["scene_flows"].device,
        )
        self.world_model._current_domain_indices = idx

    def encode_context(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        wm = self.world_model
        self._set_domain_indices(batch)
        scene_feat0 = wm.scene_feature_encoder(batch)  # (B,Ns,C)
        robot_feat_seq = wm.normalize_robot_features(batch["robot_features"])
        robot_raw = wm.robot_proj(robot_feat_seq)  # (B,T,Nr,C)
        B, T, Nr, _ = robot_raw.shape
        if T != wm.time_steps.numel():
            # The pretrained PointWorld release is fixed to horizon 11. This branch
            # permits shorter smoke tests but horizon 11 is strongly recommended.
            time_steps = torch.linspace(0, 1, T, device=robot_raw.device, dtype=wm.time_steps.dtype)
        else:
            time_steps = wm.time_steps.to(device=robot_raw.device)
        time_emb = wm.time_embed(time_steps.view(1, T)).unsqueeze(2).expand(B, T, Nr, -1)
        robot_feat = robot_raw + time_emb + wm.robot_type_emb.view(1, 1, 1, -1)
        return scene_feat0, robot_feat, batch["action_state"]

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        scene_feat0, robot_feat, action_state = self.encode_context(batch)
        scene_mask0 = batch["scene_exists"][:, 0].bool()
        robot_mask = batch["robot_exists"].bool()
        scene_pool = masked_mean(scene_feat0, scene_mask0, dim=1)
        robot_pool = masked_mean(robot_feat.reshape(robot_feat.shape[0], -1, robot_feat.shape[-1]), robot_mask.reshape(robot_mask.shape[0], -1), dim=1)
        state0 = action_state[:, 0]
        if state0.shape[-1] > self.state_dim:
            state0 = state0[..., : self.state_dim]
        elif state0.shape[-1] < self.state_dim:
            state0 = torch.nn.functional.pad(state0, (0, self.state_dim - state0.shape[-1]))
        x = torch.cat([scene_pool, robot_pool, state0], dim=-1)
        return self.action_decoder(x)
