from __future__ import annotations

from typing import Dict

import torch
from torch import nn


TensorDict = Dict[str, torch.Tensor]


class SceneBackbone(nn.Module):
    """
    Scene encoder.

    Input:
        past_positions:  [B, T_past, N, 2]
        past_velocities: [B, T_past, N, 2]

    Output:
        agent_embeddings: [B, N, D]
        scene_embedding:  [B, D]

    Idea:
        each agent history is encoded independently by an MLP;
        then we mean-pool over agents to get a global scene embedding.
    """

    def __init__(
        self,
        T_past: int,
        input_dim: int = 4,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()

        self.T_past = T_past
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim

        self.agent_encoder = nn.Sequential(
            nn.Linear(T_past * input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

        self.scene_projector = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

    @classmethod
    def from_config(cls, config: dict) -> "SceneBackbone":
        scene_cfg = config["scene"]
        model_cfg = config["model"]

        return cls(
            T_past=scene_cfg["T_past"],
            input_dim=4,
            embedding_dim=model_cfg["backbone_dim"],
            hidden_dim=model_cfg["planner_hidden"],
        )

    def _build_agent_features(
        self,
        past_positions: torch.Tensor,
        past_velocities: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert history into per-agent flat features.

        We use relative positions:
            position_t - position_last

        This makes the encoder less dependent on absolute map location.
        """
        if past_positions.dim() != 4:
            raise ValueError(
                f"past_positions must have shape [B,T,N,2], got {past_positions.shape}"
            )

        if past_velocities.dim() != 4:
            raise ValueError(
                f"past_velocities must have shape [B,T,N,2], got {past_velocities.shape}"
            )

        B, T, N, C = past_positions.shape

        if T != self.T_past:
            raise ValueError(f"Expected T_past={self.T_past}, got T={T}")

        if C != 2:
            raise ValueError(f"Expected position dim 2, got {C}")

        last_position = past_positions[:, -1:, :, :]       # [B, 1, N, 2]
        relative_positions = past_positions - last_position

        features = torch.cat(
            [relative_positions, past_velocities],
            dim=-1,
        )                                                  # [B, T, N, 4]

        features = features.permute(0, 2, 1, 3)             # [B, N, T, 4]
        features = features.reshape(B, N, T * 4)            # [B, N, T*4]

        return features

    def forward(
        self,
        past_positions: torch.Tensor | TensorDict,
        past_velocities: torch.Tensor | None = None,
    ) -> TensorDict:
        """
        Forward supports two modes:

        1. backbone(batch)
        2. backbone(past_positions, past_velocities)
        """
        valid_agent_mask = None
        if isinstance(past_positions, dict):
            batch = past_positions
            past_positions = batch["past_positions"]
            past_velocities = batch["past_velocities"]
            valid_agent_mask = batch.get("valid_agent_mask", batch.get("agent_mask"))

        if past_velocities is None:
            raise ValueError("past_velocities must be provided")

        features = self._build_agent_features(
            past_positions=past_positions,
            past_velocities=past_velocities,
        )

        agent_embeddings = self.agent_encoder(features)     # [B, N, D]

        if valid_agent_mask is None:
            scene_embedding = agent_embeddings.mean(dim=1)
        else:
            mask = valid_agent_mask.to(
                device=agent_embeddings.device, dtype=agent_embeddings.dtype
            )[..., None]
            agent_embeddings = agent_embeddings * mask
            scene_embedding = agent_embeddings.sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        scene_embedding = self.scene_projector(scene_embedding)

        return {
            "agent_embeddings": agent_embeddings,
            "scene_embedding": scene_embedding,
        }
