from __future__ import annotations

from typing import Dict

import torch
from torch import nn


TensorDict = Dict[str, torch.Tensor]


class SceneBackbone(nn.Module):
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
        return cls(
            T_past=config["scene"]["T_past"],
            input_dim=4,
            embedding_dim=config["model"]["backbone_dim"],
            hidden_dim=config["model"]["planner_hidden"],
        )

    def forward(self, batch: TensorDict) -> TensorDict:
        positions = batch["past_positions"]
        velocities = batch["past_velocities"]
        valid_mask = batch["valid_agent_mask"]
        if positions.dim() != 4 or velocities.shape != positions.shape:
            raise ValueError("past_positions and past_velocities must have shape [B,T,N,2]")
        if positions.shape[1] != self.T_past or positions.shape[-1] != 2:
            raise ValueError(f"Expected T_past={self.T_past} and xy coordinates")

        batch_size, time_steps, n_agents, _ = positions.shape
        relative_positions = positions - positions[:, -1:, :, :]
        features = torch.cat([relative_positions, velocities], dim=-1)
        features = features.permute(0, 2, 1, 3).reshape(batch_size, n_agents, time_steps * 4)
        agent_embeddings = self.agent_encoder(features)

        mask = valid_mask.to(agent_embeddings.device, agent_embeddings.dtype)[..., None]
        agent_embeddings = agent_embeddings * mask
        scene_embedding = agent_embeddings.sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        scene_embedding = self.scene_projector(scene_embedding)
        return {
            "agent_embeddings": agent_embeddings,
            "scene_embedding": scene_embedding,
        }
