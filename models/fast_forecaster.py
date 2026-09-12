from __future__ import annotations

import torch
from torch import nn


class ProbabilisticTrajectory:
    def __init__(
        self,
        mu: torch.Tensor,
        log_sigma: torch.Tensor,
        min_log_sigma: float = -5.0,
        max_log_sigma: float = 2.0,
    ) -> None:
        if mu.shape != log_sigma.shape:
            raise ValueError("mu and log_sigma must have the same shape")
        self.mu = mu
        self.log_sigma = log_sigma.clamp(min_log_sigma, max_log_sigma)

    @property
    def sigma(self) -> torch.Tensor:
        return torch.exp(self.log_sigma)


class FastForecaster(nn.Module):
    def __init__(
        self,
        n_agents: int,
        T_future: int,
        dt: float,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        min_log_sigma: float = -5.0,
        max_log_sigma: float = 2.0,
    ) -> None:
        super().__init__()
        self.n_agents = n_agents
        self.T_future = T_future
        self.dt = dt
        self.embedding_dim = embedding_dim
        self.min_log_sigma = min_log_sigma
        self.max_log_sigma = max_log_sigma
        self.shared_mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2 + 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.residual_head = nn.Linear(hidden_dim, T_future * 2)
        self.log_sigma_head = nn.Linear(hidden_dim, T_future * 2)

    @classmethod
    def from_config(cls, config: dict) -> "FastForecaster":
        scene_cfg = config["scene"]
        model_cfg = config["model"]
        return cls(
            n_agents=scene_cfg["n_agents"],
            T_future=scene_cfg["T_future"],
            dt=scene_cfg["dt"],
            embedding_dim=model_cfg["backbone_dim"],
            hidden_dim=model_cfg["planner_hidden"],
            min_log_sigma=model_cfg["min_log_sigma"],
            max_log_sigma=model_cfg["max_log_sigma"],
        )

    def _constant_velocity_rollout(
        self,
        last_positions: torch.Tensor,
        last_velocities: torch.Tensor,
    ) -> torch.Tensor:
        steps = torch.arange(
            1,
            self.T_future + 1,
            device=last_positions.device,
            dtype=last_positions.dtype,
        ).view(1, self.T_future, 1, 1)
        return last_positions[:, None] + steps * self.dt * last_velocities[:, None]

    def forward(
        self,
        agent_embeddings: torch.Tensor,
        scene_embedding: torch.Tensor,
        last_positions: torch.Tensor,
        last_velocities: torch.Tensor,
    ) -> ProbabilisticTrajectory:
        batch_size, n_agents, embedding_dim = agent_embeddings.shape
        if n_agents != self.n_agents or embedding_dim != self.embedding_dim:
            raise ValueError("Unexpected agent or embedding dimension")

        scene = scene_embedding[:, None, :].expand(batch_size, n_agents, embedding_dim)
        hidden = self.shared_mlp(
            torch.cat([agent_embeddings, scene, last_velocities], dim=-1)
        )
        residual = self.residual_head(hidden).view(
            batch_size, n_agents, self.T_future, 2
        ).permute(0, 2, 1, 3).contiguous()
        log_sigma = self.log_sigma_head(hidden).view(
            batch_size, n_agents, self.T_future, 2
        ).permute(0, 2, 1, 3).contiguous()
        mu = self._constant_velocity_rollout(last_positions, last_velocities) + residual
        return ProbabilisticTrajectory(
            mu,
            log_sigma,
            self.min_log_sigma,
            self.max_log_sigma,
        )
