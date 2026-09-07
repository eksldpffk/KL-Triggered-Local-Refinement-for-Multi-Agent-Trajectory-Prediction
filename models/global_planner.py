from __future__ import annotations

import math
from typing import Optional

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
            raise ValueError(
                f"mu and log_sigma must have same shape, got {mu.shape} and {log_sigma.shape}"
            )

        self.mu = mu
        self.log_sigma = torch.clamp(
            log_sigma,
            min=min_log_sigma,
            max=max_log_sigma,
        )

    @property
    def sigma(self) -> torch.Tensor:
        return torch.exp(self.log_sigma)

    def sample(self) -> torch.Tensor:
        eps = torch.randn_like(self.mu)
        return self.mu + self.sigma * eps

    def rsample(self) -> torch.Tensor:
        eps = torch.randn_like(self.mu)
        return self.mu + self.sigma * eps

    def log_prob(
        self,
        target: torch.Tensor,
        reduce: bool = True,
    ) -> torch.Tensor:
        # Gussian log probability
        if target.shape != self.mu.shape:
            raise ValueError(
                f"target must have shape {self.mu.shape}, got {target.shape}"
            )

        sigma = self.sigma

        log_prob_per_dim = (
            -0.5 * ((target - self.mu) / sigma) ** 2
            - self.log_sigma
            - 0.5 * math.log(2.0 * math.pi)
        )

        if reduce:
            return log_prob_per_dim.sum(dim=(1, 2, 3))

        return log_prob_per_dim

    def nll(
        self,
        target: torch.Tensor,
        reduction: str = "mean",
    ) -> torch.Tensor:
    
        nll_per_scene = -self.log_prob(target, reduce=True)

        if reduction == "mean":
            return nll_per_scene.mean()

        if reduction == "sum":
            return nll_per_scene.sum()

        if reduction == "none":
            return nll_per_scene

        raise ValueError(f"Unknown reduction: {reduction}")


class GlobalPlanner(nn.Module):

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

        # agent embedding + scene embedding + last velocity
        planner_input_dim = embedding_dim * 2 + 2

        self.shared_mlp = nn.Sequential(
            nn.Linear(planner_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.residual_head = nn.Linear(hidden_dim, T_future * 2)
        self.log_sigma_head = nn.Linear(hidden_dim, T_future * 2)

    @classmethod
    def from_config(cls, config: dict) -> "GlobalPlanner":
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
        B, N, _ = last_positions.shape

        steps = torch.arange(
            1,
            self.T_future + 1,
            device=last_positions.device,
            dtype=last_positions.dtype,
        )

        steps = steps.view(1, self.T_future, 1, 1)

        base_traj = (
            last_positions[:, None, :, :]
            + steps * self.dt * last_velocities[:, None, :, :]
        )

        return base_traj

    def forward(
        self,
        agent_embeddings: torch.Tensor,
        scene_embedding: torch.Tensor,
        last_positions: torch.Tensor,
        last_velocities: torch.Tensor,
    ) -> ProbabilisticTrajectory:
        B, N, D = agent_embeddings.shape

        if N != self.n_agents:
            raise ValueError(f"Expected n_agents={self.n_agents}, got {N}")

        if D != self.embedding_dim:
            raise ValueError(f"Expected embedding_dim={self.embedding_dim}, got {D}")

        scene_expanded = scene_embedding[:, None, :].expand(B, N, D)

        planner_features = torch.cat(
            [
                agent_embeddings,
                scene_expanded,
                last_velocities,
            ],
            dim=-1,
        )                                                  

        hidden = self.shared_mlp(planner_features)

        residual = self.residual_head(hidden)
        log_sigma = self.log_sigma_head(hidden)

        residual = residual.view(B, N, self.T_future, 2)
        log_sigma = log_sigma.view(B, N, self.T_future, 2)

        residual = residual.permute(0, 2, 1, 3).contiguous()
        log_sigma = log_sigma.permute(0, 2, 1, 3).contiguous()

        base_traj = self._constant_velocity_rollout(
            last_positions=last_positions,
            last_velocities=last_velocities,
        )

        mu = base_traj + residual

        return ProbabilisticTrajectory(
            mu=mu,
            log_sigma=log_sigma,
            min_log_sigma=self.min_log_sigma,
            max_log_sigma=self.max_log_sigma,
        )
