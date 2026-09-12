from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import nn

from models.fast_forecaster import ProbabilisticTrajectory


TensorDict = Dict[str, torch.Tensor]
RiskyPairs = List[List[Tuple[int, int]]]


class LocalRefiner(nn.Module):
    def __init__(
        self,
        d_min: float = 1.5,
        rho: float = 1.0,
        n_iters: int = 15,
        eps: float = 1e-4,
        projection_strength: float = 1.0,
        projection_sweeps: int = 32,
        projection_tol: float = 1e-3,
    ) -> None:
        super().__init__()
        if d_min <= 0:
            raise ValueError("d_min must be positive")
        if rho <= 0:
            raise ValueError("rho must be positive")
        if n_iters <= 0:
            raise ValueError("n_iters must be positive")
        if not 0 < projection_strength <= 1:
            raise ValueError("projection_strength must be in (0, 1]")
        if projection_sweeps <= 0:
            raise ValueError("projection_sweeps must be positive")
        if projection_tol < 0:
            raise ValueError("projection_tol must be non-negative")

        self.d_min = float(d_min)
        self.rho = float(rho)
        self.n_iters = int(n_iters)
        self.eps = float(eps)
        self.projection_strength = float(projection_strength)
        self.projection_sweeps = int(projection_sweeps)
        self.projection_tol = float(projection_tol)

    @classmethod
    def from_config(cls, config: dict) -> "LocalRefiner":
        scene_cfg = config["scene"]
        admm_cfg = config["admm"]
        return cls(
            d_min=scene_cfg["d_min"],
            rho=admm_cfg["rho"],
            n_iters=admm_cfg["n_iters"],
            eps=admm_cfg["eps"],
            projection_strength=admm_cfg.get("projection_strength", 1.0),
            projection_sweeps=admm_cfg.get("projection_sweeps", 32),
            projection_tol=admm_cfg.get("projection_tol", 1e-3),
        )

    def x_update(
        self,
        z: torch.Tensor,
        u: torch.Tensor,
        fast_traj: torch.Tensor,
    ) -> torch.Tensor:
        return (fast_traj + self.rho * (z - u)) / (1.0 + self.rho)

    @staticmethod
    def _upper_mask(n_agents: int, device: torch.device) -> torch.Tensor:
        return torch.triu(
            torch.ones(n_agents, n_agents, dtype=torch.bool, device=device),
            diagonal=1,
        )

    def risky_pairs_to_mask(
        self,
        risky_pairs: RiskyPairs,
        batch_size: int,
        n_agents: int,
        device: torch.device,
    ) -> torch.Tensor:
        mask = torch.zeros(
            batch_size, n_agents, n_agents, dtype=torch.bool, device=device
        )
        for batch_index, pairs in enumerate(risky_pairs):
            for i, j in pairs:
                if i == j or not (0 <= i < n_agents and 0 <= j < n_agents):
                    continue
                i, j = sorted((int(i), int(j)))
                mask[batch_index, i, j] = True
        return mask

    @staticmethod
    def _pairwise_distances(trajectory: torch.Tensor) -> torch.Tensor:
        diff = trajectory[:, :, :, None, :] - trajectory[:, :, None, :, :]
        return torch.linalg.norm(diff, dim=-1)

    def close_pairs_mask(self, trajectory: torch.Tensor) -> torch.Tensor:
        if trajectory.dim() != 4 or trajectory.shape[-1] != 2:
            raise ValueError(f"Expected trajectory [B,T,N,2], got {trajectory.shape}")
        n_agents = trajectory.shape[2]
        close = (self._pairwise_distances(trajectory) < self.d_min).any(dim=1)
        return close & self._upper_mask(n_agents, trajectory.device)[None]

    def _resolve_pair_mask(
        self,
        trajectory: torch.Tensor,
        risky_pairs: Optional[RiskyPairs],
        pair_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        batch_size, _, n_agents, _ = trajectory.shape
        if pair_mask is None:
            pair_mask = (
                self.risky_pairs_to_mask(
                    risky_pairs, batch_size, n_agents, trajectory.device
                )
                if risky_pairs is not None
                else self.close_pairs_mask(trajectory)
            )
        pair_mask = pair_mask.to(device=trajectory.device, dtype=torch.bool)
        return pair_mask & self._upper_mask(n_agents, trajectory.device)[None]

    def project_to_safe(
        self,
        y: torch.Tensor,
        risky_pairs: Optional[RiskyPairs] = None,
        pair_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if y.dim() != 4 or y.shape[-1] != 2:
            raise ValueError(f"Expected y [B,T,N,2], got {y.shape}")
        pair_mask = self._resolve_pair_mask(y, risky_pairs, pair_mask)
        if not pair_mask.any():
            return y.clone()

        projected = y.clone()
        selected_pairs = torch.nonzero(pair_mask.any(dim=0), as_tuple=False)
        for _ in range(self.projection_sweeps):
            for pair in selected_pairs:
                i = int(pair[0].item())
                j = int(pair[1].item())
                selected_scene = pair_mask[:, i, j]
                diff = projected[:, :, i] - projected[:, :, j]
                distance = torch.linalg.norm(diff, dim=-1)
                direction = diff / distance.clamp(min=self.eps)[..., None]
                fallback = torch.zeros_like(direction)
                fallback[..., 0] = 1.0
                direction = torch.where(
                    (distance < self.eps)[..., None], fallback, direction
                )
                active = (distance < self.d_min) & selected_scene[:, None]
                correction = (
                    0.5
                    * torch.relu(self.d_min - distance)[..., None]
                    * direction
                    * active[..., None]
                    * self.projection_strength
                )
                projected[:, :, i] += correction
                projected[:, :, j] -= correction

            min_distance = self.pair_min_distances(projected, pair_mask=pair_mask)
            active_scene = pair_mask.any(dim=(1, 2))
            if not active_scene.any() or torch.all(
                min_distance[active_scene] >= self.d_min - self.projection_tol
            ):
                break
        return projected

    def pair_min_distances(
        self,
        trajectory: torch.Tensor,
        risky_pairs: Optional[RiskyPairs] = None,
        pair_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if trajectory.dim() != 4 or trajectory.shape[-1] != 2:
            raise ValueError(
                f"Expected trajectory [B,T,N,2], got {trajectory.shape}"
            )
        pair_mask = self._resolve_pair_mask(trajectory, risky_pairs, pair_mask)
        distance = self._pairwise_distances(trajectory)
        distance = distance.masked_fill(~pair_mask[:, None], float("inf"))
        return distance.amin(dim=(1, 2, 3))

    @torch.no_grad()
    def refine(
        self,
        fast_traj: torch.Tensor,
        risky_pairs: Optional[RiskyPairs] = None,
        pair_mask: Optional[torch.Tensor] = None,
    ) -> TensorDict:
        if fast_traj.dim() != 4 or fast_traj.shape[-1] != 2:
            raise ValueError(
                f"Expected fast_traj [B,T,N,2], got {fast_traj.shape}"
            )

        pair_mask = self._resolve_pair_mask(fast_traj, risky_pairs, pair_mask)
        min_before = self.pair_min_distances(fast_traj, pair_mask=pair_mask)
        x = fast_traj.clone()
        z = fast_traj.clone()
        u = torch.zeros_like(fast_traj)

        for _ in range(self.n_iters):
            x = self.x_update(z, u, fast_traj)
            z_new = self.project_to_safe(x + u, pair_mask=pair_mask)
            u = u + x - z_new
            z = z_new

        shift = torch.linalg.norm(z - fast_traj, dim=-1)
        return {
            "refined_traj": z,
            "was_refined": pair_mask.any(dim=(1, 2)),
            "pair_mask": pair_mask,
            "min_dist_before": min_before,
            "min_dist_after": self.pair_min_distances(z, pair_mask=pair_mask),
            "mean_shift": shift.mean(dim=(1, 2)),
            "max_shift": shift.amax(dim=(1, 2)),
        }

    def forward(
        self,
        fast_output: Union[torch.Tensor, ProbabilisticTrajectory],
        risky_pairs: Optional[RiskyPairs] = None,
        pair_mask: Optional[torch.Tensor] = None,
    ) -> TensorDict:
        trajectory = (
            fast_output.mu
            if isinstance(fast_output, ProbabilisticTrajectory)
            else fast_output
        )
        return self.refine(trajectory, risky_pairs=risky_pairs, pair_mask=pair_mask)
