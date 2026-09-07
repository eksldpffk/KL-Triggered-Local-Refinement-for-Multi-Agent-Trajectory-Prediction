from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import nn

from models.global_planner import ProbabilisticTrajectory


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
    ) -> None:
        super().__init__()

        self.d_min = d_min
        self.rho = rho
        self.n_iters = n_iters
        self.eps = eps
        self.projection_strength = projection_strength

        if self.rho <= 0:
            raise ValueError("rho must be positive")

        if self.n_iters <= 0:
            raise ValueError("n_iters must be positive")

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
        )

    def x_update(
        self,
        z: torch.Tensor,
        u: torch.Tensor,
        global_traj: torch.Tensor,
    ) -> torch.Tensor:

        return (global_traj + self.rho * (z - u)) / (1.0 + self.rho)

    def _upper_triangular_mask(
        self,
        n_agents: int,
        device: torch.device,
    ) -> torch.Tensor:
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
            batch_size,
            n_agents,
            n_agents,
            dtype=torch.bool,
            device=device,
        )

        for b, pairs in enumerate(risky_pairs):
            for i, j in pairs:
                if i == j:
                    continue

                if not (0 <= i < n_agents and 0 <= j < n_agents):
                    continue

                a = min(int(i), int(j))
                c = max(int(i), int(j))

                mask[b, a, c] = True

        return mask

    def _pairwise_distances(self, traj: torch.Tensor) -> torch.Tensor:
        diff = traj[:, :, :, None, :] - traj[:, :, None, :, :]
        dist = torch.linalg.norm(diff, dim=-1)
        return dist

    def close_pairs_mask(self, traj: torch.Tensor) -> torch.Tensor:
        if traj.dim() != 4 or traj.shape[-1] != 2:
            raise ValueError(f"Expected traj shape [B,T,N,2], got {traj.shape}")

        B, _, N, _ = traj.shape

        dist = self._pairwise_distances(traj)
        close = (dist < self.d_min).any(dim=1)

        upper = self._upper_triangular_mask(N, traj.device)

        return close & upper[None, :, :]

    def project_to_safe(
        self,
        y: torch.Tensor,
        risky_pairs: Optional[RiskyPairs] = None,
        pair_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if y.dim() != 4 or y.shape[-1] != 2:
            raise ValueError(f"Expected y shape [B,T,N,2], got {y.shape}")

        B, T, N, _ = y.shape

        if pair_mask is None:
            if risky_pairs is not None:
                pair_mask = self.risky_pairs_to_mask(
                    risky_pairs=risky_pairs,
                    batch_size=B,
                    n_agents=N,
                    device=y.device,
                )
            else:
                pair_mask = self.close_pairs_mask(y)

        pair_mask = pair_mask.to(device=y.device, dtype=torch.bool)

        upper = self._upper_triangular_mask(N, y.device)
        pair_mask = pair_mask & upper[None, :, :]

        if not pair_mask.any():
            return y.clone()

        diff = y[:, :, :, None, :] - y[:, :, None, :, :]        
        dist = torch.linalg.norm(diff, dim=-1)

        dist_safe = dist.clamp(min=self.eps)
        direction = diff / dist_safe[..., None]

        fallback = torch.zeros_like(direction)
        fallback[..., 0] = 1.0
        direction = torch.where(
            (dist < self.eps)[..., None],
            fallback,
            direction,
        )

        active = (dist < self.d_min) & pair_mask[:, None, :, :]

        violation = torch.relu(self.d_min - dist)

        pair_correction = (
            0.5
            * violation[..., None]
            * direction
            * active[..., None]
            * self.projection_strength
        )

        # For pair (i,j):
        #   +correction goes to agent i
        #   -correction goes to agent j
        correction_as_first = pair_correction.sum(dim=3)
        correction_as_second = pair_correction.sum(dim=2)

        total_correction = correction_as_first - correction_as_second

        return y + total_correction

    def pair_min_distances(
        self,
        traj: torch.Tensor,
        risky_pairs: Optional[RiskyPairs] = None,
        pair_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # If a scene has no selected pairs, returns inf for that scene.

        if traj.dim() != 4 or traj.shape[-1] != 2:
            raise ValueError(f"Expected traj shape [B,T,N,2], got {traj.shape}")

        B, _, N, _ = traj.shape

        if pair_mask is None:
            if risky_pairs is not None:
                pair_mask = self.risky_pairs_to_mask(
                    risky_pairs=risky_pairs,
                    batch_size=B,
                    n_agents=N,
                    device=traj.device,
                )
            else:
                pair_mask = self.close_pairs_mask(traj)

        pair_mask = pair_mask.to(device=traj.device, dtype=torch.bool)

        upper = self._upper_triangular_mask(N, traj.device)
        pair_mask = pair_mask & upper[None, :, :]

        dist = self._pairwise_distances(traj)

        active = pair_mask[:, None, :, :] 
        dist = dist.masked_fill(~active, float("inf"))

        return dist.amin(dim=(1, 2, 3))

    @torch.no_grad()
    def refine(
        self,
        global_traj: torch.Tensor,
        risky_pairs: Optional[RiskyPairs] = None,
        pair_mask: Optional[torch.Tensor] = None,
    ) -> TensorDict:
        if global_traj.dim() != 4 or global_traj.shape[-1] != 2:
            raise ValueError(
                f"Expected global_traj shape [B,T,N,2], got {global_traj.shape}"
            )

        B, _, N, _ = global_traj.shape

        if pair_mask is None:
            if risky_pairs is not None:
                pair_mask = self.risky_pairs_to_mask(
                    risky_pairs=risky_pairs,
                    batch_size=B,
                    n_agents=N,
                    device=global_traj.device,
                )
            else:
                pair_mask = self.close_pairs_mask(global_traj)

        pair_mask = pair_mask.to(device=global_traj.device, dtype=torch.bool)

        was_refined = pair_mask.any(dim=(1, 2))

        min_before = self.pair_min_distances(
            traj=global_traj,
            pair_mask=pair_mask,
        )

        x = global_traj.clone()
        z = global_traj.clone()
        u = torch.zeros_like(global_traj)

        for _ in range(self.n_iters):
            x = self.x_update(
                z=z,
                u=u,
                global_traj=global_traj,
            )

            z_new = self.project_to_safe(
                y=x + u,
                pair_mask=pair_mask,
            )

            u = u + x - z_new
            z = z_new

        refined_traj = z

        min_after = self.pair_min_distances(
            traj=refined_traj,
            pair_mask=pair_mask,
        )

        shift = torch.linalg.norm(refined_traj - global_traj, dim=-1)

        mean_shift = shift.mean(dim=(1, 2))
        max_shift = shift.amax(dim=(1, 2))

        return {
            "refined_traj": refined_traj,
            "was_refined": was_refined,
            "pair_mask": pair_mask,
            "min_dist_before": min_before,
            "min_dist_after": min_after,
            "mean_shift": mean_shift,
            "max_shift": max_shift,
        }

    def forward(
        self,
        global_output: Union[torch.Tensor, ProbabilisticTrajectory],
        risky_pairs: Optional[RiskyPairs] = None,
        pair_mask: Optional[torch.Tensor] = None,
    ) -> TensorDict:
        if isinstance(global_output, ProbabilisticTrajectory):
            global_traj = global_output.mu
        else:
            global_traj = global_output

        return self.refine(
            global_traj=global_traj,
            risky_pairs=risky_pairs,
            pair_mask=pair_mask,
        )
