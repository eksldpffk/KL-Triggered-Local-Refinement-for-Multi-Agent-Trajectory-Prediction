from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from models.fast_forecaster import ProbabilisticTrajectory


RiskOutput = Dict[str, object]
RiskyPairs = List[List[Tuple[int, int]]]


def inverse_softplus(value: float) -> float:
    if value <= 0:
        raise ValueError("value must be positive")
    return value if value > 20.0 else math.log(math.expm1(value))


def kl_gaussian(
    mu_q: torch.Tensor,
    sigma_q: torch.Tensor,
    mu_p: torch.Tensor,
    sigma_p: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    sigma_q = sigma_q.clamp(min=eps)
    sigma_p = sigma_p.clamp(min=eps)
    kl = (
        torch.log(sigma_p / sigma_q)
        + (sigma_q.square() + (mu_q - mu_p).square()) / (2.0 * sigma_p.square())
        - 0.5
    )
    return kl.sum(dim=-1)


class RiskDetector(nn.Module):
    def __init__(
        self,
        d_min: float = 1.5,
        kl_threshold_init: float = 0.5,
        posterior_sigma_scale: float = 0.7,
        min_sigma: float = 0.05,
        safety_margin: float = 0.35,
        sharpness: float = 8.0,
        top_k_pairs: int = 4,
        kl_reduction: str = "mean",
        threshold_mode: str = "fixed",
        contextual_hidden: int = 16,
        safety_sensitivity: float = 1.0,
        theta_min: float = 1e-4,
        theta_max: float = 1e4,
    ) -> None:
        super().__init__()
        if d_min <= 0:
            raise ValueError("d_min must be positive")
        if safety_margin < 0:
            raise ValueError("safety_margin must be non-negative")
        if not 0 < posterior_sigma_scale <= 1:
            raise ValueError("posterior_sigma_scale must be in (0, 1]")
        if min_sigma <= 0:
            raise ValueError("min_sigma must be positive")
        if sharpness <= 0:
            raise ValueError("sharpness must be positive")
        if top_k_pairs < 0:
            raise ValueError("top_k_pairs must be non-negative")
        if contextual_hidden <= 0:
            raise ValueError("contextual_hidden must be positive")
        if theta_min <= 0 or theta_max < theta_min:
            raise ValueError("theta bounds are invalid")
        if kl_reduction not in {"mean", "sum"}:
            raise ValueError("kl_reduction must be 'mean' or 'sum'")
        if threshold_mode not in {"fixed", "contextual"}:
            raise ValueError("threshold_mode must be 'fixed' or 'contextual'")

        self.d_min = float(d_min)
        self.posterior_sigma_scale = float(posterior_sigma_scale)
        self.min_sigma = float(min_sigma)
        self.safety_margin = float(safety_margin)
        self.sharpness = float(sharpness)
        self.top_k_pairs = int(top_k_pairs)
        self.kl_reduction = kl_reduction
        self.threshold_mode = threshold_mode
        self.safety_sensitivity = float(safety_sensitivity)
        self.theta_min = float(theta_min)
        self.theta_max = float(theta_max)

        self.theta_raw = nn.Parameter(
            torch.tensor(inverse_softplus(float(kl_threshold_init)), dtype=torch.float32)
        )
        self.context_net = nn.Sequential(
            nn.Linear(2, contextual_hidden),
            nn.Tanh(),
            nn.Linear(contextual_hidden, 1),
        )
        nn.init.zeros_(self.context_net[-1].weight)
        nn.init.zeros_(self.context_net[-1].bias)

    @classmethod
    def from_config(cls, config: dict) -> "RiskDetector":
        scene_cfg = config["scene"]
        risk_cfg = config["risk"]
        return cls(
            d_min=scene_cfg["d_min"],
            kl_threshold_init=risk_cfg["kl_threshold_init"],
            posterior_sigma_scale=risk_cfg.get("posterior_sigma_scale", 0.7),
            min_sigma=risk_cfg.get("min_sigma", 0.05),
            safety_margin=risk_cfg.get("safety_margin", 0.35),
            sharpness=risk_cfg.get("sharpness", 8.0),
            top_k_pairs=risk_cfg.get("top_k_pairs", 4),
            kl_reduction=risk_cfg.get("kl_reduction", "mean"),
            threshold_mode=risk_cfg.get("threshold_mode", "fixed"),
            contextual_hidden=risk_cfg.get("contextual_hidden", 16),
            safety_sensitivity=risk_cfg.get("safety_sensitivity", 1.0),
            theta_min=risk_cfg.get("theta_min", 1e-4),
            theta_max=risk_cfg.get("theta_max", 1e4),
        )

    @property
    def theta(self) -> torch.Tensor:
        return F.softplus(self.theta_raw).clamp(self.theta_min, self.theta_max)

    def set_theta(self, value: float) -> None:
        value = max(float(value), self.theta_min)
        with torch.no_grad():
            self.theta_raw.copy_(
                torch.tensor(
                    inverse_softplus(value),
                    dtype=self.theta_raw.dtype,
                    device=self.theta_raw.device,
                )
            )

    @torch.no_grad()
    def calibrate_theta_from_max_kl(
        self,
        max_kl_values: torch.Tensor,
        target_refine_rate: float,
    ) -> float:
        if max_kl_values.numel() == 0:
            return float(self.theta.item())
        target_refine_rate = min(max(float(target_refine_rate), 0.0), 1.0)
        value = float(torch.quantile(max_kl_values.detach().flatten(), 1.0 - target_refine_rate).item())
        self.set_theta(max(value, self.theta_min))
        return float(self.theta.item())

    def _valid_agents(
        self,
        mu: torch.Tensor,
        valid_agent_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        batch_size, _, n_agents, _ = mu.shape
        if valid_agent_mask is None:
            return torch.ones(batch_size, n_agents, dtype=torch.bool, device=mu.device)
        if valid_agent_mask.shape != (batch_size, n_agents):
            raise ValueError(
                f"valid_agent_mask must have shape {(batch_size, n_agents)}, "
                f"got {valid_agent_mask.shape}"
            )
        return valid_agent_mask.to(device=mu.device, dtype=torch.bool)

    def distance_gate(
        self,
        mu: torch.Tensor,
        valid_agent_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if mu.dim() != 4 or mu.shape[-1] != 2:
            raise ValueError(f"Expected mu shape [B,T,N,2], got {mu.shape}")

        batch_size, _, n_agents, _ = mu.shape
        valid_agents = self._valid_agents(mu, valid_agent_mask)
        diff = mu[:, :, :, None, :] - mu[:, :, None, :, :]
        min_distance = torch.linalg.norm(diff, dim=-1).amin(dim=1)
        upper = torch.triu(
            torch.ones(n_agents, n_agents, dtype=torch.bool, device=mu.device),
            diagonal=1,
        ).view(1, n_agents, n_agents)
        valid_pairs = valid_agents[:, :, None] & valid_agents[:, None, :] & upper
        candidate_mask = (min_distance < self.d_min + self.safety_margin) & valid_pairs
        return candidate_mask, min_distance, valid_pairs

    def _context_features(
        self,
        prior_dist: ProbabilisticTrajectory,
        candidate_mask: torch.Tensor,
        valid_pair_mask: torch.Tensor,
        valid_agent_mask: torch.Tensor,
    ) -> torch.Tensor:
        candidate_count = candidate_mask.sum(dim=(1, 2)).float()
        pair_count = valid_pair_mask.sum(dim=(1, 2)).float().clamp(min=1.0)
        density = candidate_count / pair_count

        sigma = prior_dist.sigma
        valid = valid_agent_mask[:, None, :, None].to(dtype=sigma.dtype)
        uncertainty = (sigma * valid).sum(dim=(1, 2, 3)) / (
            valid.sum(dim=(1, 2, 3)).clamp(min=1.0) * sigma.shape[1] * sigma.shape[-1]
        )
        return torch.stack([density, torch.log1p(uncertainty)], dim=-1)

    def scene_threshold(
        self,
        context_features: torch.Tensor,
        safety_requirement: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size = context_features.shape[0]
        if self.threshold_mode == "fixed":
            return self.theta.expand(batch_size)

        if safety_requirement is None:
            safety_requirement = torch.full(
                (batch_size,), 0.5, device=context_features.device, dtype=context_features.dtype
            )
        else:
            safety_requirement = safety_requirement.to(
                device=context_features.device, dtype=context_features.dtype
            ).reshape(batch_size).clamp(0.0, 1.0)

        context_offset = self.context_net(context_features).squeeze(-1)
        safety_offset = -self.safety_sensitivity * (safety_requirement - 0.5)
        raw = self.theta_raw + context_offset + safety_offset
        return F.softplus(raw).clamp(self.theta_min, self.theta_max)

    def compute_candidate_pairwise_kl(
        self,
        prior_dist: ProbabilisticTrajectory,
        candidate_mask: torch.Tensor,
    ) -> torch.Tensor:
        mu_p = prior_dist.mu
        sigma_p = prior_dist.sigma
        batch_size, _, n_agents, _ = mu_p.shape
        kl_matrix = torch.zeros(
            batch_size, n_agents, n_agents, device=mu_p.device, dtype=mu_p.dtype
        )

        index = torch.nonzero(candidate_mask, as_tuple=False)
        if index.numel() == 0:
            return kl_matrix

        b_idx, i_idx, j_idx = index[:, 0], index[:, 1], index[:, 2]
        mu_i_p = mu_p[b_idx, :, i_idx, :]
        mu_j_p = mu_p[b_idx, :, j_idx, :]
        sigma_i_p = sigma_p[b_idx, :, i_idx, :]
        sigma_j_p = sigma_p[b_idx, :, j_idx, :]

        diff = mu_i_p - mu_j_p
        distance = torch.linalg.norm(diff, dim=-1)
        direction = diff / distance.clamp(min=1e-6)[..., None]
        fallback = torch.zeros_like(direction)
        fallback[..., 0] = 1.0
        direction = torch.where((distance < 1e-6)[..., None], fallback, direction)

        penetration = torch.relu(self.d_min - distance)
        active = (distance < self.d_min).to(distance.dtype)
        close_weight = torch.sigmoid(self.sharpness * (self.d_min - distance)) * active
        push = penetration

        mu_i_q = mu_i_p + 0.5 * push[..., None] * direction
        mu_j_q = mu_j_p - 0.5 * push[..., None] * direction
        sigma_scale = 1.0 - close_weight[..., None] * (1.0 - self.posterior_sigma_scale)
        sigma_i_floor = torch.minimum(sigma_i_p, torch.full_like(sigma_i_p, self.min_sigma))
        sigma_j_floor = torch.minimum(sigma_j_p, torch.full_like(sigma_j_p, self.min_sigma))
        sigma_i_q = torch.maximum(sigma_i_p * sigma_scale, sigma_i_floor)
        sigma_j_q = torch.maximum(sigma_j_p * sigma_scale, sigma_j_floor)

        kl_per_time = kl_gaussian(
            torch.cat([mu_i_q, mu_j_q], dim=-1),
            torch.cat([sigma_i_q, sigma_j_q], dim=-1),
            torch.cat([mu_i_p, mu_j_p], dim=-1),
            torch.cat([sigma_i_p, sigma_j_p], dim=-1),
        )
        kl_values = (
            kl_per_time.sum(dim=1)
            if self.kl_reduction == "sum"
            else kl_per_time.mean(dim=1) / 4.0
        )
        kl_matrix[b_idx, i_idx, j_idx] = kl_values
        return kl_matrix

    def detect(
        self,
        prior_dist: ProbabilisticTrajectory,
        valid_agent_mask: Optional[torch.Tensor] = None,
        safety_requirement: Optional[torch.Tensor] = None,
    ) -> RiskOutput:
        mu = prior_dist.mu
        batch_size = mu.shape[0]
        valid_agents = self._valid_agents(mu, valid_agent_mask)
        candidate_mask, min_pair_distance, valid_pair_mask = self.distance_gate(
            mu, valid_agent_mask=valid_agents
        )
        context_features = self._context_features(
            prior_dist, candidate_mask, valid_pair_mask, valid_agents
        )
        theta_scene = self.scene_threshold(context_features, safety_requirement)
        kl_matrix = self.compute_candidate_pairwise_kl(prior_dist, candidate_mask)
        max_kl = kl_matrix.flatten(1).max(dim=1).values
        risk_logits = max_kl - theta_scene
        eligible = (kl_matrix > theta_scene[:, None, None].detach()) & candidate_mask

        risky_pairs: RiskyPairs = []
        pair_trigger_mask = torch.zeros_like(candidate_mask)
        for batch_index in range(batch_size):
            candidate_index = torch.nonzero(eligible[batch_index], as_tuple=False)
            selected: List[Tuple[int, int]] = []
            if candidate_index.numel() > 0:
                scores = kl_matrix[
                    batch_index, candidate_index[:, 0], candidate_index[:, 1]
                ]
                order = torch.argsort(scores, descending=True)
                if self.top_k_pairs > 0:
                    order = order[: self.top_k_pairs]
                for rank in order:
                    i = int(candidate_index[rank, 0].item())
                    j = int(candidate_index[rank, 1].item())
                    selected.append((i, j))
                    pair_trigger_mask[batch_index, i, j] = True
            risky_pairs.append(selected)

        risk_flags = pair_trigger_mask.any(dim=(1, 2))
        return {
            "kl_matrix": kl_matrix,
            "max_kl": max_kl,
            "risk_logits": risk_logits,
            "risk_flags": risk_flags,
            "risky_pairs": risky_pairs,
            "pair_trigger_mask": pair_trigger_mask,
            "candidate_mask": candidate_mask,
            "min_pair_dist": min_pair_distance,
            "theta_scene": theta_scene,
            "density": context_features[:, 0],
            "uncertainty": torch.expm1(context_features[:, 1]),
        }

    def threshold_loss(
        self,
        risk_logits: torch.Tensor,
        interaction_heavy: torch.Tensor,
    ) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(
            risk_logits,
            interaction_heavy.float(),
        )

    def forward(
        self,
        prior_dist: ProbabilisticTrajectory,
        valid_agent_mask: Optional[torch.Tensor] = None,
        safety_requirement: Optional[torch.Tensor] = None,
    ) -> RiskOutput:
        return self.detect(
            prior_dist,
            valid_agent_mask=valid_agent_mask,
            safety_requirement=safety_requirement,
        )
