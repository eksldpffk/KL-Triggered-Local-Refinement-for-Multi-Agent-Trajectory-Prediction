from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from models.global_planner import ProbabilisticTrajectory


TensorDict = Dict[str, object]
RiskyPairs = List[List[Tuple[int, int]]]


def inverse_softplus(x: float) -> float:
    """Return y such that softplus(y) ~= x for x > 0."""
    if x <= 0:
        raise ValueError("inverse_softplus input must be positive")
    # expm1 overflows for large thresholds; softplus(y) ~= y in that regime.
    if x > 20.0:
        return x
    return math.log(math.expm1(x))


def kl_gaussian(
    mu_q: torch.Tensor,
    sigma_q: torch.Tensor,
    mu_p: torch.Tensor,
    sigma_p: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Analytic KL(q || p) for diagonal Gaussians, summed over the last dim."""
    sigma_q = torch.clamp(sigma_q, min=eps)
    sigma_p = torch.clamp(sigma_p, min=eps)
    kl = (
        torch.log(sigma_p / sigma_q)
        + (sigma_q.square() + (mu_q - mu_p).square()) / (2.0 * sigma_p.square())
        - 0.5
    )
    return kl.sum(dim=-1)


class RiskDetector(nn.Module):
    """
    Pair-level KL risk detector with an optional context-dependent threshold.

    The final matched-budget archive accidentally made the KL threshold irrelevant:
    candidate pairs were already inside d_min and an emergency distance override then
    refined every candidate. This implementation separates three concepts:

      1) candidate radius: d_pair + safety_margin;
      2) KL decision: normalized KL > theta(scene);
      3) optional emergency override: only for deep penetration below
         d_pair - collision_margin.

    Contextual theta uses scene density and predicted uncertainty. A stricter
    safety_requirement in [0, 1] monotonically lowers theta and therefore calls the
    refiner more often.
    """

    def __init__(
        self,
        d_min: float = 1.5,
        kl_threshold_init: float = 0.5,
        posterior_sigma_scale: float = 0.7,
        min_sigma: float = 0.05,
        safety_margin: float = 0.35,
        sharpness: float = 8.0,
        top_k_pairs: int = 4,
        force_refine_collisions: bool = False,
        collision_margin: float = 0.25,
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
        if collision_margin < 0:
            raise ValueError("collision_margin must be non-negative")
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
        self.force_refine_collisions = bool(force_refine_collisions)
        self.collision_margin = float(collision_margin)
        self.kl_reduction = kl_reduction
        self.threshold_mode = threshold_mode
        self.safety_sensitivity = float(safety_sensitivity)
        self.theta_min = float(theta_min)
        self.theta_max = float(theta_max)

        theta_raw_init = inverse_softplus(float(kl_threshold_init))
        self.theta_raw = nn.Parameter(torch.tensor(theta_raw_init, dtype=torch.float32))

        # Density and log-uncertainty offsets. Zero initialization makes the new
        # detector exactly scalar-thresholded until it is calibrated/trained.
        self.context_net = nn.Sequential(
            nn.Linear(2, contextual_hidden),
            nn.Tanh(),
            nn.Linear(contextual_hidden, 1),
        )
        nn.init.zeros_(self.context_net[-1].weight)
        nn.init.zeros_(self.context_net[-1].bias)

        self.register_buffer("context_mean", torch.zeros(2, dtype=torch.float32))
        self.register_buffer("context_std", torch.ones(2, dtype=torch.float32))

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
            force_refine_collisions=risk_cfg.get("force_refine_collisions", False),
            collision_margin=risk_cfg.get("collision_margin", 0.25),
            kl_reduction=risk_cfg.get("kl_reduction", "mean"),
            threshold_mode=risk_cfg.get("threshold_mode", "fixed"),
            contextual_hidden=risk_cfg.get("contextual_hidden", 16),
            safety_sensitivity=risk_cfg.get("safety_sensitivity", 1.0),
            theta_min=risk_cfg.get("theta_min", 1e-4),
            theta_max=risk_cfg.get("theta_max", 1e4),
        )

    @property
    def theta(self) -> torch.Tensor:
        """Backward-compatible scalar/base threshold."""
        return F.softplus(self.theta_raw).clamp(self.theta_min, self.theta_max)

    def set_theta(self, theta_value: float) -> None:
        theta_value = max(float(theta_value), self.theta_min)
        raw = inverse_softplus(theta_value)
        with torch.no_grad():
            self.theta_raw.copy_(
                torch.tensor(raw, dtype=self.theta_raw.dtype, device=self.theta_raw.device)
            )

    @torch.no_grad()
    def set_context_stats(self, features: torch.Tensor) -> None:
        """Fit normalization statistics for [density, log1p(uncertainty)]."""
        if features.ndim != 2 or features.shape[1] != 2:
            raise ValueError(f"Expected context features [M,2], got {features.shape}")
        self.context_mean.copy_(features.mean(dim=0).to(self.context_mean))
        std = features.std(dim=0, unbiased=False).clamp(min=1e-4)
        self.context_std.copy_(std.to(self.context_std))

    @torch.no_grad()
    def calibrate_theta_from_max_kl(
        self,
        max_kl_values: torch.Tensor,
        target_refine_rate: float = 0.25,
    ) -> float:
        """Set the base theta to a requested global refinement-rate quantile."""
        if max_kl_values.numel() == 0:
            return float(self.theta.item())
        target_refine_rate = max(0.0, min(1.0, float(target_refine_rate)))
        q = 1.0 - target_refine_rate
        theta_value = torch.quantile(max_kl_values.detach().flatten(), q).item()
        theta_value = max(theta_value, self.theta_min)
        self.set_theta(theta_value)
        return theta_value

    def _valid_agent_mask(
        self,
        mu: torch.Tensor,
        valid_agent_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        B, _, N, _ = mu.shape
        if valid_agent_mask is None:
            return torch.ones(B, N, dtype=torch.bool, device=mu.device)
        if valid_agent_mask.shape != (B, N):
            raise ValueError(
                f"valid_agent_mask must have shape {(B, N)}, got {valid_agent_mask.shape}"
            )
        return valid_agent_mask.to(device=mu.device, dtype=torch.bool)

    def _pairwise_safety_distance(
        self,
        mu: torch.Tensor,
        pairwise_d_min: Optional[torch.Tensor],
    ) -> torch.Tensor:
        B, _, N, _ = mu.shape
        if pairwise_d_min is None:
            return torch.full(
                (B, N, N), self.d_min, dtype=mu.dtype, device=mu.device
            )
        if pairwise_d_min.shape != (B, N, N):
            raise ValueError(
                f"pairwise_d_min must have shape {(B, N, N)}, got {pairwise_d_min.shape}"
            )
        return pairwise_d_min.to(device=mu.device, dtype=mu.dtype)

    def _pairwise_distances(self, mu: torch.Tensor) -> torch.Tensor:
        diff = mu[:, :, :, None, :] - mu[:, :, None, :, :]
        return torch.linalg.norm(diff, dim=-1)

    def _distance_masks(
        self,
        mu: torch.Tensor,
        valid_agent_mask: Optional[torch.Tensor] = None,
        pairwise_d_min: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if mu.dim() != 4 or mu.shape[-1] != 2:
            raise ValueError(f"Expected mu shape [B,T,N,2], got {mu.shape}")
        B, _, N, _ = mu.shape
        valid_agents = self._valid_agent_mask(mu, valid_agent_mask)
        d_pair = self._pairwise_safety_distance(mu, pairwise_d_min)

        dist = self._pairwise_distances(mu)
        eye = torch.eye(N, dtype=torch.bool, device=mu.device)
        dist = dist.masked_fill(eye.view(1, 1, N, N), float("inf"))
        min_pair_dist = dist.amin(dim=1)

        upper = torch.triu(
            torch.ones(N, N, dtype=torch.bool, device=mu.device), diagonal=1
        ).view(1, N, N)
        valid_pairs = valid_agents[:, :, None] & valid_agents[:, None, :] & upper

        candidate_mask = (
            min_pair_dist < (d_pair + self.safety_margin)
        ) & valid_pairs

        # Emergency override means deep penetration, not every candidate pair.
        emergency_radius = torch.clamp(d_pair - self.collision_margin, min=0.0)
        critical_mask = (min_pair_dist < emergency_radius) & valid_pairs
        return candidate_mask, critical_mask, min_pair_dist, valid_pairs

    def candidate_pair_mask(
        self,
        mu: torch.Tensor,
        valid_agent_mask: Optional[torch.Tensor] = None,
        pairwise_d_min: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        candidate_mask, _, _, _ = self._distance_masks(
            mu, valid_agent_mask=valid_agent_mask, pairwise_d_min=pairwise_d_min
        )
        return candidate_mask

    def _context_features(
        self,
        prior_dist: ProbabilisticTrajectory,
        candidate_mask: torch.Tensor,
        valid_pair_mask: torch.Tensor,
        valid_agent_mask: torch.Tensor,
    ) -> torch.Tensor:
        candidate_count = candidate_mask.sum(dim=(1, 2)).float()
        valid_pair_count = valid_pair_mask.sum(dim=(1, 2)).float().clamp(min=1.0)
        density = candidate_count / valid_pair_count

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
        B = context_features.shape[0]
        if safety_requirement is None:
            safety_requirement = torch.full(
                (B,), 0.5, device=context_features.device, dtype=context_features.dtype
            )
        else:
            safety_requirement = safety_requirement.to(
                device=context_features.device, dtype=context_features.dtype
            ).reshape(B)
            safety_requirement = safety_requirement.clamp(0.0, 1.0)

        if self.threshold_mode == "fixed":
            return self.theta.expand(B)

        normalized = (context_features - self.context_mean) / self.context_std
        context_offset = self.context_net(normalized).squeeze(-1)

        # Strict safety (1.0) lowers theta; permissive safety (0.0) raises it.
        safety_offset = -self.safety_sensitivity * (safety_requirement - 0.5)
        raw = self.theta_raw + context_offset + safety_offset
        return F.softplus(raw).clamp(self.theta_min, self.theta_max)

    def compute_candidate_pairwise_kl(
        self,
        prior_dist: ProbabilisticTrajectory,
        candidate_mask: torch.Tensor,
        pairwise_d_min: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute normalized KL(q_safe || p_fast) only for candidate pairs."""
        mu_p = prior_dist.mu
        sigma_p = prior_dist.sigma
        B, T, N, _ = mu_p.shape
        kl_matrix = torch.zeros(B, N, N, device=mu_p.device, dtype=mu_p.dtype)

        idx = torch.nonzero(candidate_mask, as_tuple=False)
        if idx.numel() == 0:
            return kl_matrix

        b_idx, i_idx, j_idx = idx[:, 0], idx[:, 1], idx[:, 2]
        mu_i_p = mu_p[b_idx, :, i_idx, :]
        mu_j_p = mu_p[b_idx, :, j_idx, :]
        sigma_i_p = sigma_p[b_idx, :, i_idx, :]
        sigma_j_p = sigma_p[b_idx, :, j_idx, :]

        diff = mu_i_p - mu_j_p
        dist = torch.linalg.norm(diff, dim=-1)
        dist_safe = dist.clamp(min=1e-6)
        direction = diff / dist_safe[..., None]
        fallback = torch.zeros_like(direction)
        fallback[..., 0] = 1.0
        direction = torch.where((dist < 1e-6)[..., None], fallback, direction)

        if pairwise_d_min is None:
            pair_d = torch.full(
                (idx.shape[0],), self.d_min, device=mu_p.device, dtype=mu_p.dtype
            )
        else:
            pair_d = pairwise_d_min.to(mu_p)[b_idx, i_idx, j_idx]

        safety_radius = pair_d[:, None] + self.safety_margin
        penetration = torch.relu(safety_radius - dist)
        close_weight = torch.sigmoid(self.sharpness * (safety_radius - dist))
        push = penetration * close_weight

        mu_i_q = mu_i_p + 0.5 * push[..., None] * direction
        mu_j_q = mu_j_p - 0.5 * push[..., None] * direction

        sigma_scale = 1.0 - close_weight[..., None] * (1.0 - self.posterior_sigma_scale)
        sigma_i_q = torch.clamp(sigma_i_p * sigma_scale, min=self.min_sigma)
        sigma_j_q = torch.clamp(sigma_j_p * sigma_scale, min=self.min_sigma)

        pair_mu_p = torch.cat([mu_i_p, mu_j_p], dim=-1)
        pair_mu_q = torch.cat([mu_i_q, mu_j_q], dim=-1)
        pair_sigma_p = torch.cat([sigma_i_p, sigma_j_p], dim=-1)
        pair_sigma_q = torch.cat([sigma_i_q, sigma_j_q], dim=-1)

        kl_per_time = kl_gaussian(
            mu_q=pair_mu_q,
            sigma_q=pair_sigma_q,
            mu_p=pair_mu_p,
            sigma_p=pair_sigma_p,
        )
        if self.kl_reduction == "sum":
            kl_values = kl_per_time.sum(dim=1)
        else:
            # Mean over time and the four pair coordinates. This keeps theta
            # comparable when changing the prediction horizon (10 -> 30 frames).
            kl_values = kl_per_time.mean(dim=1) / 4.0

        kl_matrix[b_idx, i_idx, j_idx] = kl_values
        return kl_matrix

    def compute_pairwise_kl(
        self,
        prior_dist: ProbabilisticTrajectory,
        valid_agent_mask: Optional[torch.Tensor] = None,
        pairwise_d_min: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        mu = prior_dist.mu
        B, _, N, _ = mu.shape
        valid_agents = self._valid_agent_mask(mu, valid_agent_mask)
        upper = torch.triu(
            torch.ones(N, N, dtype=torch.bool, device=mu.device), diagonal=1
        ).view(1, N, N)
        all_pairs = valid_agents[:, :, None] & valid_agents[:, None, :] & upper
        return self.compute_candidate_pairwise_kl(
            prior_dist, all_pairs, pairwise_d_min=pairwise_d_min
        )

    def detect(
        self,
        prior_dist: ProbabilisticTrajectory,
        valid_agent_mask: Optional[torch.Tensor] = None,
        pairwise_d_min: Optional[torch.Tensor] = None,
        safety_requirement: Optional[torch.Tensor] = None,
    ) -> TensorDict:
        mu_p = prior_dist.mu
        B, _, N, _ = mu_p.shape
        valid_agents = self._valid_agent_mask(mu_p, valid_agent_mask)
        candidate_mask, critical_mask, min_pair_dist, valid_pair_mask = self._distance_masks(
            mu_p,
            valid_agent_mask=valid_agents,
            pairwise_d_min=pairwise_d_min,
        )

        context_features = self._context_features(
            prior_dist, candidate_mask, valid_pair_mask, valid_agents
        )
        theta_scene = self.scene_threshold(context_features, safety_requirement)

        kl_matrix = self.compute_candidate_pairwise_kl(
            prior_dist, candidate_mask, pairwise_d_min=pairwise_d_min
        )
        max_kl = kl_matrix.view(B, -1).max(dim=1).values
        risk_logits = max_kl - theta_scene
        kl_pair_mask = (
            kl_matrix > theta_scene[:, None, None].detach()
        ) & candidate_mask

        risky_pairs: RiskyPairs = []
        final_pair_mask = torch.zeros_like(candidate_mask)
        for b in range(B):
            emergency_idx = torch.nonzero(critical_mask[b], as_tuple=False)
            kl_idx = torch.nonzero(kl_pair_mask[b] & ~critical_mask[b], as_tuple=False)

            selected: List[Tuple[int, int]] = []
            # Emergency pairs are never dropped by top-k.
            if self.force_refine_collisions and emergency_idx.numel() > 0:
                for row in emergency_idx:
                    selected.append((int(row[0].item()), int(row[1].item())))

            if kl_idx.numel() > 0:
                scores = kl_matrix[b, kl_idx[:, 0], kl_idx[:, 1]]
                order = torch.argsort(scores, descending=True)
                if self.top_k_pairs > 0:
                    remaining = max(self.top_k_pairs - len(selected), 0)
                    order = order[:remaining]
                for k in order:
                    pair = (int(kl_idx[k, 0].item()), int(kl_idx[k, 1].item()))
                    if pair not in selected:
                        selected.append(pair)

            for i, j in selected:
                final_pair_mask[b, i, j] = True
            risky_pairs.append(selected)

        risk_flags = final_pair_mask.any(dim=(1, 2))
        return {
            "kl_matrix": kl_matrix,
            "max_kl": max_kl,
            "risk_logits": risk_logits,
            "risk_flags": risk_flags,
            "risky_pairs": risky_pairs,
            "pair_trigger_mask": final_pair_mask,
            "candidate_mask": candidate_mask,
            "critical_mask": critical_mask,
            "min_pair_dist": min_pair_dist,
            "theta_scene": theta_scene,
            "context_features": context_features,
            "density": context_features[:, 0],
            "uncertainty": torch.expm1(context_features[:, 1]),
        }

    def threshold_loss(
        self,
        kl_matrix: Optional[torch.Tensor] = None,
        is_hard: Optional[torch.Tensor] = None,
        risk_logits: Optional[torch.Tensor] = None,
        theta_scene: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """BCE loss for scalar/contextual threshold calibration."""
        if is_hard is None:
            raise ValueError("is_hard/target labels must be provided")
        if risk_logits is None:
            if kl_matrix is None:
                raise ValueError("Provide risk_logits or kl_matrix")
            B = kl_matrix.shape[0]
            max_kl = kl_matrix.view(B, -1).max(dim=1).values.detach()
            if theta_scene is None:
                theta_scene = self.theta.expand(B)
            risk_logits = max_kl - theta_scene
        return F.binary_cross_entropy_with_logits(risk_logits, is_hard.float())

    def forward(
        self,
        prior_dist: ProbabilisticTrajectory,
        valid_agent_mask: Optional[torch.Tensor] = None,
        pairwise_d_min: Optional[torch.Tensor] = None,
        safety_requirement: Optional[torch.Tensor] = None,
    ) -> TensorDict:
        return self.detect(
            prior_dist,
            valid_agent_mask=valid_agent_mask,
            pairwise_d_min=pairwise_d_min,
            safety_requirement=safety_requirement,
        )
