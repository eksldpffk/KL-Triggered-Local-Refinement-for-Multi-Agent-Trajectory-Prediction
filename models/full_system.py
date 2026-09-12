from __future__ import annotations

from typing import Dict, List, Tuple

import torch
from torch import nn

from models.backbone import SceneBackbone
from models.fast_forecaster import FastForecaster
from models.local_refiner import LocalRefiner
from models.risk_detector import RiskDetector


TensorDict = Dict[str, torch.Tensor]
RiskyPairs = List[List[Tuple[int, int]]]


class FullSystem(nn.Module):
    def __init__(
        self,
        backbone: SceneBackbone,
        fast_forecaster: FastForecaster,
        risk_detector: RiskDetector,
        local_refiner: LocalRefiner,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.fast_forecaster = fast_forecaster
        self.risk_detector = risk_detector
        self.local_refiner = local_refiner

    @classmethod
    def from_config(cls, config: dict) -> "FullSystem":
        return cls(
            backbone=SceneBackbone.from_config(config),
            fast_forecaster=FastForecaster.from_config(config),
            risk_detector=RiskDetector.from_config(config),
            local_refiner=LocalRefiner.from_config(config),
        )

    def forecast_parameters(self):
        yield from self.backbone.parameters()
        yield from self.fast_forecaster.parameters()

    def theta_parameters(self):
        yield from self.risk_detector.parameters()

    def encode_and_forecast(self, batch: TensorDict) -> Dict[str, object]:
        encoded = self.backbone(batch)
        prior_dist = self.fast_forecaster(
            agent_embeddings=encoded["agent_embeddings"],
            scene_embedding=encoded["scene_embedding"],
            last_positions=batch["past_positions"][:, -1],
            last_velocities=batch["past_velocities"][:, -1],
        )
        return {"encoded": encoded, "prior_dist": prior_dist}

    def _empty_refiner_output(
        self,
        trajectory: torch.Tensor,
        scene_mask: torch.Tensor,
    ) -> TensorDict:
        batch_size, _, n_agents, _ = trajectory.shape
        return {
            "refined_traj": trajectory.clone(),
            "was_refined": scene_mask.clone(),
            "min_dist_before": torch.full(
                (batch_size,), float("inf"), device=trajectory.device, dtype=trajectory.dtype
            ),
            "min_dist_after": torch.full(
                (batch_size,), float("inf"), device=trajectory.device, dtype=trajectory.dtype
            ),
            "mean_shift": torch.zeros(batch_size, device=trajectory.device, dtype=trajectory.dtype),
            "max_shift": torch.zeros(batch_size, device=trajectory.device, dtype=trajectory.dtype),
            "pair_mask": torch.zeros(
                batch_size, n_agents, n_agents, device=trajectory.device, dtype=torch.bool
            ),
        }

    def _run_local_refiner(
        self,
        trajectory: torch.Tensor,
        scene_mask: torch.Tensor,
        risky_pairs: RiskyPairs,
    ) -> TensorDict:
        scene_mask = scene_mask.to(device=trajectory.device, dtype=torch.bool)
        output = self._empty_refiner_output(trajectory, scene_mask)
        selected = torch.nonzero(scene_mask, as_tuple=False).flatten()
        if selected.numel() == 0:
            return output

        selected_indices = selected.detach().cpu().tolist()
        subset = self.local_refiner(
            fast_output=trajectory.index_select(0, selected),
            risky_pairs=[risky_pairs[int(index)] for index in selected_indices],
        )
        for key in ["refined_traj", "min_dist_before", "min_dist_after", "mean_shift", "max_shift", "pair_mask"]:
            output[key][selected] = subset[key]
        return output

    def _all_pairs_mask(
        self,
        batch_size: int,
        n_agents: int,
        device: torch.device,
        valid_agent_mask: torch.Tensor,
    ) -> torch.Tensor:
        upper = torch.triu(
            torch.ones(n_agents, n_agents, dtype=torch.bool, device=device), diagonal=1
        ).view(1, n_agents, n_agents)
        valid = valid_agent_mask.to(device=device, dtype=torch.bool)
        return upper & valid[:, :, None] & valid[:, None, :]

    def _run_full_scene_refiner(
        self,
        trajectory: torch.Tensor,
        scene_mask: torch.Tensor,
        valid_agent_mask: torch.Tensor,
    ) -> TensorDict:
        scene_mask = scene_mask.to(device=trajectory.device, dtype=torch.bool)
        output = self._empty_refiner_output(trajectory, scene_mask)
        selected = torch.nonzero(scene_mask, as_tuple=False).flatten()
        if selected.numel() == 0:
            return output

        selected_traj = trajectory.index_select(0, selected)
        selected_valid = valid_agent_mask.to(trajectory.device).index_select(0, selected)
        pair_mask = self._all_pairs_mask(
            selected_traj.shape[0],
            selected_traj.shape[2],
            trajectory.device,
            selected_valid,
        )
        subset = self.local_refiner(fast_output=selected_traj, pair_mask=pair_mask)
        for key in ["refined_traj", "min_dist_before", "min_dist_after", "mean_shift", "max_shift", "pair_mask"]:
            output[key][selected] = subset[key]
        return output

    def _distance_local_selection(
        self,
        trajectory: torch.Tensor,
        valid_agent_mask: torch.Tensor,
    ) -> Tuple[RiskyPairs, torch.Tensor]:
        candidate_mask, min_distance, _ = self.risk_detector.distance_gate(
            trajectory, valid_agent_mask=valid_agent_mask
        )
        risky_pairs: RiskyPairs = []
        for batch_index in range(trajectory.shape[0]):
            candidate_index = torch.nonzero(candidate_mask[batch_index], as_tuple=False)
            selected: List[Tuple[int, int]] = []
            if candidate_index.numel() > 0:
                score = min_distance[
                    batch_index, candidate_index[:, 0], candidate_index[:, 1]
                ]
                order = torch.argsort(score)
                if self.risk_detector.top_k_pairs > 0:
                    order = order[: self.risk_detector.top_k_pairs]
                for rank in order:
                    selected.append(
                        (
                            int(candidate_index[rank, 0].item()),
                            int(candidate_index[rank, 1].item()),
                        )
                    )
            risky_pairs.append(selected)

        flags = torch.tensor(
            [bool(pairs) for pairs in risky_pairs],
            dtype=torch.bool,
            device=trajectory.device,
        )
        return risky_pairs, flags

    def _stats(
        self,
        mode: str,
        final_traj: torch.Tensor,
        risk_output: Dict[str, object] | None,
        refiner_output: TensorDict | None,
        refiner_called: torch.Tensor,
    ) -> Dict[str, object]:
        batch_size = final_traj.shape[0]
        stats: Dict[str, object] = {
            "mode": mode,
            "refiner_called": refiner_called,
            "refine_rate": float(refiner_called.float().mean().item()),
            "theta": float(self.risk_detector.theta.detach().cpu().item()),
            "mean_max_kl": 0.0,
        }

        if risk_output is not None:
            stats.update(risk_output)
            stats["mean_theta"] = float(
                risk_output["theta_scene"].detach().mean().cpu().item()
            )
            stats["mean_max_kl"] = float(
                risk_output["max_kl"].detach().mean().cpu().item()
            )
        else:
            stats["risky_pairs"] = [[] for _ in range(batch_size)]

        if refiner_output is not None:
            stats.update(
                {
                    "min_dist_before": refiner_output["min_dist_before"],
                    "min_dist_after": refiner_output["min_dist_after"],
                    "mean_shift": refiner_output["mean_shift"],
                    "max_shift": refiner_output["max_shift"],
                }
            )
        return stats

    def forward(self, batch: TensorDict, mode: str = "kl_triggered") -> Dict[str, object]:
        modes = {
            "fast_only",
            "scene_switching",
            "distance_local",
            "kl_triggered",
            "always_refine",
        }
        if mode not in modes:
            raise ValueError(f"Unknown mode {mode!r}; expected one of {sorted(modes)}")

        forecast_output = self.encode_and_forecast(batch)
        prior_dist = forecast_output["prior_dist"]
        fast_traj = prior_dist.mu
        batch_size = fast_traj.shape[0]
        valid_agent_mask = batch["valid_agent_mask"].bool()

        risk_output = None
        refiner_output = None
        refiner_called = torch.zeros(batch_size, dtype=torch.bool, device=fast_traj.device)
        final_traj = fast_traj

        if mode == "scene_switching":
            risk_output = self.risk_detector(
                prior_dist,
                valid_agent_mask=valid_agent_mask,
                safety_requirement=batch.get("safety_requirement"),
            )
            refiner_called = risk_output["risk_flags"].bool()
            refiner_output = self._run_full_scene_refiner(
                fast_traj, refiner_called, valid_agent_mask
            )
            final_traj = refiner_output["refined_traj"]

        elif mode == "distance_local":
            risky_pairs, refiner_called = self._distance_local_selection(
                fast_traj, valid_agent_mask
            )
            refiner_output = self._run_local_refiner(
                fast_traj, refiner_called, risky_pairs
            )
            final_traj = refiner_output["refined_traj"]

        elif mode == "kl_triggered":
            risk_output = self.risk_detector(
                prior_dist,
                valid_agent_mask=valid_agent_mask,
                safety_requirement=batch.get("safety_requirement"),
            )
            refiner_called = risk_output["risk_flags"].bool()
            refiner_output = self._run_local_refiner(
                fast_traj,
                refiner_called,
                risk_output["risky_pairs"],
            )
            final_traj = refiner_output["refined_traj"]

        elif mode == "always_refine":
            refiner_called = torch.ones(
                batch_size, dtype=torch.bool, device=fast_traj.device
            )
            refiner_output = self._run_full_scene_refiner(
                fast_traj, refiner_called, valid_agent_mask
            )
            final_traj = refiner_output["refined_traj"]

        stats = self._stats(
            mode, final_traj, risk_output, refiner_output, refiner_called
        )
        return {
            "encoded": forecast_output["encoded"],
            "prior_dist": prior_dist,
            "fast_traj": fast_traj,
            "final_traj": final_traj,
            "risk": risk_output,
            "refiner": refiner_output,
            "stats": stats,
        }
