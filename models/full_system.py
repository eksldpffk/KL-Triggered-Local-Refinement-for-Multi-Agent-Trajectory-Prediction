from __future__ import annotations

from typing import Dict, List, Tuple

import torch
from torch import nn

from models.backbone import SceneBackbone
from models.global_planner import GlobalPlanner, ProbabilisticTrajectory
from models.risk_detector import RiskDetector
from models.local_refiner import LocalRefiner


TensorDict = Dict[str, torch.Tensor]
RiskyPairs = List[List[Tuple[int, int]]]


class FullSystem(nn.Module):
    """
    Full KL-triggered trajectory refinement system.

    Pipeline:
        past scene
            -> SceneBackbone
            -> GlobalPlanner
            -> RiskDetector
            -> LocalRefiner if KL > theta

    Supported modes:
        global_only:
            no local refinement, baseline mode.

        kl_triggered:
            local refiner is called only for scenes where KL > theta.

        always_refine:
            local refiner is called for every scene.
            This is the expensive upper baseline.

        oracle_refine:
            local refiner uses ground-truth hard_pairs from synthetic generator.
            This is only for debugging / sanity check, not final method.
    """

    def __init__(
        self,
        backbone: SceneBackbone,
        global_planner: GlobalPlanner,
        risk_detector: RiskDetector,
        local_refiner: LocalRefiner,
        scene_switch_margin: float = 0.5,
    ) -> None:
        super().__init__()

        self.backbone = backbone
        self.global_planner = global_planner
        self.risk_detector = risk_detector
        self.local_refiner = local_refiner

        # Used only for existing-style scene-level switching baseline.
        # This is NOT KL-based.
        self.scene_switch_margin = float(scene_switch_margin)

    @classmethod
    def from_config(cls, config: dict) -> "FullSystem":
        backbone = SceneBackbone.from_config(config)
        global_planner = GlobalPlanner.from_config(config)
        risk_detector = RiskDetector.from_config(config)
        local_refiner = LocalRefiner.from_config(config)

        scene_switch_cfg = config.get("scene_switching", {})

        return cls(
            backbone=backbone,
            global_planner=global_planner,
            risk_detector=risk_detector,
            local_refiner=local_refiner,
            scene_switch_margin=scene_switch_cfg.get("risk_margin", 0.5),
        )

    def planner_parameters(self):
        """
        Parameters of the neural planner part:
            backbone + global planner.

        Used for main NLL / distillation optimizer.
        """
        yield from self.backbone.parameters()
        yield from self.global_planner.parameters()

    def theta_parameters(self):
        """
        Parameters of adaptive KL threshold theta.

        Used for separate theta optimizer.
        """
        yield from self.risk_detector.parameters()

    def encode_and_plan(self, batch: TensorDict) -> Dict[str, object]:
        """
        Run backbone and global planner.

        Returns:
            encoded:
                agent_embeddings, scene_embedding

            prior_dist:
                ProbabilisticTrajectory from global planner
        """
        encoded = self.backbone(batch)

        last_positions = batch["past_positions"][:, -1]
        last_velocities = batch["past_velocities"][:, -1]

        prior_dist = self.global_planner(
            agent_embeddings=encoded["agent_embeddings"],
            scene_embedding=encoded["scene_embedding"],
            last_positions=last_positions,
            last_velocities=last_velocities,
        )

        return {
            "encoded": encoded,
            "prior_dist": prior_dist,
        }

    def _pairs_from_hard_pairs_tensor(
        self,
        hard_pairs: torch.Tensor,
    ) -> RiskyPairs:
        """
        Convert synthetic generator hard_pairs tensor [B, 2]
        into list format used by LocalRefiner.
        """
        risky_pairs: RiskyPairs = []

        for pair in hard_pairs:
            i, j = pair.detach().cpu().tolist()

            if i >= 0 and j >= 0:
                risky_pairs.append([(int(i), int(j))])
            else:
                risky_pairs.append([])

        return risky_pairs
    
    def _run_refiner_on_subset(
        self,
        global_traj: torch.Tensor,
        scene_mask: torch.Tensor,
        risky_pairs: RiskyPairs,
    ) -> TensorDict:
        """
        Run LocalRefiner only on selected scenes.

        This makes KL-triggered latency honest:
        if only 20% of scenes are risky, ADMM is applied only to that subset,
        not to the whole batch.
        """
        if global_traj.dim() != 4:
            raise ValueError(
                f"global_traj must have shape [B,T,N,2], got {global_traj.shape}"
            )

        B, T, N, C = global_traj.shape
        device = global_traj.device
        dtype = global_traj.dtype

        scene_mask = scene_mask.to(device=device, dtype=torch.bool)

        refined_traj = global_traj.clone()

        full_output: TensorDict = {
            "refined_traj": refined_traj,
            "was_refined": scene_mask.clone(),
            "min_dist_before": torch.full(
                (B,), float("inf"), device=device, dtype=dtype
            ),
            "min_dist_after": torch.full(
                (B,), float("inf"), device=device, dtype=dtype
            ),
            "mean_shift": torch.zeros(B, device=device, dtype=dtype),
            "max_shift": torch.zeros(B, device=device, dtype=dtype),
            "pair_mask": torch.zeros(B, N, N, device=device, dtype=torch.bool),
        }

        selected = torch.nonzero(scene_mask, as_tuple=False).flatten()

        if selected.numel() == 0:
            return full_output

        selected_traj = global_traj.index_select(0, selected)

        selected_indices = selected.detach().cpu().tolist()
        selected_pairs: RiskyPairs = [risky_pairs[int(idx)] for idx in selected_indices]

        subset_output = self.local_refiner(
            global_output=selected_traj,
            risky_pairs=selected_pairs,
        )

        full_output["refined_traj"][selected] = subset_output["refined_traj"]
        full_output["min_dist_before"][selected] = subset_output["min_dist_before"]
        full_output["min_dist_after"][selected] = subset_output["min_dist_after"]
        full_output["mean_shift"][selected] = subset_output["mean_shift"]
        full_output["max_shift"][selected] = subset_output["max_shift"]

        if "pair_mask" in subset_output:
            full_output["pair_mask"][selected] = subset_output["pair_mask"]

        return full_output

    def _empty_refiner_output(
        self,
        global_traj: torch.Tensor,
        scene_mask: torch.Tensor,
    ) -> TensorDict:
        """
        Create an empty refiner output with the same structure as LocalRefiner.
        """
        B, _, N, _ = global_traj.shape
        device = global_traj.device
        dtype = global_traj.dtype

        scene_mask = scene_mask.to(device=device, dtype=torch.bool)

        return {
            "refined_traj": global_traj.clone(),
            "was_refined": scene_mask.clone(),
            "min_dist_before": torch.full(
                (B,), float("inf"), device=device, dtype=dtype
            ),
            "min_dist_after": torch.full(
                (B,), float("inf"), device=device, dtype=dtype
            ),
            "mean_shift": torch.zeros(B, device=device, dtype=dtype),
            "max_shift": torch.zeros(B, device=device, dtype=dtype),
            "pair_mask": torch.zeros(B, N, N, device=device, dtype=torch.bool),
        }


    def _all_pairs_mask(
        self,
        batch_size: int,
        n_agents: int,
        device: torch.device,
        valid_agent_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Full-scene mask: all upper-triangular agent pairs.
        Used for honest scene-level / always-refine baselines.
        """
        upper = torch.triu(
            torch.ones(n_agents, n_agents, dtype=torch.bool, device=device),
            diagonal=1,
        )

        mask = upper.view(1, n_agents, n_agents).expand(
            batch_size, n_agents, n_agents
        ).clone()
        if valid_agent_mask is not None:
            valid = valid_agent_mask.to(device=device, dtype=torch.bool)
            mask = mask & valid[:, :, None] & valid[:, None, :]
        return mask


    def _scene_risk_flags_from_distance(
        self,
        global_traj: torch.Tensor,
        valid_agent_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Existing-style scene-level switching baseline.

        A scene is risky if any pair in the predicted trajectory
        becomes closer than d_min + margin.

        This trigger is intentionally NOT KL-based.
        """
        if global_traj.dim() != 4 or global_traj.shape[-1] != 2:
            raise ValueError(
                f"global_traj must have shape [B,T,N,2], got {global_traj.shape}"
            )

        B, _, N, _ = global_traj.shape
        device = global_traj.device

        diff = global_traj[:, :, :, None, :] - global_traj[:, :, None, :, :]
        dist = torch.linalg.norm(diff, dim=-1)  # [B,T,N,N]

        eye = torch.eye(N, dtype=torch.bool, device=device).view(1, 1, N, N)
        invalid = eye
        if valid_agent_mask is not None:
            valid = valid_agent_mask.to(device=device, dtype=torch.bool)
            valid_pairs = valid[:, :, None] & valid[:, None, :]
            invalid = invalid | ~valid_pairs[:, None, :, :]
        dist = dist.masked_fill(invalid, float("inf"))

        min_scene_dist = dist.amin(dim=(1, 2, 3))  # [B]

        scene_threshold = self.local_refiner.d_min + self.scene_switch_margin

        return min_scene_dist < scene_threshold


    def _run_full_scene_refiner_on_subset(
        self,
        global_traj: torch.Tensor,
        scene_mask: torch.Tensor,
        valid_agent_mask: torch.Tensor | None = None,
    ) -> TensorDict:
        """
        Run refiner on selected scenes with ALL agent pairs active.

        This is used for:
            - scene-level switching baseline
            - always-refine baseline

        Difference from our method:
            once the scene is selected, correction is full-scene,
            not local-crop.
        """
        if global_traj.dim() != 4 or global_traj.shape[-1] != 2:
            raise ValueError(
                f"global_traj must have shape [B,T,N,2], got {global_traj.shape}"
            )

        B, _, N, _ = global_traj.shape
        device = global_traj.device

        scene_mask = scene_mask.to(device=device, dtype=torch.bool)

        full_output = self._empty_refiner_output(
            global_traj=global_traj,
            scene_mask=scene_mask,
        )

        selected = torch.nonzero(scene_mask, as_tuple=False).flatten()

        if selected.numel() == 0:
            return full_output

        selected_traj = global_traj.index_select(0, selected)

        selected_valid = None
        if valid_agent_mask is not None:
            selected_valid = valid_agent_mask.to(device=device, dtype=torch.bool).index_select(0, selected)
        selected_pair_mask = self._all_pairs_mask(
            batch_size=selected_traj.shape[0],
            n_agents=N,
            device=device,
            valid_agent_mask=selected_valid,
        )

        subset_output = self.local_refiner(
            global_output=selected_traj,
            pair_mask=selected_pair_mask,
        )

        full_output["refined_traj"][selected] = subset_output["refined_traj"]
        full_output["min_dist_before"][selected] = subset_output["min_dist_before"]
        full_output["min_dist_after"][selected] = subset_output["min_dist_after"]
        full_output["mean_shift"][selected] = subset_output["mean_shift"]
        full_output["max_shift"][selected] = subset_output["max_shift"]

        if "pair_mask" in subset_output:
            full_output["pair_mask"][selected] = subset_output["pair_mask"]

        return full_output


    def _run_local_refiner_cropped(
        self,
        global_traj: torch.Tensor,
        scene_mask: torch.Tensor,
        risky_pairs: RiskyPairs,
    ) -> TensorDict:
        """
        Fast local refinement for KL-triggered mode.

        Old version:
            loop over selected scenes and call LocalRefiner once per scene.
            On CPU this is slow because many tiny Python calls dominate latency.

        New version:
            build one batched tensor of risky pairs [K, T, 2, 2], run the
            refiner once on those local pair-crops, then scatter the local
            corrections back into the full [B, T, N, 2] trajectory.

        This keeps the method local: only agents from KL-risky pairs are changed.
        """
        if global_traj.dim() != 4 or global_traj.shape[-1] != 2:
            raise ValueError(
                f"global_traj must have shape [B,T,N,2], got {global_traj.shape}"
            )

        B, T, N, C = global_traj.shape
        device = global_traj.device
        dtype = global_traj.dtype

        scene_mask = scene_mask.to(device=device, dtype=torch.bool)

        full_output = self._empty_refiner_output(
            global_traj=global_traj,
            scene_mask=scene_mask,
        )

        selected = torch.nonzero(scene_mask, as_tuple=False).flatten()
        if selected.numel() == 0:
            return full_output

        pair_records: List[Tuple[int, int, int]] = []

        selected_cpu = selected.detach().cpu().tolist()
        for b in selected_cpu:
            if b >= len(risky_pairs):
                continue

            seen = set()
            for i, j in risky_pairs[b]:
                i = int(i)
                j = int(j)

                if i == j:
                    continue
                if not (0 <= i < N and 0 <= j < N):
                    continue

                a = min(i, j)
                c = max(i, j)
                key = (a, c)

                if key in seen:
                    continue
                seen.add(key)

                pair_records.append((int(b), a, c))

        if len(pair_records) == 0:
            return full_output

        b_idx = torch.tensor([r[0] for r in pair_records], dtype=torch.long, device=device)
        i_idx = torch.tensor([r[1] for r in pair_records], dtype=torch.long, device=device)
        j_idx = torch.tensor([r[2] for r in pair_records], dtype=torch.long, device=device)

        K = b_idx.numel()

        # [K,T,2,2]: each local problem contains only the selected pair.
        pair_traj = torch.stack(
            [
                global_traj[b_idx, :, i_idx, :],
                global_traj[b_idx, :, j_idx, :],
            ],
            dim=2,
        )

        local_pair_mask = torch.zeros(K, 2, 2, dtype=torch.bool, device=device)
        local_pair_mask[:, 0, 1] = True

        local_output = self.local_refiner(
            global_output=pair_traj,
            pair_mask=local_pair_mask,
        )

        refined_pairs = local_output["refined_traj"]       # [K,T,2,2]
        pair_delta = refined_pairs - pair_traj              # [K,T,2,2]

        # If the same agent appears in several risky pairs, average its local
        # corrections rather than overwriting the previous correction.
        delta_accum = torch.zeros_like(global_traj)
        count_accum = torch.zeros(B, N, device=device, dtype=dtype)

        time_idx = torch.arange(T, device=device).view(1, T).expand(K, T)
        b_time = b_idx.view(K, 1).expand(K, T)

        i_time = i_idx.view(K, 1).expand(K, T)
        j_time = j_idx.view(K, 1).expand(K, T)

        delta_accum.index_put_(
            (b_time.reshape(-1), time_idx.reshape(-1), i_time.reshape(-1)),
            pair_delta[:, :, 0, :].reshape(-1, C),
            accumulate=True,
        )
        delta_accum.index_put_(
            (b_time.reshape(-1), time_idx.reshape(-1), j_time.reshape(-1)),
            pair_delta[:, :, 1, :].reshape(-1, C),
            accumulate=True,
        )

        ones = torch.ones(K, device=device, dtype=dtype)
        count_accum.index_put_((b_idx, i_idx), ones, accumulate=True)
        count_accum.index_put_((b_idx, j_idx), ones, accumulate=True)

        denom = count_accum.clamp(min=1.0).view(B, 1, N, 1)
        averaged_delta = delta_accum / denom

        full_output["refined_traj"] = global_traj + averaged_delta
        full_output["pair_mask"][b_idx, i_idx, j_idx] = True

        # Stats only; small loop over K pair records is fine and does not call ADMM.
        for k, b in enumerate(b_idx.detach().cpu().tolist()):
            before = local_output["min_dist_before"][k]
            after = local_output["min_dist_after"][k]

            full_output["min_dist_before"][b] = torch.minimum(
                full_output["min_dist_before"][b],
                before,
            )
            full_output["min_dist_after"][b] = torch.minimum(
                full_output["min_dist_after"][b],
                after,
            )

        shift = torch.linalg.norm(full_output["refined_traj"] - global_traj, dim=-1)
        full_output["mean_shift"] = shift.mean(dim=(1, 2))
        full_output["max_shift"] = shift.amax(dim=(1, 2))

        return full_output

    def _make_stats(
        self,
        mode: str,
        final_traj: torch.Tensor,
        prior_dist: ProbabilisticTrajectory,
        risk_output: Dict[str, object] | None,
        refiner_output: TensorDict | None,
        refiner_called: torch.Tensor,
    ) -> Dict[str, object]:
        """
        Collect lightweight runtime statistics.
        """
        B = final_traj.shape[0]

        stats: Dict[str, object] = {
            "mode": mode,
            "batch_size": B,
            "refiner_called": refiner_called,
            "n_refiner_called": int(refiner_called.sum().item()),
            "refine_rate": float(refiner_called.float().mean().item()),
            "theta": float(self.risk_detector.theta.detach().cpu().item()),
        }

        if risk_output is not None:
            stats["kl_matrix"] = risk_output["kl_matrix"]
            stats["max_kl"] = risk_output["max_kl"]
            stats["risk_logits"] = risk_output["risk_logits"]
            stats["risk_flags"] = risk_output["risk_flags"]
            stats["risky_pairs"] = risk_output["risky_pairs"]
            stats["theta_scene"] = risk_output.get("theta_scene")
            stats["mean_theta"] = float(
                risk_output.get("theta_scene", self.risk_detector.theta.expand(B))
                .detach().mean().cpu().item()
            )
            stats["density"] = risk_output.get("density")
            stats["uncertainty"] = risk_output.get("uncertainty")
            stats["mean_max_kl"] = float(
                risk_output["max_kl"].detach().mean().cpu().item()
            )
        else:
            stats["kl_matrix"] = None
            stats["max_kl"] = None
            stats["risk_logits"] = None
            stats["risk_flags"] = None
            stats["risky_pairs"] = [[] for _ in range(B)]
            stats["theta_scene"] = None
            stats["mean_theta"] = float(self.risk_detector.theta.detach().cpu().item())
            stats["density"] = None
            stats["uncertainty"] = None
            stats["mean_max_kl"] = 0.0

        if refiner_output is not None:
            stats["min_dist_before"] = refiner_output["min_dist_before"]
            stats["min_dist_after"] = refiner_output["min_dist_after"]
            stats["mean_shift"] = refiner_output["mean_shift"]
            stats["max_shift"] = refiner_output["max_shift"]
        else:
            stats["min_dist_before"] = None
            stats["min_dist_after"] = None
            stats["mean_shift"] = torch.zeros(
                B,
                dtype=final_traj.dtype,
                device=final_traj.device,
            )
            stats["max_shift"] = torch.zeros(
                B,
                dtype=final_traj.dtype,
                device=final_traj.device,
            )

        return stats

    def forward(
        self,
        batch: TensorDict,
        mode: str = "kl_triggered",
        training: bool = False,
    ) -> Dict[str, object]:
        """
        Full forward pass.

        Args:
            batch:
                Output of SceneGenerator.generate_batch(...)

            mode:
                global_only | kl_triggered | always_refine | oracle_refine

            training:
                Kept for future train.py logic.
                The current forward returns all objects needed for training.

        Returns:
            {
                prior_dist:
                    ProbabilisticTrajectory from global planner

                global_traj:
                    prior mean trajectory [B, T_future, N, 2]

                final_traj:
                    final trajectory after optional refinement

                refined_traj:
                    refined trajectory if refiner was used, otherwise global_traj

                risk:
                    risk detector output or None

                refiner:
                    local refiner output or None

                stats:
                    summary dictionary
            }
        """
        if mode not in {
            "global_only",
            "scene_switching",
            "kl_triggered",
            "always_refine",
            "oracle_refine",
        }:
            raise ValueError(
                "mode must be one of: global_only, scene_switching, kl_triggered, always_refine, oracle_refine"
            )

        plan_output = self.encode_and_plan(batch)
        encoded = plan_output["encoded"]
        prior_dist = plan_output["prior_dist"]

        global_traj = prior_dist.mu
        B = global_traj.shape[0]
        valid_agent_mask = batch.get("valid_agent_mask", batch.get("agent_mask"))

        risk_output = None
        refiner_output = None

        refiner_called = torch.zeros(
            B,
            dtype=torch.bool,
            device=global_traj.device,
        )

        final_traj = global_traj
        refined_traj = global_traj

        if mode == "global_only":
            pass

        elif mode == "scene_switching":
            # Existing-style baseline:
            # cheap scene-level distance trigger.
            # If the scene is risky, refine the whole scene / all pairs.
            refiner_called = self._scene_risk_flags_from_distance(
                global_traj, valid_agent_mask=valid_agent_mask
            )

            refiner_output = self._run_full_scene_refiner_on_subset(
                global_traj=global_traj,
                scene_mask=refiner_called,
                valid_agent_mask=valid_agent_mask,
            )

            refined_traj = refiner_output["refined_traj"]
            final_traj = refined_traj

        elif mode == "kl_triggered":
            pairwise_d_min = batch.get("pairwise_d_min")
            safety_requirement = batch.get("safety_requirement")
            risk_output = self.risk_detector(
                prior_dist,
                valid_agent_mask=valid_agent_mask,
                pairwise_d_min=pairwise_d_min,
                safety_requirement=safety_requirement,
            )
            risky_pairs = risk_output["risky_pairs"]
            refiner_called = risk_output["risk_flags"].bool()

            # Run a single batched local correction over only KL-selected pairs,
            # then scatter the pair deltas back into the full scene.
            refiner_output = self._run_local_refiner_cropped(
                global_traj=global_traj,
                scene_mask=refiner_called,
                risky_pairs=risky_pairs,
            )
            refined_traj = refiner_output["refined_traj"]
            final_traj = refined_traj

        elif mode == "always_refine":
            # Heavy upper baseline:
            # full-scene correction for every scene.
            refiner_called = torch.ones(
                B,
                dtype=torch.bool,
                device=global_traj.device,
            )

            refiner_output = self._run_full_scene_refiner_on_subset(
                global_traj=global_traj,
                scene_mask=refiner_called,
                valid_agent_mask=valid_agent_mask,
            )

            refined_traj = refiner_output["refined_traj"]
            final_traj = refined_traj

        elif mode == "oracle_refine":
            if "hard_pairs" not in batch:
                raise ValueError(
                    "oracle_refine requires batch['hard_pairs'] from SceneGenerator"
                )

            risky_pairs = self._pairs_from_hard_pairs_tensor(batch["hard_pairs"])

            refiner_called = torch.tensor(
                [len(pairs) > 0 for pairs in risky_pairs],
                dtype=torch.bool,
                device=global_traj.device,
            )

            refiner_output = self._run_refiner_on_subset(
                global_traj=global_traj,
                scene_mask=refiner_called,
                risky_pairs=risky_pairs,
            )

            refined_traj = refiner_output["refined_traj"]
            final_traj = refined_traj

        stats = self._make_stats(
            mode=mode,
            final_traj=final_traj,
            prior_dist=prior_dist,
            risk_output=risk_output,
            refiner_output=refiner_output,
            refiner_called=refiner_called,
        )

        return {
            "encoded": encoded,
            "prior_dist": prior_dist,
            "global_traj": global_traj,
            "refined_traj": refined_traj,
            "final_traj": final_traj,
            "risk": risk_output,
            "refiner": refiner_output,
            "stats": stats,
        }
