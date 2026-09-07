from __future__ import annotations

import argparse
import copy
import csv
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import torch
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from data.factory import make_scene_source
from evaluation.metrics import compute_prediction_metrics, move_batch_to_device, refinement_rate_decay
from models.full_system import FullSystem
from utils import ensure_dirs, get_device, load_config, set_seed


TensorDict = Dict[str, torch.Tensor]


def detect_risk(system: FullSystem, prior_dist, batch: TensorDict):
    """Run the detector with optional real-data masks/context."""
    return system.risk_detector(
        prior_dist,
        valid_agent_mask=batch.get("valid_agent_mask", batch.get("agent_mask")),
        pairwise_d_min=batch.get("pairwise_d_min"),
        safety_requirement=batch.get("safety_requirement"),
    )


def gaussian_nll_masked(
    mu: torch.Tensor,
    log_sigma: torch.Tensor,
    target: torch.Tensor,
    scene_mask: Optional[torch.Tensor] = None,
    min_log_sigma: float = -5.0,
    max_log_sigma: float = 2.0,
    valid_agent_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Gaussian NLL, optionally masking scenes and padded agents."""
    if mu.shape != target.shape:
        raise ValueError(f"mu and target shape mismatch: {mu.shape} vs {target.shape}")

    log_sigma = torch.clamp(log_sigma, min=min_log_sigma, max=max_log_sigma)
    sigma = torch.exp(log_sigma)
    const = 0.5 * math.log(2.0 * math.pi)

    nll_per_dim = 0.5 * ((target - mu) / sigma) ** 2 + log_sigma + const
    if valid_agent_mask is None:
        nll_per_scene = nll_per_dim.mean(dim=(1, 2, 3))
    else:
        mask = valid_agent_mask[:, None, :, None].to(mu.device, mu.dtype).expand_as(nll_per_dim)
        nll_per_scene = (nll_per_dim * mask).sum(dim=(1, 2, 3)) / mask.sum(dim=(1, 2, 3)).clamp(min=1.0)

    if scene_mask is None:
        return nll_per_scene.mean()

    scene_mask = scene_mask.bool()
    if scene_mask.sum() == 0:
        return nll_per_scene.mean() * 0.0

    return nll_per_scene[scene_mask].mean()


def gaussian_nll_agent_masked(
    mu: torch.Tensor,
    log_sigma: torch.Tensor,
    target: torch.Tensor,
    agent_mask: torch.Tensor,
    min_log_sigma: float = -5.0,
    max_log_sigma: float = 2.0,
) -> torch.Tensor:
    """Gaussian NLL only on selected agents [B,N]."""
    if mu.shape != target.shape:
        raise ValueError(f"mu and target shape mismatch: {mu.shape} vs {target.shape}")
    if agent_mask.shape != (mu.shape[0], mu.shape[2]):
        raise ValueError(
            f"agent_mask must have shape {(mu.shape[0], mu.shape[2])}, got {agent_mask.shape}"
        )
    log_sigma = torch.clamp(log_sigma, min=min_log_sigma, max=max_log_sigma)
    sigma = torch.exp(log_sigma)
    const = 0.5 * math.log(2.0 * math.pi)
    nll = 0.5 * ((target - mu) / sigma) ** 2 + log_sigma + const
    mask = agent_mask[:, None, :, None].to(device=mu.device, dtype=mu.dtype)
    mask = mask.expand_as(nll)
    denom = mask.sum().clamp(min=1.0)
    return (nll * mask).sum() / denom


def differentiable_collision_loss(
    traj: torch.Tensor,
    d_min: float,
    safety_margin: float = 0.25,
    valid_agent_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Soft collision penalty over all pairs."""
    if traj.dim() != 4 or traj.shape[-1] != 2:
        raise ValueError(f"Expected traj shape [B,T,N,2], got {traj.shape}")

    _, _, N, _ = traj.shape
    diff = traj[:, :, :, None, :] - traj[:, :, None, :, :]
    dist = torch.linalg.norm(diff, dim=-1)

    eye = torch.eye(N, dtype=torch.bool, device=traj.device).view(1, 1, N, N)
    invalid = eye
    if valid_agent_mask is not None:
        valid = valid_agent_mask.to(traj.device, torch.bool)
        valid_pairs = valid[:, :, None] & valid[:, None, :]
        invalid = invalid | ~valid_pairs[:, None, :, :]
    dist_no_self = dist.masked_fill(invalid, float("inf"))

    safe_radius = d_min + safety_margin
    violation = torch.relu(safe_radius - dist_no_self)

    return 0.5 * violation.pow(2).mean()


def hard_pair_safety_loss(
    traj: torch.Tensor,
    hard_pairs: torch.Tensor,
    d_min: float,
    safety_margin: float = 0.30,
) -> torch.Tensor:
    """Focused soft safety loss for synthetic hard pairs, used only in Ours branch."""
    if traj.dim() != 4 or traj.shape[-1] != 2:
        raise ValueError(f"Expected traj shape [B,T,N,2], got {traj.shape}")

    B, _, N, _ = traj.shape
    device = traj.device
    safe_radius = d_min + safety_margin
    losses: List[torch.Tensor] = []

    for b in range(B):
        i = int(hard_pairs[b, 0].item())
        j = int(hard_pairs[b, 1].item())
        if i < 0 or j < 0 or i >= N or j >= N or i == j:
            continue

        dist = torch.linalg.norm(traj[b, :, i, :] - traj[b, :, j, :], dim=-1)
        violation = torch.relu(safe_radius - dist)
        losses.append(violation.pow(2).max())

    if len(losses) == 0:
        return torch.zeros((), device=device, dtype=traj.dtype)

    return torch.stack(losses).mean()


def mean_ignore_nan(values: List[float]) -> float:
    clean = [float(v) for v in values if not math.isnan(float(v))]
    if len(clean) == 0:
        return float("nan")
    return float(sum(clean) / len(clean))


def init_stats_accum() -> Dict[str, List[float]]:
    return {
        "loss_total": [],
        "loss_gt": [],
        "loss_easy": [],
        "loss_hard_gt": [],
        "loss_distill": [],
        "loss_safety": [],
        "loss_pair_safety": [],
        "loss_theta": [],
        "ade": [],
        "fde": [],
        "ade_hard": [],
        "fde_hard": [],
        "collision": [],
        "collision_hard": [],
        "refine_rate": [],
        "theta": [],
        "mean_max_kl": [],
        "n_hard": [],
        "n_distill": [],
    }


def make_epoch_row(
    branch: str,
    stage: str,
    epoch: int,
    global_epoch: int,
    label_type: str,
    stats_accum: Dict[str, List[float]],
    val_metrics: Optional[Dict[str, float]] = None,
) -> Dict[str, float | str]:
    row: Dict[str, float | str] = {
        "branch": branch,
        "stage": stage,
        "epoch": epoch,
        "global_epoch": global_epoch,
        "label_type": label_type,
    }

    for key, values in stats_accum.items():
        row[key] = mean_ignore_nan(values)

    if val_metrics is not None:
        for key, value in val_metrics.items():
            row[f"val_{key}"] = float(value)

    return row


def save_train_log(rows: List[Dict[str, float | str]], path: str) -> None:
    if len(rows) == 0:
        return

    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)

    all_fields = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                all_fields.append(key)

    with open(path_obj, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_fields)
        writer.writeheader()
        writer.writerows(rows)


def save_checkpoint(
    system: FullSystem,
    optimizer_planner: Optional[torch.optim.Optimizer],
    optimizer_theta: Optional[torch.optim.Optimizer],
    epoch: int,
    stage: str,
    branch: str,
    config: dict,
    path: str,
) -> None:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "epoch": epoch,
        "stage": stage,
        "branch": branch,
        "config": config,
        "system_state_dict": system.state_dict(),
    }

    if optimizer_planner is not None:
        payload["optimizer_planner_state_dict"] = optimizer_planner.state_dict()
    if optimizer_theta is not None:
        payload["optimizer_theta_state_dict"] = optimizer_theta.state_dict()

    torch.save(payload, path_obj)


def matched_score(metrics: Dict[str, float]) -> float:
    """
    Same checkpoint-selection score for GT-control and Ours.
    Lower is better. Safety matters most, but accuracy/refine rate also count.
    """
    collision = float(metrics.get("Collision", 1.0))
    collision_hard = float(metrics.get("Collision_hard", 1.0))
    ade_hard = float(metrics.get("ADE_hard", 1.0))
    fde_hard = float(metrics.get("FDE_hard", 1.0))
    refine_rate = float(metrics.get("RefineRate", 1.0))

    for name, value in [
        ("collision", collision),
        ("collision_hard", collision_hard),
        ("ade_hard", ade_hard),
        ("fde_hard", fde_hard),
        ("refine_rate", refine_rate),
    ]:
        if math.isnan(value):
            if name in {"collision", "collision_hard"}:
                value = 1.0

    return (
        1.5 * collision
        + 1.0 * collision_hard
        + 0.25 * ade_hard
        + 0.10 * fde_hard
        + 0.05 * refine_rate
    )


@torch.no_grad()
def evaluate_small(
    system: FullSystem,
    val_batches: List[TensorDict],
    config: dict,
    device: torch.device,
    mode: str = "kl_triggered",
) -> Dict[str, float]:
    system.eval()
    d_min = config["scene"]["d_min"]

    rows: List[Dict[str, float]] = []
    refine_rates: List[float] = []
    mean_max_kls: List[float] = []

    for batch_cpu in val_batches:
        batch = move_batch_to_device(batch_cpu, device)
        output = system(batch=batch, mode=mode, training=False)
        metrics = compute_prediction_metrics(
            pred=output["final_traj"],
            gt=batch["future_positions"],
            d_min=d_min,
            is_hard=batch["is_hard"],
            valid_agent_mask=batch.get("valid_agent_mask", batch.get("agent_mask")),
        )
        rows.append(metrics)
        refine_rates.append(float(output["stats"]["refine_rate"]))
        mean_max_kls.append(float(output["stats"]["mean_max_kl"]))

    out: Dict[str, float] = {}
    for key in rows[0].keys():
        vals = [float(r[key]) for r in rows if not math.isnan(float(r[key]))]
        out[key] = float(sum(vals) / len(vals)) if vals else float("nan")

    out["RefineRate"] = float(sum(refine_rates) / len(refine_rates))
    out["MeanMaxKL"] = float(sum(mean_max_kls) / len(mean_max_kls))
    out["Score"] = matched_score(out)
    return out


def append_train_metrics(
    stats: Dict[str, List[float]],
    system: FullSystem,
    prior_dist,
    risk_output: Optional[Dict[str, object]],
    pred: torch.Tensor,
    gt: torch.Tensor,
    is_hard: torch.Tensor,
    valid_agent_mask: Optional[torch.Tensor],
    loss_total: torch.Tensor,
    loss_gt: torch.Tensor,
    loss_easy: torch.Tensor,
    loss_hard_gt: torch.Tensor,
    loss_distill: torch.Tensor,
    loss_safety: torch.Tensor,
    loss_pair_safety: torch.Tensor,
    loss_theta: torch.Tensor,
    n_distill: float,
    config: dict,
) -> None:
    with torch.no_grad():
        metrics = compute_prediction_metrics(
            pred=pred,
            gt=gt,
            d_min=config["scene"]["d_min"],
            is_hard=is_hard,
            valid_agent_mask=valid_agent_mask,
        )

        if risk_output is None:
            refine_rate = 0.0
            mean_max_kl = 0.0
        else:
            refine_rate = float(risk_output["risk_flags"].float().mean().detach().cpu().item())
            mean_max_kl = float(risk_output["max_kl"].mean().detach().cpu().item())

        stats["loss_total"].append(float(loss_total.detach().cpu().item()))
        stats["loss_gt"].append(float(loss_gt.detach().cpu().item()))
        stats["loss_easy"].append(float(loss_easy.detach().cpu().item()))
        stats["loss_hard_gt"].append(float(loss_hard_gt.detach().cpu().item()))
        stats["loss_distill"].append(float(loss_distill.detach().cpu().item()))
        stats["loss_safety"].append(float(loss_safety.detach().cpu().item()))
        stats["loss_pair_safety"].append(float(loss_pair_safety.detach().cpu().item()))
        stats["loss_theta"].append(float(loss_theta.detach().cpu().item()))

        stats["ade"].append(metrics["ADE"])
        stats["fde"].append(metrics["FDE"])
        stats["ade_hard"].append(metrics["ADE_hard"])
        stats["fde_hard"].append(metrics["FDE_hard"])
        stats["collision"].append(metrics["Collision"])
        stats["collision_hard"].append(metrics["Collision_hard"])
        stats["refine_rate"].append(refine_rate)
        stats["theta"].append(float(system.risk_detector.theta.detach().cpu().item()))
        stats["mean_max_kl"].append(mean_max_kl)
        stats["n_hard"].append(float(is_hard.sum().detach().cpu().item()))
        stats["n_distill"].append(float(n_distill))


def train_gt_epoch(
    system: FullSystem,
    generator: Any,
    optimizer_planner: torch.optim.Optimizer,
    optimizer_theta: Optional[torch.optim.Optimizer],
    config: dict,
    device: torch.device,
    batch_size: int,
    steps_per_epoch: int,
    grad_clip: float,
    epoch: int,
    n_epochs: int,
    stage_name: str,
    train_theta: bool,
) -> Dict[str, List[float]]:
    """Ordinary GT training. Used for warmup and GT-control continuation."""
    system.train()
    model_cfg = config["model"]
    stats = init_stats_accum()

    progress = tqdm(range(steps_per_epoch), desc=f"{stage_name} {epoch}/{n_epochs}", leave=False)
    for _ in progress:
        batch = move_batch_to_device(generator.generate_batch(batch_size=batch_size), device)
        plan_output = system.encode_and_plan(batch)
        prior_dist = plan_output["prior_dist"]

        gt = batch["future_positions"]
        is_hard = batch["is_hard"].bool()
        is_easy = ~is_hard

        loss_gt = gaussian_nll_masked(
            prior_dist.mu,
            prior_dist.log_sigma,
            gt,
            None,
            model_cfg["min_log_sigma"],
            model_cfg["max_log_sigma"],
            valid_agent_mask=batch.get("valid_agent_mask", batch.get("agent_mask")),
        )
        loss_easy = gaussian_nll_masked(
            prior_dist.mu,
            prior_dist.log_sigma,
            gt,
            is_easy,
            model_cfg["min_log_sigma"],
            model_cfg["max_log_sigma"],
            valid_agent_mask=batch.get("valid_agent_mask", batch.get("agent_mask")),
        )
        loss_hard_gt = gaussian_nll_masked(
            prior_dist.mu,
            prior_dist.log_sigma,
            gt,
            is_hard,
            model_cfg["min_log_sigma"],
            model_cfg["max_log_sigma"],
            valid_agent_mask=batch.get("valid_agent_mask", batch.get("agent_mask")),
        )

        loss_total = loss_gt
        optimizer_planner.zero_grad(set_to_none=True)
        loss_total.backward()
        clip_grad_norm_(list(system.planner_parameters()), max_norm=grad_clip)
        optimizer_planner.step()

        with torch.no_grad():
            prior_for_theta = system.encode_and_plan(batch)["prior_dist"]
        risk_output = detect_risk(system, prior_for_theta, batch)

        loss_theta = system.risk_detector.threshold_loss(
            risk_logits=risk_output["risk_logits"],
            is_hard=is_hard,
        )

        if train_theta and optimizer_theta is not None:
            optimizer_theta.zero_grad(set_to_none=True)
            loss_theta.backward()
            optimizer_theta.step()

        zero = torch.zeros((), device=device)
        append_train_metrics(
            stats=stats,
            system=system,
            prior_dist=prior_dist,
            risk_output=risk_output,
            pred=prior_dist.mu,
            gt=gt,
            is_hard=is_hard,
            valid_agent_mask=batch.get("valid_agent_mask", batch.get("agent_mask")),
            loss_total=loss_total,
            loss_gt=loss_gt,
            loss_easy=loss_easy,
            loss_hard_gt=loss_hard_gt,
            loss_distill=zero,
            loss_safety=zero,
            loss_pair_safety=zero,
            loss_theta=loss_theta.detach(),
            n_distill=0.0,
            config=config,
        )

        progress.set_postfix({"gt": f"{loss_gt.item():.3f}", "ref": f"{stats['refine_rate'][-1]:.2f}"})

    return stats


@torch.no_grad()
def calibrate_theta(
    system: FullSystem,
    generator: Any,
    config: dict,
    device: torch.device,
    batch_size: int,
    n_batches: int,
) -> float:
    system.eval()
    max_kls: List[torch.Tensor] = []

    for _ in range(n_batches):
        batch = move_batch_to_device(generator.generate_batch(batch_size=batch_size), device)
        prior_dist = system.encode_and_plan(batch)["prior_dist"]
        risk_out = detect_risk(system, prior_dist, batch)
        max_kls.append(risk_out["max_kl"].detach())

    max_kl_values = torch.cat(max_kls, dim=0)
    target_refine_rate = config["risk"].get("target_refine_rate", config["scene"].get("hard_ratio", 0.2))
    theta_value = system.risk_detector.calibrate_theta_from_max_kl(
        max_kl_values=max_kl_values,
        target_refine_rate=target_refine_rate,
    )
    print(f"Calibrated shared theta = {theta_value:.4f} for target refine rate {target_refine_rate:.2f}")
    return theta_value


@torch.no_grad()
def make_kl_local_teacher(
    system: FullSystem,
    batch: TensorDict,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, object]]:
    """
    Build the teacher from exactly the KL-selected local corrections.

    Returns a scene mask and, crucially, an agent mask. The old code blended the
    teacher into every agent of a selected scene, including agents that had not
    been refined. That was not local distillation and could damage ADE.
    """
    system.eval()
    output = system(batch=batch, mode="kl_triggered", training=False)
    teacher_traj = output["final_traj"].detach()
    risk_output = output["risk"]
    B, _, N, _ = teacher_traj.shape
    agent_mask = torch.zeros(B, N, dtype=torch.bool, device=teacher_traj.device)
    if risk_output is not None:
        for b, pairs in enumerate(risk_output["risky_pairs"]):
            for i, j in pairs:
                if 0 <= i < N:
                    agent_mask[b, i] = True
                if 0 <= j < N:
                    agent_mask[b, j] = True
    scene_mask = agent_mask.any(dim=1)
    return teacher_traj, scene_mask, agent_mask, risk_output


def train_safety_distill_epoch(
    system: FullSystem,
    generator: Any,
    optimizer_planner: torch.optim.Optimizer,
    config: dict,
    device: torch.device,
    batch_size: int,
    steps_per_epoch: int,
    grad_clip: float,
    epoch: int,
    n_epochs: int,
) -> Dict[str, List[float]]:
    """
    Ours continuation:
      - same training budget as GT-control;
      - after warmup, supervision changes only for KL-triggered local risky scenes;
      - pseudo-labels are local-refined trajectories from the current system;
      - theta is frozen, so refine-rate decay means the planner itself became safer.
    """
    system.train()
    model_cfg = config["model"]
    train_cfg = config["train"]
    stats = init_stats_accum()

    lambda_gt_all = train_cfg.get("lambda_gt_all", 0.20)
    lambda_easy_gt = train_cfg.get("lambda_easy_gt", 1.0)
    lambda_hard_gt = train_cfg.get("lambda_hard_gt", 0.35)
    lambda_distill = train_cfg.get("lambda_distill", 1.0)
    lambda_safety = train_cfg.get("lambda_safety", 0.75)
    lambda_pair_safety = train_cfg.get("lambda_pair_safety", 3.0)
    safety_loss_margin = train_cfg.get("safety_loss_margin", 0.20)
    pair_safety_margin = train_cfg.get("pair_safety_margin", 0.30)
    safe_target_blend = train_cfg.get("safe_target_blend", 0.70)

    progress = tqdm(range(steps_per_epoch), desc=f"OURS safety distill {epoch}/{n_epochs}", leave=False)
    for _ in progress:
        batch = move_batch_to_device(generator.generate_batch(batch_size=batch_size), device)
        gt = batch["future_positions"]
        is_hard = batch["is_hard"].bool()
        is_easy = ~is_hard

        # Local teacher from KL-triggered local refiner. No gradient through teacher.
        teacher_traj, distill_mask, distill_agent_mask, teacher_risk = make_kl_local_teacher(
            system=system, batch=batch
        )

        # Student forward with gradient.
        plan_output = system.encode_and_plan(batch)
        prior_dist = plan_output["prior_dist"]
        risk_output = detect_risk(system, prior_dist, batch)

        # Blend the local teacher with GT to avoid destroying ADE while still learning safety.
        safe_target = gt.clone()
        if distill_agent_mask.any():
            agent_select = distill_agent_mask[:, None, :, None].expand_as(gt)
            blended = (1.0 - safe_target_blend) * gt + safe_target_blend * teacher_traj
            safe_target = torch.where(agent_select, blended, gt)

        loss_gt_all = gaussian_nll_masked(
            prior_dist.mu,
            prior_dist.log_sigma,
            gt,
            None,
            model_cfg["min_log_sigma"],
            model_cfg["max_log_sigma"],
            valid_agent_mask=batch.get("valid_agent_mask", batch.get("agent_mask")),
        )
        loss_easy = gaussian_nll_masked(
            prior_dist.mu,
            prior_dist.log_sigma,
            gt,
            is_easy,
            model_cfg["min_log_sigma"],
            model_cfg["max_log_sigma"],
            valid_agent_mask=batch.get("valid_agent_mask", batch.get("agent_mask")),
        )
        loss_hard_gt = gaussian_nll_masked(
            prior_dist.mu,
            prior_dist.log_sigma,
            gt,
            is_hard,
            model_cfg["min_log_sigma"],
            model_cfg["max_log_sigma"],
            valid_agent_mask=batch.get("valid_agent_mask", batch.get("agent_mask")),
        )
        valid_agents = batch.get("valid_agent_mask", batch.get("agent_mask"))
        if valid_agents is not None:
            distill_agent_mask = distill_agent_mask & valid_agents.bool()
        loss_distill = gaussian_nll_agent_masked(
            prior_dist.mu,
            prior_dist.log_sigma,
            safe_target,
            distill_agent_mask,
            model_cfg["min_log_sigma"],
            model_cfg["max_log_sigma"],
        )
        loss_safety = differentiable_collision_loss(
            traj=prior_dist.mu,
            d_min=config["scene"]["d_min"],
            safety_margin=safety_loss_margin,
            valid_agent_mask=batch.get("valid_agent_mask", batch.get("agent_mask")),
        )
        if lambda_pair_safety > 0.0 and "hard_pairs" in batch:
            loss_pair_safety = hard_pair_safety_loss(
                traj=prior_dist.mu,
                hard_pairs=batch["hard_pairs"],
                d_min=config["scene"]["d_min"],
                safety_margin=pair_safety_margin,
            )
        else:
            loss_pair_safety = torch.zeros((), device=device, dtype=prior_dist.mu.dtype)

        loss_total = (
            lambda_gt_all * loss_gt_all
            + lambda_easy_gt * loss_easy
            + lambda_hard_gt * loss_hard_gt
            + lambda_distill * loss_distill
            + lambda_safety * loss_safety
            + lambda_pair_safety * loss_pair_safety
        )

        optimizer_planner.zero_grad(set_to_none=True)
        loss_total.backward()
        clip_grad_norm_(list(system.planner_parameters()), max_norm=grad_clip)
        optimizer_planner.step()

        with torch.no_grad():
            loss_theta = system.risk_detector.threshold_loss(
                risk_logits=risk_output["risk_logits"],
                is_hard=is_hard,
            )

            eval_out = system(batch=batch, mode="kl_triggered", training=False)
            pred = eval_out["final_traj"]

        append_train_metrics(
            stats=stats,
            system=system,
            prior_dist=prior_dist,
            risk_output=risk_output,
            pred=pred,
            gt=gt,
            is_hard=is_hard,
            valid_agent_mask=batch.get("valid_agent_mask", batch.get("agent_mask")),
            loss_total=loss_total,
            loss_gt=loss_gt_all,
            loss_easy=loss_easy,
            loss_hard_gt=loss_hard_gt,
            loss_distill=loss_distill,
            loss_safety=loss_safety,
            loss_pair_safety=loss_pair_safety,
            loss_theta=loss_theta.detach(),
            n_distill=float(distill_agent_mask.sum().detach().cpu().item()),
            config=config,
        )

        progress.set_postfix({
            "loss": f"{loss_total.item():.3f}",
            "distill_n": int(distill_agent_mask.sum().item()),
            "ref": f"{stats['refine_rate'][-1]:.2f}",
        })

    return stats


def plot_training_curves(rows: List[Dict[str, float | str]], save_dir: str) -> None:
    if len(rows) == 0:
        return

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    branches = sorted(set(str(r["branch"]) for r in rows))
    keys = [
        "loss_total",
        "loss_gt",
        "loss_distill",
        "loss_safety",
        "collision",
        "collision_hard",
        "refine_rate",
        "val_Collision",
        "val_Collision_hard",
        "val_RefineRate",
        "val_Score",
    ]

    for key in keys:
        if not any(key in r for r in rows):
            continue

        plt.figure(figsize=(7, 4))
        for branch in branches:
            br = [r for r in rows if str(r["branch"]) == branch and key in r]
            if not br:
                continue
            x = [float(r["global_epoch"]) for r in br]
            y = [float(r[key]) for r in br]
            plt.plot(x, y, marker="o", label=branch)

        plt.xlabel("Global epoch")
        plt.ylabel(key)
        plt.title(key)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path / f"matched_{key}.png", dpi=160)
        plt.close()

    # Specific refinement-rate decay plot for Ours.
    ours = [r for r in rows if str(r["branch"]) == "ours_safety" and "refine_rate" in r]
    if ours:
        y = [float(r["refine_rate"]) for r in ours]
        x = [float(r["global_epoch"]) for r in ours]
        decay = refinement_rate_decay(y, window=min(5, len(y)))
        plt.figure(figsize=(7, 4))
        plt.plot(x, y, marker="o")
        plt.xlabel("Global epoch")
        plt.ylabel("Refinement Rate")
        plt.title(f"Ours refine-rate decay: {decay['start_rate']:.3f} → {decay['end_rate']:.3f}")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path / "ours_refinement_rate_decay.png", dpi=160)
        plt.close()


def run_matched_budget_training(
    config_path: str,
    split_epochs_override: Optional[int] = None,
    branch_epochs_override: Optional[int] = None,
    steps_per_epoch_override: Optional[int] = None,
    batch_size_override: Optional[int] = None,
) -> List[Dict[str, float | str]]:
    config = load_config(config_path)
    set_seed(config["seed"])
    ensure_dirs(config)

    device = get_device(config)
    train_cfg = config["train"]

    split_epochs = split_epochs_override or train_cfg.get("split_epochs", train_cfg.get("base_epochs", 8))
    branch_epochs = branch_epochs_override or train_cfg.get("branch_epochs", train_cfg.get("cotune_epochs", 35))
    steps_per_epoch = steps_per_epoch_override or train_cfg["steps_per_epoch"]
    batch_size = batch_size_override or train_cfg["batch_size"]
    grad_clip = train_cfg.get("grad_clip", 5.0)
    val_batches_n = train_cfg.get("val_batches", 4)
    theta_calib_batches = train_cfg.get("theta_calib_batches", 20)

    generator = make_scene_source(config, split="train")
    val_generator = make_scene_source(config, split="val")
    val_batches = [val_generator.generate_batch(batch_size=batch_size) for _ in range(val_batches_n)]

    checkpoint_dir = Path(config["paths"]["checkpoint_dir"])
    train_log_path = config["paths"]["train_log"]
    results_dir = Path(config["paths"]["results_dir"])

    rows: List[Dict[str, float | str]] = []

    print("\nMatched-budget training settings")
    print("device:           ", device)
    print("split_epochs:     ", split_epochs)
    print("branch_epochs:    ", branch_epochs)
    print("total/model:      ", split_epochs + branch_epochs)
    print("steps_per_epoch:  ", steps_per_epoch)
    print("batch_size:       ", batch_size)
    print("results_dir:      ", results_dir)
    print("checkpoint_dir:   ", checkpoint_dir)

    # ---------------------------
    # Stage 0: shared GT warmup.
    # ---------------------------
    system_shared = FullSystem.from_config(config).to(device)
    optimizer_planner = torch.optim.Adam(system_shared.planner_parameters(), lr=train_cfg["lr"])
    optimizer_theta = torch.optim.Adam(system_shared.theta_parameters(), lr=config["risk"]["lr_theta"])

    global_epoch = 0

    for epoch in range(1, split_epochs + 1):
        global_epoch += 1
        stats = train_gt_epoch(
            system=system_shared,
            generator=generator,
            optimizer_planner=optimizer_planner,
            optimizer_theta=optimizer_theta,
            config=config,
            device=device,
            batch_size=batch_size,
            steps_per_epoch=steps_per_epoch,
            grad_clip=grad_clip,
            epoch=epoch,
            n_epochs=split_epochs,
            stage_name="Shared GT warmup",
            train_theta=True,
        )

        val_metrics = evaluate_small(system_shared, val_batches, config, device, mode="kl_triggered")
        row = make_epoch_row(
            branch="shared_warmup",
            stage="warmup_gt",
            epoch=epoch,
            global_epoch=global_epoch,
            label_type="GT",
            stats_accum=stats,
            val_metrics=val_metrics,
        )
        rows.append(row)
        save_train_log(rows, train_log_path)

        print(
            f"[WARMUP] Epoch {epoch:03d} | "
            f"loss={row['loss_total']:.4f} | "
            f"val_coll={row['val_Collision']:.3f} | "
            f"val_ref={row['val_RefineRate']:.3f} | "
            f"theta={row['theta']:.3f}"
        )

    # Calibrate theta ONCE after the shared warmup and freeze it for both branches.
    calibrate_theta(
        system=system_shared,
        generator=generator,
        config=config,
        device=device,
        batch_size=batch_size,
        n_batches=theta_calib_batches,
    )

    warmup_val = evaluate_small(system_shared, val_batches, config, device, mode="kl_triggered")
    print(
        f"Shared warmup KL validation: collision={warmup_val['Collision']:.3f}, "
        f"refine_rate={warmup_val['RefineRate']:.3f}, score={warmup_val['Score']:.4f}"
    )

    save_checkpoint(
        system=system_shared,
        optimizer_planner=optimizer_planner,
        optimizer_theta=optimizer_theta,
        epoch=split_epochs,
        stage="warmup_gt",
        branch="shared_warmup",
        config=config,
        path=str(checkpoint_dir / "shared_warmup.pt"),
    )

    shared_state = copy.deepcopy(system_shared.state_dict())
    theta_value = float(system_shared.risk_detector.theta.detach().cpu().item())

    # Fairness: both post-split branches receive the exact same random batch
    # sequence, not merely the same number of epochs from the same distribution.
    branch_config = copy.deepcopy(config)
    branch_config["seed"] = int(config.get("seed", 42)) + 2000
    generator_gt = make_scene_source(branch_config, split="train")
    generator_ours = make_scene_source(branch_config, split="train")

    # --------------------------------
    # Branch A: extra GT control.
    # --------------------------------
    system_gt = FullSystem.from_config(config).to(device)
    system_gt.load_state_dict(shared_state)
    system_gt.risk_detector.set_theta(theta_value)
    optimizer_gt = torch.optim.Adam(system_gt.planner_parameters(), lr=train_cfg["lr"])

    best_gt_score = float("inf")

    for epoch in range(1, branch_epochs + 1):
        global_epoch += 1
        stats = train_gt_epoch(
            system=system_gt,
            generator=generator_gt,
            optimizer_planner=optimizer_gt,
            optimizer_theta=None,
            config=config,
            device=device,
            batch_size=batch_size,
            steps_per_epoch=steps_per_epoch,
            grad_clip=grad_clip,
            epoch=epoch,
            n_epochs=branch_epochs,
            stage_name="GT-control continuation",
            train_theta=False,
        )

        val_metrics = evaluate_small(system_gt, val_batches, config, device, mode="kl_triggered")
        row = make_epoch_row(
            branch="gt_control",
            stage="matched_budget_gt",
            epoch=epoch,
            global_epoch=global_epoch,
            label_type="GT",
            stats_accum=stats,
            val_metrics=val_metrics,
        )
        rows.append(row)
        save_train_log(rows, train_log_path)

        save_checkpoint(
            system=system_gt,
            optimizer_planner=optimizer_gt,
            optimizer_theta=None,
            epoch=epoch,
            stage="matched_budget_gt",
            branch="gt_control",
            config=config,
            path=str(checkpoint_dir / "gt_control_last.pt"),
        )

        if val_metrics["Score"] < best_gt_score:
            best_gt_score = val_metrics["Score"]
            save_checkpoint(
                system=system_gt,
                optimizer_planner=optimizer_gt,
                optimizer_theta=None,
                epoch=epoch,
                stage="matched_budget_gt",
                branch="gt_control",
                config=config,
                path=str(checkpoint_dir / "gt_control_best.pt"),
            )

        print(
            f"[GT-CONTROL] Epoch {epoch:03d} | "
            f"val_coll={val_metrics['Collision']:.3f} | "
            f"val_hcoll={val_metrics['Collision_hard']:.3f} | "
            f"val_ref={val_metrics['RefineRate']:.3f} | "
            f"score={val_metrics['Score']:.4f}"
        )

    save_checkpoint(
        system=system_gt,
        optimizer_planner=optimizer_gt,
        optimizer_theta=None,
        epoch=branch_epochs,
        stage="matched_budget_gt",
        branch="gt_control",
        config=config,
        path=str(checkpoint_dir / "gt_control.pt"),
    )

    # ----------------------------------------
    # Branch B: Ours safety-distillation branch.
    # ----------------------------------------
    system_ours = FullSystem.from_config(config).to(device)
    system_ours.load_state_dict(shared_state)
    system_ours.risk_detector.set_theta(theta_value)
    optimizer_ours = torch.optim.Adam(system_ours.planner_parameters(), lr=train_cfg["lr"])

    best_ours_score = float("inf")

    for epoch in range(1, branch_epochs + 1):
        global_epoch += 1
        stats = train_safety_distill_epoch(
            system=system_ours,
            generator=generator_ours,
            optimizer_planner=optimizer_ours,
            config=config,
            device=device,
            batch_size=batch_size,
            steps_per_epoch=steps_per_epoch,
            grad_clip=grad_clip,
            epoch=epoch,
            n_epochs=branch_epochs,
        )

        val_metrics = evaluate_small(system_ours, val_batches, config, device, mode="kl_triggered")
        row = make_epoch_row(
            branch="ours_safety",
            stage="matched_budget_safety_distill",
            epoch=epoch,
            global_epoch=global_epoch,
            label_type="KL-local-refined pseudo-labels",
            stats_accum=stats,
            val_metrics=val_metrics,
        )
        rows.append(row)
        save_train_log(rows, train_log_path)

        save_checkpoint(
            system=system_ours,
            optimizer_planner=optimizer_ours,
            optimizer_theta=None,
            epoch=epoch,
            stage="matched_budget_safety_distill",
            branch="ours_safety",
            config=config,
            path=str(checkpoint_dir / "ours_last.pt"),
        )

        if val_metrics["Score"] < best_ours_score:
            best_ours_score = val_metrics["Score"]
            save_checkpoint(
                system=system_ours,
                optimizer_planner=optimizer_ours,
                optimizer_theta=None,
                epoch=epoch,
                stage="matched_budget_safety_distill",
                branch="ours_safety",
                config=config,
                path=str(checkpoint_dir / "ours_best.pt"),
            )

        print(
            f"[OURS] Epoch {epoch:03d} | "
            f"val_coll={val_metrics['Collision']:.3f} | "
            f"val_hcoll={val_metrics['Collision_hard']:.3f} | "
            f"val_ref={val_metrics['RefineRate']:.3f} | "
            f"score={val_metrics['Score']:.4f}"
        )

    save_checkpoint(
        system=system_ours,
        optimizer_planner=optimizer_ours,
        optimizer_theta=None,
        epoch=branch_epochs,
        stage="matched_budget_safety_distill",
        branch="ours_safety",
        config=config,
        path=str(checkpoint_dir / "ours.pt"),
    )

    save_train_log(rows, train_log_path)
    plot_training_curves(rows, str(results_dir))

    print("\nDone matched-budget training.")
    print(f"Shared warmup:       {checkpoint_dir / 'shared_warmup.pt'}")
    print(f"GT-control best:     {checkpoint_dir / 'gt_control_best.pt'}")
    print(f"Ours safety best:    {checkpoint_dir / 'ours_best.pt'}")
    print(f"Train log:           {train_log_path}")

    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/experiment.yaml")
    parser.add_argument("--split_epochs", type=int, default=None)
    parser.add_argument("--branch_epochs", type=int, default=None)
    parser.add_argument("--steps_per_epoch", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_matched_budget_training(
        config_path=args.config,
        split_epochs_override=args.split_epochs,
        branch_epochs_override=args.branch_epochs,
        steps_per_epoch_override=args.steps_per_epoch,
        batch_size_override=args.batch_size,
    )
