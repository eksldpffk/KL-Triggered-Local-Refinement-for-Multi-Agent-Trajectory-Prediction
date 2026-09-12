from __future__ import annotations

import argparse
import copy
import csv
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from data.factory import make_scene_source
from evaluation.metrics import compute_prediction_metrics, move_batch_to_device
from models.full_system import FullSystem
from utils import ensure_dirs, get_device, load_config, set_seed


TensorDict = Dict[str, torch.Tensor]


def gaussian_nll(
    mu: torch.Tensor,
    log_sigma: torch.Tensor,
    target: torch.Tensor,
    valid_agent_mask: torch.Tensor,
    agent_mask: Optional[torch.Tensor] = None,
    min_log_sigma: float = -5.0,
    max_log_sigma: float = 2.0,
) -> torch.Tensor:
    log_sigma = log_sigma.clamp(min_log_sigma, max_log_sigma)
    sigma = torch.exp(log_sigma)
    loss = 0.5 * ((target - mu) / sigma).square() + log_sigma + 0.5 * math.log(2.0 * math.pi)
    mask = valid_agent_mask.bool()
    if agent_mask is not None:
        mask = mask & agent_mask.bool()
    mask = mask[:, None, :, None].to(mu.device, mu.dtype).expand_as(loss)
    if not mask.any():
        return loss.sum() * 0.0
    return (loss * mask).sum() / mask.sum()


def masked_position_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_agent_mask: torch.Tensor,
    agent_mask: torch.Tensor,
) -> torch.Tensor:
    error = (prediction - target).square()
    mask = (valid_agent_mask.bool() & agent_mask.bool())[:, None, :, None]
    mask = mask.to(prediction.device, prediction.dtype).expand_as(error)
    if not mask.any():
        return error.sum() * 0.0
    return (error * mask).sum() / mask.sum()


def _weighted_average(rows: List[Dict[str, float]]) -> Dict[str, float]:
    output: Dict[str, float] = {}
    if not rows:
        return output
    for key in [name for name in rows[0] if not name.startswith("_")]:
        if key.endswith("_interaction"):
            count_key = "_interaction_count"
        elif key.endswith("_regular"):
            count_key = "_regular_count"
        else:
            count_key = "_count"
        numerator = 0.0
        denominator = 0.0
        for row in rows:
            value = float(row.get(key, float("nan")))
            weight = float(row.get(count_key, 0.0))
            if math.isnan(value) or weight <= 0:
                continue
            numerator += value * weight
            denominator += weight
        output[key] = numerator / denominator if denominator else float("nan")
    return output


@torch.no_grad()
def evaluate_small(
    system: FullSystem,
    batches: List[TensorDict],
    config: dict,
    device: torch.device,
) -> Dict[str, float]:
    system.eval()
    rows: List[Dict[str, float]] = []
    refined = 0
    scenes = 0
    max_kl_sum = 0.0

    for batch_cpu in batches:
        batch = move_batch_to_device(batch_cpu, device)
        output = system(batch, mode="kl_triggered")
        metrics_cfg = config.get("metrics", {})
        metrics = compute_prediction_metrics(
            output["final_traj"],
            batch["future_positions"],
            config["scene"]["d_min"],
            interaction_heavy=batch["interaction_heavy"],
            valid_agent_mask=batch["valid_agent_mask"],
            initial_velocity=batch["past_velocities"][:, -1],
            vehicle_length=metrics_cfg.get("vehicle_length", 4.0),
            vehicle_width=metrics_cfg.get("vehicle_width", 2.0),
        )
        rows.append(metrics)
        batch_size = output["final_traj"].shape[0]
        scenes += batch_size
        refined += int(output["stats"]["refiner_called"].sum().cpu())
        max_kl_sum += float(output["stats"]["mean_max_kl"]) * batch_size

    metrics = _weighted_average(rows)
    metrics["RefineRate"] = refined / scenes
    metrics["MeanMaxKL"] = max_kl_sum / scenes
    return metrics


def save_checkpoint(
    system: FullSystem,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    branch: str,
    config: dict,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "branch": branch,
            "config": config,
            "system_state_dict": system.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )


def save_log(rows: List[Dict[str, float | str]], path: str) -> None:
    if not rows:
        return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def train_gt_epoch(
    system: FullSystem,
    source,
    optimizer: torch.optim.Optimizer,
    theta_optimizer: Optional[torch.optim.Optimizer],
    config: dict,
    device: torch.device,
    batch_size: int,
    steps: int,
    train_theta: bool,
) -> Dict[str, float]:
    system.train()
    model_cfg = config["model"]
    grad_clip = config["train"].get("grad_clip", 5.0)
    total_loss = 0.0
    refine_rate = 0.0

    for _ in tqdm(range(steps), desc="GT", leave=False):
        batch = move_batch_to_device(source.generate_batch(batch_size), device)
        prior = system.encode_and_forecast(batch)["prior_dist"]
        loss = gaussian_nll(
            prior.mu,
            prior.log_sigma,
            batch["future_positions"],
            batch["valid_agent_mask"],
            min_log_sigma=model_cfg["min_log_sigma"],
            max_log_sigma=model_cfg["max_log_sigma"],
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        clip_grad_norm_(list(system.forecast_parameters()), grad_clip)
        optimizer.step()

        with torch.no_grad():
            prior_for_theta = system.encode_and_forecast(batch)["prior_dist"]
        risk = system.risk_detector(
            prior_for_theta,
            valid_agent_mask=batch["valid_agent_mask"],
            safety_requirement=batch["safety_requirement"],
        )
        theta_loss = system.risk_detector.threshold_loss(
            risk["risk_logits"], batch["interaction_heavy"]
        )
        if train_theta and theta_optimizer is not None:
            theta_optimizer.zero_grad(set_to_none=True)
            theta_loss.backward()
            theta_optimizer.step()

        total_loss += float(loss.detach().cpu())
        refine_rate += float(risk["risk_flags"].float().mean().detach().cpu())

    return {
        "loss": total_loss / steps,
        "refine_rate": refine_rate / steps,
        "theta": float(system.risk_detector.theta.detach().cpu()),
    }


@torch.no_grad()
def calibrate_theta(
    system: FullSystem,
    source,
    config: dict,
    device: torch.device,
    batch_size: int,
    n_batches: int,
) -> float:
    system.eval()
    values = []
    for _ in range(n_batches):
        batch = move_batch_to_device(source.generate_batch(batch_size), device)
        prior = system.encode_and_forecast(batch)["prior_dist"]
        risk = system.risk_detector(
            prior,
            valid_agent_mask=batch["valid_agent_mask"],
            safety_requirement=batch["safety_requirement"],
        )
        values.append(risk["max_kl"])
    return system.risk_detector.calibrate_theta_from_max_kl(
        torch.cat(values),
        config["risk"]["target_refine_rate"],
    )


@torch.no_grad()
def make_kl_local_teacher(
    system: FullSystem,
    batch: TensorDict,
) -> Tuple[torch.Tensor, torch.Tensor]:
    was_training = system.training
    system.eval()
    output = system(batch, mode="kl_triggered")
    teacher = output["final_traj"].detach()
    batch_size, _, n_agents, _ = teacher.shape
    agent_mask = torch.zeros(batch_size, n_agents, dtype=torch.bool, device=teacher.device)
    for batch_index, pairs in enumerate(output["risk"]["risky_pairs"]):
        for i, j in pairs:
            agent_mask[batch_index, i] = True
            agent_mask[batch_index, j] = True
    system.train(was_training)
    return teacher, agent_mask


def train_distillation_epoch(
    system: FullSystem,
    source,
    optimizer: torch.optim.Optimizer,
    config: dict,
    device: torch.device,
    batch_size: int,
    steps: int,
) -> Dict[str, float]:
    system.train()
    model_cfg = config["model"]
    train_cfg = config["train"]
    grad_clip = train_cfg.get("grad_clip", 5.0)
    blend = float(train_cfg.get("safe_target_blend", 0.5))
    lambda_distill = float(train_cfg.get("lambda_distill", 1.0))
    total_loss = 0.0
    total_gt = 0.0
    total_distill = 0.0
    refine_rate = 0.0

    for _ in tqdm(range(steps), desc="GT+distill", leave=False):
        batch = move_batch_to_device(source.generate_batch(batch_size), device)
        teacher, distill_agents = make_kl_local_teacher(system, batch)
        distill_agents &= batch["valid_agent_mask"].bool()

        prior = system.encode_and_forecast(batch)["prior_dist"]
        gt = batch["future_positions"]
        gt_loss = gaussian_nll(
            prior.mu,
            prior.log_sigma,
            gt,
            batch["valid_agent_mask"],
            min_log_sigma=model_cfg["min_log_sigma"],
            max_log_sigma=model_cfg["max_log_sigma"],
        )

        safe_target = (1.0 - blend) * gt + blend * teacher
        distill_loss = masked_position_mse(
            prior.mu,
            safe_target,
            batch["valid_agent_mask"],
            distill_agents,
        )
        loss = gt_loss + lambda_distill * distill_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        clip_grad_norm_(list(system.forecast_parameters()), grad_clip)
        optimizer.step()

        with torch.no_grad():
            risk = system.risk_detector(
                prior,
                valid_agent_mask=batch["valid_agent_mask"],
                safety_requirement=batch["safety_requirement"],
            )
        total_loss += float(loss.detach().cpu())
        total_gt += float(gt_loss.detach().cpu())
        total_distill += float(distill_loss.detach().cpu())
        refine_rate += float(risk["risk_flags"].float().mean().detach().cpu())

    return {
        "loss": total_loss / steps,
        "gt_loss": total_gt / steps,
        "distill_loss": total_distill / steps,
        "refine_rate": refine_rate / steps,
        "theta": float(system.risk_detector.theta.detach().cpu()),
    }


def run_matched_budget_training(
    config_path: str = "configs/experiment.yaml",
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

    split_epochs = split_epochs_override or train_cfg["split_epochs"]
    branch_epochs = branch_epochs_override or train_cfg["branch_epochs"]
    steps = steps_per_epoch_override or train_cfg["steps_per_epoch"]
    batch_size = batch_size_override or train_cfg["batch_size"]

    train_source = make_scene_source(config, split="train")
    val_source = make_scene_source(config, split="val")
    val_batches = list(
        val_source.iter_batches(
            batch_size=train_cfg.get("val_batch_size", batch_size),
            shuffle=False,
            max_batches=train_cfg.get("val_batches", 8),
        )
    )
    if not val_batches:
        raise RuntimeError("Validation split produced no usable scenes")

    results_dir = Path(config["paths"]["results_dir"])
    checkpoint_dir = Path(config["paths"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, float | str]] = []

    shared = FullSystem.from_config(config).to(device)
    forecast_optimizer = torch.optim.Adam(shared.forecast_parameters(), lr=train_cfg["lr"])
    theta_optimizer = torch.optim.Adam(
        shared.theta_parameters(), lr=config["risk"]["lr_theta"]
    )

    for epoch in range(1, split_epochs + 1):
        stats = train_gt_epoch(
            shared,
            train_source,
            forecast_optimizer,
            theta_optimizer,
            config,
            device,
            batch_size,
            steps,
            train_theta=True,
        )
        val = evaluate_small(shared, val_batches, config, device)
        row = {
            "branch": "shared_warmup",
            "epoch": epoch,
            **stats,
            **{f"val_{key}": value for key, value in val.items()},
        }
        rows.append(row)
        save_log(rows, config["paths"]["train_log"])

    calibrate_theta(
        shared,
        train_source,
        config,
        device,
        batch_size,
        int(train_cfg.get("theta_calib_batches", 20)),
    )
    shared_state = copy.deepcopy(shared.state_dict())
    save_checkpoint(
        shared,
        forecast_optimizer,
        split_epochs,
        "shared_warmup",
        config,
        checkpoint_dir / "shared_warmup.pt",
    )

    branch_config = copy.deepcopy(config)
    branch_config["seed"] = int(config["seed"]) + 2000
    gt_source = make_scene_source(branch_config, split="train")
    distill_source = make_scene_source(branch_config, split="train")

    gt_system = FullSystem.from_config(config).to(device)
    gt_system.load_state_dict(shared_state)
    gt_optimizer = torch.optim.Adam(gt_system.forecast_parameters(), lr=train_cfg["lr"])
    for epoch in range(1, branch_epochs + 1):
        stats = train_gt_epoch(
            gt_system,
            gt_source,
            gt_optimizer,
            None,
            config,
            device,
            batch_size,
            steps,
            train_theta=False,
        )
        val = evaluate_small(gt_system, val_batches, config, device)
        rows.append(
            {
                "branch": "gt_control",
                "epoch": epoch,
                **stats,
                **{f"val_{key}": value for key, value in val.items()},
            }
        )
        save_log(rows, config["paths"]["train_log"])

    ours = FullSystem.from_config(config).to(device)
    ours.load_state_dict(shared_state)
    ours_optimizer = torch.optim.Adam(ours.forecast_parameters(), lr=train_cfg["lr"])
    for epoch in range(1, branch_epochs + 1):
        stats = train_distillation_epoch(
            ours,
            distill_source,
            ours_optimizer,
            config,
            device,
            batch_size,
            steps,
        )
        val = evaluate_small(ours, val_batches, config, device)
        rows.append(
            {
                "branch": "ours_safety",
                "epoch": epoch,
                **stats,
                **{f"val_{key}": value for key, value in val.items()},
            }
        )
        save_log(rows, config["paths"]["train_log"])

    save_checkpoint(
        gt_system,
        gt_optimizer,
        branch_epochs,
        "gt_control",
        config,
        checkpoint_dir / "gt_control_last.pt",
    )
    save_checkpoint(
        ours,
        ours_optimizer,
        branch_epochs,
        "ours_safety",
        config,
        checkpoint_dir / "ours_last.pt",
    )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--split_epochs", type=int, default=None)
    parser.add_argument("--branch_epochs", type=int, default=None)
    parser.add_argument("--steps_per_epoch", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_matched_budget_training(
        args.config,
        args.split_epochs,
        args.branch_epochs,
        args.steps_per_epoch,
        args.batch_size,
    )
