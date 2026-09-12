from __future__ import annotations

import time
from typing import Dict, List, Optional, Sequence

import torch


TensorDict = Dict[str, torch.Tensor]


def move_batch_to_device(batch: TensorDict, device: torch.device | str) -> TensorDict:
    device = torch.device(device)
    return {key: value.to(device) for key, value in batch.items()}


def displacement_error(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    if pred.shape != gt.shape:
        raise ValueError(f"pred and gt shapes differ: {pred.shape} vs {gt.shape}")
    if pred.dim() != 4 or pred.shape[-1] != 2:
        raise ValueError(f"Expected [B,T,N,2], got {pred.shape}")
    return torch.linalg.norm(pred - gt, dim=-1)


def ade(
    pred: torch.Tensor,
    gt: torch.Tensor,
    reduction: str = "mean",
    valid_agent_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    error = displacement_error(pred, gt)
    if valid_agent_mask is None:
        per_scene = error.mean(dim=(1, 2))
    else:
        mask = valid_agent_mask.to(error.device, error.dtype)[:, None, :]
        per_scene = (error * mask).sum(dim=(1, 2)) / (
            mask.sum(dim=(1, 2)).clamp(min=1.0) * error.shape[1]
        )
    if reduction == "mean":
        return per_scene.mean()
    if reduction == "none":
        return per_scene
    raise ValueError(f"Unknown reduction: {reduction}")


def fde(
    pred: torch.Tensor,
    gt: torch.Tensor,
    reduction: str = "mean",
    valid_agent_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    error = displacement_error(pred, gt)[:, -1]
    if valid_agent_mask is None:
        per_scene = error.mean(dim=1)
    else:
        mask = valid_agent_mask.to(error.device, error.dtype)
        per_scene = (error * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
    if reduction == "mean":
        return per_scene.mean()
    if reduction == "none":
        return per_scene
    raise ValueError(f"Unknown reduction: {reduction}")


def _valid_pair_mask(
    trajectory: torch.Tensor,
    valid_agent_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    batch_size, _, n_agents, _ = trajectory.shape
    upper = torch.triu(
        torch.ones(n_agents, n_agents, dtype=torch.bool, device=trajectory.device),
        diagonal=1,
    ).view(1, n_agents, n_agents)
    if valid_agent_mask is None:
        return upper.expand(batch_size, n_agents, n_agents)
    valid = valid_agent_mask.to(trajectory.device, torch.bool)
    return upper & valid[:, :, None] & valid[:, None, :]


def separation_violation_rate(
    trajectory: torch.Tensor,
    d_min: float,
    reduction: str = "mean",
    valid_agent_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if trajectory.dim() != 4 or trajectory.shape[-1] != 2:
        raise ValueError(f"Expected [B,T,N,2], got {trajectory.shape}")
    diff = trajectory[:, :, :, None, :] - trajectory[:, :, None, :, :]
    distance = torch.linalg.norm(diff, dim=-1)
    valid_pairs = _valid_pair_mask(trajectory, valid_agent_mask)
    per_scene = ((distance < float(d_min)) & valid_pairs[:, None]).any(dim=(1, 2, 3)).float()
    if reduction == "mean":
        return per_scene.mean()
    if reduction == "none":
        return per_scene
    raise ValueError(f"Unknown reduction: {reduction}")


def _trajectory_directions(
    trajectory: torch.Tensor,
    initial_velocity: Optional[torch.Tensor] = None,
    eps: float = 0.2,
) -> torch.Tensor:
    batch_size, time_steps, n_agents, _ = trajectory.shape
    if initial_velocity is not None and initial_velocity.shape != (batch_size, n_agents, 2):
        raise ValueError(
            f"initial_velocity must have shape {(batch_size, n_agents, 2)}, got {initial_velocity.shape}"
        )

    if initial_velocity is None:
        first_delta = trajectory[:, 1] - trajectory[:, 0] if time_steps > 1 else torch.zeros_like(trajectory[:, 0])
    else:
        first_delta = initial_velocity

    default = torch.zeros_like(first_delta)
    default[..., 0] = 1.0
    speed = torch.linalg.norm(first_delta, dim=-1, keepdim=True)
    previous = torch.where(speed > eps, first_delta / speed.clamp(min=1e-6), default)
    directions = [previous]

    for step in range(1, time_steps):
        delta = trajectory[:, step] - trajectory[:, step - 1]
        speed = torch.linalg.norm(delta, dim=-1, keepdim=True)
        current = torch.where(speed > eps, delta / speed.clamp(min=1e-6), previous)
        directions.append(current)
        previous = current

    return torch.stack(directions, dim=1)


def approximate_collision_rate(
    trajectory: torch.Tensor,
    reduction: str = "mean",
    valid_agent_mask: Optional[torch.Tensor] = None,
    initial_velocity: Optional[torch.Tensor] = None,
    vehicle_length: float = 4.0,
    vehicle_width: float = 2.0,
) -> torch.Tensor:
    if vehicle_length <= 0 or vehicle_width <= 0:
        raise ValueError("vehicle dimensions must be positive")
    if trajectory.dim() != 4 or trajectory.shape[-1] != 2:
        raise ValueError(f"Expected [B,T,N,2], got {trajectory.shape}")

    forward = _trajectory_directions(trajectory, initial_velocity)
    lateral = torch.stack([-forward[..., 1], forward[..., 0]], dim=-1)
    center_delta = trajectory[:, :, None, :, :] - trajectory[:, :, :, None, :]
    fi = forward[:, :, :, None, :]
    fj = forward[:, :, None, :, :]
    li = lateral[:, :, :, None, :]
    lj = lateral[:, :, None, :, :]
    half_length = 0.5 * float(vehicle_length)
    half_width = 0.5 * float(vehicle_width)

    def overlaps(axis: torch.Tensor) -> torch.Tensor:
        center_projection = torch.abs((center_delta * axis).sum(dim=-1))
        radius_i = half_length * torch.abs((fi * axis).sum(dim=-1)) + half_width * torch.abs((li * axis).sum(dim=-1))
        radius_j = half_length * torch.abs((fj * axis).sum(dim=-1)) + half_width * torch.abs((lj * axis).sum(dim=-1))
        return center_projection <= radius_i + radius_j

    collision = overlaps(fi) & overlaps(li) & overlaps(fj) & overlaps(lj)
    valid_pairs = _valid_pair_mask(trajectory, valid_agent_mask)
    per_scene = (collision & valid_pairs[:, None]).any(dim=(1, 2, 3)).float()
    if reduction == "mean":
        return per_scene.mean()
    if reduction == "none":
        return per_scene
    raise ValueError(f"Unknown reduction: {reduction}")


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.bool()
    if not mask.any():
        return torch.tensor(float("nan"), device=values.device, dtype=values.dtype)
    return values[mask].mean()


def compute_prediction_metrics(
    pred: torch.Tensor,
    gt: torch.Tensor,
    d_min: float,
    interaction_heavy: Optional[torch.Tensor] = None,
    valid_agent_mask: Optional[torch.Tensor] = None,
    initial_velocity: Optional[torch.Tensor] = None,
    vehicle_length: float = 4.0,
    vehicle_width: float = 2.0,
) -> Dict[str, float]:
    ade_scene = ade(pred, gt, "none", valid_agent_mask)
    fde_scene = fde(pred, gt, "none", valid_agent_mask)
    separation_scene = separation_violation_rate(pred, d_min, "none", valid_agent_mask)
    collision_scene = approximate_collision_rate(
        pred,
        "none",
        valid_agent_mask,
        initial_velocity,
        vehicle_length,
        vehicle_width,
    )

    output: Dict[str, float] = {
        "ADE": float(ade_scene.mean().cpu()),
        "FDE": float(fde_scene.mean().cpu()),
        "SeparationViolation": float(separation_scene.mean().cpu()),
        "ApproxCollision": float(collision_scene.mean().cpu()),
        "_count": float(pred.shape[0]),
        "_interaction_count": 0.0,
        "_regular_count": 0.0,
    }

    if interaction_heavy is not None:
        interaction_heavy = interaction_heavy.bool()
        regular = ~interaction_heavy
        output["_interaction_count"] = float(interaction_heavy.sum().cpu())
        output["_regular_count"] = float(regular.sum().cpu())
        for name, values in [
            ("ADE", ade_scene),
            ("FDE", fde_scene),
            ("SeparationViolation", separation_scene),
            ("ApproxCollision", collision_scene),
        ]:
            output[f"{name}_interaction"] = float(_masked_mean(values, interaction_heavy).cpu())
            output[f"{name}_regular"] = float(_masked_mean(values, regular).cpu())
    return output


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def latency_benchmark_ms(
    model,
    scene_batches: Sequence[TensorDict],
    mode: str,
    warmup: int = 10,
    runs_per_scene: int = 1,
    device: Optional[torch.device | str] = None,
) -> Dict[str, float]:
    if not scene_batches:
        return {"mean_ms": float("nan"), "std_ms": float("nan"), "p50_ms": float("nan"), "p95_ms": float("nan"), "n": 0.0}

    device = next(model.parameters()).device if device is None else torch.device(device)
    model.eval()
    prepared = [move_batch_to_device(batch, device) for batch in scene_batches]
    if any(next(iter(batch.values())).shape[0] != 1 for batch in prepared):
        raise ValueError("Latency benchmark requires batch size 1")

    for _ in range(warmup):
        model(prepared[0], mode=mode)
    _sync(device)

    times: List[float] = []
    for batch in prepared:
        for _ in range(max(1, int(runs_per_scene))):
            _sync(device)
            start = time.perf_counter()
            model(batch, mode=mode)
            _sync(device)
            times.append((time.perf_counter() - start) * 1000.0)

    values = torch.tensor(times, dtype=torch.float64)
    return {
        "mean_ms": float(values.mean()),
        "std_ms": float(values.std(unbiased=False)),
        "p50_ms": float(torch.quantile(values, 0.50)),
        "p95_ms": float(torch.quantile(values, 0.95)),
        "n": float(values.numel()),
    }
