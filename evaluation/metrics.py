from __future__ import annotations

import time
from typing import Dict, Iterable, List, Optional

import torch


TensorDict = Dict[str, torch.Tensor]


def move_batch_to_device(batch: TensorDict, device: torch.device | str) -> TensorDict:
    """
    Move all tensor values in a batch dictionary to device.
    """
    device = torch.device(device)

    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def displacement_error(
    pred: torch.Tensor,
    gt: torch.Tensor,
) -> torch.Tensor:
    if pred.shape != gt.shape:
        raise ValueError(f"pred and gt must have same shape, got {pred.shape} and {gt.shape}")

    if pred.dim() != 4 or pred.shape[-1] != 2:
        raise ValueError(f"Expected shape [B,T,N,2], got {pred.shape}")

    return torch.linalg.norm(pred - gt, dim=-1)


def ade(
    pred: torch.Tensor,
    gt: torch.Tensor,
    reduction: str = "mean",
    valid_agent_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Average displacement error with optional padded-agent masking."""
    err = displacement_error(pred, gt)
    if valid_agent_mask is None:
        per_scene = err.mean(dim=(1, 2))
    else:
        mask = valid_agent_mask.to(err.device, err.dtype)[:, None, :]
        per_scene = (err * mask).sum(dim=(1, 2)) / (
            mask.sum(dim=(1, 2)).clamp(min=1.0) * err.shape[1]
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
    # Final displacement error with optional padded-agent masking
    final_err = displacement_error(pred, gt)[:, -1, :]
    if valid_agent_mask is None:
        per_scene = final_err.mean(dim=1)
    else:
        mask = valid_agent_mask.to(final_err.device, final_err.dtype)
        per_scene = (final_err * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
    if reduction == "mean":
        return per_scene.mean()
    if reduction == "none":
        return per_scene
    raise ValueError(f"Unknown reduction: {reduction}")

def min_pairwise_distance(
    traj: torch.Tensor,
    valid_agent_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    # Minimum pairwise point distance, ignoring padded agents
    if traj.dim() != 4 or traj.shape[-1] != 2:
        raise ValueError(f"Expected traj shape [B,T,N,2], got {traj.shape}")
    B, _, N, _ = traj.shape
    diff = traj[:, :, :, None, :] - traj[:, :, None, :, :]
    dist = torch.linalg.norm(diff, dim=-1)
    eye = torch.eye(N, dtype=torch.bool, device=traj.device).view(1, 1, N, N)
    invalid = eye.expand(B, dist.shape[1], N, N)
    if valid_agent_mask is not None:
        valid = valid_agent_mask.to(traj.device, torch.bool)
        valid_pairs = valid[:, :, None] & valid[:, None, :]
        invalid = invalid | ~valid_pairs[:, None, :, :]
    dist = dist.masked_fill(invalid, float("inf"))
    return dist.amin(dim=(1, 2, 3))

def collision_rate(
    traj: torch.Tensor,
    d_min: float = 1.5,
    reduction: str = "mean",
    valid_agent_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    # Fraction of scenes with a point-distance violation
    min_dist = min_pairwise_distance(traj, valid_agent_mask=valid_agent_mask)
    collision = min_dist < d_min
    if reduction == "mean":
        return collision.float().mean()
    if reduction == "none":
        return collision.float()
    raise ValueError(f"Unknown reduction: {reduction}")

def masked_mean(
    values: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    mask = mask.bool()

    if mask.sum() == 0:
        return torch.tensor(float("nan"), device=values.device, dtype=values.dtype)

    return values[mask].mean()


def compute_prediction_metrics(
    pred: torch.Tensor,
    gt: torch.Tensor,
    d_min: float,
    is_hard: Optional[torch.Tensor] = None,
    valid_agent_mask: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    # Compute ADE, FDE, collision rate, hard/easy (in some cases is_hard) metrics
    ade_scene = ade(pred, gt, reduction="none", valid_agent_mask=valid_agent_mask)
    fde_scene = fde(pred, gt, reduction="none", valid_agent_mask=valid_agent_mask)
    coll_scene = collision_rate(
        pred, d_min=d_min, reduction="none", valid_agent_mask=valid_agent_mask
    )

    out: Dict[str, float] = {
        "ADE": float(ade_scene.mean().detach().cpu().item()),
        "FDE": float(fde_scene.mean().detach().cpu().item()),
        "Collision": float(coll_scene.mean().detach().cpu().item()),
        "MinDist": float(
            min_pairwise_distance(pred, valid_agent_mask=valid_agent_mask)
            .mean().detach().cpu().item()
        ),
    }

    if is_hard is not None:
        is_hard = is_hard.bool()
        is_easy = ~is_hard

        out.update(
            {
                "ADE_hard": float(masked_mean(ade_scene, is_hard).detach().cpu().item()),
                "FDE_hard": float(masked_mean(fde_scene, is_hard).detach().cpu().item()),
                "Collision_hard": float(masked_mean(coll_scene, is_hard).detach().cpu().item()),
                "ADE_easy": float(masked_mean(ade_scene, is_easy).detach().cpu().item()),
                "FDE_easy": float(masked_mean(fde_scene, is_easy).detach().cpu().item()),
                "Collision_easy": float(masked_mean(coll_scene, is_easy).detach().cpu().item()),
            }
        )

    return out


def refinement_rate_from_stats(stats: Dict[str, object]) -> float:
    if "refine_rate" in stats:
        return float(stats["refine_rate"])

    if "refiner_called" in stats:
        called = stats["refiner_called"]

        if torch.is_tensor(called):
            return float(called.float().mean().detach().cpu().item())

    raise KeyError("stats must contain either 'refine_rate' or 'refiner_called'")


def refinement_rate_decay(
    refine_rates: Iterable[float],
    window: int = 5,
) -> Dict[str, float]:
    rates: List[float] = [float(x) for x in refine_rates]

    if len(rates) == 0:
        return {
            "start_rate": float("nan"),
            "end_rate": float("nan"),
            "absolute_drop": float("nan"),
            "relative_drop": float("nan"),
        }

    w = min(window, len(rates))

    start_rate = sum(rates[:w]) / w
    end_rate = sum(rates[-w:]) / w

    absolute_drop = start_rate - end_rate

    if abs(start_rate) < 1e-12:
        relative_drop = float("nan")
    else:
        relative_drop = absolute_drop / start_rate

    return {
        "start_rate": start_rate,
        "end_rate": end_rate,
        "absolute_drop": absolute_drop,
        "relative_drop": relative_drop,
    }


@torch.no_grad()
def latency_ms(
    model,
    batch: TensorDict,
    mode: str = "kl_triggered",
    warmup: int = 5,
    runs: int = 30,
    device: Optional[torch.device | str] = None,
) -> Dict[str, float]
    if device is None:
        device = next(model.parameters()).device
    else:
        device = torch.device(device)

    batch = move_batch_to_device(batch, device)

    model.eval()

    for _ in range(warmup):
        _ = model(batch, mode=mode, training=False)

    if device.type == "cuda":
        torch.cuda.synchronize()

    times = []

    for _ in range(runs):
        start = time.perf_counter()

        _ = model(batch, mode=mode, training=False)

        if device.type == "cuda":
            torch.cuda.synchronize()

        end = time.perf_counter()
        times.append((end - start) * 1000.0)

    times_tensor = torch.tensor(times, dtype=torch.float32)

    batch_size = None

    for value in batch.values():
        if torch.is_tensor(value):
            batch_size = value.shape[0]
            break

    if batch_size is None:
        batch_size = 1

    mean_ms = float(times_tensor.mean().item())
    std_ms = float(times_tensor.std(unbiased=False).item())

    return {
        "mean_ms": mean_ms,
        "std_ms": std_ms,
        "mean_per_scene_ms": mean_ms / batch_size,
        "std_per_scene_ms": std_ms / batch_size,
    }
