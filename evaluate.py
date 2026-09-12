from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch
from tqdm import tqdm

from data.factory import make_scene_source
from evaluation.metrics import (
    compute_prediction_metrics,
    latency_benchmark_ms,
    move_batch_to_device,
)
from models.full_system import FullSystem
from utils import ensure_dirs, get_device, load_config, set_seed


TensorDict = Dict[str, torch.Tensor]


def average_dicts(rows: List[Dict[str, float]]) -> Dict[str, float]:
    if not rows:
        return {}
    output: Dict[str, float] = {}
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


def load_system(config: dict, checkpoint_path: str, device: torch.device) -> FullSystem:
    system = FullSystem.from_config(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    system.load_state_dict(checkpoint["system_state_dict"])
    system.eval()
    return system


@torch.no_grad()
def evaluate_method(
    system: FullSystem,
    batches: List[TensorDict],
    config: dict,
    mode: str,
    device: torch.device,
) -> Dict[str, float]:
    rows: List[Dict[str, float]] = []
    scenes = 0
    refined = 0
    max_kl_sum = 0.0

    for batch_cpu in tqdm(batches, desc=mode, leave=False):
        batch = move_batch_to_device(batch_cpu, device)
        output = system(batch, mode=mode)
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

    if scenes == 0:
        raise RuntimeError("No usable evaluation scenes")

    metrics = average_dicts(rows)
    metrics["RefineRate"] = refined / scenes
    metrics["MeanMaxKL"] = max_kl_sum / scenes
    metrics["NScenes"] = float(scenes)
    return metrics


def format_rate(start: float, end: float) -> str:
    if math.isnan(start) or math.isnan(end):
        return "N/A"
    return f"{start:.3f} → {end:.3f}"


@torch.no_grad()
def evaluate_matched_budget_table(
    config_path: str = "configs/experiment.yaml",
    warmup_checkpoint: str = "results_av2/checkpoints/shared_warmup.pt",
    gt_checkpoint: str = "results_av2/checkpoints/gt_control_last.pt",
    ours_checkpoint: str = "results_av2/checkpoints/ours_last.pt",
    n_batches: int | None = None,
    batch_size: int | None = None,
    split: str = "val",
) -> pd.DataFrame:
    config = load_config(config_path)
    set_seed(config["seed"])
    ensure_dirs(config)
    device = get_device(config)

    n_batches = config["eval"].get("n_batches") if n_batches is None else n_batches
    batch_size = int(config["eval"].get("batch_size", 128)) if batch_size is None else batch_size

    for checkpoint in [warmup_checkpoint, gt_checkpoint, ours_checkpoint]:
        if not Path(checkpoint).exists():
            raise FileNotFoundError(checkpoint)

    source = make_scene_source(config, split=split)
    eval_batches = list(
        source.iter_batches(
            batch_size=batch_size,
            shuffle=False,
            max_batches=n_batches,
        )
    )
    latency_batches = list(
        source.iter_batches(
            batch_size=1,
            shuffle=False,
            max_batches=int(config["eval"].get("latency_scenes", 50)),
        )
    )
    if not eval_batches or not latency_batches:
        raise RuntimeError("Selected split produced no usable scenes")

    warmup_system = load_system(config, warmup_checkpoint, device)
    gt_system = load_system(config, gt_checkpoint, device)
    ours_system = load_system(config, ours_checkpoint, device)
    warmup_rate = evaluate_method(
        warmup_system, eval_batches, config, "kl_triggered", device
    )["RefineRate"]

    methods = [
        ("Fast only", gt_system, gt_checkpoint, "fast_only", "8 GT + 35 GT"),
        ("Always refine", gt_system, gt_checkpoint, "always_refine", "8 GT + 35 GT"),
        ("Scene-level switching", gt_system, gt_checkpoint, "scene_switching", "8 GT + 35 GT"),
        ("KL-triggered local (Ours)", gt_system, gt_checkpoint, "kl_triggered", "8 GT + 35 GT"),
        ("Ours + Safety Distillation", ours_system, ours_checkpoint, "kl_triggered", "8 GT + 35 GT+distill"),
    ]

    detailed_rows = []
    main_rows = []
    for name, system, checkpoint, mode, budget in methods:
        metrics = evaluate_method(system, eval_batches, config, mode, device)
        latency = latency_benchmark_ms(
            system,
            latency_batches,
            mode,
            warmup=int(config["eval"].get("latency_warmup", 10)),
            runs_per_scene=int(config["eval"].get("latency_runs_per_scene", 1)),
            device=device,
        )
        refine_rate = metrics["RefineRate"]
        if mode == "fast_only":
            start_rate, end_rate = 0.0, 0.0
        elif mode == "always_refine":
            start_rate, end_rate = 1.0, 1.0
        elif name == "Ours + Safety Distillation":
            start_rate, end_rate = warmup_rate, refine_rate
        elif mode == "kl_triggered":
            start_rate, end_rate = warmup_rate, refine_rate
        else:
            start_rate, end_rate = refine_rate, refine_rate

        detailed_rows.append(
            {
                "Method": name,
                "TrainingBudget": budget,
                "Checkpoint": checkpoint,
                "Mode": mode,
                "NScenes": metrics["NScenes"],
                "ADE": metrics["ADE"],
                "FDE": metrics["FDE"],
                "ADE_interaction": metrics.get("ADE_interaction", float("nan")),
                "FDE_interaction": metrics.get("FDE_interaction", float("nan")),
                "SeparationViolation": metrics["SeparationViolation"],
                "SeparationViolation_interaction": metrics.get(
                    "SeparationViolation_interaction", float("nan")
                ),
                "ApproxCollision": metrics["ApproxCollision"],
                "ApproxCollision_interaction": metrics.get(
                    "ApproxCollision_interaction", float("nan")
                ),
                "Latency_mean_ms_batch1": latency["mean_ms"],
                "Latency_p50_ms_batch1": latency["p50_ms"],
                "Latency_p95_ms_batch1": latency["p95_ms"],
                "RefineRate": refine_rate,
                "MeanMaxKL": metrics["MeanMaxKL"],
                "Theta": float(system.risk_detector.theta.detach().cpu()),
            }
        )
        main_rows.append(
            {
                "Method": name,
                "Training Budget": budget,
                "ADE ↓ (interaction)": metrics.get("ADE_interaction", float("nan")),
                "FDE ↓ (interaction)": metrics.get("FDE_interaction", float("nan")),
                "Approx collision ↓": metrics["ApproxCollision"],
                "Separation violation ↓": metrics["SeparationViolation"],
                "Latency P50 ms ↓": latency["p50_ms"],
                "Latency P95 ms ↓": latency["p95_ms"],
                "Refine Rate": format_rate(start_rate, end_rate),
            }
        )

    results_dir = Path(config["paths"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(detailed_rows).to_csv(results_dir / "detailed_results.csv", index=False)
    main_df = pd.DataFrame(main_rows)
    main_df.to_csv(results_dir / "project_table.csv", index=False)
    main_df[
        [
            "Method",
            "Training Budget",
            "Approx collision ↓",
            "Separation violation ↓",
            "Latency P50 ms ↓",
            "Latency P95 ms ↓",
            "Refine Rate",
        ]
    ].to_csv(results_dir / "slide_table.csv", index=False)

    ablation_rows = []
    for mode, name in [
        ("distance_local", "Distance-triggered local"),
        ("kl_triggered", "KL-triggered local"),
    ]:
        metrics = evaluate_method(gt_system, eval_batches, config, mode, device)
        latency = latency_benchmark_ms(
            gt_system,
            latency_batches,
            mode,
            warmup=int(config["eval"].get("latency_warmup", 10)),
            runs_per_scene=int(config["eval"].get("latency_runs_per_scene", 1)),
            device=device,
        )
        ablation_rows.append(
            {
                "Method": name,
                "ApproxCollision": metrics["ApproxCollision"],
                "SeparationViolation": metrics["SeparationViolation"],
                "ADE_interaction": metrics.get("ADE_interaction", float("nan")),
                "RefineRate": metrics["RefineRate"],
                "Latency_p50_ms": latency["p50_ms"],
                "Latency_p95_ms": latency["p95_ms"],
            }
        )
    pd.DataFrame(ablation_rows).to_csv(
        results_dir / "distance_vs_kl_ablation.csv", index=False
    )
    print(main_df.to_string(index=False))
    return main_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--warmup_checkpoint", default="results_av2/checkpoints/shared_warmup.pt")
    parser.add_argument("--gt_checkpoint", default="results_av2/checkpoints/gt_control_last.pt")
    parser.add_argument("--ours_checkpoint", default="results_av2/checkpoints/ours_last.pt")
    parser.add_argument("--n_batches", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_matched_budget_table(
        args.config,
        args.warmup_checkpoint,
        args.gt_checkpoint,
        args.ours_checkpoint,
        args.n_batches,
        args.batch_size,
        args.split,
    )
