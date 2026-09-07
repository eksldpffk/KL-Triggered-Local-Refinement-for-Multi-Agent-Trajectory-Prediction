from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import torch
from tqdm import tqdm

from data.factory import make_scene_source
from evaluation.metrics import compute_prediction_metrics, latency_ms, move_batch_to_device, refinement_rate_decay
from models.full_system import FullSystem
from utils import ensure_dirs, get_device, load_config, set_seed


TensorDict = Dict[str, torch.Tensor]


def average_dicts(rows: List[Dict[str, float]]) -> Dict[str, float]:
    if len(rows) == 0:
        return {}
    out: Dict[str, float] = {}
    for key in rows[0].keys():
        vals = [float(row[key]) for row in rows if not math.isnan(float(row[key]))]
        out[key] = float(sum(vals) / len(vals)) if vals else float("nan")
    return out


def load_system_from_checkpoint(config: dict, checkpoint_path: str, device: torch.device) -> FullSystem:
    system = FullSystem.from_config(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    system.load_state_dict(checkpoint["system_state_dict"])
    system.eval()
    return system


@torch.no_grad()
def make_safe_target(system: FullSystem, batch: TensorDict) -> torch.Tensor:
    risky_pairs = system._pairs_from_hard_pairs_tensor(batch["hard_pairs"])
    refiner_output = system.local_refiner(
        global_output=batch["future_positions"],
        risky_pairs=risky_pairs,
    )
    return refiner_output["refined_traj"]


@torch.no_grad()
def evaluate_method(
    system: FullSystem,
    batches: List[TensorDict],
    config: dict,
    mode: str,
    device: torch.device,
) -> Dict[str, float]:
    system.eval()
    d_min = config["scene"]["d_min"]

    metric_rows: List[Dict[str, float]] = []
    refine_rates: List[float] = []
    mean_max_kls: List[float] = []

    for batch_cpu in tqdm(batches, desc=f"Evaluating {mode}", leave=False):
        batch = move_batch_to_device(batch_cpu, device)
        output = system(batch=batch, mode=mode, training=False)

        pred = output["final_traj"]
        gt = batch["future_positions"]

        valid_mask = batch.get("valid_agent_mask", batch.get("agent_mask"))
        metrics = compute_prediction_metrics(
            pred=pred, gt=gt, d_min=d_min, is_hard=batch["is_hard"],
            valid_agent_mask=valid_mask,
        )

        safe_target = make_safe_target(system=system, batch=batch)
        safe_metrics = compute_prediction_metrics(
            pred=pred, gt=safe_target, d_min=d_min, is_hard=batch["is_hard"],
            valid_agent_mask=valid_mask,
        )
        metrics["SafeADE"] = safe_metrics["ADE"]
        metrics["SafeFDE"] = safe_metrics["FDE"]
        metrics["SafeADE_hard"] = safe_metrics["ADE_hard"]
        metrics["SafeFDE_hard"] = safe_metrics["FDE_hard"]

        metric_rows.append(metrics)
        refine_rates.append(float(output["stats"]["refine_rate"]))
        mean_max_kls.append(float(output["stats"]["mean_max_kl"]))

    avg = average_dicts(metric_rows)
    avg["RefineRate"] = float(sum(refine_rates) / len(refine_rates))
    avg["MeanMaxKL"] = float(sum(mean_max_kls) / len(mean_max_kls))
    return avg


@torch.no_grad()
def average_latency_ms(
    model: FullSystem,
    batches: List[TensorDict],
    mode: str,
    config: dict,
    device: torch.device,
    max_batches: int = 5,
) -> Dict[str, float]:
    rows = []
    for batch_cpu in batches[:max_batches]:
        batch = move_batch_to_device(batch_cpu, device)
        row = latency_ms(
            model=model,
            batch=batch,
            mode=mode,
            warmup=config["eval"].get("latency_warmup", 5),
            runs=config["eval"].get("latency_runs", 30),
            device=device,
        )
        rows.append(row)

    return {
        "mean_ms": sum(r["mean_ms"] for r in rows) / len(rows),
        "std_ms": sum(r["std_ms"] for r in rows) / len(rows),
        "mean_per_scene_ms": sum(r["mean_per_scene_ms"] for r in rows) / len(rows),
        "std_per_scene_ms": sum(r["std_per_scene_ms"] for r in rows) / len(rows),
    }


def format_pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def format_refine_rate(start: float, end: float) -> str:
    if math.isnan(start) or math.isnan(end):
        return "N/A"
    return f"{start:.3f} → {end:.3f}"


def read_branch_decay(train_log_path: str, branch: str) -> Tuple[float, float]:
    path = Path(train_log_path)
    if not path.exists():
        return float("nan"), float("nan")

    df = pd.read_csv(path)
    if "branch" not in df.columns or "refine_rate" not in df.columns:
        return float("nan"), float("nan")

    branch_df = df[df["branch"] == branch]
    if len(branch_df) == 0:
        return float("nan"), float("nan")

    rates = branch_df["refine_rate"].astype(float).tolist()
    decay = refinement_rate_decay(rates, window=min(5, len(rates)))
    return decay["start_rate"], decay["end_rate"]


def plot_refinement_rates(train_log_path: str, save_path: str) -> None:
    path = Path(train_log_path)
    if not path.exists():
        print(f"Train log not found: {train_log_path}")
        return

    df = pd.read_csv(path)
    if "branch" not in df.columns or "refine_rate" not in df.columns:
        print("branch/refine_rate columns not found in train log.")
        return

    plt.figure(figsize=(7, 4))
    for branch in ["shared_warmup", "gt_control", "ours_safety"]:
        sub = df[df["branch"] == branch]
        if len(sub) == 0:
            continue
        plt.plot(sub["global_epoch"], sub["refine_rate"], marker="o", label=branch)

    plt.xlabel("Global epoch")
    plt.ylabel("Refinement Rate")
    plt.title("Matched-budget refinement-rate comparison")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    save_obj = Path(save_path)
    save_obj.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_obj, dpi=160)
    plt.close()
    print(f"Saved refinement-rate plot to: {save_obj}")


@torch.no_grad()
def evaluate_matched_budget_table(
    config_path: str = "configs/experiment.yaml",
    warmup_checkpoint: str = "results_interaction/checkpoints/shared_warmup.pt",
    gt_checkpoint: str = "results_interaction/checkpoints/gt_control_best.pt",
    ours_checkpoint: str = "results_interaction/checkpoints/ours_best.pt",
    n_batches: int | None = None,
    batch_size: int | None = None,
    split: str = "test",
) -> pd.DataFrame:
    config = load_config(config_path)
    set_seed(config["seed"])
    ensure_dirs(config)

    device = get_device(config)
    if n_batches is None:
        n_batches = config["eval"]["n_batches"]
    if batch_size is None:
        batch_size = config["eval"]["batch_size"]

    for p in [warmup_checkpoint, gt_checkpoint, ours_checkpoint]:
        if not Path(p).exists():
            raise FileNotFoundError(f"Checkpoint not found: {p}")

    generator = make_scene_source(config, split=split)
    eval_batches = [generator.generate_batch(batch_size=batch_size) for _ in range(n_batches)]

    warmup_system = load_system_from_checkpoint(config, warmup_checkpoint, device)
    gt_system = load_system_from_checkpoint(config, gt_checkpoint, device)
    ours_system = load_system_from_checkpoint(config, ours_checkpoint, device)

    # Evaluate warmup only to define start rates and optional detailed reference.
    warmup_kl_metrics = evaluate_method(warmup_system, eval_batches, config, "kl_triggered", device)
    warmup_rate = warmup_kl_metrics["RefineRate"]

    methods = [
        {
            "Method": "Global only + extra GT",
            "System": gt_system,
            "Checkpoint": gt_checkpoint,
            "Mode": "global_only",
            "Training Budget": "8 GT + 35 GT",
            "Post-split labels": "GT",
        },
        {
            "Method": "Always refine + extra GT",
            "System": gt_system,
            "Checkpoint": gt_checkpoint,
            "Mode": "always_refine",
            "Training Budget": "8 GT + 35 GT",
            "Post-split labels": "GT",
        },
        {
            "Method": "Scene-level switching + extra GT",
            "System": gt_system,
            "Checkpoint": gt_checkpoint,
            "Mode": "scene_switching",
            "Training Budget": "8 GT + 35 GT",
            "Post-split labels": "GT",
        },
        {
            "Method": "KL-triggered + extra GT",
            "System": gt_system,
            "Checkpoint": gt_checkpoint,
            "Mode": "kl_triggered",
            "Training Budget": "8 GT + 35 GT",
            "Post-split labels": "GT",
        },
        {
            "Method": "Ours + selective safety distillation",
            "System": ours_system,
            "Checkpoint": ours_checkpoint,
            "Mode": "kl_triggered",
            "Training Budget": "8 GT + 35 safety",
            "Post-split labels": "KL-local safe pseudo-labels",
        },
    ]

    detailed_rows = []
    main_rows = []

    print("\nMatched-budget evaluation settings:")
    print("device: ", device)
    print("warmup ckpt: ", warmup_checkpoint)
    print("gt ckpt: ", gt_checkpoint)
    print("ours ckpt: ", ours_checkpoint)
    print("evaluation split: ", split)
    print("n_batches: ", n_batches)
    print("batch_size: ", batch_size)
    print("warmup KL refine: ", f"{warmup_rate:.3f}")

    for item in methods:
        metrics = evaluate_method(item["System"], eval_batches, config, item["Mode"], device)
        latency = average_latency_ms(
            model=item["System"],
            batches=eval_batches,
            mode=item["Mode"],
            config=config,
            device=device,
            max_batches=min(5, len(eval_batches)),
        )
        refine_rate = metrics["RefineRate"]

        if item["Mode"] == "global_only":
            refine_start, refine_end = 0.0, 0.0
        elif item["Mode"] == "always_refine":
            refine_start, refine_end = 1.0, 1.0
        elif item["Method"] == "Ours + selective safety distillation":
            refine_start, refine_end = warmup_rate, refine_rate
        elif item["Method"] == "KL-triggered + extra GT":
            refine_start, refine_end = warmup_rate, refine_rate
        else:
            refine_start, refine_end = refine_rate, refine_rate

        detailed_row = {
            "Method": item["Method"],
            "Training Budget": item["Training Budget"],
            "Post-split labels": item["Post-split labels"],
            "Checkpoint": item["Checkpoint"],
            "Mode": item["Mode"],
            "ADE": metrics["ADE"],
            "FDE": metrics["FDE"],
            "ADE_hard": metrics["ADE_hard"],
            "FDE_hard": metrics["FDE_hard"],
            "SafeADE": metrics["SafeADE"],
            "SafeFDE": metrics["SafeFDE"],
            "SafeADE_hard": metrics["SafeADE_hard"],
            "SafeFDE_hard": metrics["SafeFDE_hard"],
            "Collision": metrics["Collision"],
            "Collision_hard": metrics["Collision_hard"],
            "Collision_easy": metrics["Collision_easy"],
            "Latency_ms": latency["mean_per_scene_ms"],
            "LatencyBatch_ms": latency["mean_ms"],
            "Latency_std_ms": latency["std_per_scene_ms"],
            "RefineRate_eval": refine_rate,
            "RefineRate_start": refine_start,
            "RefineRate_end": refine_end,
            "MeanMaxKL": metrics["MeanMaxKL"],
            "Theta": float(item["System"].risk_detector.theta.detach().cpu().item()),
        }

        main_row = {
            "Method": item["Method"],
            "Training Budget": item["Training Budget"],
            "Post-split labels": item["Post-split labels"],
            "ADE ↓ (hard)": metrics["ADE_hard"],
            "FDE ↓ (hard)": metrics["FDE_hard"],
            "Collision ↓": metrics["Collision"],
            "Latency (Mean) ↓": latency["mean_per_scene_ms"],
            "Refine Rate (Start → End)": format_refine_rate(refine_start, refine_end),
        }

        detailed_rows.append(detailed_row)
        main_rows.append(main_row)

    detailed_df = pd.DataFrame(detailed_rows)
    main_df = pd.DataFrame(main_rows)

    results_dir = Path(config["paths"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    detailed_path = results_dir / "detailed_results.csv"
    main_path = results_dir / "project_table.csv"
    slide_path = results_dir / "slide_table.csv"

    detailed_df.to_csv(detailed_path, index=False)
    main_df.to_csv(main_path, index=False)

    slide_cols = [
        "Method",
        "Training Budget",
        "Collision ↓",
        "Latency (Mean) ↓",
        "Refine Rate (Start → End)",
    ]
    main_df[slide_cols].to_csv(slide_path, index=False)

    print("\nMatched-budget main project table:")
    print(main_df.to_string(index=False))

    print("\nMatched-budget detailed results:")
    print(detailed_df.to_string(index=False))

    print(f"\nSaved main table to:     {main_path}")
    print(f"Saved detailed table to: {detailed_path}")
    print(f"Saved slide table to:    {slide_path}")

    plot_refinement_rates(
        train_log_path=config["paths"]["train_log"],
        save_path=str(results_dir / "refinement_rates.png"),
    )

    return main_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/experiment.yaml")
    parser.add_argument("--warmup_checkpoint", type=str, default="results_interaction/checkpoints/shared_warmup.pt")
    parser.add_argument("--gt_checkpoint", type=str, default="results_interaction/checkpoints/gt_control_best.pt")
    parser.add_argument("--ours_checkpoint", type=str, default="results_interaction/checkpoints/ours_best.pt")
    parser.add_argument("--n_batches", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_matched_budget_table(
        config_path=args.config,
        warmup_checkpoint=args.warmup_checkpoint,
        gt_checkpoint=args.gt_checkpoint,
        ours_checkpoint=args.ours_checkpoint,
        n_batches=args.n_batches,
        batch_size=args.batch_size,
        split=args.split,
    )
