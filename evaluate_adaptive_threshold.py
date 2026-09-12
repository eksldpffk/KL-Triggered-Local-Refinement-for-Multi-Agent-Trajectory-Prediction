from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch

from data.factory import make_scene_source
from evaluation.metrics import ade, approximate_collision_rate, fde, move_batch_to_device, separation_violation_rate
from models.full_system import FullSystem
from utils import get_device, load_config, set_seed


@torch.no_grad()
def evaluate_threshold_contexts(
    config_path: str,
    checkpoint_path: str,
    safety_levels: List[float],
    n_batches: int | None = None,
    batch_size: int | None = None,
    output_csv: str | None = None,
    split: str = "val",
) -> pd.DataFrame:
    config = load_config(config_path)
    set_seed(config.get("seed", 42))
    device = get_device(config)
    source = make_scene_source(config, split=split)
    n_batches = config["eval"].get("n_batches") if n_batches is None else n_batches
    batch_size = int(config["eval"].get("batch_size", 128)) if batch_size is None else batch_size
    batches = list(
        source.iter_batches(
            batch_size=batch_size,
            shuffle=False,
            max_batches=n_batches,
        )
    )
    if not batches:
        raise RuntimeError("Selected split produced no usable batches")

    system = FullSystem.from_config(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    system.load_state_dict(checkpoint["system_state_dict"])
    system.eval()

    rows: List[Dict[str, float]] = []
    for safety in safety_levels:
        for batch_index, batch_cpu in enumerate(batches):
            batch = move_batch_to_device(batch_cpu, device)
            batch["safety_requirement"] = torch.full(
                (batch["future_positions"].shape[0],), float(safety), device=device
            )
            output = system(batch, mode="kl_triggered")
            valid = batch["valid_agent_mask"]
            ade_scene = ade(output["final_traj"], batch["future_positions"], "none", valid)
            fde_scene = fde(output["final_traj"], batch["future_positions"], "none", valid)
            separation_scene = separation_violation_rate(
                output["final_traj"], config["scene"]["d_min"], "none", valid
            )
            metrics_cfg = config.get("metrics", {})
            collision_scene = approximate_collision_rate(
                output["final_traj"],
                "none",
                valid,
                batch["past_velocities"][:, -1],
                metrics_cfg.get("vehicle_length", 4.0),
                metrics_cfg.get("vehicle_width", 2.0),
            )
            risk = output["risk"]
            for scene_index in range(output["final_traj"].shape[0]):
                rows.append(
                    {
                        "safety_requirement": float(safety),
                        "batch": float(batch_index),
                        "ADE": float(ade_scene[scene_index].cpu()),
                        "FDE": float(fde_scene[scene_index].cpu()),
                        "SeparationViolation": float(separation_scene[scene_index].cpu()),
                        "ApproxCollision": float(collision_scene[scene_index].cpu()),
                        "Refined": float(risk["risk_flags"][scene_index].float().cpu()),
                        "Theta": float(risk["theta_scene"][scene_index].cpu()),
                        "Density": float(risk["density"][scene_index].cpu()),
                        "Uncertainty": float(risk["uncertainty"][scene_index].cpu()),
                        "MaxKL": float(risk["max_kl"][scene_index].cpu()),
                    }
                )

    df = pd.DataFrame(rows)
    summary = (
        df.groupby("safety_requirement")
        .agg(
            scenes=("ApproxCollision", "size"),
            approximate_collision_rate=("ApproxCollision", "mean"),
            separation_violation_rate=("SeparationViolation", "mean"),
            refine_rate=("Refined", "mean"),
            mean_theta=("Theta", "mean"),
            ADE=("ADE", "mean"),
            FDE=("FDE", "mean"),
        )
        .reset_index()
    )
    if output_csv is None:
        output_csv = str(Path(config["paths"]["results_dir"]) / "adaptive_threshold_summary.csv")
    target = Path(output_csv)
    target.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(target, index=False)
    df.to_csv(target.with_name(target.stem + "_scenes.csv"), index=False)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--safety-levels", nargs="+", type=float, default=[0.25, 0.5, 0.75])
    parser.add_argument("--n-batches", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    args = parser.parse_args()
    print(
        evaluate_threshold_contexts(
            args.config,
            args.checkpoint,
            args.safety_levels,
            args.n_batches,
            args.batch_size,
            args.output,
            args.split,
        ).to_string(index=False)
    )


if __name__ == "__main__":
    main()
