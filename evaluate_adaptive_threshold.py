from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch

from data.factory import make_scene_source
from evaluation.metrics import (
    ade,
    collision_rate,
    fde,
    move_batch_to_device,
)
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
    split: str = "test",
) -> pd.DataFrame:
    config = load_config(config_path)
    set_seed(config.get("seed", 42))
    device = get_device(config)
    source = make_scene_source(config, split=split)
    n_batches = n_batches or config["eval"].get("n_batches", 20)
    batch_size = batch_size or config["eval"].get("batch_size", 128)

    system = FullSystem.from_config(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    system.load_state_dict(checkpoint["system_state_dict"])
    system.eval()

    rows: List[Dict[str, float]] = []
    for safety in safety_levels:
        for batch_idx in range(n_batches):
            batch = move_batch_to_device(source.generate_batch(batch_size), device)
            batch["safety_requirement"] = torch.full(
                (batch_size,), float(safety), device=device
            )
            out = system(batch, mode="kl_triggered", training=False)
            valid = batch.get("valid_agent_mask", batch.get("agent_mask"))
            ade_scene = ade(out["final_traj"], batch["future_positions"], "none", valid)
            fde_scene = fde(out["final_traj"], batch["future_positions"], "none", valid)
            coll_scene = collision_rate(
                out["final_traj"], config["scene"]["d_min"], "none", valid
            )
            risk = out["risk"]
            for b in range(batch_size):
                rows.append({
                    "safety_requirement": float(safety),
                    "batch": batch_idx,
                    "ADE": float(ade_scene[b].cpu()),
                    "FDE": float(fde_scene[b].cpu()),
                    "Collision": float(coll_scene[b].cpu()),
                    "Refined": float(risk["risk_flags"][b].float().cpu()),
                    "Theta": float(risk["theta_scene"][b].cpu()),
                    "Density": float(risk["density"][b].cpu()),
                    "Uncertainty": float(risk["uncertainty"][b].cpu()),
                    "MaxKL": float(risk["max_kl"][b].cpu()),
                })

    df = pd.DataFrame(rows)

    def robust_quantile_bin(values: pd.Series, name: str) -> pd.Series:
        # qcut returns no usable groups when every scene has the same value.
        # Keep a single explicit bin in that case so small/debug evaluations and
        # homogeneous subsets still produce a summary table.
        if values.nunique(dropna=True) < 2:
            return pd.Series([f"{name}_all"] * len(values), index=values.index)
        try:
            binned = pd.qcut(values, q=3, duplicates="drop")
            return binned.astype(str)
        except ValueError:
            return pd.Series([f"{name}_all"] * len(values), index=values.index)

    df["density_bin"] = robust_quantile_bin(df["Density"], "density")
    df["uncertainty_bin"] = robust_quantile_bin(df["Uncertainty"], "uncertainty")
    summary = (
        df.groupby(["safety_requirement", "density_bin", "uncertainty_bin"], observed=True)
        .agg(
            scenes=("Collision", "size"),
            collision_rate=("Collision", "mean"),
            refine_rate=("Refined", "mean"),
            mean_theta=("Theta", "mean"),
            ADE=("ADE", "mean"),
            FDE=("FDE", "mean"),
        )
        .reset_index()
    )
    if output_csv is None:
        output_csv = str(Path(config["paths"]["results_dir"]) / "adaptive_threshold_summary.csv")
    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_path, index=False)
    df.to_csv(out_path.with_name(out_path.stem + "_scenes.csv"), index=False)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--safety-levels", nargs="+", type=float, default=[0.25, 0.5, 0.75])
    parser.add_argument("--n-batches", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    args = parser.parse_args()
    table = evaluate_threshold_contexts(
        args.config, args.checkpoint, args.safety_levels,
        args.n_batches, args.batch_size, args.output, args.split,
    )
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
