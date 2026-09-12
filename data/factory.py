from __future__ import annotations

from data.av2_dataset import AV2SceneGenerator


def make_scene_source(config: dict, split: str | None = None) -> AV2SceneGenerator:
    source = config.get("data", {}).get("source", "av2").lower()
    if source not in {"av2", "argoverse2", "argoverse_2"}:
        raise ValueError(f"Unsupported data.source={source!r}; expected 'av2'")
    return AV2SceneGenerator.from_config(config, split=split)
