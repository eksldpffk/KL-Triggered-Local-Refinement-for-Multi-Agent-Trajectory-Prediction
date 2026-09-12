from __future__ import annotations

from dataclasses import dataclass
from collections import OrderedDict
import io
import math
from pathlib import Path
import tarfile
from typing import Dict, Iterator, List, Optional, Sequence, Union

import pandas as pd
import torch


TensorDict = Dict[str, torch.Tensor]

# места нет, берем тар файлы датасета, 55ГБ ту мач для распаковки
@dataclass(frozen=True)
class _TarParquet:
    name: str
    offset: int
    size: int

_REQUIRED_COLUMNS = {
    "track_id",
    "object_type",
    "timestep",
    "position_x",
    "position_y",
    "heading",
    "velocity_x",
    "velocity_y",
}


def _rotation(theta: float) -> torch.Tensor:
    c, s = math.cos(theta), math.sin(theta)
    return torch.tensor([[c, -s], [s, c]], dtype=torch.float32)


def _split_alias(split: str) -> str:
    split = split.lower()
    return "val" if split in {"validation", "valid"} else split


def _split_limit(data_cfg: dict, split: str) -> Optional[int]:
    value = data_cfg.get(f"max_scenes_{split}")
    return None if value is None else int(value)


@dataclass
class AV2SceneGenerator:
    root: str
    split: str = "train"
    n_agents: int = 10
    T_past: int = 10
    T_future: int = 30
    dt: float = 0.1
    local_radius: float = 45.0
    interaction_distance_threshold: float = 5.5
    safety_requirement: float = 0.5
    max_scenes: Optional[int] = None
    cache_size: int = 2048
    seed: int = 42

    def __post_init__(self) -> None:
        if self.T_past <= 0 or self.T_future <= 0:
            raise ValueError("T_past and T_future must be positive")
        if self.n_agents < 2:
            raise ValueError("n_agents must be >= 2")
        if self.cache_size < 0:
            raise ValueError("cache_size must be >= 0")

        self.split = _split_alias(self.split)
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(self.seed)
        self._archive_file = None
        self._paths = self._discover_parquets()
        self._cache: OrderedDict[int, Optional[TensorDict]] = OrderedDict()
        if not self._paths:
            raise RuntimeError(
                f"No AV2 Motion Forecasting parquet files found under {self.root!r} "
                f"for split {self.split!r}"
            )

    @classmethod
    def from_config(cls, config: dict, split: Optional[str] = None) -> "AV2SceneGenerator":
        scene_cfg = config["scene"]
        data_cfg = config["data"]
        chosen_split = _split_alias(split or data_cfg.get("split", "train"))
        root = data_cfg.get(f"{chosen_split}_path") or data_cfg.get("root")

        if not root:
            raise ValueError(
                f"Set data.{chosen_split}_path to the AV2 {chosen_split}.tar archive "
                "or set data.root to an extracted AV2 Motion Forecasting directory"
    )

        return cls(
            root=root,
            split=chosen_split,
            n_agents=scene_cfg["n_agents"],
            T_past=scene_cfg["T_past"],
            T_future=scene_cfg["T_future"],
            dt=scene_cfg["dt"],
            local_radius=scene_cfg.get("local_radius", 45.0),
            interaction_distance_threshold=data_cfg.get(
                "interaction_distance_threshold",
                scene_cfg["d_min"] + config.get("risk", {}).get("safety_margin", 0.0),
            ),
            safety_requirement=data_cfg.get("safety_requirement", 0.5),
            max_scenes=_split_limit(data_cfg, chosen_split),
            cache_size=int(data_cfg.get("cache_size", 2048)),
            seed=int(config.get("seed", 42)) + (0 if chosen_split == "train" else 1000),
        )

    def __len__(self) -> int:
        return len(self._paths)

    # despite the name, u can change it (but no neccesarry - it works with both type of dir)
    def _discover_parquets(self) -> List[Union[Path, _TarParquet]]:
        root = Path(self.root).expanduser()

        if root.is_file():
            if root.suffix.lower() != ".tar":
                raise ValueError(f"Expected an AV2 .tar archive, got {root}")

            entries: List[Union[Path, _TarParquet]] = []

            with tarfile.open(root, mode="r:") as archive:
                for member in archive:
                    if not member.isfile():
                        continue

                    name = member.name.replace("\\", "/")
                    filename = Path(name).name

                    if filename.startswith("scenario_") and filename.endswith(".parquet"):
                        entries.append(
                            _TarParquet(
                                name=name,
                                offset=int(member.offset_data),
                                size=int(member.size),
                            )
                        )

            paths = sorted(entries, key=lambda item: item.name)

        else:
            split_dir = root / self.split
            base = split_dir if split_dir.exists() else root

            paths = sorted(base.rglob("scenario_*.parquet"))

            if not paths:
                paths = sorted(base.rglob("*.parquet"))

        if self.max_scenes is not None and len(paths) > self.max_scenes:
            gen = torch.Generator(device="cpu")
            gen.manual_seed(self.seed)

            order = torch.randperm(
                len(paths),
                generator=gen,
            )[: self.max_scenes].tolist()

            paths = [paths[i] for i in order]

        return paths

    def _read_scene(
        self,
        path: Union[Path, _TarParquet],
    ) -> pd.DataFrame:

        try:
            if isinstance(path, _TarParquet):
                if self._archive_file is None:
                    self._archive_file = open(
                        Path(self.root).expanduser(),
                        "rb",
                    )

                self._archive_file.seek(path.offset)
                raw = self._archive_file.read(path.size)

                if len(raw) != path.size:
                    raise IOError(
                        f"Could not read full parquet member {path.name!r}"
                    )

                df = pd.read_parquet(
                    io.BytesIO(raw),
                    engine="pyarrow",
                )

                label = path.name

            else:
                df = pd.read_parquet(path)
                label = str(path)

        except ImportError as exc:
            raise ImportError(
                "Install pyarrow to read AV2 parquet files"
            ) from exc

        missing = _REQUIRED_COLUMNS - set(df.columns)

        if missing:
            raise ValueError(
                f"{label}: missing columns {sorted(missing)}"
            )

        return df

    def _choose_anchor(
        self,
        df: pd.DataFrame,
        complete_ids: Sequence[str],
        last_history_step: int,
    ) -> str:
        if "focal_track_id" in df.columns and len(df):
            focal = str(df["focal_track_id"].iloc[0])
            if focal in complete_ids:
                return focal

        if "object_category" in df.columns:
            category = pd.to_numeric(df["object_category"], errors="coerce")
            focal_rows = df[category == 3]
            if len(focal_rows):
                focal = str(focal_rows.iloc[0]["track_id"])
                if focal in complete_ids:
                    return focal

        last = df[df["timestep"] == last_history_step]
        last = last[last["track_id"].astype(str).isin(complete_ids)]
        if len(last) == 0:
            return str(complete_ids[0])
        xy = torch.as_tensor(last[["position_x", "position_y"]].to_numpy(), dtype=torch.float32)
        ids = last["track_id"].astype(str).tolist()
        if len(ids) == 1:
            return ids[0]
        distances = torch.cdist(xy, xy)
        neighbors = ((distances < self.local_radius) & (distances > 0)).sum(dim=1)
        return ids[int(neighbors.argmax().item())]

    def _prepare_scene(self, df: pd.DataFrame) -> Optional[TensorDict]:
        df = df[df["object_type"].astype(str).str.lower() == "vehicle"].copy()
        if len(df) == 0:
            return None

        if "observed" in df.columns and df["observed"].astype(bool).any():
            observed_end = int(df.loc[df["observed"].astype(bool), "timestep"].max())
        else:
            observed_end = 49

        past_steps = list(range(observed_end - self.T_past + 1, observed_end + 1))
        future_steps = list(range(observed_end + 1, observed_end + self.T_future + 1))
        if past_steps[0] < 0:
            return None

        selected_steps = past_steps + future_steps
        use = df[df["timestep"].isin(selected_steps)].copy()
        counts = use.assign(track_id_str=use["track_id"].astype(str)).groupby("track_id_str")["timestep"].nunique()
        complete_ids = counts[counts == len(selected_steps)].index.tolist()
        if len(complete_ids) < 2:
            return None

        use["track_id_str"] = use["track_id"].astype(str)
        if "focal_track_id" in use.columns and len(use):
            focal = str(use["focal_track_id"].iloc[0])
            if focal not in complete_ids:
                return None
            anchor = focal
        else:
            anchor = self._choose_anchor(use, complete_ids, observed_end)
        last = use[use["timestep"] == observed_end].set_index("track_id_str")
        if anchor not in last.index:
            return None

        anchor_row = last.loc[anchor]
        if isinstance(anchor_row, pd.DataFrame):
            anchor_row = anchor_row.iloc[0]
        anchor_xy = torch.tensor(
            [float(anchor_row["position_x"]), float(anchor_row["position_y"])],
            dtype=torch.float32,
        )

        candidates = []
        for track_id in complete_ids:
            if track_id not in last.index:
                continue
            row = last.loc[track_id]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            xy = torch.tensor([float(row["position_x"]), float(row["position_y"])])
            distance = float(torch.linalg.norm(xy - anchor_xy).item())
            if track_id == anchor or distance <= self.local_radius:
                candidates.append((0.0 if track_id == anchor else distance, track_id))

        candidates.sort(key=lambda item: item[0])
        selected_ids = [anchor] + [
            track_id for _, track_id in candidates if track_id != anchor
        ][: self.n_agents - 1]
        if len(selected_ids) < 2:
            return None

        total_steps = self.T_past + self.T_future
        positions = torch.zeros(total_steps, self.n_agents, 2, dtype=torch.float32)
        velocities = torch.zeros_like(positions)
        valid_mask = torch.zeros(self.n_agents, dtype=torch.bool)
        step_to_index = {step: index for index, step in enumerate(selected_steps)}

        for agent_index, track_id in enumerate(selected_ids):
            track = use[use["track_id_str"] == track_id].sort_values("timestep")
            for row in track.itertuples(index=False):
                time_index = step_to_index.get(int(row.timestep))
                if time_index is None:
                    continue
                positions[time_index, agent_index] = torch.tensor(
                    [float(row.position_x), float(row.position_y)], dtype=torch.float32
                )
                velocities[time_index, agent_index] = torch.tensor(
                    [float(row.velocity_x), float(row.velocity_y)], dtype=torch.float32
                )
            valid_mask[agent_index] = True

        origin = positions[self.T_past - 1, 0].clone()
        anchor_velocity = velocities[self.T_past - 1, 0]
        if float(torch.linalg.norm(anchor_velocity).item()) > 0.2:
            anchor_heading = math.atan2(float(anchor_velocity[1]), float(anchor_velocity[0]))
        else:
            anchor_heading = float(anchor_row["heading"])

        rotation = _rotation(-anchor_heading)
        positions[:, valid_mask] = (positions[:, valid_mask] - origin) @ rotation.T
        velocities[:, valid_mask] = velocities[:, valid_mask] @ rotation.T

        n_valid = int(valid_mask.sum().item())
        future = positions[self.T_past :, :n_valid]
        pair_diff = future[:, :, None, :] - future[:, None, :, :]
        pair_distance = torch.linalg.norm(pair_diff, dim=-1)
        eye = torch.eye(n_valid, dtype=torch.bool)
        pair_distance = pair_distance.masked_fill(eye[None, :, :], float("inf"))
        min_distance = float(pair_distance.amin().item())
        interaction_heavy = min_distance < self.interaction_distance_threshold

        return {
            "past_positions": positions[: self.T_past],
            "past_velocities": velocities[: self.T_past],
            "future_positions": positions[self.T_past :],
            "valid_agent_mask": valid_mask,
            "interaction_heavy": torch.tensor(interaction_heavy, dtype=torch.bool),
            "safety_requirement": torch.tensor(self.safety_requirement, dtype=torch.float32),
        }

    def _get_scene(self, index: int) -> Optional[TensorDict]:
        index = int(index)
        if index in self._cache:
            scene = self._cache.pop(index)
            self._cache[index] = scene
            return scene

        scene = self._prepare_scene(self._read_scene(self._paths[index]))
        if self.cache_size > 0:
            self._cache[index] = scene
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        return scene

    @staticmethod
    def _stack(rows: List[TensorDict]) -> TensorDict:
        return {key: torch.stack([row[key] for row in rows]) for key in rows[0]}

    def generate_batch(self, batch_size: int) -> TensorDict:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        rows: List[TensorDict] = []
        attempts = 0
        max_attempts = max(batch_size * 100, 1000)
        while len(rows) < batch_size and attempts < max_attempts:
            index = int(torch.randint(len(self._paths), (1,), generator=self.generator).item())
            scene = self._get_scene(index)
            attempts += 1
            if scene is not None:
                rows.append(scene)

        if len(rows) < batch_size:
            raise RuntimeError(f"Could only build {len(rows)}/{batch_size} usable AV2 scenes")
        return self._stack(rows)

    def iter_batches(
        self,
        batch_size: int,
        *,
        shuffle: bool = False,
        seed: Optional[int] = None,
        max_batches: Optional[int] = None,
        drop_last: bool = False,
    ) -> Iterator[TensorDict]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        if shuffle:
            gen = torch.Generator(device="cpu")
            gen.manual_seed(self.seed if seed is None else int(seed))
            indices = torch.randperm(len(self._paths), generator=gen).tolist()
        else:
            indices = list(range(len(self._paths)))

        rows: List[TensorDict] = []
        produced = 0
        for index in indices:
            scene = self._get_scene(index)
            if scene is None:
                continue
            rows.append(scene)
            if len(rows) == batch_size:
                yield self._stack(rows)
                rows = []
                produced += 1
                if max_batches is not None and produced >= max_batches:
                    return

        if rows and not drop_last and (max_batches is None or produced < max_batches):
            yield self._stack(rows)


    def close(self) -> None:
        if self._archive_file is not None:
            self._archive_file.close()
            self._archive_file = None

    def __del__(self) -> None:
        handle = getattr(self, "_archive_file", None)

        if handle is not None:
            handle.close()