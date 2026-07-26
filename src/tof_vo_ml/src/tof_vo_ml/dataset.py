from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


DEFAULT_MAX_DISTANCE_M = 4.0
DEFAULT_WINDOW_SIZE = 8
LABEL_NAMES = ("delta_x", "delta_y", "delta_theta")


@dataclass(frozen=True)
class DatasetInfo:
    npz_path: Path
    max_distance_m: float
    frame_count: int
    sample_count: int
    height: int
    width: int


class ToFWindowDataset(Dataset):
    """8-frame ToF windows with labels from the final frame transition."""

    def __init__(
        self,
        npz_path: str | Path,
        window_size: int = DEFAULT_WINDOW_SIZE,
        max_distance_m: float | None = None,
    ):
        if window_size < 2:
            raise ValueError("window_size must be at least 2")

        self.npz_path = Path(npz_path)
        self.window_size = window_size
        self.max_distance_m = resolve_max_distance_m(self.npz_path, max_distance_m)

        with np.load(self.npz_path) as data:
            missing = {"distance_m", "valid", "pose_xytheta"} - set(data.files)
            if missing:
                missing_names = ", ".join(sorted(missing))
                raise ValueError(f"{self.npz_path} is missing arrays: {missing_names}")

            distance_m = data["distance_m"].astype(np.float32)
            valid = data["valid"].astype(bool)
            pose_xytheta = data["pose_xytheta"].astype(np.float32)

        self._validate_arrays(distance_m, valid, pose_xytheta)

        normalized = (distance_m / self.max_distance_m).astype(np.float32)
        self.inputs = np.where(valid, normalized, 0.0).astype(np.float32)
        self.labels = make_final_transition_labels(pose_xytheta, window_size)
        self.info = DatasetInfo(
            npz_path=self.npz_path,
            max_distance_m=self.max_distance_m,
            frame_count=int(distance_m.shape[0]),
            sample_count=int(self.labels.shape[0]),
            height=int(distance_m.shape[1]),
            width=int(distance_m.shape[2]),
        )

    def _validate_arrays(
        self,
        distance_m: np.ndarray,
        valid: np.ndarray,
        pose_xytheta: np.ndarray,
    ) -> None:
        if distance_m.ndim != 3:
            raise ValueError(
                f"distance_m must have shape (frames, height, width); got {distance_m.shape}"
            )
        if valid.shape != distance_m.shape:
            raise ValueError(
                f"valid must have the same shape as distance_m; got {valid.shape}"
            )
        if pose_xytheta.ndim != 2 or pose_xytheta.shape[1] != 3:
            raise ValueError(
                f"pose_xytheta must have shape (frames, 3); got {pose_xytheta.shape}"
            )
        if pose_xytheta.shape[0] != distance_m.shape[0]:
            raise ValueError(
                "pose_xytheta and distance_m must contain the same number of frames"
            )
        if distance_m.shape[0] < self.window_size:
            raise ValueError(
                f"{self.npz_path} has {distance_m.shape[0]} frames, "
                f"but window_size is {self.window_size}"
            )
        if self.max_distance_m <= 0.0:
            raise ValueError("max_distance_m must be greater than 0.0")

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        end = index + self.window_size - 1
        start = end - self.window_size + 1
        window = self.inputs[start : end + 1]
        label = self.labels[index]
        return torch.from_numpy(window), torch.from_numpy(label)


def resolve_max_distance_m(
    npz_path: str | Path,
    override: float | None = None,
) -> float:
    if override is not None:
        return float(override)

    metadata_path = Path(npz_path).with_suffix(".json")
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if "max_distance_m" in metadata:
            return float(metadata["max_distance_m"])

    return DEFAULT_MAX_DISTANCE_M


def make_final_transition_labels(
    pose_xytheta: np.ndarray,
    window_size: int,
) -> np.ndarray:
    labels = []
    for end in range(window_size - 1, pose_xytheta.shape[0]):
        start_pose = pose_xytheta[end - 1]
        end_pose = pose_xytheta[end]
        labels.append(relative_pose_label(start_pose, end_pose))
    return np.asarray(labels, dtype=np.float32)


def relative_pose_label(start: Sequence[float], end: Sequence[float]) -> np.ndarray:
    dx_world = float(end[0] - start[0])
    dy_world = float(end[1] - start[1])
    start_theta = float(start[2])
    cos_theta = math.cos(start_theta)
    sin_theta = math.sin(start_theta)
    dx_body = cos_theta * dx_world + sin_theta * dy_world
    dy_body = -sin_theta * dx_world + cos_theta * dy_world
    dtheta = normalize_angle(float(end[2] - start[2]))
    return np.asarray([dx_body, dy_body, dtheta], dtype=np.float32)


def normalize_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi
