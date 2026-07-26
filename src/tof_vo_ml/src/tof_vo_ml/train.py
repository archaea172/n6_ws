from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Sequence

import torch
from torch import nn
from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import Subset
from torch.utils.data import random_split

from .dataset import DEFAULT_MAX_DISTANCE_M
from .dataset import DEFAULT_WINDOW_SIZE
from .dataset import LABEL_NAMES
from .dataset import ToFWindowDataset
from .model import ToFOdometryCNN


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a ToF odometry CNN from 8-frame distance windows. "
            "Each label is the final transition in the window."
        ),
    )
    parser.add_argument(
        "datasets",
        nargs="+",
        type=Path,
        help="One or more .npz datasets exported by tof_dataset.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs. Default: 100.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Mini-batch size. Default: 32.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Adam learning rate. Default: 1e-3.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
        help="AdamW weight decay. Default: 0.0.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/tof_vo_model.pt"),
        help="Path where the .pt checkpoint is written. Default: runs/tof_vo_model.pt.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=DEFAULT_WINDOW_SIZE,
        help=f"Number of consecutive ToF frames per sample. Default: {DEFAULT_WINDOW_SIZE}.",
    )
    parser.add_argument(
        "--max-distance",
        type=float,
        default=None,
        help=(
            "Distance normalization scale in meters. If omitted, each dataset's "
            f"same-name .json metadata is used, then {DEFAULT_MAX_DISTANCE_M}."
        ),
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.2,
        help="Fraction of samples used for validation. Default: 0.2.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/validation split. Default: 42.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader worker processes. Default: 0.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Training device. Default: auto.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    validate_args(args)

    device = select_device(args.device)
    dataset = load_datasets(args.datasets, args.window_size, args.max_distance)
    train_dataset, val_dataset = split_dataset(dataset, args.val_split, args.seed)

    sample_input, sample_label = dataset[0]
    model = ToFOdometryCNN(
        input_channels=int(sample_input.shape[0]),
        output_dim=int(sample_label.shape[0]),
    ).to(device)

    target_mean, target_std = compute_target_stats(train_dataset)
    target_mean = target_mean.to(device)
    target_std = target_std.to(device)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = None
    if len(val_dataset) > 0:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    criterion = nn.MSELoss()

    history = []
    best_val_loss = None
    best_state_dict = None
    best_epoch = 0

    print(f"device: {device}")
    print(f"samples: train={len(train_dataset)} val={len(val_dataset)}")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_raw_mse = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            target_mean,
            target_std,
            optimizer,
        )
        val_loss = None
        val_raw_mse = None
        if val_loader is not None:
            val_loss, val_raw_mse = evaluate(
                model,
                val_loader,
                criterion,
                device,
                target_mean,
                target_std,
            )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_raw_mse": train_raw_mse,
                "val_loss": val_loss,
                "val_raw_mse": val_raw_mse,
            }
        )

        if val_loss is not None:
            if best_val_loss is None or val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state_dict = deepcopy(model.state_dict())
                best_epoch = epoch
            print(
                f"epoch {epoch:04d} "
                f"train_loss={train_loss:.6f} train_raw_mse={train_raw_mse:.6f} "
                f"val_loss={val_loss:.6f} val_raw_mse={val_raw_mse:.6f}"
            )
        else:
            print(
                f"epoch {epoch:04d} "
                f"train_loss={train_loss:.6f} train_raw_mse={train_raw_mse:.6f}"
            )

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
    else:
        best_epoch = args.epochs

    save_checkpoint(
        args.output,
        model,
        optimizer,
        args,
        dataset,
        target_mean.cpu(),
        target_std.cpu(),
        history,
        best_epoch,
    )
    print(f"saved: {args.output}")
    return 0


def validate_args(args: argparse.Namespace) -> None:
    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if args.window_size < 2:
        raise ValueError("--window-size must be at least 2")
    if not 0.0 <= args.val_split < 1.0:
        raise ValueError("--val-split must be in the range [0.0, 1.0)")
    if args.max_distance is not None and args.max_distance <= 0.0:
        raise ValueError("--max-distance must be greater than 0.0")


def select_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but it is not available")
    return torch.device(device_name)


def load_datasets(
    paths: Sequence[Path],
    window_size: int,
    max_distance_m: float | None,
) -> Dataset:
    datasets = [
        ToFWindowDataset(path, window_size=window_size, max_distance_m=max_distance_m)
        for path in paths
    ]
    max_distances = {round(item.info.max_distance_m, 9) for item in datasets}
    if len(max_distances) != 1:
        raise ValueError(
            "datasets use different max_distance_m values; pass --max-distance "
            "to force one normalization scale"
        )
    if len(datasets) == 1:
        return datasets[0]
    return ConcatDataset(datasets)


def split_dataset(
    dataset: Dataset,
    val_split: float,
    seed: int,
) -> tuple[Dataset, Dataset]:
    val_count = int(round(len(dataset) * val_split))
    if val_count >= len(dataset):
        val_count = len(dataset) - 1
    train_count = len(dataset) - val_count
    if train_count <= 0:
        raise ValueError("not enough samples for the requested validation split")
    if val_count == 0:
        return dataset, Subset(dataset, [])

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(
        dataset,
        [train_count, val_count],
        generator=generator,
    )
    return train_dataset, val_dataset


def compute_target_stats(dataset: Dataset) -> tuple[torch.Tensor, torch.Tensor]:
    labels = torch.stack([dataset[index][1] for index in range(len(dataset))])
    mean = labels.mean(dim=0)
    std = labels.std(dim=0, unbiased=False)
    std = torch.clamp(std, min=1e-6)
    return mean, std


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
    optimizer: torch.optim.Optimizer,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_raw_mse = 0.0
    total_samples = 0

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        normalized_targets = (targets - target_mean) / target_std

        optimizer.zero_grad(set_to_none=True)
        predictions = model(inputs)
        loss = criterion(predictions, normalized_targets)
        loss.backward()
        optimizer.step()

        batch_size = int(inputs.shape[0])
        total_loss += float(loss.item()) * batch_size
        total_raw_mse += raw_mse(predictions, targets, target_mean, target_std) * batch_size
        total_samples += batch_size

    return total_loss / total_samples, total_raw_mse / total_samples


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_raw_mse = 0.0
    total_samples = 0

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        normalized_targets = (targets - target_mean) / target_std
        predictions = model(inputs)
        loss = criterion(predictions, normalized_targets)

        batch_size = int(inputs.shape[0])
        total_loss += float(loss.item()) * batch_size
        total_raw_mse += raw_mse(predictions, targets, target_mean, target_std) * batch_size
        total_samples += batch_size

    return total_loss / total_samples, total_raw_mse / total_samples


def raw_mse(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
) -> float:
    raw_predictions = predictions * target_std + target_mean
    return float(torch.mean((raw_predictions - targets) ** 2).item())


def save_checkpoint(
    output_path: Path,
    model: ToFOdometryCNN,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    dataset: Dataset,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
    history: list[dict[str, float | int | None]],
    best_epoch: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sample_input, _ = dataset[0]
    max_distance_m = get_dataset_max_distance_m(dataset)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "model_config": {
            "input_channels": int(sample_input.shape[0]),
            "output_dim": len(LABEL_NAMES),
            "window_size": args.window_size,
        },
        "target_mean": target_mean,
        "target_std": target_std,
        "max_distance_m": max_distance_m,
        "label_names": list(LABEL_NAMES),
        "history": history,
        "best_epoch": best_epoch,
    }
    torch.save(checkpoint, output_path)


def get_dataset_max_distance_m(dataset: Dataset) -> float:
    if isinstance(dataset, ToFWindowDataset):
        return dataset.info.max_distance_m
    if isinstance(dataset, ConcatDataset):
        first = dataset.datasets[0]
        if isinstance(first, ToFWindowDataset):
            return first.info.max_distance_m
    raise TypeError("unsupported dataset type")


if __name__ == "__main__":
    raise SystemExit(main())
