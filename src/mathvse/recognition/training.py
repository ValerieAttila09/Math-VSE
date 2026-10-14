from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


DEFAULT_MATH_LABELS = [
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "x",
    "y",
    "+",
    "-",
    "*",
    "/",
    "=",
    "(",
    ")",
]


@dataclass(frozen=True)
class TrainingConfig:
    dataset_root: str
    train_ratio: float = 0.7
    validation_ratio: float = 0.15
    test_ratio: float = 0.15
    seed: int = 42
    max_epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 1e-3
    device: str = "cpu"


def build_label_map() -> dict[str, str]:
    return {str(index): label for index, label in enumerate(DEFAULT_MATH_LABELS)}


def validate_dataset_root(dataset_root: str | Path) -> Path:
    root = Path(dataset_root)
    if not root.exists():
        raise FileNotFoundError(
            f"Dataset root does not exist: {root}. Acquire a local HASYv2-like dataset before training."
        )
    return root


def split_dataset_indices(total_items: int, *, train_ratio: float = 0.7, validation_ratio: float = 0.15, test_ratio: float = 0.15) -> tuple[int, int, int]:
    if abs(train_ratio + validation_ratio + test_ratio - 1.0) > 1e-9:
        raise ValueError("Dataset split ratios must sum to 1.0")
    train_count = int(total_items * train_ratio)
    validation_count = int(total_items * validation_ratio)
    test_count = total_items - train_count - validation_count
    return train_count, validation_count, test_count


def training_guide() -> str:
    return (
        "Offline training workflow: 1) acquire a local HASYv2 or similar mathematical-symbol dataset under a license you have reviewed; "
        "2) create a train/val/test split with no leakage; 3) register label mapping in mathvse.recognition.training.DEFAULT_MATH_LABELS; "
        "4) train on CPU with deterministic seeds; 5) save weights to a local path such as models/math_symbols.pt; 6) point the classifier to that file in code or config."
    )
