from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

import numpy as np
from PIL import Image, UnidentifiedImageError


LABEL_MAP_VERSION = "mathvse-symbols-v1"
CHECKPOINT_VERSION = 1
IMAGE_SIZE = (32, 32)
PREPROCESSING_METADATA = {
    "mode": "grayscale",
    "image_size": [32, 32],
    "normalization": "zero_one",
    "foreground": "as_stored",
}

# This subset maps only symbols already representable by the baseline spatial parser.
SUPPORTED_LABELS = (
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
)

LABEL_ALIASES = {
    "\\cdot": "*",
    "\\times": "*",
    "\\div": "/",
    "\\left(": "(",
    "\\right)": ")",
}
LABEL_TO_ID = {label: index for index, label in enumerate(SUPPORTED_LABELS)}
ID_TO_LABEL = {str(index): label for index, label in enumerate(SUPPORTED_LABELS)}
DEFAULT_MATH_LABELS = list(SUPPORTED_LABELS)


class DatasetFormatError(ValueError):
    """Raised when a local HASYv2-style dataset cannot be read."""


class CheckpointCompatibilityError(ValueError):
    """Raised when a checkpoint does not match the supported inference contract."""


@dataclass(frozen=True)
class DatasetSample:
    path: Path
    label: str
    label_id: int
    group_id: str


@dataclass(frozen=True)
class InvalidSample:
    path: str
    reason: str


@dataclass(frozen=True)
class DatasetReport:
    root: Path
    samples: tuple[DatasetSample, ...]
    class_counts: dict[str, int]
    unsupported_labels: dict[str, int]
    invalid_samples: tuple[InvalidSample, ...]
    duplicate_count: int
    source_rows: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "valid_samples": len(self.samples),
            "class_counts": self.class_counts,
            "unsupported_labels": self.unsupported_labels,
            "invalid_samples": [sample.__dict__ for sample in self.invalid_samples],
            "duplicate_count": self.duplicate_count,
            "source_rows": self.source_rows,
        }


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
    output_path: str = "models/math_symbols.pt"


def build_label_map() -> dict[str, str]:
    return dict(ID_TO_LABEL)


def validate_dataset_root(dataset_root: str | Path) -> Path:
    root = Path(dataset_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")
    for filename in ("hasy-data.csv", "symbols.csv"):
        if not (root / filename).is_file():
            raise DatasetFormatError(f"Expected {filename} in HASYv2 dataset root: {root}")
    return root


def _canonical_label(label: str) -> str | None:
    stripped = label.strip()
    canonical = LABEL_ALIASES.get(stripped, stripped)
    return canonical if canonical in LABEL_TO_ID else None


def _read_symbol_names(root: Path) -> dict[str, str]:
    names: dict[str, str] = {}
    with (root / "symbols.csv").open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "symbol_id" not in reader.fieldnames or "latex" not in reader.fieldnames:
            raise DatasetFormatError("symbols.csv must have symbol_id and latex columns")
        for row in reader:
            names[row["symbol_id"].strip()] = row["latex"].strip()
    return names


def load_hasy_dataset(dataset_root: str | Path, *, image_size: tuple[int, int] = IMAGE_SIZE) -> DatasetReport:
    """Validate and index a local HASYv2 directory; no dataset files are downloaded."""
    root = validate_dataset_root(dataset_root)
    symbol_names = _read_symbol_names(root)
    samples: list[DatasetSample] = []
    invalid: list[InvalidSample] = []
    unsupported: Counter[str] = Counter()
    counts: Counter[str] = Counter()
    seen_hashes: set[str] = set()
    duplicate_count = 0
    source_rows = 0

    with (root / "hasy-data.csv").open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "path" not in reader.fieldnames:
            raise DatasetFormatError("hasy-data.csv must have a path column")
        if "symbol_id" not in reader.fieldnames and "latex" not in reader.fieldnames:
            raise DatasetFormatError("hasy-data.csv must have symbol_id or latex column")
        for row_number, row in enumerate(reader, start=2):
            source_rows += 1
            relative_path = row.get("path", "").strip()
            raw_label = row.get("latex", "").strip()
            if not raw_label:
                raw_label = symbol_names.get(row.get("symbol_id", "").strip(), "")
            label = _canonical_label(raw_label)
            if label is None:
                unsupported[raw_label or "<missing>"] += 1

            posix_path = PurePosixPath(relative_path)
            if not relative_path or posix_path.is_absolute() or ".." in posix_path.parts:
                invalid.append(InvalidSample(relative_path or f"row:{row_number}", "invalid relative path"))
                continue
            sample_path = (root / Path(*posix_path.parts)).resolve()
            if not sample_path.is_relative_to(root):
                invalid.append(InvalidSample(relative_path, "path escapes dataset root"))
                continue
            try:
                with Image.open(sample_path) as image:
                    if image.size != image_size:
                        raise ValueError(f"expected {image_size[0]}x{image_size[1]}, got {image.width}x{image.height}")
                    grayscale = image.convert("L")
                    grayscale.load()
                    digest = hashlib.sha256(grayscale.tobytes()).hexdigest()
            except (OSError, ValueError, UnidentifiedImageError) as exc:
                invalid.append(InvalidSample(relative_path, str(exc)))
                continue

            if label is None:
                continue
            if digest in seen_hashes:
                duplicate_count += 1
                continue
            seen_hashes.add(digest)
            samples.append(DatasetSample(sample_path, label, LABEL_TO_ID[label], digest))
            counts[label] += 1

    return DatasetReport(
        root=root,
        samples=tuple(samples),
        class_counts={label: counts[label] for label in SUPPORTED_LABELS if counts[label]},
        unsupported_labels=dict(sorted(unsupported.items())),
        invalid_samples=tuple(invalid),
        duplicate_count=duplicate_count,
        source_rows=source_rows,
    )


def _validate_ratios(train_ratio: float, validation_ratio: float, test_ratio: float) -> None:
    ratios = (train_ratio, validation_ratio, test_ratio)
    if any(ratio < 0 or not math.isfinite(ratio) for ratio in ratios) or abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError("Dataset split ratios must be finite, non-negative, and sum to 1.0")
    if any(ratio == 0 for ratio in ratios):
        raise ValueError("Training, validation, and test ratios must all be greater than zero")


def split_dataset_indices(
    total_items: int,
    *,
    train_ratio: float = 0.7,
    validation_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> tuple[int, int, int]:
    _validate_ratios(train_ratio, validation_ratio, test_ratio)
    train_count = int(total_items * train_ratio)
    validation_count = int(total_items * validation_ratio)
    return train_count, validation_count, total_items - train_count - validation_count


def split_samples(
    samples: Iterable[DatasetSample],
    *,
    train_ratio: float = 0.7,
    validation_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[list[DatasetSample], list[DatasetSample], list[DatasetSample]]:
    """Split per class with a fixed seed; each duplicate-image group stays in one split."""
    _validate_ratios(train_ratio, validation_ratio, test_ratio)
    by_label: dict[int, dict[str, DatasetSample]] = defaultdict(dict)
    unique_groups: dict[str, DatasetSample] = {}
    for sample in sorted(samples, key=lambda item: (item.group_id, item.label_id, str(item.path))):
        unique_groups.setdefault(sample.group_id, sample)
    for sample in unique_groups.values():
        by_label[sample.label_id].setdefault(sample.group_id, sample)

    rng = random.Random(seed)
    result: tuple[list[DatasetSample], list[DatasetSample], list[DatasetSample]] = ([], [], [])
    for label_id in sorted(by_label):
        groups = sorted(by_label[label_id].values(), key=lambda sample: sample.group_id)
        rng.shuffle(groups)
        sizes = list(split_dataset_indices(len(groups), train_ratio=train_ratio, validation_ratio=validation_ratio, test_ratio=test_ratio))
        if len(groups) >= 3:
            for index, size in enumerate(sizes):
                if size == 0:
                    donor = max(range(3), key=lambda part: sizes[part])
                    sizes[donor] -= 1
                    sizes[index] = 1
        offset = 0
        for part, size in zip(result, sizes):
            part.extend(groups[offset : offset + size])
            offset += size
    for part in result:
        part.sort(key=lambda sample: (sample.label_id, sample.group_id))
    return result


def validate_checkpoint_metadata(metadata: dict[str, Any]) -> None:
    if metadata.get("checkpoint_version") != CHECKPOINT_VERSION:
        raise CheckpointCompatibilityError("Unsupported or missing checkpoint version")
    if metadata.get("label_map_version") != LABEL_MAP_VERSION or metadata.get("label_mapping") != ID_TO_LABEL:
        raise CheckpointCompatibilityError("Checkpoint label mapping is incompatible with this MathVSE version")
    if metadata.get("input_shape") != [1, *IMAGE_SIZE]:
        raise CheckpointCompatibilityError("Checkpoint input dimensions must be [1, 32, 32]")
    if metadata.get("preprocessing") != PREPROCESSING_METADATA:
        raise CheckpointCompatibilityError("Checkpoint preprocessing metadata is incompatible")
    if metadata.get("output_classes") != len(SUPPORTED_LABELS):
        raise CheckpointCompatibilityError("Checkpoint output class count does not match the label mapping")
    if metadata.get("architecture") != "small_cnn_v1":
        raise CheckpointCompatibilityError("Checkpoint architecture is unsupported")


def validate_output_shape(shape: tuple[int, ...], *, batch_size: int, class_count: int) -> None:
    if tuple(shape) != (batch_size, class_count):
        raise CheckpointCompatibilityError(
            f"CNN output shape must be {(batch_size, class_count)}, got {tuple(shape)}"
        )


def prepare_classifier_input(image: np.ndarray) -> np.ndarray:
    """Return one as-stored grayscale sample in the CNN's (1, 32, 32) [0, 1] contract."""
    if image is None or np.asarray(image).size == 0:
        raise ValueError("Classifier image patch is empty")
    pixels = np.asarray(image)
    if pixels.ndim == 2:
        gray = pixels
    elif pixels.ndim == 3 and pixels.shape[2] in {1, 3, 4}:
        if pixels.dtype != np.uint8:
            pixels = np.clip(pixels, 0, 255).astype(np.uint8)
        gray = np.asarray(Image.fromarray(pixels).convert("L"))
    else:
        raise ValueError(f"Unsupported classifier image shape: {pixels.shape}")
    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)
    if gray.shape != IMAGE_SIZE:
        gray = np.asarray(Image.fromarray(gray).resize(IMAGE_SIZE, Image.Resampling.LANCZOS))
    return (gray.astype(np.float32) / 255.0)[None, :, :]


def _torch_modules():
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise RuntimeError("PyTorch is optional. Install it with: python -m pip install -e '.[training]'") from exc
    return torch, nn


def create_cnn(class_count: int | None = None):
    torch, nn = _torch_modules()
    classes = class_count or len(SUPPORTED_LABELS)

    class SmallSymbolCNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(32 * 8 * 8, 64),
                nn.ReLU(),
                nn.Linear(64, classes),
            )

        def forward(self, inputs):
            return self.classifier(self.features(inputs))

    return SmallSymbolCNN()


class _ImageDataset:
    def __init__(self, samples: list[DatasetSample]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        with Image.open(sample.path) as image:
            pixels = np.asarray(image.convert("L"))
        return prepare_classifier_input(pixels), sample.label_id


def _evaluate(model, samples: list[DatasetSample], *, batch_size: int) -> dict[str, Any]:
    torch, _ = _torch_modules()
    from torch.utils.data import DataLoader

    matrix = np.zeros((len(SUPPORTED_LABELS), len(SUPPORTED_LABELS)), dtype=np.int64)
    loader = DataLoader(_ImageDataset(samples), batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()
    with torch.inference_mode():
        for images, labels in loader:
            logits = model(images)
            validate_output_shape(tuple(logits.shape), batch_size=len(labels), class_count=len(SUPPORTED_LABELS))
            predictions = logits.argmax(dim=1)
            for expected, predicted in zip(labels.tolist(), predictions.tolist()):
                matrix[expected, predicted] += 1

    per_class: dict[str, dict[str, float | int]] = {}
    for class_id, label in enumerate(SUPPORTED_LABELS):
        true_positive = int(matrix[class_id, class_id])
        support = int(matrix[class_id, :].sum())
        predicted_count = int(matrix[:, class_id].sum())
        per_class[label] = {
            "precision": true_positive / predicted_count if predicted_count else 0.0,
            "recall": true_positive / support if support else 0.0,
            "count": support,
        }
    total = int(matrix.sum())
    return {
        "accuracy": float(np.trace(matrix) / total) if total else 0.0,
        "sample_count": total,
        "class_counts": {label: values["count"] for label, values in per_class.items()},
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
    }


def train_model(config: TrainingConfig) -> dict[str, Any]:
    torch, nn = _torch_modules()
    if config.device != "cpu":
        raise ValueError("Phase 5A training currently supports CPU only")
    if config.max_epochs < 1 or config.batch_size < 1 or config.learning_rate <= 0:
        raise ValueError("epochs, batch size, and learning rate must be positive")
    report = load_hasy_dataset(config.dataset_root)
    train_samples, validation_samples, test_samples = split_samples(
        report.samples,
        train_ratio=config.train_ratio,
        validation_ratio=config.validation_ratio,
        test_ratio=config.test_ratio,
        seed=config.seed,
    )
    if not train_samples or not validation_samples or not test_samples:
        raise DatasetFormatError("Dataset is too small for non-empty training, validation, and test splits")

    torch.manual_seed(config.seed)
    rng = torch.Generator().manual_seed(config.seed)
    model = create_cnn()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()
    from torch.utils.data import DataLoader

    train_loader = DataLoader(_ImageDataset(train_samples), batch_size=config.batch_size, shuffle=True, generator=rng, num_workers=0)
    epoch_metrics: list[dict[str, float]] = []
    for epoch in range(config.max_epochs):
        model.train()
        loss_total = 0.0
        correct = 0
        seen = 0
        for images, labels in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            validate_output_shape(tuple(logits.shape), batch_size=len(labels), class_count=len(SUPPORTED_LABELS))
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            loss_total += float(loss.item()) * len(labels)
            correct += int((logits.argmax(dim=1) == labels).sum().item())
            seen += len(labels)
        validation_metrics = _evaluate(model, validation_samples, batch_size=config.batch_size)
        epoch_metrics.append({
            "epoch": epoch + 1,
            "training_loss": loss_total / seen,
            "training_accuracy": correct / seen,
            "validation_accuracy": validation_metrics["accuracy"],
        })

    test_metrics = _evaluate(model, test_samples, batch_size=config.batch_size)
    checkpoint = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "architecture": "small_cnn_v1",
        "label_map_version": LABEL_MAP_VERSION,
        "label_mapping": ID_TO_LABEL,
        "input_shape": [1, *IMAGE_SIZE],
        "output_classes": len(SUPPORTED_LABELS),
        "preprocessing": PREPROCESSING_METADATA,
        "config": {
            "seed": config.seed,
            "max_epochs": config.max_epochs,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "train_ratio": config.train_ratio,
            "validation_ratio": config.validation_ratio,
            "test_ratio": config.test_ratio,
        },
        "metrics": {
            "epochs": epoch_metrics,
            "held_out_test": test_metrics,
            "split_counts": {
                "training": len(train_samples),
                "validation": len(validation_samples),
                "test": len(test_samples),
            },
        },
        "state_dict": model.state_dict(),
    }
    validate_checkpoint_metadata(checkpoint)
    output = Path(config.output_path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)
    return {"checkpoint": str(output), "dataset": report.as_dict(), "metrics": checkpoint["metrics"]}


def load_checkpoint(checkpoint_path: str | Path) -> dict[str, Any]:
    torch, _ = _torch_modules()
    checkpoint = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise CheckpointCompatibilityError("Checkpoint must contain a metadata dictionary")
    validate_checkpoint_metadata(checkpoint)
    if "state_dict" not in checkpoint:
        raise CheckpointCompatibilityError("Checkpoint has no CNN weights")
    return checkpoint


def evaluate_checkpoint(checkpoint_path: str | Path, dataset_root: str | Path) -> dict[str, Any]:
    checkpoint = load_checkpoint(checkpoint_path)
    config = checkpoint["config"]
    report = load_hasy_dataset(dataset_root)
    _, _, test_samples = split_samples(
        report.samples,
        train_ratio=config["train_ratio"],
        validation_ratio=config["validation_ratio"],
        test_ratio=config["test_ratio"],
        seed=config["seed"],
    )
    if not test_samples:
        raise DatasetFormatError("Dataset has no samples in the held-out test split")
    model = create_cnn()
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return {"dataset": report.as_dict(), "held_out_test": _evaluate(model, test_samples, batch_size=config["batch_size"])}


def inspect_checkpoint(checkpoint_path: str | Path) -> dict[str, Any]:
    checkpoint = load_checkpoint(checkpoint_path)
    return {key: value for key, value in checkpoint.items() if key != "state_dict"}


def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mathvse-recognition", description="Local mathematical-symbol CNN dataset and checkpoint tools")
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate-dataset", help="validate a local HASYv2 dataset")
    validate.add_argument("dataset_root")
    train = commands.add_parser("train", help="train the local CPU CNN")
    train.add_argument("dataset_root")
    train.add_argument("--output", default="models/math_symbols.pt")
    train.add_argument("--epochs", type=int, default=20)
    train.add_argument("--batch-size", type=int, default=32)
    train.add_argument("--learning-rate", type=float, default=1e-3)
    train.add_argument("--seed", type=int, default=42)
    evaluate = commands.add_parser("evaluate", help="evaluate checkpoint on the deterministic held-out split")
    evaluate.add_argument("checkpoint")
    evaluate.add_argument("dataset_root")
    inspect = commands.add_parser("inspect", help="inspect a compatible checkpoint")
    inspect.add_argument("checkpoint")
    infer = commands.add_parser("infer-image", help="run local image-to-expression inference")
    infer.add_argument("image_path")
    infer.add_argument("--checkpoint")
    infer.add_argument("--confidence-threshold", type=float, default=0.55)
    infer.add_argument("--ambiguity-margin", type=float, default=0.08)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_cli().parse_args(argv)
    if args.command == "validate-dataset":
        result = load_hasy_dataset(args.dataset_root).as_dict()
    elif args.command == "train":
        result = train_model(TrainingConfig(
            dataset_root=args.dataset_root,
            max_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            seed=args.seed,
            output_path=args.output,
        ))
    elif args.command == "evaluate":
        result = evaluate_checkpoint(args.checkpoint, args.dataset_root)
    elif args.command == "inspect":
        result = inspect_checkpoint(args.checkpoint)
    else:
        from mathvse.pipeline.image_inference import infer_image_file

        result = infer_image_file(
            args.image_path,
            model_path=args.checkpoint,
            confidence_threshold=args.confidence_threshold,
            ambiguity_margin=args.ambiguity_margin,
        ).as_dict()
    print(json.dumps(result, ensure_ascii=True, indent=2))
    return 0


def training_guide() -> str:
    return (
        "Use a locally acquired HASYv2 dataset after reviewing its license and terms. "
        "Run mathvse-recognition validate-dataset DATASET_ROOT, then train with a fixed seed; "
        "the versioned checkpoint stores the mapping, preprocessing contract, configuration, and held-out test metrics."
    )


if __name__ == "__main__":
    raise SystemExit(main())
