import csv

import numpy as np
import pytest
from PIL import Image

from mathvse.recognition.classifier import TorchSymbolClassifier
from mathvse.recognition.training import (
    CHECKPOINT_VERSION,
    ID_TO_LABEL,
    LABEL_MAP_VERSION,
    PREPROCESSING_METADATA,
    SUPPORTED_LABELS,
    CheckpointCompatibilityError,
    DatasetSample,
    load_hasy_dataset,
    split_samples,
    validate_checkpoint_metadata,
    validate_output_shape,
)


def _write_dataset(root, rows):
    (root / "hasy-data").mkdir(exist_ok=True)
    with (root / "symbols.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["symbol_id", "latex"])
        writer.writeheader()
        writer.writerows([{"symbol_id": 1, "latex": "0"}, {"symbol_id": 2, "latex": "\\cdot"}, {"symbol_id": 3, "latex": "\\alpha"}])
    with (root / "hasy-data.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "symbol_id"])
        writer.writeheader()
        writer.writerows(rows)


def test_hasy_loader_maps_subset_and_reports_invalid_or_unsupported_samples(tmp_path):
    (tmp_path / "hasy-data").mkdir()
    rows = []
    for index, symbol_id in enumerate([1, 2, 1, 3, 1]):
        path = f"hasy-data/{index}.png"
        if index == 2:
            (tmp_path / path).write_bytes(b"not an image")
        else:
            image = Image.new("L", (32, 32), index * 40)
            image.save(tmp_path / path)
        rows.append({"path": path, "symbol_id": symbol_id})
    _write_dataset(tmp_path, rows)

    report = load_hasy_dataset(tmp_path)

    assert {sample.label for sample in report.samples} == {"0", "*"}
    assert report.unsupported_labels == {"\\alpha": 1}
    assert len(report.invalid_samples) == 1
    assert report.class_counts == {"0": 2, "*": 1}


def test_split_is_seeded_and_duplicate_group_never_crosses_splits():
    samples = [
        DatasetSample(path=None, label="0", label_id=0, group_id=f"image-{index // 2}")
        for index in range(12)
    ]
    first = split_samples(samples, seed=91)
    second = split_samples(samples, seed=91)
    reordered = split_samples(list(reversed(samples)), seed=91)

    assert [[sample.group_id for sample in part] for part in first] == [[sample.group_id for sample in part] for part in second]
    assert [[sample.group_id for sample in part] for part in first] == [[sample.group_id for sample in part] for part in reordered]
    group_sets = [set(sample.group_id for sample in part) for part in first]
    assert not (group_sets[0] & group_sets[1] or group_sets[0] & group_sets[2] or group_sets[1] & group_sets[2])


def test_checkpoint_metadata_and_output_shape_contract():
    metadata = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "architecture": "small_cnn_v1",
        "label_map_version": LABEL_MAP_VERSION,
        "label_mapping": ID_TO_LABEL,
        "input_shape": [1, 32, 32],
        "output_classes": len(SUPPORTED_LABELS),
        "preprocessing": PREPROCESSING_METADATA,
    }

    validate_checkpoint_metadata(metadata)
    validate_output_shape((4, len(SUPPORTED_LABELS)), batch_size=4, class_count=len(SUPPORTED_LABELS))
    with pytest.raises(CheckpointCompatibilityError, match="label mapping"):
        validate_checkpoint_metadata({**metadata, "label_mapping": {"0": "x"}})
    with pytest.raises(CheckpointCompatibilityError, match="output shape"):
        validate_output_shape((4, 1), batch_size=4, class_count=len(SUPPORTED_LABELS))


def test_checkpoint_rejects_wrong_dimensions_or_normalization():
    metadata = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "architecture": "small_cnn_v1",
        "label_map_version": LABEL_MAP_VERSION,
        "label_mapping": ID_TO_LABEL,
        "input_shape": [1, 28, 28],
        "output_classes": len(SUPPORTED_LABELS),
        "preprocessing": PREPROCESSING_METADATA,
    }
    with pytest.raises(CheckpointCompatibilityError, match="dimensions"):
        validate_checkpoint_metadata(metadata)


def test_dataset_loader_reports_wrong_image_dimensions(tmp_path):
    (tmp_path / "hasy-data").mkdir()
    Image.new("L", (31, 32), 255).save(tmp_path / "hasy-data" / "small.png")
    _write_dataset(tmp_path, [{"path": "hasy-data/small.png", "symbol_id": 1}])

    report = load_hasy_dataset(tmp_path)

    assert not report.samples
    assert report.invalid_samples[0].reason == "expected 32x32, got 31x32"


def test_torch_classifier_keeps_missing_weights_unavailable(tmp_path):
    classifier = TorchSymbolClassifier(model_path=str(tmp_path / "missing.pt"))

    result = classifier.predict(
        np.full((32, 32), 255, dtype=np.uint8),
        bbox=(0, 0, 32, 32),
    )

    assert result.status == "model_unavailable"
    assert result.label == "unknown"