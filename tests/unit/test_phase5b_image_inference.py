from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from mathvse.contracts.image import CandidateRegion, PreprocessingResult
from mathvse.contracts.recognition import SymbolCandidate, SymbolPrediction
from mathvse.pipeline.image_inference import extract_symbol_crops, infer_image, infer_image_file
from mathvse.recognition import training
from mathvse.recognition.classifier import (
    InvalidModelConfigError,
    ModelUnavailableError,
    SymbolClassifier,
    TorchSymbolClassifier,
)
from mathvse.recognition.training import (
    CHECKPOINT_VERSION,
    ID_TO_LABEL,
    LABEL_MAP_VERSION,
    PREPROCESSING_METADATA,
    SUPPORTED_LABELS,
    DatasetSample,
    _ImageDataset,
    prepare_classifier_input,
    validate_output_shape,
)
from mathvse.vision.preprocessing import preprocess_image


class FakeTestClassifier(SymbolClassifier):
    """Test double for pipeline wiring; it does not perform OCR or model inference."""

    def __init__(self, labels, confidence=0.99, alternatives=None):
        super().__init__(model_path="test-double")
        self.labels = labels
        self.confidence = confidence
        self.alternatives = alternatives
        self.calls = []

    @property
    def is_available(self):
        return True

    def predict(self, patch, *, bbox, label_map=None):
        index = len(self.calls)
        self.calls.append((patch.copy(), bbox))
        label = self.labels[index]
        alternatives = self.alternatives or (
            SymbolCandidate(label, self.confidence, bbox),
            SymbolCandidate("unknown", max(0.0, self.confidence - 0.5), bbox),
        )
        return SymbolPrediction(
            label=label,
            confidence=self.confidence,
            bbox=bbox,
            alternatives=alternatives,
            source="fake_test_double",
        )


def _equation_image():
    image = np.full((120, 180, 3), 255, dtype=np.uint8)
    image[42:72, 18:38] = 0
    image[42:72, 72:92] = 0
    image[42:72, 126:146] = 0
    return image


def test_synthetic_image_pipeline_uses_test_double_and_preserves_ids_and_boxes():
    classifier = FakeTestClassifier(["x", "=", "1"])

    result = infer_image(_equation_image(), classifier=classifier)

    assert result.status == "success"
    assert result.normalized_expression == "x=1"
    assert result.confirmed_expression is None
    assert result.solution is None
    assert [prediction.component_id for prediction in result.predictions] == [
        crop.component_id for crop in result.crops
    ]
    assert [prediction.bbox for prediction in result.predictions] == [
        crop.bbox for crop in result.crops
    ]
    assert [prediction.source for prediction in result.predictions] == ["fake_test_double"] * 3
    assert len(classifier.calls) == 3


def test_low_confidence_and_close_top_classes_require_review():
    image = _equation_image()
    low_confidence = infer_image(
        image,
        classifier=FakeTestClassifier(["x", "=", "1"], confidence=0.4),
        confidence_threshold=0.55,
    )
    ambiguous = infer_image(
        image,
        classifier=FakeTestClassifier(
            ["x", "=", "1"],
            confidence=0.8,
            alternatives=(
                SymbolCandidate("x", 0.8, (0, 0, 10, 10)),
                SymbolCandidate("y", 0.77, (0, 0, 10, 10)),
            ),
        ),
        confidence_threshold=0.5,
        ambiguity_margin=0.08,
    )

    assert low_confidence.status == "needs_review"
    assert low_confidence.normalized_expression == ""
    assert all(prediction.status == "needs_review" for prediction in low_confidence.predictions)
    assert ambiguous.status == "needs_review"
    assert ambiguous.normalized_expression == ""
    assert any("review required" in warning for warning in ambiguous.warnings)


def test_thin_stroke_component_cannot_be_accepted_as_complete_symbol():
    image = np.full((80, 140, 3), 255, dtype=np.uint8)
    image[36:40, 30:100] = 0

    result = infer_image(image, classifier=FakeTestClassifier(["-"]))

    assert result.status == "needs_review"
    assert result.normalized_expression == ""
    assert result.crops[0].status == "needs_review"


def test_wide_connected_component_requires_review():
    image = np.full((80, 140, 3), 255, dtype=np.uint8)
    image[25:55, 25:90] = 0

    result = infer_image(image, classifier=FakeTestClassifier(["x"]))

    assert result.status == "needs_review"
    assert result.crops[0].status == "needs_review"


def test_crop_extraction_preserves_coordinates_and_marks_elongated_components():
    gray = np.full((80, 160), 255, dtype=np.uint8)
    gray[35:40, 30:110] = 0
    preprocessing = PreprocessingResult(
        status="success",
        strategy="test",
        variants={"grayscale": gray},
        variant_metadata={},
        candidate_regions=[CandidateRegion("component_stroke", (30, 35, 80, 5), 400)],
        bounding_boxes=[(30, 35, 80, 5)],
    )

    crops, warnings = extract_symbol_crops(preprocessing)

    assert crops[0].component_id == "component_stroke"
    assert crops[0].bbox == (30, 35, 80, 5)
    assert crops[0].image.shape == (21, 96)
    assert crops[0].status == "needs_review"
    assert warnings


def test_crop_extraction_clamps_component_box_to_image_bounds():
    gray = np.full((40, 40), 255, dtype=np.uint8)
    preprocessing = PreprocessingResult(
        status="success",
        strategy="test",
        variants={"grayscale": gray},
        variant_metadata={},
        candidate_regions=[CandidateRegion("edge_component", (-3, 4, 12, 10), 60)],
        bounding_boxes=[(-3, 4, 12, 10)],
    )

    crops, _ = extract_symbol_crops(preprocessing)

    assert crops[0].bbox == (0, 4, 9, 10)
    assert crops[0].image.shape[0] <= gray.shape[0]
    assert crops[0].image.shape[1] <= gray.shape[1]
    assert crops[0].status == "needs_review"


def test_crop_extraction_enforces_component_count_bound():
    gray = np.full((80, 160), 255, dtype=np.uint8)
    regions = [CandidateRegion(f"component_{index:04d}", (index, 10, 2, 20), 40) for index in range(5)]
    preprocessing = PreprocessingResult(
        status="success",
        strategy="test",
        variants={"grayscale": gray},
        variant_metadata={},
        candidate_regions=regions,
        bounding_boxes=[region.bbox for region in regions],
    )

    crops, warnings = extract_symbol_crops(preprocessing, max_components=2)

    assert len(crops) == 2
    assert warnings[0] == "Component limit reached (2); remaining candidates require review."


def test_missing_weights_and_missing_torch_have_distinct_unavailable_status(tmp_path, monkeypatch):
    missing = infer_image(_equation_image(), classifier=TorchSymbolClassifier(model_path=str(tmp_path / "missing.pt")))
    assert missing.status == "model_unavailable"
    assert missing.predictions == []

    checkpoint = tmp_path / "present.pt"
    checkpoint.write_bytes(b"local placeholder")
    monkeypatch.setattr(training, "_torch_modules", lambda: (_ for _ in ()).throw(RuntimeError("PyTorch is optional")))
    pipeline_result = infer_image(
        _equation_image(), classifier=TorchSymbolClassifier(model_path=str(checkpoint))
    )
    assert pipeline_result.status == "model_unavailable"
    with pytest.raises(ModelUnavailableError, match="PyTorch is not installed"):
        TorchSymbolClassifier(model_path=str(checkpoint)).predict(np.zeros((32, 32), dtype=np.uint8), bbox=(0, 0, 32, 32))


def test_torch_classifier_rejects_unsupported_checkpoint_extension(tmp_path):
    checkpoint = tmp_path / "weights.onnx"
    checkpoint.write_bytes(b"not a torch checkpoint")

    with pytest.raises(InvalidModelConfigError, match=".pt or .pth"):
        TorchSymbolClassifier(model_path=str(checkpoint)).predict(
            np.zeros((32, 32), dtype=np.uint8), bbox=(0, 0, 32, 32)
        )


@pytest.mark.parametrize(
    "bad_metadata, message",
    [
        ({"label_mapping": {"0": "bad"}}, "label mapping"),
        ({"input_shape": [1, 28, 28]}, "dimensions"),
        ({"output_classes": 1}, "output class count"),
    ],
)
def test_malformed_or_incompatible_checkpoint_fails_explicitly(tmp_path, monkeypatch, bad_metadata, message):
    checkpoint_path = tmp_path / "broken.pt"
    checkpoint_path.write_bytes(b"checkpoint test data")
    metadata = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "architecture": "small_cnn_v1",
        "label_map_version": LABEL_MAP_VERSION,
        "label_mapping": ID_TO_LABEL,
        "input_shape": [1, 32, 32],
        "output_classes": len(SUPPORTED_LABELS),
        "preprocessing": PREPROCESSING_METADATA,
        **bad_metadata,
    }
    fake_torch = SimpleNamespace(load=lambda *args, **kwargs: metadata)
    monkeypatch.setattr(training, "_torch_modules", lambda: (fake_torch, object()))

    with pytest.raises(InvalidModelConfigError, match=message):
        TorchSymbolClassifier(model_path=str(checkpoint_path)).predict(
            np.zeros((32, 32), dtype=np.uint8), bbox=(0, 0, 32, 32)
        )


def test_checkpoint_loader_rejects_malformed_checkpoint(tmp_path, monkeypatch):
    checkpoint_path = tmp_path / "malformed.pt"
    checkpoint_path.write_bytes(b"malformed")
    monkeypatch.setattr(training, "_torch_modules", lambda: (SimpleNamespace(load=lambda *args, **kwargs: []), object()))

    with pytest.raises(InvalidModelConfigError, match="metadata dictionary"):
        TorchSymbolClassifier(model_path=str(checkpoint_path)).predict(
            np.zeros((32, 32), dtype=np.uint8), bbox=(0, 0, 32, 32)
        )

    failed = infer_image(_equation_image(), classifier=TorchSymbolClassifier(model_path=str(checkpoint_path)))
    assert failed.status == "recognition_failed"


def test_training_and_inference_share_grayscale_normalization(tmp_path):
    values = np.arange(32 * 32, dtype=np.uint8).reshape(32, 32)
    image_path = tmp_path / "sample.png"
    Image.fromarray(values).save(image_path)
    dataset = _ImageDataset([DatasetSample(image_path, "0", 0, "sample")])
    training_pixels, _ = dataset[0]
    inference_pixels = prepare_classifier_input(values)

    assert inference_pixels.shape == (1, 32, 32)
    assert inference_pixels.dtype == np.float32
    np.testing.assert_array_equal(training_pixels, inference_pixels)
    assert inference_pixels.min() == 0.0
    assert inference_pixels.max() == 1.0


def test_incompatible_logit_shape_is_rejected():
    with pytest.raises(training.CheckpointCompatibilityError, match="output shape"):
        validate_output_shape((1, 1), batch_size=1, class_count=len(SUPPORTED_LABELS))


def test_blank_image_is_unsupported_and_invalid_image_is_distinct():
    blank = np.full((80, 120, 3), 255, dtype=np.uint8)
    unsupported = infer_image(blank, classifier=FakeTestClassifier([]))
    invalid = infer_image(np.empty((0, 0, 3), dtype=np.uint8))

    assert unsupported.status == "unsupported_structure"
    assert invalid.status == "invalid_image"


def test_invalid_image_file_has_typed_status(tmp_path):
    result = infer_image_file(tmp_path / "missing.png")

    assert result.status == "invalid_image"
    assert result.loaded_image is None