import numpy as np

from mathvse.contracts.recognition import SymbolPrediction
from mathvse.recognition.classifier import BaselineSymbolClassifier
from mathvse.spatial.parser import parse_symbols


def test_classifier_reports_missing_model():
    classifier = BaselineSymbolClassifier(model_path="/tmp/does_not_exist.pt")
    result = classifier.predict(np.zeros((32, 32), dtype=np.uint8), bbox=(0, 0, 32, 32))

    assert result.status == "model_unavailable"
    assert result.label == "unknown"


def test_classifier_predicts_when_model_file_exists(tmp_path):
    model_path = tmp_path / "mock_model.pt"
    model_path.write_text("placeholder weights")

    classifier = BaselineSymbolClassifier(
        model_path=str(model_path),
        label_map={"default": "x"},
        expected_input_shape=(1, 32, 32),
    )
    result = classifier.predict(np.ones((32, 32), dtype=np.uint8), bbox=(10, 10, 20, 20))

    assert result.status == "recognized"
    assert result.label == "x"
    assert result.confidence > 0.9


def test_spatial_parser_builds_baseline_expression():
    predictions = [
        SymbolPrediction(label="2", confidence=0.99, bbox=(0, 10, 10, 20), status="recognized"),
        SymbolPrediction(label="x", confidence=0.98, bbox=(15, 10, 10, 20), status="recognized"),
        SymbolPrediction(label="+", confidence=0.99, bbox=(30, 10, 10, 20), status="recognized"),
        SymbolPrediction(label="5", confidence=0.99, bbox=(45, 10, 10, 20), status="recognized"),
        SymbolPrediction(label="=", confidence=0.99, bbox=(60, 10, 10, 20), status="recognized"),
        SymbolPrediction(label="1", confidence=0.99, bbox=(75, 10, 10, 20), status="recognized"),
        SymbolPrediction(label="5", confidence=0.99, bbox=(90, 10, 10, 20), status="recognized"),
    ]

    result = parse_symbols(predictions)

    assert result.status == "success"
    assert "2*x+5=15" in result.normalized_text or "2*x + 5 = 15" in result.normalized_text


def test_spatial_parser_requires_review_for_low_confidence():
    predictions = [
        SymbolPrediction(label="x", confidence=0.1, bbox=(0, 20, 12, 18), status="recognized"),
        SymbolPrediction(label="=", confidence=0.9, bbox=(20, 20, 10, 18), status="recognized"),
        SymbolPrediction(label="1", confidence=0.9, bbox=(35, 20, 12, 18), status="recognized"),
    ]

    result = parse_symbols(predictions)
    assert result.status == "needs_review"


def test_spatial_parser_marks_overlapping_symbols_as_ambiguous():
    predictions = [
        SymbolPrediction(label="x", confidence=0.95, bbox=(0, 0, 20, 20), status="recognized"),
        SymbolPrediction(label="=", confidence=0.95, bbox=(5, 0, 20, 20), status="recognized"),
    ]

    result = parse_symbols(predictions)
    assert result.status == "needs_review"


def test_spatial_parser_rejects_superscript_structure():
    predictions = [
        SymbolPrediction(label="x", confidence=0.95, bbox=(5, 40, 12, 18), status="recognized"),
        SymbolPrediction(label="2", confidence=0.95, bbox=(15, 10, 8, 12), status="recognized"),
    ]

    result = parse_symbols(predictions)
    assert result.status == "unsupported_structure"


def test_spatial_parser_handles_unknown_symbol_status():
    predictions = [
        SymbolPrediction(label="unknown", confidence=0.0, bbox=(0, 0, 10, 10), status="unknown"),
    ]

    result = parse_symbols(predictions)
    assert result.status == "needs_review"
