from pathlib import Path

import pytest
from PIL import Image

from mathvse.exceptions import ImageValidationError
from mathvse.ui.app import evaluate_expression, evaluate_uploaded_image


def test_manual_expression_routes_to_solver():
    result = evaluate_expression("3*x - 6 = 12")
    assert result.status == "success"
    assert result.solution == "x = 6"


def test_manual_expression_invalid_input_is_reported():
    result = evaluate_expression("2 *")
    assert result.status == "failed"


def test_uploaded_image_requires_real_file(tmp_path):
    missing = tmp_path / "missing.png"
    with pytest.raises(ImageValidationError):
        evaluate_uploaded_image(missing)


def test_missing_ocr_model_is_explicitly_unavailable(tmp_path):
    image_path = tmp_path / "sample.png"
    image = Image.new("RGB", (80, 60), "white")
    image.save(image_path)

    payload = evaluate_uploaded_image(image_path, model_path="/tmp/does_not_exist.pt")
    assert payload["recognition"].model_status == "model_unavailable"
