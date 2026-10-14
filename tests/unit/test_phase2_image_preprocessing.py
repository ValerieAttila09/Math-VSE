from __future__ import annotations

import io

import numpy as np
from PIL import Image

from mathvse.vision.input import ImageValidationError, load_image_bytes, load_image_file
from mathvse.vision.preprocessing import preprocess_image


def _make_test_image(size=(240, 180), color=(255, 255, 255), background_color=(0, 0, 0)):
    image = Image.new("RGB", size, color)
    for x in range(60, 180):
        for y in range(50, 140):
            image.putpixel((x, y), background_color)
    return image


def test_load_valid_png_image(tmp_path):
    image = _make_test_image()
    path = tmp_path / "equation.png"
    image.save(path)

    result = load_image_file(path)

    assert result.metadata.width > 0
    assert result.metadata.height > 0
    assert result.image.shape[2] == 3
    assert result.metadata.file_format in {"PNG", "png"}


def test_load_invalid_empty_file_raises(tmp_path):
    path = tmp_path / "empty.png"
    path.write_bytes(b"")

    try:
        load_image_file(path)
        assert False, "Expected ImageValidationError for empty file"
    except ImageValidationError:
        pass


def test_load_corrupted_image_raises(tmp_path):
    path = tmp_path / "bad.png"
    path.write_bytes(b"not-a-real-image")

    try:
        load_image_file(path)
        assert False, "Expected ImageValidationError for corrupted file"
    except ImageValidationError:
        pass


def test_size_and_dimension_limits(tmp_path):
    image = Image.new("RGB", (128, 128), "white")
    path = tmp_path / "large.png"
    image.save(path)

    try:
        load_image_file(path, max_bytes=1, max_width=64, max_height=64)
        assert False, "Expected validation error for oversized image"
    except ImageValidationError:
        pass


def test_load_image_bytes_supports_grayscale():
    image = Image.new("L", (120, 80), 200)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    result = load_image_bytes(buffer.getvalue(), filename="gray.png")

    assert result.image.dtype == np.uint8
    assert result.image.ndim == 3
    assert result.image.shape[0] == 80
    assert result.image.shape[1] == 120


def test_preprocessing_variants_for_synthetic_equation():
    image = _make_test_image()
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    source = load_image_bytes(buffer.getvalue(), filename="equation.png")

    result = preprocess_image(source.image, strategy_name="balanced")

    assert result.status == "success"
    assert set(result.variants.keys()) >= {"grayscale", "otsu", "adaptive"}
    assert result.variants["grayscale"].shape[0] > 0
    assert result.candidate_regions
    assert result.bounding_boxes


def test_preprocessing_handles_empty_input():
    empty = np.zeros((0, 0, 3), dtype=np.uint8)
    try:
        preprocess_image(empty)
        assert False, "Expected validation error for empty input"
    except ValueError:
        pass
