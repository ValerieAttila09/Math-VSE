from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

from mathvse.contracts.image import ImageMetadata, LoadedImage
from mathvse.exceptions import (
    EmptyImageError,
    ImageTooLargeError,
    ImageValidationError,
    InvalidImageDataError,
    UnsupportedImageFormatError,
)

SUPPORTED_FORMATS = {"PNG", "JPEG", "JPG"}


def _coerce_image_to_bgr(pil_image: Image.Image) -> np.ndarray:
    if pil_image.mode in {"1", "L", "I;16"}:
        gray = np.asarray(pil_image.convert("L"), dtype=np.uint8)
        return cvt_gray_to_bgr(gray)
    rgb = np.asarray(pil_image.convert("RGB"), dtype=np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def cvt_gray_to_bgr(gray: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(gray.astype(np.uint8), cv2.COLOR_GRAY2BGR)


def load_image_file(
    path: str | Path,
    *,
    max_bytes: int = 25 * 1024 * 1024,
    max_width: int = 5000,
    max_height: int = 5000,
) -> LoadedImage:
    source = Path(path)
    if not source.exists() or not source.is_file():
        raise ImageValidationError(f"Image file not found: {source}")

    file_size = source.stat().st_size
    if file_size == 0:
        raise EmptyImageError(f"Image file is empty: {source}")
    if file_size > max_bytes:
        raise ImageTooLargeError(f"Image exceeds file-size limit: {file_size} > {max_bytes} bytes")

    try:
        data = source.read_bytes()
        return load_image_bytes(data, filename=source.name, max_bytes=max_bytes, max_width=max_width, max_height=max_height)
    except OSError as exc:
        raise InvalidImageDataError(f"Unable to read image file: {source}") from exc


def load_image_bytes(
    payload: bytes,
    *,
    filename: str = "upload",
    max_bytes: int = 25 * 1024 * 1024,
    max_width: int = 5000,
    max_height: int = 5000,
) -> LoadedImage:
    if not payload:
        raise EmptyImageError(f"Image payload is empty: {filename}")
    if len(payload) > max_bytes:
        raise ImageTooLargeError(f"Image exceeds file-size limit: {len(payload)} > {max_bytes} bytes")

    try:
        image = Image.open(io.BytesIO(payload))
        image.verify()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise InvalidImageDataError(f"Image payload is corrupted or unreadable: {filename}") from exc

    try:
        image = Image.open(io.BytesIO(payload))
        image.load()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise InvalidImageDataError(f"Image payload could not be decoded: {filename}") from exc

    image_format = (image.format or "").upper()
    if image_format not in SUPPORTED_FORMATS:
        raise UnsupportedImageFormatError(f"Unsupported image format: {image_format or 'unknown'}")

    width, height = image.size
    if width <= 0 or height <= 0:
        raise InvalidImageDataError(f"Decoded image dimensions are invalid: {width}x{height}")
    if width > max_width or height > max_height:
        raise ImageTooLargeError(f"Decoded dimensions exceed limits: {width}x{height} > {max_width}x{max_height}")

    arr = _coerce_image_to_bgr(image)
    if arr.size == 0 or arr.shape[0] == 0 or arr.shape[1] == 0:
        raise EmptyImageError(f"Decoded image is empty: {filename}")

    metadata = ImageMetadata(
        file_name=filename,
        file_format=image_format,
        file_size_bytes=len(payload),
        width=int(width),
        height=int(height),
        channels=int(arr.shape[2]) if arr.ndim == 3 else 1,
        color_mode="BGR",
    )
    return LoadedImage(image=arr, metadata=metadata, warnings=[])


try:
    import cv2  # type: ignore  # noqa: F401
except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard
    raise RuntimeError("OpenCV is required for MathVSE Phase 2 image preprocessing.") from exc
