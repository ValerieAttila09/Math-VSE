from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from mathvse.contracts.image import CandidateRegion, PreprocessingResult


def _ensure_2d_gray(image: np.ndarray) -> np.ndarray:
    if image.size == 0:
        raise ValueError("Input image is empty.")
    if image.ndim == 2:
        return image.astype(np.uint8)
    if image.ndim == 3 and image.shape[2] == 1:
        return image[:, :, 0].astype(np.uint8)
    if image.ndim == 3 and image.shape[2] in {3, 4}:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    raise ValueError(f"Unsupported image shape for preprocessing: {image.shape}")


def _resize_with_limit(gray: np.ndarray, resize_limit: int) -> np.ndarray:
    height, width = gray.shape[:2]
    long_side = max(height, width)
    if long_side <= resize_limit:
        return gray
    scale = resize_limit / float(long_side)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    return cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_AREA)


def preprocess_image(
    image: np.ndarray,
    *,
    strategy_name: str = "balanced",
    resize_limit: int = 2048,
    denoise: bool = True,
    invert: bool = False,
    normalize: bool = True,
) -> PreprocessingResult:
    if image is None or image.size == 0:
        raise ValueError("Input image is empty.")

    gray = _ensure_2d_gray(image)
    resized = _resize_with_limit(gray, resize_limit)
    denoised = cv2.GaussianBlur(resized, (3, 3), 0) if denoise else resized
    otsu = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    adaptive = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        10,
    )
    normalized = cv2.normalize(denoised, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    if invert:
        otsu = cv2.bitwise_not(otsu)
        adaptive = cv2.bitwise_not(adaptive)
        normalized = cv2.bitwise_not(normalized)

    variant_metadata: dict[str, dict[str, Any]] = {
        "grayscale": {"operation": "convert_to_grayscale", "params": {"resize_limit": resize_limit}},
        "denoised": {"operation": "gaussian_blur", "params": {"kernel_size": (3, 3), "sigma_x": 0}},
        "otsu": {"operation": "threshold_otsu", "params": {"mode": "binary", "threshold": 0}},
        "adaptive": {"operation": "threshold_adaptive", "params": {"method": "gaussian", "block_size": 31, "c": 10}},
        "normalized": {"operation": "normalize", "params": {"norm_type": "minmax"}},
    }

    variants = {
        "grayscale": resized,
        "denoised": denoised,
        "otsu": otsu,
        "adaptive": adaptive,
        "normalized": normalized if normalize else denoised,
    }

    contour_map, _ = cv2.findContours(otsu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidate_regions: list[CandidateRegion] = []
    for idx, contour in enumerate(contour_map):
        area = cv2.contourArea(contour)
        if area < 30:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        candidate_regions.append(
            CandidateRegion(
                region_id=f"candidate_{idx + 1}",
                bbox=(int(x), int(y), int(w), int(h)),
                area=int(area),
                label="preliminary_candidate",
                status="preliminary",
            )
        )

    warnings: list[str] = []
    if not candidate_regions:
        warnings.append("No preliminary contour candidates were detected; this is expected for blank or very sparse inputs.")

    return PreprocessingResult(
        status="success",
        strategy=strategy_name,
        variants=variants,
        variant_metadata=variant_metadata,
        candidate_regions=candidate_regions,
        bounding_boxes=[region.bbox for region in candidate_regions],
        warnings=warnings,
    )
