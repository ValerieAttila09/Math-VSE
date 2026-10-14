from __future__ import annotations

from dataclasses import replace
from math import isfinite
from pathlib import Path

import numpy as np

from mathvse.contracts.image import LoadedImage, PreprocessingResult
from mathvse.contracts.image_inference import ImageInferenceResult, ImageInferenceStatus, SymbolCrop
from mathvse.contracts.recognition import RecognitionResult, SymbolCandidate, SymbolPrediction
from mathvse.contracts.spatial import SpatialParseResult
from mathvse.exceptions import ImageValidationError, InvalidImageDataError
from mathvse.recognition.classifier import (
    InvalidModelConfigError,
    ModelUnavailableError,
    SymbolClassifier,
    TorchSymbolClassifier,
)
from mathvse.spatial.parser import parse_symbols
from mathvse.vision.input import load_image_file
from mathvse.vision.preprocessing import preprocess_image


class CropExtractionError(ValueError):
    """Raised when candidate crop extraction cannot honor its safety bounds."""


def extract_symbol_crops(
    preprocessing: PreprocessingResult,
    *,
    margin_ratio: float = 0.1,
    max_components: int = 128,
    max_crop_dimension: int = 512,
    max_total_crop_pixels: int = 16_777_216,
) -> tuple[list[SymbolCrop], list[str]]:
    if not 0 <= margin_ratio <= 0.5 or max_components < 1 or max_crop_dimension < 1 or max_total_crop_pixels < 1:
        raise ValueError("Crop extraction limits must be positive and margin_ratio must be between 0 and 0.5")
    grayscale = preprocessing.variants.get("grayscale")
    if grayscale is None or grayscale.ndim != 2 or grayscale.size == 0:
        raise CropExtractionError("Preprocessing did not provide a non-empty grayscale image")

    height, width = grayscale.shape
    regions = sorted(preprocessing.candidate_regions, key=lambda region: (region.bbox[1], region.bbox[0], region.region_id))
    warnings: list[str] = []
    crops: list[SymbolCrop] = []
    total_pixels = 0
    if len(regions) > max_components:
        warnings.append(f"Component limit reached ({max_components}); remaining candidates require review.")

    for region_index, region in enumerate(regions[:max_components]):
        x, y, box_width, box_height = region.bbox
        left = max(0, x)
        top = max(0, y)
        right = min(width, x + box_width)
        bottom = min(height, y + box_height)
        if right <= left or bottom <= top:
            warnings.append(f"{region.region_id}: component box is outside the processed image.")
            crops.append(SymbolCrop(region.region_id, region.bbox, status="not_cropped", warning="Invalid component box."))
            continue

        clipped_bbox = (left, top, right - left, bottom - top)
        margin = max(2, min(12, int(round(max(right - left, bottom - top) * margin_ratio))))
        crop_left = max(0, left - margin)
        crop_top = max(0, top - margin)
        crop_right = min(width, right + margin)
        crop_bottom = min(height, bottom + margin)
        crop_width = crop_right - crop_left
        crop_height = crop_bottom - crop_top
        reasons: list[str] = []
        touches_edge = left == 0 or top == 0 or right == width or bottom == height
        aspect_ratio = max(clipped_bbox[2] / max(clipped_bbox[3], 1), clipped_bbox[3] / max(clipped_bbox[2], 1))
        if touches_edge:
            reasons.append("component touches the image boundary")
        if aspect_ratio > 1.8:
            reasons.append("elongated component may contain a stroke or merged symbols")
        if crop_width > max_crop_dimension or crop_height > max_crop_dimension:
            reasons.append("component exceeds the per-crop size limit")
        if total_pixels + crop_width * crop_height > max_total_crop_pixels:
            reasons.append("total crop pixel limit reached")
        if reasons:
            warning = f"{region.region_id}: " + "; ".join(reasons) + "."
            warnings.append(warning)
            if crop_width > max_crop_dimension or crop_height > max_crop_dimension or total_pixels + crop_width * crop_height > max_total_crop_pixels:
                crops.append(SymbolCrop(region.region_id, clipped_bbox, status="not_cropped", warning=warning))
                continue

        crop_image = np.ascontiguousarray(grayscale[crop_top:crop_bottom, crop_left:crop_right].copy())
        total_pixels += crop_image.size
        crops.append(
            SymbolCrop(
                component_id=region.region_id,
                bbox=clipped_bbox,
                image=crop_image,
                status="needs_review" if reasons else "ready",
                warning="; ".join(reasons) if reasons else None,
            )
        )

    return crops, warnings


def _empty_recognition(status: str, warning: str, model_path: str | None = None) -> RecognitionResult:
    return RecognitionResult(
        status=status,
        symbols=[],
        normalized_text="",
        warnings=[warning],
        model_path=model_path,
        model_status="model_unavailable" if status == "model_unavailable" else "unavailable",
    )


def _result(
    status: ImageInferenceStatus,
    *,
    loaded_image: LoadedImage | None,
    preprocessing: PreprocessingResult | None,
    crops: list[SymbolCrop],
    predictions: list[SymbolPrediction],
    recognition: RecognitionResult,
    spatial: SpatialParseResult | None,
    expression: str = "",
    warnings: list[str] | None = None,
) -> ImageInferenceResult:
    return ImageInferenceResult(
        status=status,
        loaded_image=loaded_image,
        preprocessing=preprocessing,
        crops=crops,
        predictions=predictions,
        recognition=recognition,
        spatial=spatial,
        normalized_expression=expression,
        warnings=warnings or [],
    )


def infer_image(
    image: np.ndarray,
    *,
    classifier: SymbolClassifier | None = None,
    model_path: str | None = None,
    confidence_threshold: float = 0.55,
    ambiguity_margin: float = 0.08,
    loaded_image: LoadedImage | None = None,
) -> ImageInferenceResult:
    if not 0.0 <= confidence_threshold <= 1.0 or not 0.0 <= ambiguity_margin <= 1.0:
        raise ValueError("Confidence threshold and ambiguity margin must be between 0 and 1")
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        warning = "Image input is empty or invalid."
        return _result(
            "invalid_image",
            loaded_image=loaded_image,
            preprocessing=None,
            crops=[],
            predictions=[],
            recognition=_empty_recognition("invalid_image", warning, model_path),
            spatial=None,
            warnings=[warning],
        )

    try:
        preprocessing = preprocess_image(image, strategy_name="balanced")
        crops, crop_warnings = extract_symbol_crops(preprocessing)
    except (ValueError, CropExtractionError) as exc:
        warning = f"Image preprocessing failed: {exc}"
        return _result(
            "invalid_image",
            loaded_image=loaded_image,
            preprocessing=None,
            crops=[],
            predictions=[],
            recognition=_empty_recognition("invalid_image", warning, model_path),
            spatial=None,
            warnings=[warning],
        )

    active_classifier = classifier or TorchSymbolClassifier(model_path=model_path)
    active_model_path = getattr(active_classifier, "model_path", model_path)
    if not active_classifier.is_available:
        warning = "A local compatible OCR checkpoint is not available; symbol predictions were not generated."
        recognition = _empty_recognition("model_unavailable", warning, active_model_path)
        recognition.details["component_count"] = len(crops)
        return _result(
            "model_unavailable",
            loaded_image=loaded_image,
            preprocessing=preprocessing,
            crops=crops,
            predictions=[],
            recognition=recognition,
            spatial=None,
            warnings=crop_warnings + [warning],
        )
    if not crops:
        warning = "No foreground components were detected for symbol classification."
        spatial = parse_symbols([])
        recognition = _empty_recognition("unsupported_structure", warning, active_model_path)
        return _result(
            "unsupported_structure",
            loaded_image=loaded_image,
            preprocessing=preprocessing,
            crops=[],
            predictions=[],
            recognition=recognition,
            spatial=spatial,
            warnings=crop_warnings + [warning],
        )

    predictions: list[SymbolPrediction] = []
    warnings = list(crop_warnings)
    try:
        for crop in crops:
            if crop.image is None:
                prediction = SymbolPrediction(
                    label="unknown",
                    confidence=0.0,
                    bbox=crop.bbox,
                    status="needs_review",
                    alternatives=(SymbolCandidate("unknown", 0.0, crop.bbox, status="unknown"),),
                    source="component_review",
                    component_id=crop.component_id,
                )
                predictions.append(prediction)
                continue

            prediction = active_classifier.predict(crop.image, bbox=crop.bbox)
            if not isinstance(prediction, SymbolPrediction):
                raise InvalidModelConfigError("Classifier returned an invalid prediction object")
            if not isfinite(prediction.confidence) or not 0.0 <= prediction.confidence <= 1.0:
                raise InvalidModelConfigError(f"Classifier returned invalid confidence for {crop.component_id}")
            ranked = sorted(prediction.alternatives, key=lambda candidate: candidate.confidence, reverse=True)
            ambiguous = len(ranked) > 1 and ranked[0].confidence - ranked[1].confidence < ambiguity_margin
            uncertain = (
                prediction.status != "recognized"
                or prediction.label == "unknown"
                or prediction.confidence < confidence_threshold
                or ambiguous
                or crop.status == "needs_review"
            )
            status = "needs_review" if uncertain else prediction.status
            prediction = replace(prediction, status=status, component_id=crop.component_id)
            predictions.append(prediction)
            if uncertain:
                reason = crop.warning or "confidence or top-class ambiguity is below the configured threshold"
                warnings.append(f"{crop.component_id}: review required ({reason}).")
    except ModelUnavailableError as exc:
        warning = str(exc)
        recognition = _empty_recognition("model_unavailable", warning, active_model_path)
        return _result(
            "model_unavailable",
            loaded_image=loaded_image,
            preprocessing=preprocessing,
            crops=crops,
            predictions=predictions,
            recognition=recognition,
            spatial=None,
            warnings=warnings + [warning],
        )
    except (InvalidModelConfigError, InvalidImageDataError, OSError, ValueError) as exc:
        warning = f"Symbol recognition failed: {exc}"
        recognition = _empty_recognition("recognition_failed", warning, active_model_path)
        recognition.symbols = predictions
        return _result(
            "recognition_failed",
            loaded_image=loaded_image,
            preprocessing=preprocessing,
            crops=crops,
            predictions=predictions,
            recognition=recognition,
            spatial=None,
            warnings=warnings + [warning],
        )
    except Exception as exc:
        warning = f"Symbol recognition failed unexpectedly: {exc}"
        recognition = _empty_recognition("recognition_failed", warning, active_model_path)
        recognition.symbols = predictions
        return _result(
            "recognition_failed",
            loaded_image=loaded_image,
            preprocessing=preprocessing,
            crops=crops,
            predictions=predictions,
            recognition=recognition,
            spatial=None,
            warnings=warnings + [warning],
        )

    spatial = parse_symbols(predictions)
    uncertain_symbols = any(prediction.status != "recognized" for prediction in predictions)
    if spatial.status == "unsupported_structure":
        status: ImageInferenceStatus = "unsupported_structure"
    elif spatial.status != "success" or uncertain_symbols:
        status = "needs_review"
    else:
        status = "success"
    expression = spatial.normalized_text if status == "success" else ""
    recognition = RecognitionResult(
        status=status,
        symbols=predictions,
        normalized_text=expression,
        warnings=list(dict.fromkeys(warnings + spatial.warnings)),
        model_path=active_model_path,
        model_status="ready",
        details={"component_count": len(crops), "confidence_threshold": confidence_threshold, "ambiguity_margin": ambiguity_margin},
    )
    return _result(
        status,
        loaded_image=loaded_image,
        preprocessing=preprocessing,
        crops=crops,
        predictions=predictions,
        recognition=recognition,
        spatial=spatial,
        expression=expression,
        warnings=list(dict.fromkeys(warnings + spatial.warnings)),
    )


def infer_image_file(
    image_path: str | Path,
    *,
    model_path: str | None = None,
    confidence_threshold: float = 0.55,
    ambiguity_margin: float = 0.08,
) -> ImageInferenceResult:
    try:
        loaded = load_image_file(image_path)
    except ImageValidationError as exc:
        warning = f"Image validation failed: {exc}"
        return _result(
            "invalid_image",
            loaded_image=None,
            preprocessing=None,
            crops=[],
            predictions=[],
            recognition=_empty_recognition("invalid_image", warning, model_path),
            spatial=None,
            warnings=[warning],
        )
    return infer_image(
        loaded.image,
        model_path=model_path,
        confidence_threshold=confidence_threshold,
        ambiguity_margin=ambiguity_margin,
        loaded_image=loaded,
    )