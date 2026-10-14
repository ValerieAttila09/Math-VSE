from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from mathvse.contracts.image import LoadedImage, PreprocessingResult
from mathvse.contracts.recognition import RecognitionResult, SymbolPrediction
from mathvse.contracts.solution import SolutionResult
from mathvse.contracts.spatial import SpatialParseResult


ImageInferenceStatus = Literal[
    "success",
    "needs_review",
    "unsupported_structure",
    "model_unavailable",
    "invalid_image",
    "recognition_failed",
]


@dataclass(frozen=True)
class SymbolCrop:
    component_id: str
    bbox: tuple[int, int, int, int]
    image: np.ndarray | None = field(default=None, repr=False, compare=False)
    status: Literal["ready", "needs_review", "not_cropped"] = "ready"
    warning: str | None = None


@dataclass
class ImageInferenceResult:
    status: ImageInferenceStatus
    loaded_image: LoadedImage | None
    preprocessing: PreprocessingResult | None
    crops: list[SymbolCrop]
    predictions: list[SymbolPrediction]
    recognition: RecognitionResult
    spatial: SpatialParseResult | None
    normalized_expression: str = ""
    confirmed_expression: str | None = None
    solution: SolutionResult | None = None
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "normalized_expression": self.normalized_expression,
            "confirmed_expression": self.confirmed_expression,
            "warnings": self.warnings,
            "crops": [
                {
                    "component_id": crop.component_id,
                    "bbox": crop.bbox,
                    "status": crop.status,
                    "has_image": crop.image is not None,
                    "warning": crop.warning,
                }
                for crop in self.crops
            ],
            "predictions": [
                {
                    "component_id": prediction.component_id,
                    "label": prediction.label,
                    "confidence": prediction.confidence,
                    "bbox": prediction.bbox,
                    "status": prediction.status,
                    "source": prediction.source,
                    "alternatives": [
                        {"label": candidate.label, "confidence": candidate.confidence}
                        for candidate in prediction.alternatives
                    ],
                }
                for prediction in self.predictions
            ],
            "spatial_status": self.spatial.status if self.spatial else None,
        }