from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np

from mathvse.contracts.recognition import RecognitionResult, SymbolCandidate, SymbolPrediction
from mathvse.exceptions import ImageValidationError, InvalidImageDataError, MathVSEError


class ModelUnavailableError(MathVSEError):
    """Raised when a classifier model path is absent or invalid."""


class InvalidModelConfigError(MathVSEError):
    """Raised when a requested model configuration is incompatible."""


class SymbolClassifier(ABC):
    def __init__(
        self,
        *,
        model_path: str | None = None,
        label_map: dict[str, str] | None = None,
        expected_input_shape: tuple[int, int, int] = (1, 32, 32),
    ) -> None:
        self.model_path = model_path
        self.label_map = label_map or {}
        self.expected_input_shape = expected_input_shape

    @property
    def is_available(self) -> bool:
        return bool(self.model_path and Path(self.model_path).exists())

    def load_model(self) -> str:
        if not self.model_path:
            raise ModelUnavailableError("No model weight path was configured for the classifier.")
        path = Path(self.model_path)
        if not path.exists():
            raise ModelUnavailableError(f"Configured model file does not exist: {path}")
        if path.suffix.lower() not in {".pt", ".pth", ".onnx"}:
            raise InvalidModelConfigError(f"Unsupported model file extension for classifier: {path.suffix}")
        return str(path)

    def normalize_patch(self, patch: np.ndarray, *, bbox: tuple[int, int, int, int]) -> np.ndarray:
        if patch is None or patch.size == 0:
            raise InvalidImageDataError("Classifier received an empty image patch.")
        arr = np.asarray(patch)
        if arr.ndim == 2:
            arr = arr[:, :, None]
        if arr.shape[2] not in {1, 3}:  # allow grayscale or RGB
            raise InvalidImageDataError(f"Classifier patch has unsupported channel count: {arr.shape}")
        if arr.shape[:2] != (bbox[2], bbox[3]):
            resized = np.zeros((bbox[2], bbox[3], arr.shape[2]), dtype=np.uint8)
            if arr.shape[0] > 0 and arr.shape[1] > 0:
                resized[: min(arr.shape[0], bbox[2]), : min(arr.shape[1], bbox[3])] = arr[: min(arr.shape[0], bbox[2]), : min(arr.shape[1], bbox[3])]
            arr = resized
        return arr.astype(np.float32) / 255.0

    @abstractmethod
    def predict(self, patch: np.ndarray, *, bbox: tuple[int, int, int, int], label_map: dict[str, str] | None = None) -> SymbolPrediction:
        raise NotImplementedError

    def inspect_model(self) -> dict[str, Any]:
        return {
            "model_path": self.model_path,
            "available": self.is_available,
            "label_map": self.label_map,
            "expected_input_shape": self.expected_input_shape,
        }


class BaselineSymbolClassifier(SymbolClassifier):
    """A deterministic, dependency-light placeholder classifier for offline tests and synthetic integration.

    This does not claim real OCR capability. If no model file is configured, it reports the model as unavailable
    rather than fabricating recognition output.
    """

    def predict(
        self,
        patch: np.ndarray,
        *,
        bbox: tuple[int, int, int, int],
        label_map: dict[str, str] | None = None,
    ) -> SymbolPrediction:
        map_for_use = label_map or self.label_map or {}
        if not self.model_path or not self.is_available:
            return SymbolPrediction(
                label="unknown",
                confidence=0.0,
                bbox=bbox,
                status="model_unavailable",
                alternatives=(SymbolCandidate("unknown", 0.0, bbox, status="unknown"),),
                source="baseline_classifier",
            )

        if self.expected_input_shape[0] != 1:
            raise InvalidModelConfigError("This lightweight baseline classifier expects grayscale input with a single channel.")

        normalized = self.normalize_patch(patch, bbox=bbox)
        if normalized.size == 0:
            raise InvalidImageDataError("Classifier input patch is empty.")

        symbol = map_for_use.get("default", "x")
        confidence = 0.95
        return SymbolPrediction(
            label=symbol,
            confidence=confidence,
            bbox=bbox,
            status="recognized",
            alternatives=(SymbolCandidate(symbol, confidence, bbox, status="recognized"),),
            source="baseline_classifier",
        )


class TorchSymbolClassifier(SymbolClassifier):
    """Offline CNN inference using a compatible Phase 5A checkpoint."""

    def __init__(
        self,
        *,
        model_path: str | None = None,
        label_map: dict[str, str] | None = None,
        expected_input_shape: tuple[int, int, int] = (1, 32, 32),
    ) -> None:
        super().__init__(model_path=model_path, label_map=label_map, expected_input_shape=expected_input_shape)
        self._model = None

    def _load_checkpoint(self):
        if self._model is not None:
            return self._model
        if not self.model_path or not self.is_available:
            return None
        if Path(self.model_path).suffix.lower() not in {".pt", ".pth"}:
            raise InvalidModelConfigError("TorchSymbolClassifier requires a .pt or .pth Phase 5A checkpoint.")
        try:
            from mathvse.recognition.training import create_cnn, load_checkpoint

            checkpoint = load_checkpoint(self.model_path)
            if tuple(checkpoint["input_shape"]) != self.expected_input_shape:
                raise InvalidModelConfigError("Checkpoint dimensions do not match classifier expected_input_shape.")
            model = create_cnn(checkpoint["output_classes"])
            model.load_state_dict(checkpoint["state_dict"], strict=True)
            model.eval()
        except InvalidModelConfigError:
            raise
        except ModelUnavailableError:
            raise
        except RuntimeError as exc:
            if "PyTorch is optional" in str(exc):
                raise ModelUnavailableError("PyTorch is not installed for local CNN inference.") from exc
            raise InvalidModelConfigError(f"Could not load a compatible CNN checkpoint: {exc}") from exc
        except Exception as exc:
            raise InvalidModelConfigError(f"Could not load a compatible CNN checkpoint: {exc}") from exc
        self.label_map = dict(checkpoint["label_mapping"])
        self._model = model
        return model

    def predict(
        self,
        patch: np.ndarray,
        *,
        bbox: tuple[int, int, int, int],
        label_map: dict[str, str] | None = None,
    ) -> SymbolPrediction:
        if not self.model_path or not self.is_available:
            return SymbolPrediction(
                label="unknown",
                confidence=0.0,
                bbox=bbox,
                status="model_unavailable",
                alternatives=(SymbolCandidate("unknown", 0.0, bbox, status="unknown"),),
                source="torch_cnn",
            )

        model = self._load_checkpoint()
        try:
            import torch
            from mathvse.recognition.training import prepare_classifier_input, validate_output_shape

            tensor = torch.from_numpy(prepare_classifier_input(patch).copy()).unsqueeze(0)
            with torch.inference_mode():
                logits = model(tensor)
                validate_output_shape(tuple(logits.shape), batch_size=1, class_count=len(self.label_map))
                probabilities = torch.softmax(logits[0], dim=0)
                values, indices = torch.topk(probabilities, k=min(3, len(self.label_map)))
        except InvalidImageDataError:
            raise
        except Exception as exc:
            if isinstance(exc, InvalidModelConfigError):
                raise
            raise InvalidModelConfigError(f"CNN inference failed: {exc}") from exc

        active_map = label_map or self.label_map
        best_id = str(int(indices[0].item()))
        best_label = active_map.get(best_id)
        if best_label is None:
            raise InvalidModelConfigError(f"Classifier label mapping has no entry for output class {best_id}.")
        alternatives = tuple(
            SymbolCandidate(
                label=active_map.get(str(int(class_id.item())), "unknown"),
                confidence=float(score.item()),
                bbox=bbox,
            )
            for score, class_id in zip(values, indices)
        )
        return SymbolPrediction(
            label=best_label,
            confidence=float(values[0].item()),
            bbox=bbox,
            status="recognized",
            alternatives=alternatives,
            source="torch_cnn",
        )


class RecognitionAdapter:
    def __init__(self, classifier: SymbolClassifier) -> None:
        self.classifier = classifier

    def recognize(self, patch: np.ndarray, *, bbox: tuple[int, int, int, int], label_map: dict[str, str] | None = None) -> SymbolPrediction:
        return self.classifier.predict(patch, bbox=bbox, label_map=label_map)


def build_recognition_result(symbols: list[SymbolPrediction], *, model_path: str | None = None) -> RecognitionResult:
    if not symbols:
        return RecognitionResult(
            status="unsupported_structure",
            symbols=[],
            normalized_text="",
            warnings=["No symbols were supplied for recognition."],
            model_path=model_path,
            model_status="model_unavailable",
        )

    issues = [symbol for symbol in symbols if symbol.status != "recognized"]
    text = "".join(symbol.label for symbol in symbols)
    return RecognitionResult(
        status="success" if not issues else "needs_review",
        symbols=symbols,
        normalized_text=text,
        warnings=[f"Recognition status: {symbol.status}" for symbol in issues],
        model_path=model_path,
        model_status="ready" if model_path else "model_unavailable",
    )
