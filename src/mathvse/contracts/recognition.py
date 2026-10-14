from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SymbolCandidate:
    label: str
    confidence: float
    bbox: tuple[int, int, int, int]
    status: str = "candidate"


@dataclass(frozen=True)
class SymbolPrediction:
    label: str
    confidence: float
    bbox: tuple[int, int, int, int]
    status: str = "recognized"
    alternatives: tuple[SymbolCandidate, ...] = ()
    source: str = "synthetic"


@dataclass
class RecognitionResult:
    status: str
    symbols: list[SymbolPrediction]
    normalized_text: str = ""
    warnings: list[str] = field(default_factory=list)
    model_path: str | None = None
    model_status: str = "model_unavailable"
    details: dict[str, Any] = field(default_factory=dict)
