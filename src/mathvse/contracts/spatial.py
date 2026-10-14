from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from mathvse.contracts.recognition import SymbolPrediction


@dataclass(frozen=True)
class SpatialSymbol:
    label: str
    bbox: tuple[int, int, int, int]
    confidence: float
    status: str = "recognized"


@dataclass
class SpatialParseResult:
    status: str
    normalized_text: str
    symbols: list[SpatialSymbol] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)
    source: str = "spatial_parser"


@dataclass
class ParsedExpressionCandidate:
    text: str
    status: str
    confidence: float = 0.0
    symbols: list[SymbolPrediction] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
