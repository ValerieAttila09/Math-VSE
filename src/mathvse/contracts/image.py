from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ImageMetadata:
    file_name: str
    file_format: str
    file_size_bytes: int
    width: int
    height: int
    channels: int
    color_mode: str


@dataclass(frozen=True)
class CandidateRegion:
    region_id: str
    bbox: tuple[int, int, int, int]
    area: int
    label: str = "preliminary_candidate"
    confidence: float | None = None
    status: str = "preliminary"


@dataclass
class LoadedImage:
    image: np.ndarray
    metadata: ImageMetadata
    warnings: list[str] = field(default_factory=list)


@dataclass
class PreprocessingResult:
    status: str
    strategy: str
    variants: dict[str, np.ndarray]
    variant_metadata: dict[str, dict[str, Any]]
    candidate_regions: list[CandidateRegion]
    bounding_boxes: list[tuple[int, int, int, int]]
    warnings: list[str] = field(default_factory=list)
    original_metadata: ImageMetadata | None = None
