from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ParsedExpression:
    raw_text: str
    normalized_text: str
    variable: Optional[str] = None
    is_supported: bool = True
    status: str = "ready"
