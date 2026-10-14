from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class TraceEvent:
    rule_id: str
    step_index: int
    before: str
    operation: dict[str, Any]
    after: str
    reason_code: str
    preconditions: list[str] = field(default_factory=list)
    validated: bool = True


@dataclass
class SolutionResult:
    status: str
    solution: str
    trace: list[TraceEvent]
    is_valid: bool = False
    message: str = ""
    solution_value: Optional[Any] = None


@dataclass
class ValidationResult:
    is_valid: bool
    message: str
    details: dict[str, Any] = field(default_factory=dict)
