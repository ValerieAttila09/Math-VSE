from __future__ import annotations

from mathvse.contracts.solution import SolutionResult
from mathvse.explanation.templates import DEFAULT_TEMPLATE, TEMPLATE_MAP


def generate_explanation(result: SolutionResult) -> str:
    if not result.trace:
        return "Tidak ada langkah yang dapat dijelaskan untuk persamaan ini."

    steps: list[str] = []
    for event in result.trace:
        template = TEMPLATE_MAP.get(event.rule_id, DEFAULT_TEMPLATE)
        if event.rule_id == "divide_by_coefficient":
            template = "Bagi kedua ruas dengan koefisien variabel untuk mengisolasi x."
        steps.append(template)

    message = " ".join(steps)
    if result.solution:
        message = f"{message} Jadi, {result.solution}."
    return message
