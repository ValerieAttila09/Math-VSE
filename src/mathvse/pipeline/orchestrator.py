from __future__ import annotations

from mathvse.contracts.solution import SolutionResult
from mathvse.math_engine.lexer import Lexer
from mathvse.math_engine.parser import Parser
from mathvse.math_engine.solver import solve_equation


def solve_text_expression(expression: str) -> SolutionResult:
    if not expression or not expression.strip():
        return SolutionResult(
            status="failed",
            solution="",
            trace=[],
            is_valid=False,
            message="Ekspresi tidak boleh kosong.",
        )
    try:
        tree = Parser(Lexer(expression).tokenize()).parse()
        return solve_equation(tree)
    except Exception as exc:  # pragma: no cover - expected boundary handling
        return SolutionResult(
            status="failed",
            solution="",
            trace=[],
            is_valid=False,
            message=f"Gagal memproses ekspresi: {exc}",
        )
