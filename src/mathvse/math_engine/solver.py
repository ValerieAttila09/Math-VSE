from __future__ import annotations

import sympy as sp

from mathvse.contracts.solution import SolutionResult, TraceEvent
from mathvse.math_engine.ast_nodes import BinaryNode, EquationNode, ExprNode, NumberNode, UnaryNode, VariableNode
from mathvse.math_engine.sympy_adapter import ast_to_sympy


def _collect_variables(node: ExprNode) -> set[str]:
    if isinstance(node, VariableNode):
        return {node.name}
    if isinstance(node, NumberNode):
        return set()
    if isinstance(node, UnaryNode):
        return _collect_variables(node.operand)
    if isinstance(node, BinaryNode):
        return _collect_variables(node.left) | _collect_variables(node.right)
    if isinstance(node, EquationNode):
        return _collect_variables(node.left) | _collect_variables(node.right)
    return set()


def _linear_trace(event_id: str, step_index: int, before: str, operation: dict, after: str, reason_code: str, preconditions: list[str]) -> TraceEvent:
    return TraceEvent(
        rule_id=event_id,
        step_index=step_index,
        before=before,
        operation=operation,
        after=after,
        reason_code=reason_code,
        preconditions=preconditions,
        validated=True,
    )


def solve_equation(tree: EquationNode) -> SolutionResult:
    variables = _collect_variables(tree)
    if len(variables) != 1:
        return SolutionResult(
            status="unsupported",
            solution="",
            trace=[],
            is_valid=False,
            message="Persamaan harus memiliki tepat satu variabel yang tidak diketahui.",
        )

    variable = next(iter(variables))
    expr = sp.expand(ast_to_sympy(tree.left) - ast_to_sympy(tree.right))
    symbol = sp.Symbol(variable)

    if expr == 0:
        return SolutionResult(
            status="infinite_solutions",
            solution="semua nilai x memenuhi",
            trace=[
                _linear_trace(
                    "normalize_linear_terms",
                    1,
                    tree.to_string(),
                    {"type": "simplify", "target": "both_sides"},
                    "0 = 0",
                    "collect_like_terms",
                    ["equation_is_linear_in_one_variable"],
                )
            ],
            is_valid=True,
            message="Persamaan ini berlaku untuk semua nilai x.",
        )

    if not expr.is_polynomial(symbol):
        return SolutionResult(
            status="unsupported",
            solution="",
            trace=[
                _linear_trace(
                    "normalize_linear_terms",
                    1,
                    tree.to_string(),
                    {"type": "check_linear", "target": variable},
                    tree.to_string(),
                    "unsupported_structure",
                    ["equation_is_linear_in_one_variable"],
                )
            ],
            is_valid=False,
            message="Persamaan ini di luar cakupan linear satu variabel yang didukung.",
        )

    coeff = sp.expand(expr).coeff(symbol, 1)
    constant = sp.expand(expr).coeff(symbol, 0)

    if coeff == 0:
        return SolutionResult(
            status="no_solution",
            solution="tidak ada solusi",
            trace=[
                _linear_trace(
                    "normalize_linear_terms",
                    1,
                    tree.to_string(),
                    {"type": "simplify", "target": "both_sides"},
                    f"{constant} = 0",
                    "collect_like_terms",
                    ["equation_is_linear_in_one_variable"],
                )
            ],
            is_valid=True,
            message="Setelah penyederhanaan, persamaan menghasilkan kontradiksi.",
        )

    value = sp.simplify(-constant / coeff)
    formatted_value = str(value)
    if value.is_integer:
        formatted_value = str(int(value))
    elif value.q != 1:
        formatted_value = f"{value.p}/{value.q}"

    trace = [
        _linear_trace(
            "normalize_linear_terms",
            1,
            tree.to_string(),
            {"type": "collect_like_terms", "variable": variable},
            f"{coeff}*{variable} + {constant} = 0",
            "collect_like_terms",
            ["equation_is_linear_in_one_variable"],
        ),
        _linear_trace(
            "divide_by_coefficient",
            2,
            f"{coeff}*{variable} + {constant} = 0",
            {"type": "divide", "coefficient": str(coeff), "scope": "both_sides"},
            f"{variable} = {formatted_value}",
            "isolate_variable",
            ["coefficient_is_nonzero"],
        ),
    ]

    return SolutionResult(
        status="success",
        solution=f"{variable} = {formatted_value}",
        trace=trace,
        is_valid=True,
        message="Solusi ditemukan dengan operasi aljabar linear dasar.",
        solution_value=value,
    )
