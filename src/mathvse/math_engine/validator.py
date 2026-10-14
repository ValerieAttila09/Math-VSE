from __future__ import annotations

import re

import sympy as sp

from mathvse.contracts.solution import ValidationResult
from mathvse.math_engine.ast_nodes import BinaryNode, EquationNode, ExprNode, NumberNode, UnaryNode, VariableNode
from mathvse.math_engine.sympy_adapter import ast_to_sympy


def _extract_variable_name(node: ExprNode):
    if isinstance(node, VariableNode):
        return node.name
    if isinstance(node, NumberNode):
        return None
    if isinstance(node, UnaryNode):
        return _extract_variable_name(node.operand)
    if isinstance(node, BinaryNode):
        left_var = _extract_variable_name(node.left)
        if left_var is not None:
            return left_var
        return _extract_variable_name(node.right)
    if isinstance(node, EquationNode):
        return _extract_variable_name(node.left) or _extract_variable_name(node.right)
    return None


def _extract_candidate_value(candidate):
    if isinstance(candidate, (int, float, sp.Integer, sp.Float, sp.Rational)):
        return sp.sympify(candidate)
    if isinstance(candidate, str):
        match = re.search(r"[xX]\s*=\s*([-+]?\d+/\d+|[-+]?\d+)", candidate)
        if match:
            raw = match.group(1)
            if "/" in raw:
                numerator, denominator = raw.split("/")
                return sp.Rational(int(numerator), int(denominator))
            return sp.Integer(int(raw))
        return sp.sympify(candidate)
    return candidate


def validate_solution(equation: EquationNode, candidate) -> ValidationResult:
    try:
        if not isinstance(equation, EquationNode):
            return ValidationResult(False, "Objek yang divalidasi bukan persamaan.")

        variable = _extract_variable_name(equation)
        if variable is None:
            return ValidationResult(False, "Persamaan tidak memiliki variabel yang dapat divalidasi.")

        candidate_value = _extract_candidate_value(candidate)
        lhs = ast_to_sympy(equation.left).subs(sp.Symbol(variable), candidate_value)
        rhs = ast_to_sympy(equation.right).subs(sp.Symbol(variable), candidate_value)
        is_valid = sp.simplify(lhs - rhs) == 0
        if is_valid:
            return ValidationResult(True, "Solusi valid dan memenuhi persamaan awal.", {"variable": variable, "value": candidate_value})
        return ValidationResult(False, "Solusi tidak memenuhi persamaan awal.", {"variable": variable, "value": candidate_value})
    except Exception as exc:  # pragma: no cover - validation failure path
        return ValidationResult(False, f"Validasi gagal: {exc}")
