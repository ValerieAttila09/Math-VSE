from __future__ import annotations

import sympy as sp

from mathvse.math_engine.ast_nodes import BinaryNode, EquationNode, ExprNode, NumberNode, UnaryNode, VariableNode


def ast_to_sympy(node: ExprNode):
    if isinstance(node, NumberNode):
        if isinstance(node.value, int):
            return sp.Integer(node.value)
        return sp.Rational(node.value.numerator, node.value.denominator)
    if isinstance(node, VariableNode):
        return sp.Symbol(node.name)
    if isinstance(node, UnaryNode):
        operand = ast_to_sympy(node.operand)
        if node.operator == "+":
            return +operand
        if node.operator == "-":
            return -operand
        raise ValueError(f"Unsupported unary operator: {node.operator}")
    if isinstance(node, BinaryNode):
        left = ast_to_sympy(node.left)
        right = ast_to_sympy(node.right)
        if node.operator == "+":
            return left + right
        if node.operator == "-":
            return left - right
        if node.operator == "*":
            return left * right
        if node.operator == "/":
            return left / right
        raise ValueError(f"Unsupported binary operator: {node.operator}")
    if isinstance(node, EquationNode):
        return ast_to_sympy(node.left) - ast_to_sympy(node.right)
    raise TypeError(f"Unsupported AST node: {type(node)!r}")
