from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Union

Numeric = Union[int, Fraction]


class ExprNode:
    def to_string(self) -> str:
        raise NotImplementedError


@dataclass(frozen=True)
class NumberNode(ExprNode):
    value: Numeric

    def to_string(self) -> str:
        if isinstance(self.value, Fraction):
            if self.value.denominator == 1:
                return str(self.value.numerator)
            return f"{self.value.numerator}/{self.value.denominator}"
        return str(self.value)


@dataclass(frozen=True)
class VariableNode(ExprNode):
    name: str

    def to_string(self) -> str:
        return self.name


@dataclass(frozen=True)
class UnaryNode(ExprNode):
    operator: str
    operand: ExprNode

    def to_string(self) -> str:
        inner = self.operand.to_string()
        if isinstance(self.operand, BinaryNode):
            inner = f"({inner})"
        return f"{self.operator}{inner}"


@dataclass(frozen=True)
class BinaryNode(ExprNode):
    operator: str
    left: ExprNode
    right: ExprNode

    def to_string(self) -> str:
        left_str = self.left.to_string()
        right_str = self.right.to_string()
        if isinstance(self.left, BinaryNode) and self._precedence(self.left.operator) < self._precedence(self.operator):
            left_str = f"({left_str})"
        if isinstance(self.right, BinaryNode) and self._precedence(self.right.operator) < self._precedence(self.operator):
            right_str = f"({right_str})"
        if self.operator in {"+", "-"}:
            return f"{left_str} {self.operator} {right_str}"
        return f"{left_str}{self.operator}{right_str}"

    @staticmethod
    def _precedence(operator: str) -> int:
        if operator in {"+", "-"}:
            return 1
        if operator in {"*", "/"}:
            return 2
        return 0


@dataclass(frozen=True)
class EquationNode(ExprNode):
    left: ExprNode
    right: ExprNode

    def to_string(self) -> str:
        return f"{self.left.to_string()} = {self.right.to_string()}"
