from __future__ import annotations

from mathvse.exceptions import InvalidExpressionError
from mathvse.math_engine.ast_nodes import BinaryNode, EquationNode, ExprNode, NumberNode, UnaryNode, VariableNode
from mathvse.math_engine.tokens import Token, TokenType


class Parser:
    def __init__(self, tokens: list[Token]):
        self.tokens = tokens
        self.index = 0

    def parse(self) -> ExprNode:
        expr = self.parse_equality()
        if self.peek().type != TokenType.EOF:
            raise InvalidExpressionError(f"Unexpected token '{self.peek().value}' after expression.")
        return expr

    def peek(self) -> Token:
        return self.tokens[self.index]

    def advance(self) -> Token:
        token = self.tokens[self.index]
        self.index += 1
        return token

    def match(self, *token_types: TokenType) -> bool:
        if self.peek().type in token_types:
            self.index += 1
            return True
        return False

    def parse_equality(self) -> ExprNode:
        left = self.parse_additive()
        if self.peek().type == TokenType.EQUALS:
            self.advance()
            right = self.parse_additive()
            return EquationNode(left, right)
        return left

    def parse_additive(self) -> ExprNode:
        left = self.parse_multiplicative()
        while self.peek().type in {TokenType.PLUS, TokenType.MINUS}:
            operator = self.advance().type
            right = self.parse_multiplicative()
            if operator == TokenType.PLUS:
                left = BinaryNode("+", left, right)
            else:
                left = BinaryNode("-", left, right)
        return left

    def parse_multiplicative(self) -> ExprNode:
        left = self.parse_unary()
        while True:
            token = self.peek()
            if token.type == TokenType.MULTIPLY:
                self.advance()
                right = self.parse_unary()
                left = BinaryNode("*", left, right)
                continue
            if token.type == TokenType.DIVIDE:
                self.advance()
                right = self.parse_unary()
                left = BinaryNode("/", left, right)
                continue
            if token.type in {TokenType.NUMBER, TokenType.IDENT, TokenType.LPAREN}:
                right = self.parse_unary()
                left = BinaryNode("*", left, right)
                continue
            break
        return left

    def parse_unary(self) -> ExprNode:
        token = self.peek()
        if token.type in {TokenType.PLUS, TokenType.MINUS}:
            operator = self.advance().value
            operand = self.parse_unary()
            return UnaryNode(operator, operand)
        return self.parse_primary()

    def parse_primary(self) -> ExprNode:
        token = self.peek()
        if token.type == TokenType.NUMBER:
            self.advance()
            return NumberNode(int(token.value))
        if token.type == TokenType.IDENT:
            self.advance()
            return VariableNode(token.value)
        if token.type == TokenType.LPAREN:
            self.advance()
            inner = self.parse_equality()
            if self.peek().type != TokenType.RPAREN:
                raise InvalidExpressionError("Missing closing parenthesis.")
            self.advance()
            return inner
        raise InvalidExpressionError(f"Unexpected token '{token.value}' in expression.")
