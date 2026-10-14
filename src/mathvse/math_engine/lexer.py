from __future__ import annotations

from mathvse.exceptions import InvalidExpressionError
from mathvse.math_engine.tokens import Token, TokenType


class Lexer:
    def __init__(self, text: str):
        self.text = text
        self.index = 0

    def tokenize(self) -> list[Token]:
        tokens: list[Token] = []
        while self.index < len(self.text):
            ch = self.text[self.index]
            if ch.isspace():
                self.index += 1
                continue
            if ch.isdigit():
                start = self.index
                while self.index < len(self.text) and self.text[self.index].isdigit():
                    self.index += 1
                tokens.append(Token(TokenType.NUMBER, self.text[start:self.index], start))
                continue
            if ch.isalpha() or ch == "_":
                start = self.index
                while self.index < len(self.text) and (self.text[self.index].isalnum() or self.text[self.index] == "_"):
                    self.index += 1
                tokens.append(Token(TokenType.IDENT, self.text[start:self.index], start))
                continue
            if ch == "+":
                tokens.append(Token(TokenType.PLUS, ch, self.index))
                self.index += 1
                continue
            if ch == "-":
                tokens.append(Token(TokenType.MINUS, ch, self.index))
                self.index += 1
                continue
            if ch == "*":
                tokens.append(Token(TokenType.MULTIPLY, ch, self.index))
                self.index += 1
                continue
            if ch == "/":
                tokens.append(Token(TokenType.DIVIDE, ch, self.index))
                self.index += 1
                continue
            if ch == "=":
                tokens.append(Token(TokenType.EQUALS, ch, self.index))
                self.index += 1
                continue
            if ch == "(":
                tokens.append(Token(TokenType.LPAREN, ch, self.index))
                self.index += 1
                continue
            if ch == ")":
                tokens.append(Token(TokenType.RPAREN, ch, self.index))
                self.index += 1
                continue
            raise InvalidExpressionError(f"Unsupported character '{ch}' at position {self.index}.")
        tokens.append(Token(TokenType.EOF, "", self.index))
        return tokens
