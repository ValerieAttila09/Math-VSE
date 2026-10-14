import pytest

from mathvse.math_engine.lexer import Lexer
from mathvse.math_engine.parser import Parser
from mathvse.math_engine.ast_nodes import EquationNode, BinaryNode, NumberNode, VariableNode


def test_parse_basic_equation():
    tokens = Lexer("2*x + 5 = 15").tokenize()
    tree = Parser(tokens).parse()

    assert isinstance(tree, EquationNode)
    assert tree.left.to_string() == "2*x + 5"
    assert tree.right.to_string() == "15"


def test_parse_implicit_multiplication():
    tree = Parser(Lexer("2x + 5 = 15").tokenize()).parse()
    assert tree.left.to_string() == "2*x + 5"


def test_parse_unary_minus_and_parentheses():
    tree = Parser(Lexer("-(x + 3) = 5").tokenize()).parse()
    assert tree.left.to_string() == "-(x + 3)"
    assert tree.right.to_string() == "5"


def test_parse_invalid_expression_raises():
    with pytest.raises(ValueError):
        Parser(Lexer("2 *").tokenize()).parse()
