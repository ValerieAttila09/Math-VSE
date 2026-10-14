from mathvse.math_engine.lexer import Lexer
from mathvse.math_engine.parser import Parser
from mathvse.math_engine.solver import solve_equation
from mathvse.explanation.generator import generate_explanation


def test_explanation_uses_trace_and_indonesian():
    tree = Parser(Lexer("x/2 + 3 = 7").tokenize()).parse()
    result = solve_equation(tree)
    explanation = generate_explanation(result)
    assert "bagi" in explanation.lower()
    assert "x" in explanation
