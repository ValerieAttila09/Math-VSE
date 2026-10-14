from mathvse.math_engine.lexer import Lexer
from mathvse.math_engine.parser import Parser
from mathvse.math_engine.solver import solve_equation


def test_unique_solution_basic():
    tree = Parser(Lexer("3*x - 6 = 12").tokenize()).parse()
    result = solve_equation(tree)
    assert result.status == "success"
    assert result.solution == "x = 6"
    assert result.is_valid is True


def test_fractional_solution():
    tree = Parser(Lexer("x/2 + 3 = 7").tokenize()).parse()
    result = solve_equation(tree)
    assert result.status == "success"
    assert result.solution == "x = 8"


def test_no_solution_case():
    tree = Parser(Lexer("2*x + 1 = 2*x + 3").tokenize()).parse()
    result = solve_equation(tree)
    assert result.status == "no_solution"


def test_infinite_solution_case():
    tree = Parser(Lexer("2*x + 1 = 2*x + 1").tokenize()).parse()
    result = solve_equation(tree)
    assert result.status == "infinite_solutions"
