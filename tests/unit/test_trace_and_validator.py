from mathvse.math_engine.lexer import Lexer
from mathvse.math_engine.parser import Parser
from mathvse.math_engine.solver import solve_equation
from mathvse.math_engine.validator import validate_solution


def test_solver_trace_is_generated():
    tree = Parser(Lexer("3*x - 6 = 12").tokenize()).parse()
    result = solve_equation(tree)
    assert len(result.trace) >= 2
    assert result.trace[0].rule_id
    assert result.trace[-1].after


def test_validate_solution_accepts_correct_answer():
    tree = Parser(Lexer("3*x - 6 = 12").tokenize()).parse()
    solved = solve_equation(tree)
    assert validate_solution(tree, solved.solution_value).is_valid is True


def test_validate_solution_rejects_wrong_answer():
    tree = Parser(Lexer("2*x + 1 = 9").tokenize()).parse()
    assert validate_solution(tree, 10).is_valid is False
