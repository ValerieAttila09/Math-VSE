from mathvse.math_engine.ast_nodes import ExprNode


def format_expression(node: ExprNode) -> str:
    return node.to_string()


def format_equation(node: ExprNode) -> str:
    return node.to_string()
