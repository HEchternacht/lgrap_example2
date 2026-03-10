"""
Safe calculator tool using AST-based expression evaluation.
Only allows numbers and basic arithmetic — no arbitrary code execution.
"""
from __future__ import annotations

import ast
import operator

from langchain_core.tools import tool

_SAFE_OPS: dict = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval(node: ast.expr) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_OPS:
        return _SAFE_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_OPS:
        return _SAFE_OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError(f"Unsupported operation: {ast.dump(node)}")


@tool
def calculator(expression: str) -> str:
    """
    Evaluate a mathematical expression safely.

    Supports: +, -, *, /, // (floor div), % (modulo), ** (power), unary - and +.
    Examples: '2 + 3 * 4', '(10 - 3) / 2', '2 ** 8', '17 % 5'
    """
    try:
        tree = ast.parse(expression.strip(), mode="eval")
        result = _safe_eval(tree.body)
        # Format as int when result is a whole number
        formatted = int(result) if result == int(result) else result
        return f"{expression} = {formatted}"
    except ZeroDivisionError:
        return "Error: Division by zero."
    except ValueError as exc:
        return f"Error: {exc}"
    except SyntaxError:
        return "Error: Invalid expression syntax."
