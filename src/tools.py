"""Stage 3: tool-calling. Each function below is a plain Python function
decorated with @tool — the docstring and type hints are what the LLM reads
to decide when and how to call it (see data/docs/04_tool_calling.md).

None of these need an API key. Run standalone smoke tests with:
    python -m src.tools
"""

import ast
import operator
from datetime import datetime

from ddgs import DDGS
from langchain_core.tools import tool

_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _eval_node(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:
        return _OPERATORS[type(node.op)](
            _eval_node(node.left), _eval_node(node.right)
        )
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPERATORS:
        return _OPERATORS[type(node.op)](_eval_node(node.operand))
    raise ValueError(f"Unsupported expression: {ast.dump(node)}")


@tool
def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression, e.g. '12 * (7 + 3)'.
    Supports +, -, *, /, %, ** and parentheses only — no variables or
    function calls. Use this for any arithmetic instead of computing it
    yourself, since exact arithmetic is unreliable to do purely in text."""
    try:
        tree = ast.parse(expression, mode="eval")
        return str(_eval_node(tree.body))
    except Exception as exc:
        return f"Error evaluating expression: {exc}"


@tool
def get_current_datetime() -> str:
    """Return the current date and time. Use this whenever a question
    depends on 'today', 'now', or the current date, since the model has
    no innate sense of the current date/time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S %A")


@tool
def web_search(query: str) -> str:
    """Search the web for current information that isn't in the local
    knowledge base or the model's training data (e.g. recent events).
    Returns a short list of result titles, snippets, and links."""
    try:
        results = list(DDGS().text(query, max_results=3))
        if not results:
            return "No results found."
        return "\n".join(
            f"- {r.get('title')}: {r.get('body')} ({r.get('href')})"
            for r in results
        )
    except Exception as exc:
        return f"Search failed: {exc}"


ALL_TOOLS = [calculator, get_current_datetime, web_search]


if __name__ == "__main__":
    print("calculator:", calculator.invoke({"expression": "12 * (7 + 3)"}))
    print("get_current_datetime:", get_current_datetime.invoke({}))
    print("web_search:")
    print(web_search.invoke({"query": "LangGraph LangChain agents"}))
