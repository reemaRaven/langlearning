"""CLI entry point: an interactive chat loop over the compiled LangGraph
agent from src/graph.py.

Requires ANTHROPIC_API_KEY (copy .env.example to .env and fill it in).
Run:
    python -m src.chat
"""

import os

from src.graph import build_graph


def main():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print(
            "ANTHROPIC_API_KEY not set. Copy .env.example to .env, add "
            "your key, then rerun."
        )
        return

    app = build_graph()
    print(
        "Agentic RAG assistant ready.\n"
        "Ask about LangChain/RAG/tool-calling/LangGraph concepts, or ask "
        "for a calculation, the date, or a web search. Ctrl+C to quit.\n"
    )
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not question:
            continue

        result = app.invoke({"question": question})
        print(f"[routed to: {result['route']}]")
        print(f"Assistant: {result['answer']}\n")


if __name__ == "__main__":
    main()
