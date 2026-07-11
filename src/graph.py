"""Stage 4: a LangGraph agent that routes each question to either the RAG
retriever (src/rag_chain.py) or the tool set (src/tools.py), then generates
a final answer. See data/docs/05_langgraph_basics.md for the concepts.

Requires ANTHROPIC_API_KEY (copy .env.example to .env and fill it in).
Run:
    python -m src.graph
"""

import os
from typing import List, TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from src.config import CHAT_MODEL_NAME
from src.rag_chain import format_docs, get_retriever
from src.tools import ALL_TOOLS


class GraphState(TypedDict):
    question: str
    route: str
    documents: List[Document]
    tool_output: str
    answer: str


ROUTER_SYSTEM_PROMPT = (
    "You are a router for a Q&A assistant. The assistant has a local "
    "knowledge base of study notes about LLMs, LangChain, RAG, tool "
    "calling, and LangGraph. It also has tools: a calculator, a "
    "current-datetime lookup, and a web search.\n\n"
    "Given the user's question, decide which path to take:\n"
    "- Reply with exactly 'RETRIEVE' if the question is about LLM / "
    "LangChain / RAG / tool-calling / LangGraph concepts (answerable from "
    "the knowledge base).\n"
    "- Reply with exactly 'TOOLS' if the question needs arithmetic, the "
    "current date/time, or current/live information from the web.\n"
    "Reply with only that one word, nothing else."
)

TOOLS_BY_NAME = {t.name: t for t in ALL_TOOLS}


def _model():
    return ChatAnthropic(model=CHAT_MODEL_NAME, temperature=0)


def router_node(state: GraphState) -> dict:
    response = _model().invoke(
        [
            SystemMessage(content=ROUTER_SYSTEM_PROMPT),
            HumanMessage(content=state["question"]),
        ]
    )
    route = "TOOLS" if "TOOLS" in response.content.upper() else "RETRIEVE"
    return {"route": route}


def route_decision(state: GraphState) -> str:
    return "call_tools" if state["route"] == "TOOLS" else "retrieve"


def retrieve_node(state: GraphState) -> dict:
    docs = get_retriever().invoke(state["question"])
    return {"documents": docs}


def call_tools_node(state: GraphState) -> dict:
    model = _model().bind_tools(ALL_TOOLS)
    response = model.invoke([HumanMessage(content=state["question"])])

    outputs = []
    for call in response.tool_calls:
        tool_fn = TOOLS_BY_NAME.get(call["name"])
        if tool_fn is None:
            continue
        result = tool_fn.invoke(call["args"])
        outputs.append(f"{call['name']}({call['args']}) -> {result}")

    return {"tool_output": "\n".join(outputs) if outputs else "No tool was called."}


def generate_node(state: GraphState) -> dict:
    if state.get("documents"):
        context, source = format_docs(state["documents"]), "knowledge base"
    else:
        context, source = state.get("tool_output", ""), "tool results"

    prompt = (
        f"Answer the question using the {source} below. If it's "
        f"insufficient, say so instead of guessing.\n\n"
        f"{source.title()}:\n{context}\n\nQuestion: {state['question']}"
    )
    response = _model().invoke([HumanMessage(content=prompt)])
    return {"answer": response.content}


def build_graph():
    graph = StateGraph(GraphState)
    graph.add_node("router", router_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("call_tools", call_tools_node)
    graph.add_node("generate", generate_node)

    graph.add_edge(START, "router")
    graph.add_conditional_edges(
        "router",
        route_decision,
        {"retrieve": "retrieve", "call_tools": "call_tools"},
    )
    graph.add_edge("retrieve", "generate")
    graph.add_edge("call_tools", "generate")
    graph.add_edge("generate", END)

    return graph.compile()


def main():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print(
            "ANTHROPIC_API_KEY not set — the graph needs it to run.\n"
            "Copy .env.example to .env and add your key, then rerun."
        )
        return

    app = build_graph()
    for question in ["What is a retriever?", "What is 45 * 12?"]:
        result = app.invoke({"question": question})
        print(f"\nQ: {question}\nRoute: {result['route']}\nA: {result['answer']}")


if __name__ == "__main__":
    main()
