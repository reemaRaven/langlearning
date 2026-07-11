# LangChain Basics

LangChain is a Python/JS framework for building LLM applications out of
composable pieces, instead of hand-rolling prompt strings and API calls
every time. The core value it adds is standardized interfaces: any chat
model, any vector store, any document loader looks the same to the rest of
your code, so you can swap providers without rewriting your application
logic.

## Core building blocks

- **Chat models** — a unified wrapper (`ChatAnthropic`, `ChatOpenAI`, ...)
  exposing the same `.invoke()` / `.stream()` methods regardless of
  provider.
- **Prompt templates** — `ChatPromptTemplate` fills placeholders into a
  message list, keeping prompt text separate from application code.
- **Output parsers** — turn raw model output into structured data (a
  Python object, JSON, a Pydantic model) instead of a plain string.
- **Retrievers** — a standard interface (`.invoke(query) -> list[Document]`)
  over "given a query, return relevant documents," backed by a vector
  store, a keyword search index, or anything else.
- **Tools** — a standard interface for functions an LLM can call (see
  `04_tool_calling.md`).

## LCEL (LangChain Expression Language)

LangChain lets you compose these pieces with the `|` (pipe) operator into a
"runnable" chain, similar to a Unix pipeline:

```python
chain = prompt | model | output_parser
result = chain.invoke({"question": "What is a retriever?"})
```

Each stage's output becomes the next stage's input. This is the pattern
`rag_chain.py` in this project uses for a simple "retrieve context, then
ask the model" pipeline. It's the right tool for a straight-line sequence
of steps.

## Where LangChain stops being enough

LCEL chains are fundamentally linear (or simple branches). Once your
application needs to make a *decision* — "should I search my documents, or
call a tool, or ask a follow-up question, and maybe loop back and try
again" — you need explicit control flow and state that persists across
steps. That's what LangGraph is for (see `05_langgraph_basics.md`):
LangChain gives you the components, LangGraph gives you the orchestration.
