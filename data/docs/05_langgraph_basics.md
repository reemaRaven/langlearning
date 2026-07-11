# LangGraph Basics

LangGraph is a library for building applications as a graph of steps, with
explicit state, rather than a single linear chain. It's built by the
LangChain team specifically for the cases plain LangChain chains handle
poorly: branching, looping, and multi-step agents that need to make
decisions about what to do next.

## Core concepts

- **State** — a shared data structure (typically a `TypedDict` or Pydantic
  model) that flows through the whole graph. Every node receives the
  current state and returns updates to it. In this project, state holds
  things like the user's question, retrieved documents, tool results, and
  the final answer.
- **Nodes** — plain Python functions (or LCEL chains) that take the state
  and return a partial update to it. Each node does one job: "retrieve
  documents," "call the LLM," "run a tool."
- **Edges** — connections between nodes describing what runs next. A
  **normal edge** always goes A → B. A **conditional edge** picks the next
  node based on the current state — e.g. "if the router decided this needs
  the knowledge base, go to `retrieve`; if it decided this needs a tool, go
  to `call_tools`."
- **Graph compilation** — you build the graph by adding nodes and edges to
  a `StateGraph`, then call `.compile()` to get a runnable object with the
  same `.invoke()` interface as any other LangChain runnable.

## Why not just an if/else chain in Python?

You could write the routing logic as plain Python control flow, and for a
one-off script that's often simpler. LangGraph earns its keep once you want
things a hand-rolled script has to reimplement every time: automatic state
management across branches, the ability to visualize the graph, built-in
support for loops (a node's conditional edge can point back to an earlier
node — e.g. "grade the retrieved documents; if they're irrelevant, rewrite
the query and retrieve again"), and checkpointing (persisting state so a
conversation can pause and resume, or be replayed for debugging).

## This project's graph

```
        ┌──────────┐
   in → │  router  │
        └────┬─────┘
     ┌────────┴────────┐
     ▼                  ▼
┌──────────┐      ┌────────────┐
│ retrieve │      │ call_tools │
└────┬─────┘      └─────┬──────┘
     └────────┬─────────┘
              ▼
        ┌──────────┐
        │ generate │ → out
        └──────────┘
```

The `router` node asks the model (or uses simple heuristics) whether the
question is best answered from the local knowledge base or needs a tool.
`retrieve` pulls chunks from the Chroma vector store (see
`03_rag_pipeline.md`); `call_tools` runs whichever tool the model asked for
(see `04_tool_calling.md`). Both feed into `generate`, which produces the
final answer from whatever context was gathered. A natural next step
(left as a stretch goal) is adding a loop: a `grade_documents` node after
`retrieve` that checks relevance and routes back to a query-rewrite step if
the retrieved chunks don't actually answer the question.
