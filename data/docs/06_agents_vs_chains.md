# Agents vs. Chains

"Chain" and "agent" describe two different amounts of control you hand to
the LLM.

## Chains: fixed control flow

A chain has a control flow decided in advance by the developer. The LCEL
example from `02_langchain_basics.md`,
`chain = prompt | model | output_parser`, always runs those three steps in
that order, every time, no matter what the input is. The model only ever
produces text — it never decides which step runs next. This project's
`rag_chain.py` is a chain: retrieve, then generate, always, in that order.

## Agents: model-decided control flow

An agent hands some of that control to the LLM itself. Instead of a fixed
sequence, the model is given a set of tools (see `04_tool_calling.md`) and,
in a loop, decides at each step: "do I have enough information to answer,
or should I call a tool (or retrieve more context) first?" The loop
continues until the model decides it's done. This is strictly more
flexible than a chain — the same agent can answer a simple question in one
step or a complex one in five, adapting to the input — but also less
predictable, since you're trusting the model's judgment about what to do
next rather than hard-coding it.

## Where LangGraph fits

LangGraph is the tool for building agents (and anything in between a rigid
chain and a fully autonomous agent). The graph in this project
(`05_langgraph_basics.md`) is a middle ground sometimes called a "router" or
"semi-structured" agent: the *shape* of the possible paths is fixed by the
developer (router → retrieve-or-tools → generate), but *which* path is
taken for a given question is decided dynamically. A fully autonomous agent
would instead let the model loop through tool calls an arbitrary number of
times with no fixed shape at all — more flexible, harder to reason about
and debug. Most production systems land somewhere on this spectrum rather
than at either extreme, picking as much structure as keeps behavior
predictable while leaving the model room to adapt.
