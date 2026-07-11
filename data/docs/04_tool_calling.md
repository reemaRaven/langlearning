# Tool Calling

Tool calling (also called "function calling") lets an LLM go beyond
generating text: the model can request that your code run a specific
function with specific arguments, see the result, and use it to continue
answering. The model never executes anything itself — it only ever outputs
"please call `get_current_datetime()`" or "please call
`calculator(expression='12*7')`"; your application is responsible for
actually running that function and feeding the result back in.

## Why this matters

LLMs are bad at things that need to be exact or current: arithmetic,
today's date, live data, anything outside their training data. Tool calling
lets you delegate those subtasks to reliable, deterministic code, while the
LLM handles the parts it's actually good at — understanding the request,
deciding which tool (if any) is needed, and turning a raw result into a
natural-language answer.

## How it works, mechanically

1. You define tools — in LangChain, typically a plain Python function
   decorated with `@tool`, with a docstring and type hints. The docstring
   and signature are what the model actually reads to decide when and how
   to call it, so they need to be clear and specific.
2. You "bind" the tools to a chat model (`model.bind_tools([...])`). The
   model provider's API now knows the available tool names, descriptions,
   and argument schemas.
3. You send a normal message. If the model decides a tool would help, it
   replies not with a text answer but with a *tool call*: a tool name plus
   arguments (as structured JSON, not free text).
4. Your code executes the matching Python function with those arguments.
5. You send the tool's return value back to the model as a new message
   (a "tool result"). The model then produces its final natural-language
   answer, now informed by that result.

Steps 3-5 can repeat — a model can call multiple tools in sequence (or in
parallel) before it has enough information to answer.

## Tools in this project

- `calculator` — evaluates arithmetic expressions safely (no `eval` of
  arbitrary code), because LLMs are unreliable at exact arithmetic.
- `get_current_datetime` — LLMs have a training-data cutoff and no innate
  sense of "now," so "what's today's date" has to come from a real clock.
- `web_search` — a DuckDuckGo search wrapper, for questions about
  current events or anything outside both the model's training data and
  the local knowledge base.

## Tool calling vs. RAG

They solve different problems and are often combined, as they are in this
project's LangGraph agent: RAG answers "what does *my* data say," backed by
a fixed, pre-indexed corpus; tool calling answers "go get me a fact / do a
computation right now," via a live function call. A well-designed agent
decides per-question which one (or neither, or both) it needs — that
decision-making is what LangGraph's routing is for (see
`05_langgraph_basics.md`).
