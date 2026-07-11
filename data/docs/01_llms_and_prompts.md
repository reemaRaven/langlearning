# LLMs and Prompts

A large language model (LLM) is a next-token predictor: given a sequence of
text, it predicts a probability distribution over what token comes next, and
repeats that process to generate text. Chat-tuned LLMs like Claude are
further trained (via instruction tuning and reinforcement learning) to follow
instructions and hold a conversation rather than just complete arbitrary text.

## Prompts

A prompt is the input you send the model. Most modern LLM APIs, including
Anthropic's, structure a prompt as a list of messages with roles:

- **system** — instructions that set the model's behavior, persona, or
  constraints for the whole conversation (e.g. "You are a terse code
  reviewer.").
- **user** — what the human is asking.
- **assistant** — the model's prior replies, included so the model has
  conversational memory across turns.

A "prompt template" is a reusable string with placeholders (e.g.
`"Answer the question using only this context:\n{context}\n\nQuestion: {question}"`)
that gets filled in with real values at run time. Templating prompts is the
first thing frameworks like LangChain standardize, because almost every
LLM application needs to combine a fixed instruction with variable data.

## Context window and tokens

Models don't read raw characters — text is broken into tokens (roughly
word-pieces), and every model has a maximum context window (a token budget
for prompt + conversation history + output). This limit is the core reason
RAG (see `03_rag_pipeline.md`) exists: you usually can't just paste an
entire document collection into a prompt, so you need a way to select only
the most relevant pieces before asking the model a question.

## Temperature and determinism

Generation is sampled from the model's output distribution; a `temperature`
parameter controls how random that sampling is. Temperature 0 is closest to
"always pick the most likely next token" (more deterministic, better for
factual/agentic tasks); higher temperature gives more varied, creative
output. Agents and tool-calling pipelines typically use low temperature
because you want reliable, predictable behavior, not creative variation.
