# RAG: Retrieval-Augmented Generation

RAG means: before asking an LLM to answer a question, first retrieve
relevant text from an external knowledge source and insert it into the
prompt as context. It exists to solve two problems plain LLMs have: they
don't know about your private/recent data (their knowledge is frozen at
training time), and you can't just paste your whole document collection
into a prompt because of the context window limit (see
`01_llms_and_prompts.md`).

## The pipeline, stage by stage

1. **Load** — read source documents (PDFs, markdown, web pages, database
   rows) into a common `Document` object (text + metadata).
2. **Split / chunk** — break long documents into smaller pieces (e.g. a few
   hundred tokens each, usually with some overlap between consecutive
   chunks so context isn't lost at a boundary). This project uses
   LangChain's `RecursiveCharacterTextSplitter`, which tries to split on
   paragraph/sentence boundaries before falling back to raw character
   counts, so chunks stay semantically coherent.
3. **Embed** — run each chunk through an embedding model, which turns text
   into a fixed-length vector of numbers such that semantically similar
   text produces vectors that are close together in that vector space.
   This project uses a local `sentence-transformers` model
   (`all-MiniLM-L6-v2`) via `langchain-huggingface`, run entirely on your
   machine — no API key or network call needed at query time.
4. **Store** — persist the chunks and their vectors in a vector database
   (this project uses Chroma, running locally on disk) that supports fast
   nearest-neighbor search.
5. **Retrieve** — at query time, embed the user's question with the same
   embedding model, then ask the vector store for the `k` chunks whose
   vectors are closest to the question's vector (cosine similarity or
   similar). This is "semantic search": it matches on meaning, not just
   keyword overlap.
6. **Generate** — stuff the retrieved chunks into a prompt template
   alongside the original question, and send that to the LLM (Claude, in
   this project) to produce a grounded answer.

## Why the embedding model doesn't have to match the chat model

Embeddings and text generation are different tasks with different models.
Anthropic doesn't publish an embeddings API at all, which is exactly why
this project uses a local HuggingFace embedding model for steps 3 and 5,
while using Claude only for the final generation step. Mixing providers
like this is common and is exactly the kind of thing LangChain's
standardized interfaces make painless.

## Failure modes worth knowing

- **Irrelevant retrieval**: if the top-k chunks don't actually contain the
  answer, the model will either say so (good) or hallucinate an answer
  anyway (bad) — prompting the model to admit "I don't know" when context
  is insufficient helps.
- **Chunking too coarse or too fine**: chunks too large dilute relevance
  (a big chunk about many topics scores lower for any one of them); chunks
  too small lose surrounding context.
- **Stale index**: the vector store only knows what you last ingested — if
  source documents change, you need to re-run ingestion.
