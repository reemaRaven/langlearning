"""Stage 2: a plain LangChain/LCEL RAG chain (no LangGraph yet) — retrieve
relevant chunks, stuff them into a prompt, ask Claude.

The retrieval half works with no API key. Run:
    python -m src.rag_chain
It will always print retrieval results; it will only call Claude (and may
print a "no API key" message instead) for the generation half.
"""

import os
from pathlib import Path

from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from src.config import CHAT_MODEL_NAME, RETRIEVER_K
from src.ingest import load_vectorstore

RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a study assistant. Answer the question using ONLY the "
            "provided context. If the context doesn't contain the answer, "
            "say you don't know instead of guessing.",
        ),
        ("human", "Context:\n{context}\n\nQuestion: {question}"),
    ]
)


def get_retriever():
    vectorstore = load_vectorstore()
    return vectorstore.as_retriever(search_kwargs={"k": RETRIEVER_K})


def format_docs(docs):
    return "\n\n".join(
        f"[{Path(d.metadata.get('source', '?')).name}]\n{d.page_content}"
        for d in docs
    )


def build_rag_chain():
    """Full retrieve-then-answer chain. Calling .invoke() on this needs
    ANTHROPIC_API_KEY set."""
    retriever = get_retriever()
    model = ChatAnthropic(model=CHAT_MODEL_NAME, temperature=0)
    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | RAG_PROMPT
        | model
        | StrOutputParser()
    )


def main():
    question = "What is a retriever?"

    retriever = get_retriever()
    docs = retriever.invoke(question)
    print(f"Retrieval-only test for: {question!r} (no API key needed)")
    for i, doc in enumerate(docs, 1):
        source = Path(doc.metadata.get("source", "?")).name
        print(f"\n[{i}] from {source}:\n{doc.page_content[:200]}")

    print("\n" + "-" * 60)
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print(
            "ANTHROPIC_API_KEY not set — skipping the generation step.\n"
            "Copy .env.example to .env and add your key to try full answers."
        )
        return

    chain = build_rag_chain()
    answer = chain.invoke(question)
    print(f"\nGenerated answer:\n{answer}")


if __name__ == "__main__":
    main()
