"""Stage 1: load the knowledge-base docs, split into chunks, embed locally,
and persist to a Chroma vector store on disk.

Run standalone with no API key required:
    python -m src.ingest
"""

from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import (
    CHROMA_COLLECTION_NAME,
    CHROMA_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DOCS_DIR,
    EMBEDDING_MODEL_NAME,
)


def load_documents():
    loader = DirectoryLoader(
        str(DOCS_DIR),
        glob="*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    return loader.load()


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    return splitter.split_documents(documents)


def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


def build_vectorstore(chunks, embeddings):
    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=CHROMA_COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
    )


def load_vectorstore():
    """Open the already-persisted collection (used by rag_chain.py / graph.py).

    Run ingest.py's main() at least once before calling this.
    """
    return Chroma(
        collection_name=CHROMA_COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=str(CHROMA_DIR),
    )


def main():
    documents = load_documents()
    chunks = split_documents(documents)
    print(f"Loaded {len(documents)} documents -> split into {len(chunks)} chunks")

    embeddings = get_embeddings()
    vectorstore = build_vectorstore(chunks, embeddings)
    print(f"Persisted vector store to {CHROMA_DIR}")

    query = "What is a retriever?"
    results = vectorstore.similarity_search(query, k=2)
    print(f"\nSample similarity search for: {query!r}")
    for i, doc in enumerate(results, 1):
        source = Path(doc.metadata.get("source", "?")).name
        print(f"\n[{i}] from {source}:\n{doc.page_content[:300]}")


if __name__ == "__main__":
    main()
