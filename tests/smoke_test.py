"""Smoke tests that need no API key: verifies ingestion + retrieval work.

Run after `python -m src.ingest` has been run at least once:
    python -m pytest tests/smoke_test.py -v
"""

from src.ingest import load_vectorstore
from src.rag_chain import get_retriever
from src.tools import calculator, get_current_datetime


def test_vectorstore_has_documents():
    vectorstore = load_vectorstore()
    results = vectorstore.similarity_search("What is RAG?", k=1)
    assert len(results) == 1
    assert len(results[0].page_content) > 0


def test_retriever_returns_relevant_chunks():
    retriever = get_retriever()
    docs = retriever.invoke("What is a vector store used for?")
    assert len(docs) > 0
    assert any("vector" in d.page_content.lower() for d in docs)


def test_calculator_tool():
    assert calculator.invoke({"expression": "2 + 2"}) == "4"
    assert calculator.invoke({"expression": "6 * 7"}) == "42"


def test_calculator_rejects_unsafe_input():
    result = calculator.invoke({"expression": "__import__('os').system('echo hi')"})
    assert "Error" in result


def test_get_current_datetime_tool():
    result = get_current_datetime.invoke({})
    assert len(result) > 0
