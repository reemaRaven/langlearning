"""Central configuration: paths, model names, and .env loading.

Every other module in src/ imports its settings from here, so there's a
single place to change e.g. the embedding model or chunk size.
"""

from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = PROJECT_ROOT / "data" / "docs"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"
CHROMA_COLLECTION_NAME = "langlearning_docs"

# Local, free, no API key required.
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Requires ANTHROPIC_API_KEY to actually call.
CHAT_MODEL_NAME = "claude-sonnet-5"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 75
RETRIEVER_K = 4
