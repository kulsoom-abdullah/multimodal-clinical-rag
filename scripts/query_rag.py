#!/usr/bin/env python3
"""
Queries the multimodal RAG system.

This script loads the persisted ChromaDB vector store and the
pickled InMemoryStore, re-creates the MultiVectorRetriever,
and sets up a RAG chain to answer questions.

Usage:
    python scripts/query_rag.py "Your question here"
"""
import sys
import pickle
from pathlib import Path
from typing import List

# LangChain Imports (corrected for LangChain 1.0+)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.stores import InMemoryStore
from langchain_core.documents import Document
from langchain_classic.retrievers.multi_vector import MultiVectorRetriever

# Imports for the RAG chain (will be used in next iteration)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

# Utilities
from dotenv import load_dotenv

load_dotenv()

# Configuration constants
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "clinical_trials"
ID_KEY = "doc_id"
MAX_DOCS_TO_DISPLAY = 5  # Number of documents to show in output


def load_retriever() -> MultiVectorRetriever:
    """Load the persisted vector store and docstore to re-create the retriever.

    Returns:
        Configured MultiVectorRetriever instance.

    Raises:
        SystemExit: If required files (docstore.pkl or ChromaDB) are not found.
    """
    print("📦 Loading components...")

    # 1. Load Embedding Model (must be the same one used for ingestion)
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # 2. Load the persistent docstore
    docstore_path = Path("data/docstore.pkl")
    if not docstore_path.exists():
        print(f"❌ Error: docstore.pkl not found at {docstore_path}")
        print("Please run the ingestion script first.")
        sys.exit(1)

    with open(docstore_path, "rb") as f:
        store_dict = pickle.load(f)
    docstore = InMemoryStore()
    docstore.mset(list(store_dict.items()))

    # 3. Load the persistent vector store
    chroma_dir = Path("data/chroma_db")
    if not chroma_dir.exists():
        print(f"❌ Error: ChromaDB not found at {chroma_dir}")
        print("Please run the ingestion script first.")
        sys.exit(1)

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(chroma_dir)
    )

    # 4. Re-create the MultiVectorRetriever
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        id_key=ID_KEY,
    )
    print("✅ Retriever loaded successfully.")
    return retriever


def display_results(docs: List[Document]) -> None:
    """Display retrieved documents in a formatted way.

    Args:
        docs: List of retrieved documents to display.
    """
    print(f"\n📊 Found {len(docs)} documents:\n")

    for i, doc in enumerate(docs[:MAX_DOCS_TO_DISPLAY]):
        print(f"{'='*60}")
        print(f"Document {i+1}:")
        print(f"  Type: {doc.metadata.get('chunk_type', 'unknown')}")
        print(f"  Source: {doc.metadata.get('source', 'unknown')}")
        print(f"  Trial: {doc.metadata.get('trial_id', 'unknown')}")
        print(f"  Content preview: {doc.page_content[:200]}...")
        print(f"{'='*60}\n")

    if len(docs) > MAX_DOCS_TO_DISPLAY:
        print(f"... and {len(docs) - MAX_DOCS_TO_DISPLAY} more documents\n")


def main() -> None:
    """Main entry point for the query script."""
    retriever = load_retriever()

    # We'll take the question from the command line
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        print(f"\n❓ Query: {question}")

        # Test the retriever
        print("\n🔍 Retriever test output:")
        docs = retriever.invoke(question)

        display_results(docs)

    else:
        print("Usage: python scripts/query_rag.py \"Your question here\"")


if __name__ == "__main__":
    main()
