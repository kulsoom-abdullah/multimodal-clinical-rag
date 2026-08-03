#!/usr/bin/env python3
"""
Queries the multimodal RAG system using a "Dynamic Router" architecture.
NOW UPDATED: Implements "Strict Mode" to prevent BM25 Leaks on ID queries.

Usage:
    python scripts/query_rag.py "Your question here"
"""
import sys
import pickle
from pathlib import Path
from typing import List

# --- LangChain Imports ---
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.stores import InMemoryStore
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- Retrieval Imports ---
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import (
    EnsembleRetriever,
    ContextualCompressionRetriever,
)
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_classic.retrievers.multi_vector import MultiVectorRetriever

import os

sys.path.append(str(Path(__file__).resolve().parent.parent))
from scripts.router import (  # noqa: E402
    ANSWER_PROMPT,
    NO_CONTEXT_MESSAGE,
    STRICT,
    UNKNOWN_ID,
    decide_route,
    detect_intent,  # noqa: F401  (re-exported for existing callers)
    load_known_ids,
)

# Suppress tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- Utilities ---
from dotenv import load_dotenv

load_dotenv()


# Configuration
EMBEDDING_MODEL = "NeuML/pubmedbert-base-embeddings"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
COLLECTION_NAME = "clinical_trials_advanced"
ID_KEY = "doc_id"

# Paths
CHROMA_DIR = Path("data/chroma_db_advanced")
DOCSTORE_PATH = Path("data/docstore_advanced.pkl")


# Routing and the answer prompt live in scripts/router.py so the CLI and the Streamlit
# app cannot drift apart again. detect_intent is re-exported for existing callers.
_KNOWN_IDS_CACHE: dict = {}


def get_known_ids(vectorstore):
    """Identifier whitelist for this index, computed once per vectorstore."""
    key = id(vectorstore)
    if key not in _KNOWN_IDS_CACHE:
        _KNOWN_IDS_CACHE[key] = load_known_ids(vectorstore)
    return _KNOWN_IDS_CACHE[key]


def load_resources():
    """
    Loads the raw components (VectorStore, DocStore, Documents) needed to build retrievers.
    Unlike before, this returns the STORES, not the pre-built retrievers.
    """
    print("📦 Loading retrieval resources...")

    if not DOCSTORE_PATH.exists() or not CHROMA_DIR.exists():
        print(f"❌ Error: Data not found at {DOCSTORE_PATH} or {CHROMA_DIR}")
        sys.exit(1)

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    with open(DOCSTORE_PATH, "rb") as f:
        store_dict = pickle.load(f)
    docstore = InMemoryStore()
    docstore.mset(list(store_dict.items()))

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )

    # Get text docs for BM25 (filtering out images)
    all_text_docs = [
        doc for doc in store_dict.values() if doc.metadata.get("chunk_type") != "image"
    ]

    return vectorstore, docstore, all_text_docs


def build_dynamic_retriever(query: str, vectorstore, docstore, all_text_docs):
    """
    Builds the retriever pipeline dynamically based on the query intent.

    STRATEGIES:
    1. STRICT MODE (ID Detected): Uses ONLY Semantic Search with a HARD Metadata Filter.
    2. HYBRID MODE (Conceptual): Uses Semantic + Keyword (BM25) + Reranking.
    """
    decision = decide_route(query, get_known_ids(vectorstore))

    # Base Semantic Retriever (MultiVector)
    semantic_retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        id_key=ID_KEY,
        search_kwargs={"k": 15},  # Fetch more candidates for reranking
    )

    # Initialize Reranker
    cross_encoder = HuggingFaceCrossEncoder(model_name=RERANKER_MODEL)
    reranker = CrossEncoderReranker(model=cross_encoder, top_n=5)

    # --- ROUTING DECISION ---

    if decision.route == STRICT:
        print(f"\n   🔀 ROUTER: {decision.reason}")
        print("   🛡️  Strategy: STRICT MODE (Hard Metadata Filter, No BM25)")

        # Apply Hard Filter to Semantic Search
        semantic_retriever.search_kwargs["filter"] = {
            decision.filter_key: decision.filter_value
        }

        # In Strict Mode, we bypass the Ensemble.
        # We trust the filter 100% and just rerank the semantic results.
        final_retriever = ContextualCompressionRetriever(
            base_compressor=reranker, base_retriever=semantic_retriever
        )

    else:
        print(f"\n   🔀 ROUTER: {decision.reason}")
        print("   🧠 Strategy: HYBRID SEARCH (Semantic + BM25)")

        # Create BM25 only if needed (saves compute if strictly ID based, though negligible here)
        keyword_retriever = BM25Retriever.from_documents(all_text_docs)
        keyword_retriever.k = 10

        # Ensemble: Mix Semantic (70%) and Keyword (30%)
        ensemble_retriever = EnsembleRetriever(
            retrievers=[semantic_retriever, keyword_retriever], weights=[0.7, 0.3]
        )

        final_retriever = ContextualCompressionRetriever(
            base_compressor=reranker, base_retriever=ensemble_retriever
        )

    return final_retriever


def format_docs_for_prompt(docs: List[Document]) -> str:
    """Format docs for the LLM."""
    formatted_docs = []
    for i, doc in enumerate(docs):
        metadata = doc.metadata
        source_info = (
            f"Source: {metadata.get('trial_id', 'Unknown')}/{metadata.get('pdf_stem', 'Unknown')}"
            f" (Type: {metadata.get('chunk_type')})"
        )
        content_preview = doc.page_content.strip()

        formatted_docs.append(
            f"--- DOCUMENT {i+1} ---\n"
            f"{source_info}\n"
            f"{content_preview}\n"
            f"-----------------------"
        )
    return "\n\n".join(formatted_docs)


def main() -> None:
    # 1. Load Resources (Vectorstore, Docstore, etc.)
    vectorstore, docstore, all_text_docs = load_resources()

    llm = ChatOpenAI(model="gpt-5-mini", temperature=0)

    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        print(f"\n❓ Query: {question}")

        # 2. A named-but-unindexed trial is answerable without retrieving anything.
        decision = decide_route(question, get_known_ids(vectorstore))
        if decision.route == UNKNOWN_ID:
            print(f"\n   🔀 ROUTER: {decision.reason}")
            print("\n" + "=" * 60)
            print("✅ FINAL ANSWER:")
            print("=" * 60)
            print(
                f"{decision.rejected_candidate} is not present in this index. "
                "No answer is possible from the indexed protocols."
            )
            print("=" * 60 + "\n")
            return

        # 3. Build the specific retriever for this query and retrieve.
        retriever = build_dynamic_retriever(
            question, vectorstore, docstore, all_text_docs
        )
        docs = retriever.invoke(question)

        # 4. Refuse rather than prompting the model with an empty context: with no
        #    evidence the model answers from priors and still emits a citation.
        if not docs:
            print("\n   ⚠️  Retrieved 0 documents.")
            print("\n" + "=" * 60)
            print("✅ FINAL ANSWER:")
            print("=" * 60)
            print(NO_CONTEXT_MESSAGE)
            print("=" * 60 + "\n")
            return

        print(f"🔗 Building RAG chain over {len(docs)} chunks...")
        prompt = ChatPromptTemplate.from_template(ANSWER_PROMPT)

        rag_chain = (
            {
                "context": lambda _: format_docs_for_prompt(docs),
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm
            | StrOutputParser()
        )

        print("🤖 Generating answer...")
        answer = rag_chain.invoke(question)

        print("\n" + "=" * 60)
        print("✅ FINAL ANSWER:")
        print("=" * 60)
        print(answer)
        print("=" * 60 + "\n")

    else:
        print('Usage: python scripts/query_rag.py "Your question here"')


if __name__ == "__main__":
    main()
