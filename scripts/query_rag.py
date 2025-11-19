#!/usr/bin/env python3
"""
Queries the multimodal RAG system using a "Dynamic Router" architecture.
NOW UPDATED: Implements "Strict Mode" to prevent BM25 Leaks on ID queries.

Usage:
    python scripts/query_rag.py "Your question here"
"""
import sys
import pickle
import re
from pathlib import Path
from typing import List, Tuple, Optional

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
from langchain_classic.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_classic.retrievers.multi_vector import MultiVectorRetriever

import os
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


def detect_intent(query: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Router Logic:
    Analyzes the query to see if it references a specific Trial ID (NCT)
    or an internal Protocol ID.
    
    Returns:
        (intent_type, value) -> e.g., ("trial_id", "NCT01234567")
    """
    # Regex for NCT IDs (e.g., NCT02423343)
    nct_match = re.search(r"(NCT\d{8})", query, re.IGNORECASE)
    if nct_match:
        return "trial_id", nct_match.group(1).upper()

    # Regex for Protocol IDs (e.g., 3000-02-005 or H9H-MC-JBEF)
    # Catches standard formats appearing in headers
    protocol_match = re.search(r"(\d{4}-\d{2}-\d{3}|[A-Z0-9]{2,5}-[A-Z0-9]{2,5}-[A-Z0-9]{3,5})", query, re.IGNORECASE)
    if protocol_match:
        return "protocol_id", protocol_match.group(1)

    return None, None


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
        persist_directory=str(CHROMA_DIR)
    )
    
    # Get text docs for BM25 (filtering out images)
    all_text_docs = [doc for doc in store_dict.values() if doc.metadata.get("chunk_type") != "image"]

    return vectorstore, docstore, all_text_docs


def build_dynamic_retriever(query: str, vectorstore, docstore, all_text_docs):
    """
    Builds the retriever pipeline dynamically based on the query intent.
    
    STRATEGIES:
    1. STRICT MODE (ID Detected): Uses ONLY Semantic Search with a HARD Metadata Filter.
    2. HYBRID MODE (Conceptual): Uses Semantic + Keyword (BM25) + Reranking.
    """
    intent_type, intent_value = detect_intent(query)
    
    # Base Semantic Retriever (MultiVector)
    semantic_retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        id_key=ID_KEY,
        search_kwargs={"k": 15} # Fetch more candidates for reranking
    )

    # Initialize Reranker
    cross_encoder = HuggingFaceCrossEncoder(model_name=RERANKER_MODEL)
    reranker = CrossEncoderReranker(model=cross_encoder, top_n=5)

    # --- ROUTING DECISION ---

    if intent_type:
        print(f"\n   🔀 ROUTER: Detected {intent_type} -> '{intent_value}'")
        print("   🛡️  Strategy: STRICT MODE (Hard Metadata Filter, No BM25)")
        
        # Apply Hard Filter to Semantic Search
        semantic_retriever.search_kwargs["filter"] = {intent_type: intent_value}
        
        # In Strict Mode, we bypass the Ensemble. 
        # We trust the filter 100% and just rerank the semantic results.
        final_retriever = ContextualCompressionRetriever(
            base_compressor=reranker, 
            base_retriever=semantic_retriever
        )
        
    else:
        print("\n   🔀 ROUTER: No ID detected.")
        print("   🧠 Strategy: HYBRID SEARCH (Semantic + BM25)")
        
        # Create BM25 only if needed (saves compute if strictly ID based, though negligible here)
        keyword_retriever = BM25Retriever.from_documents(all_text_docs)
        keyword_retriever.k = 10
        
        # Ensemble: Mix Semantic (70%) and Keyword (30%)
        ensemble_retriever = EnsembleRetriever(
            retrievers=[semantic_retriever, keyword_retriever],
            weights=[0.7, 0.3]
        )
        
        final_retriever = ContextualCompressionRetriever(
            base_compressor=reranker, 
            base_retriever=ensemble_retriever
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

        # 2. Build the specific retriever for this query
        retriever = build_dynamic_retriever(question, vectorstore, docstore, all_text_docs)

        print("🔗 Building RAG chain...")
        template = """
        You are a Senior Clinical Research Associate (CRA) assisting with protocol verification.
        Your task is to answer the user's question based *strictly* on the provided context.

        GUIDELINES:
        1. **Evidence-Based:** Answer only using the provided chunks. Do not use outside knowledge.
        2. **Hierarchy of Data:** If you see multiple versions of a protocol (e.g., v1.0 vs v2.0 or Amendment), prioritize the latest version.
        3. **Safety First:** If the user asks about Dosing, Exclusion Criteria, or Safety Signals, quote the text directly or be extremely precise.
        4. **Formatting:** Use bullet points for lists. Use **bold** for key metrics.
        5. **Uncertainty:** If the documents do not contain the exact answer, state that clearly.

        CONTEXT:
        {context}

        QUESTION:
        {question}

        ANSWER (Structured for a Clinical Audience):
        """

        prompt = ChatPromptTemplate.from_template(template)

        rag_chain = (
            {
                "context": retriever | format_docs_for_prompt,
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm
            | StrOutputParser()
        )

        print("🤖 Generating answer...")
        answer = rag_chain.invoke(question)

        print("\n" + "="*60)
        print("✅ FINAL ANSWER:")
        print("="*60)
        print(answer)
        print("="*60 + "\n")

    else:
        print("Usage: python scripts/query_rag.py \"Your question here\"")


if __name__ == "__main__":
    main()