import streamlit as st
import re
import time
import pickle
from pathlib import Path

# --- LangChain Imports ---
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.stores import InMemoryStore
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- Retrieval Imports ---
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers.document_compressors import CrossEncoderReranker

import os
# Suppress tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- Config ---
from dotenv import load_dotenv
load_dotenv()

st.set_page_config(page_title="Clinical Trial RAG", layout="wide", page_icon="🧬")

# Configuration
LLM_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "NeuML/pubmedbert-base-embeddings"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DB_PATH = "data/chroma_db_advanced"
DOCSTORE_PATH = "data/docstore_advanced.pkl"
COLLECTION_NAME = "clinical_trials_advanced"

# --- CACHED RESOURCES ---
@st.cache_resource
def load_resources():
    """Loads the raw stores (VectorStore, DocStore, TextDocs) for dynamic retriever building."""
    print("📦 Loading retrieval system...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # 1. Docstore
    if not Path(DOCSTORE_PATH).exists():
        st.error(f"❌ Docstore not found at {DOCSTORE_PATH}")
        st.stop()
    with open(DOCSTORE_PATH, "rb") as f:
        store_dict = pickle.load(f)
    docstore = InMemoryStore()
    docstore.mset(list(store_dict.items()))

    # 2. Vector Store
    if not Path(DB_PATH).exists():
        st.error(f"❌ ChromaDB not found at {DB_PATH}")
        st.stop()
    vectorstore = Chroma(
        persist_directory=DB_PATH,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME
    )

    # 3. Text Docs for BM25
    all_text_docs = [doc for doc in store_dict.values() if doc.metadata.get("chunk_type") != "image"]

    return vectorstore, docstore, all_text_docs

try:
    vectorstore, docstore, all_text_docs = load_resources()
except Exception as e:
    st.error(f"Failed to load system: {e}")
    st.stop()

# --- UI LAYOUT ---
with st.sidebar:
    st.header("🗂️ Protocol Filters")
    st.caption("Filters apply to *Conceptual* queries only.")
    
    phases = ["Phase 1", "Phase 1b/2", "Phase 2", "Phase 3"]
    selected_phase = st.multiselect("Trial Phase", phases)
    
    st.divider()
    st.info(f"📚 **{vectorstore._collection.count()}** chunks indexed")

# --- DYNAMIC ROUTER (The Brain) ---
def detect_intent(query: str):
    """Matches your CLI script logic."""
    nct_match = re.search(r"(NCT\d{8})", query, re.IGNORECASE)
    if nct_match:
        return "trial_id", nct_match.group(1).upper()

    protocol_match = re.search(r"(\d{4}-\d{2}-\d{3}|[A-Z0-9]{2,5}-[A-Z0-9]{2,5}-[A-Z0-9]{3,5})", query, re.IGNORECASE)
    if protocol_match:
        return "protocol_id", protocol_match.group(1)

    return None, None

def get_dynamic_retriever(query, phase_filters=None):
    
    intent_type, intent_value = detect_intent(query)
    
    # Base Semantic
    semantic_retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        id_key="doc_id",
        search_kwargs={"k": 15}
    )

    # Reranker
    cross_encoder = HuggingFaceCrossEncoder(model_name=RERANKER_MODEL)
    reranker = CrossEncoderReranker(model=cross_encoder, top_n=5)

    # --- STRATEGY 1: STRICT MODE (ID DETECTED) ---
    if intent_type:
        strategy = f"🎯 {intent_type} Detected ({intent_value}) -> **Strict Filtering**"
        
        # Hard Filter
        semantic_retriever.search_kwargs["filter"] = {intent_type: intent_value}
        
        # Bypass Ensemble (Fixes BM25 Leak)
        final_retriever = ContextualCompressionRetriever(
            base_compressor=reranker, 
            base_retriever=semantic_retriever
        )
        
    # --- STRATEGY 2: HYBRID MODE (CONCEPTUAL) ---
    else:
        strategy = "🧠 Conceptual Query -> **Hybrid Search**"
        
        # Apply Sidebar Filters
        if phase_filters:
            filter_dict = {}
            if len(phase_filters) == 1:
                filter_dict["trial_phase"] = phase_filters[0]
            else:
                filter_dict["trial_phase"] = {"$in": phase_filters}
            semantic_retriever.search_kwargs["filter"] = filter_dict
        
        # Build Keyword Retriever on the fly
        keyword_retriever = BM25Retriever.from_documents(all_text_docs)
        keyword_retriever.k = 10

        # Ensemble
        ensemble = EnsembleRetriever(
            retrievers=[semantic_retriever, keyword_retriever],
            weights=[0.7, 0.3]
        )

        final_retriever = ContextualCompressionRetriever(
            base_compressor=reranker,
            base_retriever=ensemble
        )
    
    return final_retriever, strategy

# --- APP LOGIC ---
st.title("🧬 Clinical Trial Assistant")
st.caption(f"Powered by {LLM_MODEL} | PubMedBERT Embeddings | Dynamic Routing")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask about a protocol..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response_container = st.empty()

        # 1. Router Visualization
        with st.status("🔍 Analyzing Query...", expanded=True) as status:
            retriever, strategy = get_dynamic_retriever(prompt, selected_phase)
            st.write(f"**Router Decision:** {strategy}")
            
            st.write("📚 Retrieving & Reranking...")
            docs = retriever.invoke(prompt)
            st.write(f"✅ Found {len(docs)} highly relevant chunks")
            status.update(label="✅ Ready", state="complete", expanded=False)

        # 2. Generation
        template = """You are a Senior Clinical Research Associate.
Answer based ONLY on the provided context.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""
        prompt_template = ChatPromptTemplate.from_template(template)
        llm = ChatOpenAI(model=LLM_MODEL, temperature=0)

        def format_docs(documents):
            formatted = []
            for d in documents:
                meta = d.metadata
                src = f"[{meta.get('trial_id')}/{meta.get('pdf_stem')}]"
                formatted.append(f"{src}\n{d.page_content}")
            return "\n\n".join(formatted)

        chain = (
            {"context": lambda x: format_docs(docs), "question": RunnablePassthrough()}
            | prompt_template
            | llm
            | StrOutputParser()
        )

        response = chain.invoke(prompt)
        response_container.markdown(response)

        # 3. Evidence Display (WITH IMAGES)
        with st.expander("📚 View Retrieved Evidence"):
            for i, doc in enumerate(docs, 1):
                meta = doc.metadata
                st.markdown(f"**Doc {i}: {meta.get('pdf_stem', 'Unknown')}**")
                st.caption(f"Trial: {meta.get('trial_id', 'N/A')} | Type: {meta.get('chunk_type', 'text')}")
                
                # --- IMAGE LOGIC ---
                if meta.get("chunk_type") == "image":
                    # Try to find the image file
                    img_path = Path(meta.get("source", ""))
                    if img_path.exists():
                        st.image(str(img_path), caption=f"Figure from {meta.get('pdf_stem')}")
                    else:
                        st.warning(f"Image file not found: {img_path}")
                    st.info(f"**AI Description:** {doc.page_content}")
                else:
                    st.text(doc.page_content[:400] + "...")
                # -------------------
                st.divider()

    st.session_state.messages.append({"role": "assistant", "content": response})