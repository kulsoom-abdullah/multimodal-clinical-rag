import streamlit as st
import re
import time
import pickle
import os
from pathlib import Path

# --- LangChain Imports ---
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.stores import InMemoryStore
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- Retrieval Imports (RunPod Optimized) ---
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import (
    EnsembleRetriever,
    ContextualCompressionRetriever,
)
from langchain_classic.retrievers.multi_vector import MultiVectorRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker

# --- Config ---
from dotenv import load_dotenv

import sys

sys.path.append(str(Path(__file__).resolve().parent))
from scripts.router import (  # noqa: E402
    ANSWER_PROMPT,
    NO_CONTEXT_MESSAGE,
    STRICT,
    UNKNOWN_ID,
    decide_route,
    load_known_ids,
)

load_dotenv()

# Suppress tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

st.set_page_config(page_title="Clinical Trial RAG", layout="wide", page_icon="🧬")

# --- DEBUG BLOCK ---
# import os
# st.sidebar.error(f"Current Working Dir: {os.getcwd()}")
# if os.path.exists("output"):
#     st.sidebar.success(f"Output folder exists! Contains: {len(os.listdir('output'))} items")
# else:
#     st.sidebar.error("Output folder NOT FOUND in current directory.")

# --- MODEL CONFIGURATION ---
# Option A: Speed & Cost (Default)
LLM_MODEL = "gpt-5-mini"

# Option B: Maximum Reasoning (Uncomment to use)
# LLM_MODEL = "claude-sonnet-4-5-20250929"

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
        collection_name=COLLECTION_NAME,
    )

    # 3. Text Docs for BM25
    all_text_docs = [
        doc for doc in store_dict.values() if doc.metadata.get("chunk_type") != "image"
    ]

    return vectorstore, docstore, all_text_docs


@st.cache_resource
def load_query_time_resources(_vectorstore, _all_text_docs):
    """Reranker, BM25 index and identifier whitelist.

    These were previously rebuilt on every chat message: the cross-encoder was
    re-instantiated and the BM25 index re-built over every text chunk per query.
    """
    cross_encoder = HuggingFaceCrossEncoder(model_name=RERANKER_MODEL)
    reranker = CrossEncoderReranker(model=cross_encoder, top_n=5)

    keyword_retriever = BM25Retriever.from_documents(_all_text_docs)
    keyword_retriever.k = 10

    known_ids = load_known_ids(_vectorstore)
    return reranker, keyword_retriever, known_ids


@st.cache_resource
def load_indexed_phases(_vectorstore):
    """Phase values actually present in the index.

    Hardcoding this list let the sidebar offer a phase that matched zero documents.
    """
    collection = _vectorstore._collection
    total = collection.count()
    phases, offset = set(), 0
    while offset < total:
        batch = collection.get(include=["metadatas"], limit=2000, offset=offset)
        for meta in batch["metadatas"]:
            value = meta.get("trial_phase")
            if value and value != "UNKNOWN":
                phases.add(value)
        offset += 2000
    return sorted(phases)


try:
    vectorstore, docstore, all_text_docs = load_resources()
    reranker, keyword_retriever, known_ids = load_query_time_resources(
        vectorstore, all_text_docs
    )
except Exception as e:
    st.error(f"Failed to load system: {e}")
    st.stop()

# --- UI LAYOUT ---
with st.sidebar:
    st.header("🗂️ Context Selector")

    # --- FEATURE 1: The Study Selector (Synced with DB) ---
    # Labels are derived from each document's extracted intervention_drug / indication
    # and confirmed by grep against the source text — not written alongside the IDs.
    # The two 205801 rows are one master protocol under two registrations; they are kept
    # adjacent so the shared identity is visible without reading the metadata.
    TRIAL_MAP = {
        "🌎 All Trials (Global Search)": None,
        "NCT02423343 — Galunisertib + Nivolumab (Ph 1b/2, solid tumors)": "NCT02423343",
        "NCT02578680 — KEYNOTE-189: Pembrolizumab + chemo (Ph 3, NSCLC)": "NCT02578680",
        "NCT05600322 — Hexvix blue-light cystoscopy (bladder)": "NCT05600322",
        "NCT05613088 — CA116001: Farletuzumab ecteribulin / MORAb-202 (Ph 2, ovarian)": "NCT05613088",
        "NCT05751629 — 3000-02-005: Dostarlimab + bevacizumab + niraparib (Ph 2, ovarian)": "NCT05751629",
        # --- Protocol 205801 (GSK NSCLC master protocol), two registrations ---
        "NCT05553808 — 205801 Amd 08: NSCLC master protocol (Ph 2)": "NCT05553808",
        "NCT06926673 — 205801 Sub-study 3 SAP (Ph 2, NSCLC)": "NCT06926673",
    }
    selected_trial_name = st.selectbox(
        "Select Clinical Trial",
        options=list(TRIAL_MAP.keys()),
        index=0,
        help="Restricts search to a specific protocol.",
    )
    selected_trial_id = TRIAL_MAP[selected_trial_name]

    st.divider()

    st.header("⚡ Quick Filters")
    st.caption("Applies only to Global Search")
    selected_phase = st.multiselect("Trial Phase", load_indexed_phases(vectorstore))

    st.divider()
    st.info(f"📚 **{vectorstore._collection.count()}** chunks indexed")


# --- DYNAMIC ROUTER (The Brain) ---
def _phase_filtered_docs(docs, phase_filters):
    """Apply the phase filter to BM25 results.

    The filter used to be attached only to the semantic retriever's search_kwargs,
    so the keyword half of the ensemble returned out-of-phase documents regardless
    of the sidebar selection.
    """
    if not phase_filters:
        return docs
    allowed = set(phase_filters)
    return [d for d in docs if d.metadata.get("trial_phase") in allowed]


def get_dynamic_retriever(query, phase_filters=None, trial_filter=None):
    """
    Args:
        trial_filter (str): If provided (from Sidebar), forces a Hard Filter.
    """
    decision = decide_route(query, known_ids)

    # Base Semantic
    semantic_retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        id_key="doc_id",
        search_kwargs={"k": 15},
    )

    # --- PRIORITY 1: SIDEBAR SELECTION (The "Context Switch") ---
    if trial_filter:
        strategy = f"🧬 **Context Locked:** {trial_filter}"
        if decision.route == STRICT and decision.filter_value != trial_filter:
            # An ID in the query that disagrees with the sidebar lock used to be
            # silently ignored, answering about a trial the user did not name.
            strategy += (
                f" — ignoring `{decision.filter_value}` from your query; "
                "clear the sidebar selection to search it."
            )
        semantic_retriever.search_kwargs["filter"] = {"trial_id": trial_filter}
        # Use Strict Semantic (No BM25 needed as scope is small)
        base_retriever = semantic_retriever

    # --- PRIORITY 2: QUERY ID DETECTION (Strict Mode) ---
    elif decision.route == STRICT:
        strategy = (
            f"🎯 {decision.filter_key} Detected ({decision.filter_value}) "
            "-> **Strict Filtering**"
        )
        semantic_retriever.search_kwargs["filter"] = {
            decision.filter_key: decision.filter_value
        }
        base_retriever = semantic_retriever

    # --- PRIORITY 2b: NAMED BUT UNINDEXED TRIAL (no search is performed) ---
    elif decision.route == UNKNOWN_ID:
        strategy = (
            f"🚫 `{decision.rejected_candidate}` is not in this index "
            "-> **No Search Performed**"
        )
        base_retriever = semantic_retriever  # unused; the caller short-circuits

    # --- PRIORITY 3: GLOBAL HYBRID SEARCH ---
    else:
        strategy = "🧠 Global Query -> **Hybrid Search**"

        # Apply Phase Filters to the semantic half via metadata filtering...
        if phase_filters:
            filter_dict = {}
            if len(phase_filters) == 1:
                filter_dict["trial_phase"] = phase_filters[0]
            else:
                filter_dict["trial_phase"] = {"$in": phase_filters}
            semantic_retriever.search_kwargs["filter"] = filter_dict

        # Ensemble (BM25 retriever is cached; see load_query_time_resources)
        base_retriever = EnsembleRetriever(
            retrievers=[semantic_retriever, keyword_retriever], weights=[0.7, 0.3]
        )

    # Final Reranking Step
    final_retriever = ContextualCompressionRetriever(
        base_compressor=reranker, base_retriever=base_retriever
    )

    return final_retriever, strategy, decision


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
            # PASS THE NEW SIDEBAR FILTER
            retriever, strategy, decision = get_dynamic_retriever(
                prompt, selected_phase, selected_trial_id
            )

            st.write(f"**Router Decision:** {strategy}")

            if decision.route == UNKNOWN_ID and not selected_trial_id:
                docs = []
                st.write(f"⚠️ {decision.reason}")
            else:
                st.write("📚 Retrieving & Reranking...")
                docs = retriever.invoke(prompt)
                # BM25 does not honour metadata filters, so enforce the phase
                # selection on the merged result set.
                if not selected_trial_id and decision.route != STRICT:
                    docs = _phase_filtered_docs(docs, selected_phase)
                st.write(f"✅ Retrieved {len(docs)} chunks")
            status.update(label="✅ Ready", state="complete", expanded=False)

        # 2. Generation — refuse rather than prompting the model with no evidence.
        if not docs:
            if decision.route == UNKNOWN_ID and not selected_trial_id:
                response = (
                    f"**{decision.rejected_candidate}** is not present in this index. "
                    "No answer is possible from the indexed protocols."
                )
            else:
                response = NO_CONTEXT_MESSAGE
            response_container.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.stop()

        prompt_template = ChatPromptTemplate.from_template(ANSWER_PROMPT)

        # --- LLM SELECTION LOGIC ---
        if "claude" in LLM_MODEL:
            llm = ChatAnthropic(model=LLM_MODEL, temperature=0)
        else:
            llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
        # ---------------------------

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

        # --- NEW: Auto-Display Images in Chat ---
        # If an image was retrieved, show it directly!
        for doc in docs:
            if doc.metadata.get("chunk_type") == "image":
                # Reuse the path finding logic
                img_filename = doc.metadata.get("filename")
                if not img_filename:
                     # Fallback logic
                     raw_path = doc.metadata.get("source", "")
                     img_filename = os.path.basename(raw_path)

                # Construct paths again (quick check)
                trial_id = doc.metadata.get("trial_id", "Unknown")
                pdf_stem = doc.metadata.get("pdf_stem", "Unknown")
                
                possible_paths = [
                    Path(f"output/{trial_id}/{pdf_stem}/images/{img_filename}"),
                    Path(f"output/{trial_id}/{img_filename}"),
                    Path(img_filename)
                ]
                
                for p in possible_paths:
                    if p.exists():
                        st.image(str(p), caption=f"🖼️ Retrieved Figure: {img_filename}")
                        break
        # ----------------------------------------

        # 3. Evidence Display (Keep this for detailed inspection)
        with st.expander("📚 View Retrieved Evidence"):
            for i, doc in enumerate(docs, 1):
                meta = doc.metadata
                st.markdown(f"**Doc {i}: {meta.get('pdf_stem', 'Unknown')}**")
                st.caption(
                    f"Trial: {meta.get('trial_id', 'N/A')} | Type: {meta.get('chunk_type', 'text')}"
                )

                if meta.get("chunk_type") == "image":
                    # 1. Try to get the direct filename from metadata (Best Method)
                    img_filename = meta.get("filename")
                    
                    # 2. Fallback: If missing, try to parse from source or text
                    if not img_filename:
                        raw_path = meta.get("source", "")
                        img_filename = os.path.basename(raw_path)
                        if img_filename.lower().endswith(".md"):
                            # Try to find image name in the description
                            match = re.search(r"([a-zA-Z0-9_]+\.(?:jpeg|jpg|png))", doc.page_content, re.IGNORECASE)
                            if match:
                                img_filename = match.group(1)

                    # 3. Construct Paths (Docker & Local)
                    # Your output structure is likely: output/TRIAL_ID/PDF_STEM/images/FILE.jpg
                    trial_id = meta.get("trial_id", "Unknown")
                    pdf_stem = meta.get("pdf_stem", "Unknown")
                    
                    possible_paths = [
                        # Docker specific standard path
                        Path(f"/app/output/{trial_id}/{pdf_stem}/images/{img_filename}"),
                        # Local/Relative standard path
                        Path(f"output/{trial_id}/{pdf_stem}/images/{img_filename}"),
                        # Fallback: Flat structure
                        Path(f"output/{trial_id}/{img_filename}"),
                        # Fallback: Direct filename
                        Path(img_filename)
                    ]
                    
                    valid_img_path = None
                    for p in possible_paths:
                        if p.exists():
                            valid_img_path = p
                            break
                            
                    if valid_img_path:
                        try:
                            st.image(str(valid_img_path), caption=f"Figure: {img_filename}")
                        except Exception as e:
                            st.error(f"⚠️ Display Error: {e}")
                    else:
                        # Debugging Helper: Show what we looked for
                        st.warning(f"⚠️ Image '{img_filename}' not found on disk.")
                        # st.caption(f"Searched paths: {[str(p) for p in possible_paths]}")

                    st.info(f"**AI Analysis:** {doc.page_content}")
                else:
                    st.text(doc.page_content[:400] + "...")
                st.divider()

    st.session_state.messages.append({"role": "assistant", "content": response})
