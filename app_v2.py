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

load_dotenv()

# Suppress tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

st.set_page_config(page_title="Clinical Trial RAG", layout="wide", page_icon="🧬")

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


try:
    vectorstore, docstore, all_text_docs = load_resources()
except Exception as e:
    st.error(f"Failed to load system: {e}")
    st.stop()

# --- UI LAYOUT ---
with st.sidebar:
    st.header("🗂️ Context Selector")

    # --- FEATURE 1: The Study Selector (Synced with DB) ---
    TRIAL_MAP = {
        "🌎 All Trials (Global Search)": None,
        # Validated IDs from your Database
        "NCT02423343 (Galunisertib + Nivo)": "NCT02423343",
        "NCT02578680 (KEYNOTE-189 / NSCLC)": "NCT02578680",
        "NCT05751629 (MORAb-202 / Ovarian)": "NCT05751629",
        "NCT05600322 (Hexvix Blue Light)": "NCT05600322",
        # The 3 previously missing trials:
        "NCT05553808 (Liso-cel / CLL)": "NCT05553808",
        "NCT05613088 (Talazoparib / PARP)": "NCT05613088",
        "NCT06926673 (Check Name in PDF)": "NCT06926673",
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

    protocol_match = re.search(
        r"(\d{4}-\d{2}-\d{3}|[A-Z0-9]{2,5}-[A-Z0-9]{2,5}-[A-Z0-9]{3,5})",
        query,
        re.IGNORECASE,
    )
    if protocol_match:
        return "protocol_id", protocol_match.group(1)

    return None, None


def get_dynamic_retriever(query, phase_filters=None, trial_filter=None):
    """
    Args:
        trial_filter (str): If provided (from Sidebar), forces a Hard Filter.
    """
    intent_type, intent_value = detect_intent(query)

    # Base Semantic
    semantic_retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        id_key="doc_id",
        search_kwargs={"k": 15},
    )

    # Reranker
    cross_encoder = HuggingFaceCrossEncoder(model_name=RERANKER_MODEL)
    reranker = CrossEncoderReranker(model=cross_encoder, top_n=5)

    # --- PRIORITY 1: SIDEBAR SELECTION (The "Context Switch") ---
    if trial_filter:
        strategy = f"🧬 **Context Locked:** {trial_filter}"
        # Apply Hard Filter
        semantic_retriever.search_kwargs["filter"] = {"trial_id": trial_filter}
        # Use Strict Semantic (No BM25 needed as scope is small)
        base_retriever = semantic_retriever

    # --- PRIORITY 2: QUERY ID DETECTION (Strict Mode) ---
    elif intent_type:
        strategy = f"🎯 {intent_type} Detected ({intent_value}) -> **Strict Filtering**"
        semantic_retriever.search_kwargs["filter"] = {intent_type: intent_value}
        base_retriever = semantic_retriever

    # --- PRIORITY 3: GLOBAL HYBRID SEARCH ---
    else:
        strategy = "🧠 Global Query -> **Hybrid Search**"

        # Apply Phase Filters
        if phase_filters:
            filter_dict = {}
            if len(phase_filters) == 1:
                filter_dict["trial_phase"] = phase_filters[0]
            else:
                filter_dict["trial_phase"] = {"$in": phase_filters}
            semantic_retriever.search_kwargs["filter"] = filter_dict

        # Build Keyword Retriever
        keyword_retriever = BM25Retriever.from_documents(all_text_docs)
        keyword_retriever.k = 10

        # Ensemble
        base_retriever = EnsembleRetriever(
            retrievers=[semantic_retriever, keyword_retriever], weights=[0.7, 0.3]
        )

    # Final Reranking Step
    final_retriever = ContextualCompressionRetriever(
        base_compressor=reranker, base_retriever=base_retriever
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
            # PASS THE NEW SIDEBAR FILTER
            retriever, strategy = get_dynamic_retriever(
                prompt, selected_phase, selected_trial_id
            )

            st.write(f"**Router Decision:** {strategy}")

            st.write("📚 Retrieving & Reranking...")
            docs = retriever.invoke(prompt)
            st.write(f"✅ Found {len(docs)} highly relevant chunks")
            status.update(label="✅ Ready", state="complete", expanded=False)

        # 2. Generation
        # --- UPGRADED PROMPT ---
        template = """You are a Senior Clinical Research Associate (CRA) assisting with protocol verification.
Your task is to answer the user's question based *strictly* on the provided context.

GUIDELINES:
1. **Evidence-Based:** Answer only using the provided chunks. Do not use outside knowledge.
2. **Hierarchy of Data:** If you see multiple versions (e.g., Protocol v1.0 vs Amendment 2), prioritize the LATEST information.
3. **Safety First:** If the user asks about Dosing, Exclusion Criteria, or Toxicity Management, quote the text/values exactly.
4. **Visuals:** If the context includes an image description (e.g., "Figure 1", "Flowchart"), refer to it in your answer.
5. **Citations:** End your answer with the specific Source Documents used (e.g., "Source: SAP_001.pdf").

CONTEXT:
{context}

QUESTION:
{question}

ANSWER (Structured for a Clinical Audience):"""

        prompt_template = ChatPromptTemplate.from_template(template)

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

        # 3. Evidence Display (Robust Path Handling)
        with st.expander("📚 View Retrieved Evidence"):
            for i, doc in enumerate(docs, 1):
                meta = doc.metadata
                st.markdown(f"**Doc {i}: {meta.get('pdf_stem', 'Unknown')}**")
                st.caption(
                    f"Trial: {meta.get('trial_id', 'N/A')} | Type: {meta.get('chunk_type', 'text')}"
                )

                if meta.get("chunk_type") == "image":
                    # --- ROBUST IMAGE LOADING ---
                    raw_path = meta.get("source", "")
                    img_filename = os.path.basename(raw_path)

                    # FIX: If metadata points to a .md file, extract the real image name from the text content
                    if img_filename.lower().endswith(".md"):
                        # Look for pattern: _page_116_Figure_2.jpeg inside the content
                        match = re.search(
                            r"(_page_.*?\.(?:jpeg|jpg|png))",
                            doc.page_content,
                            re.IGNORECASE,
                        )
                        if match:
                            img_filename = match.group(1)

                    # Check paths
                    possible_paths = [
                        # 1. The Standard Structure: output/TRIAL/STEM/images/FILE
                        Path(
                            f"output/{meta.get('trial_id')}/{meta.get('pdf_stem')}/images/{img_filename}"
                        ),
                        # 2. Flat output: output/TRIAL/FILE
                        Path(f"output/{meta.get('trial_id')}/{img_filename}"),
                        # 3. Exact path (only if it's NOT a markdown file)
                        (
                            Path(raw_path)
                            if not str(raw_path).endswith(".md")
                            else Path("NONEXISTENT")
                        ),
                        # 4. Current directory
                        Path(img_filename),
                    ]

                    valid_img_path = None
                    for p in possible_paths:
                        if p.exists() and not p.name.lower().endswith(".md"):
                            valid_img_path = p
                            break

                    if valid_img_path:
                        try:
                            st.image(
                                str(valid_img_path),
                                caption=f"Figure from {meta.get('pdf_stem')}",
                            )
                        except Exception as e:
                            st.error(
                                f"⚠️ Could not display image: {valid_img_path.name}"
                            )
                    else:
                        st.warning(f"⚠️ Image file '{img_filename}' not found on disk.")

                    st.info(f"**AI Analysis:** {doc.page_content}")
                else:
                    st.text(doc.page_content[:400] + "...")
                st.divider()

    st.session_state.messages.append({"role": "assistant", "content": response})
