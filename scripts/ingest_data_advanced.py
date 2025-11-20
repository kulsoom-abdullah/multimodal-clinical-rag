#!/usr/bin/env python3
"""
Advanced Ingestion Script: "The Clinical Researcher"

Architecture:
1. Extraction (The Brain): Uses Claude Sonnet 4.5 (superior reasoning) to read
   headers/synopses and extract structured metadata (Phase, Drugs, Endpoints).
2. Summarization (The Worker): Uses GPT-5-mini (cost-effective) to summarize
   individual chunks (text/tables/images).
3. Storage: Embeds via PubMedBERT (Domain Specific) and stores in ChromaDB.

Usage:
    python scripts/ingest_data_advanced.py
"""
import os
import sys
import re
import uuid
import pickle
import backoff
import base64
from pathlib import Path
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

# --- LangChain Core ---
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import MarkdownTextSplitter
from langchain_core.stores import InMemoryStore

# --- LangChain Integrations ---
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_classic.retrievers.multi_vector import MultiVectorRetriever
from langchain_core.messages import HumanMessage, SystemMessage

# # --- FIX: Robust Import for Retriever ---
# try:
#     from langchain.retrievers import MultiVectorRetriever
# except ImportError:
#     try:
#         from langchain.retrievers.multi_vector import MultiVectorRetriever
#     except ImportError:
#         from langchain_community.retrievers import MultiVectorRetriever

# --- Utilities ---
from dotenv import load_dotenv

load_dotenv()

# --- CONFIGURATION ---
# Models
EXTRACTION_MODEL_NAME = "claude-sonnet-4-5-20250929"
SUMMARY_MODEL_NAME = "gpt-5-mini"

# Embeddings: Switched to PubMedBERT based on Deep Research
EMBEDDING_MODEL = "NeuML/pubmedbert-base-embeddings"

# Paths
CHROMA_DIR = Path("data/chroma_db_advanced")
DOCSTORE_PATH = Path("data/docstore_advanced.pkl")
MAX_CONTEXT_CHARS = 15000


# --- METADATA SCHEMA ---
class ClinicalTrialMetadata(BaseModel):
    """Schema for extracting structured metadata from clinical trial documents."""

    trial_id: Optional[str] = Field(None, description="The NCT ID (e.g., NCT01234567).")
    protocol_id: Optional[str] = Field(
        None, description="The Sponsor Protocol Number (e.g., 3000-02-005)."
    )
    trial_phase: Optional[str] = Field(
        None, description="Phase of the trial (e.g., Phase 1, Phase 3)."
    )
    indication: Optional[str] = Field(
        None, description="Primary disease (e.g., NSCLC, Ovarian Cancer)."
    )
    intervention_drug: Optional[str] = Field(None, description="Main drugs studied.")
    primary_endpoint: Optional[str] = Field(
        None, description="Primary outcome (e.g., OS, PFS, ORR, Safety)."
    )
    sample_size: Optional[str] = Field(None, description="Target enrollment number.")
    study_design: Optional[str] = Field(
        None, description="Design type (e.g., Open-Label, Randomized)."
    )
    document_type: Optional[str] = Field(
        None, description="Type of document (e.g., Protocol, SAP)."
    )


# --- HELPERS ---
def encode_image(image_path):
    """Encodes an image to base64."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def extract_smart_context(full_text: str) -> str:
    """Extracts Header + Key Sections (Synopsis, Objectives) up to limit."""
    header_text = full_text[:3000]
    target_headers = [
        "PROTOCOL SYNOPSIS",
        "STUDY SYNOPSIS",
        "EXECUTIVE SUMMARY",
        "STUDY OBJECTIVES",
        "OBJECTIVES AND ENDPOINTS",
        "PRIMARY ENDPOINT",
        "STUDY DESIGN",
        "INVESTIGATIONAL PLAN",
        "SAMPLE SIZE",
    ]

    extracted = []
    lines = full_text.split("\n")
    capture = False
    buffer = []

    for line in lines[100:]:
        if any(h in line.upper() for h in target_headers):
            capture = True
            buffer.append(f"\n--- SECTION: {line.strip()} ---\n")

        if capture:
            buffer.append(line)
            if len(buffer) > 500:
                extracted.extend(buffer)
                buffer = []
                capture = False

    return (header_text + "\n".join(extracted))[:MAX_CONTEXT_CHARS]


def run_regex_fallbacks(text: str, current_metadata: Dict) -> Dict:
    """Fallback regex for critical IDs."""
    if not current_metadata.get("trial_id"):
        match = re.search(r"NCT\d{8}", text, re.IGNORECASE)
        if match:
            current_metadata["trial_id"] = match.group(0).upper()

    if not current_metadata.get("protocol_id"):
        match = re.search(r"\b\d{4}-\d{2}-\d{3}\b", text)
        if match:
            current_metadata["protocol_id"] = match.group(0)

    return current_metadata


def load_and_chunk_markdown(md_path: Path) -> List[Document]:
    with open(md_path, "r", encoding="utf-8") as f:
        content = f.read()
    splitter = MarkdownTextSplitter(chunk_size=1024, chunk_overlap=100)
    return splitter.create_documents([content])


def load_images(images_dir: Path) -> List[Dict[str, Any]]:
    if not images_dir.exists():
        return []
    images = []
    valid = {".jpg", ".jpeg", ".png"}
    for p in images_dir.iterdir():
        if p.suffix.lower() in valid and p.stat().st_size > 5000:
            images.append({"path": p, "filename": p.name})
    return images


# --- AI AGENTS ---


@backoff.on_exception(backoff.expo, Exception, max_time=300)
def agent_extract_metadata(extractor_llm, text_context: str, filename: str) -> Dict:
    """Uses Claude to understand the clinical document and extract fields."""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert Clinical Research Coordinator. Extract metadata from this trial document.",
            ),
            ("user", "Filename: {filename}\n\nDocument Text:\n{text}"),
        ]
    )
    structured_llm = extractor_llm.with_structured_output(ClinicalTrialMetadata)
    chain = prompt | structured_llm
    result = chain.invoke({"filename": filename, "text": text_context})
    return result.model_dump()


@backoff.on_exception(backoff.expo, Exception, max_time=300)
def agent_summarize(summarizer_llm, content: str, kind: str, meta: Dict) -> str:
    """Uses GPT to summarize a specific chunk (Text, Table, or Image)."""

    # Safety check
    if not content or not content.strip():
        return ""

    # 1. Prepare context
    trial_id = meta.get("trial_id", "Unknown")
    phase = meta.get("trial_phase", "Unknown")
    drug = meta.get("intervention_drug", "Unknown")
    protocol_id = meta.get("protocol_id", "Unknown")

    context_str = (
        f"Trial: {trial_id} (Protocol: {protocol_id}), Phase: {phase}, Drug: {drug}"
    )

    # --- CASE: IMAGE (VISION) ---
    if kind == "image":
        try:
            # content here is the file path string
            image_path = Path(content)
            if image_path.exists():
                base64_image = encode_image(image_path)

                # The Redaction-Resilient Prompt
                prompt_text = f"""
                CONTEXT:
                This image comes from Clinical Trial: {trial_id}
                Study Phase: {phase}
                Drug/Intervention: {drug}
                
                INSTRUCTIONS:
                Analyze this image for RAG retrieval. Note that some elements may be redacted.
                
                1. SANITY CHECK: If it is a Logo, Blank Page, or Noise, output ONLY: "IGNORE_IMAGE".
                2. ANALYSIS: Describe the image type (e.g., Kaplan-Meier, Flowchart), title, axes, and visible data trends.
                3. REDACTIONS: If redacted, acknowledge it but infer the context using the metadata provided above.
                """

                msg = HumanMessage(
                    content=[
                        {"type": "text", "text": prompt_text},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                    ]
                )
                # Call the model directly with the message list
                response = summarizer_llm.invoke([msg])
                return response.content
            else:
                return f"Image file not found: {image_path.name}"
        except Exception as e:
            return f"Error analyzing image {content}: {str(e)}"

    # --- CASE: TEXT/TABLE ---
    else:
        if kind == "table":
            instr = "Summarize this clinical table. Focus on data values, row/column headers, and trends."
        else:
            instr = "Summarize this clinical text chunk. Capture inclusion criteria, dosing, safety signals, or statistical methods."

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", instr),
                ("user", "Global Context: {context}\n\nContent:\n{content}"),
            ]
        )

        chain = prompt | summarizer_llm | (lambda x: x.content)
        return chain.invoke({"context": context_str, "content": content})


# --- MAIN ---


def main():
    print(f"🚀 Starting ADVANCED Ingestion (PubMedBERT + Claude 4.5)")

    if not os.getenv("ANTHROPIC_API_KEY"):
        print("❌ ANTHROPIC_API_KEY missing!")
        sys.exit(1)

    extractor_llm = ChatAnthropic(model=EXTRACTION_MODEL_NAME, temperature=0)
    summarizer_llm = ChatOpenAI(model=SUMMARY_MODEL_NAME, temperature=0)

    print(f"📦 Loading Embeddings: {EMBEDDING_MODEL}...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore = Chroma(
        collection_name="clinical_trials_advanced",
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )

    docstore = InMemoryStore()
    if DOCSTORE_PATH.exists():
        print("   📂 Loading existing docstore...")
        with open(DOCSTORE_PATH, "rb") as f:
            # FIX: Robust loading
            loaded_data = pickle.load(f)
            if isinstance(loaded_data, dict):
                docstore.mset(list(loaded_data.items()))

    retriever = MultiVectorRetriever(
        vectorstore=vectorstore, docstore=docstore, id_key="doc_id"
    )

    output_dir = Path("output")
    trial_dirs = sorted([d for d in output_dir.iterdir() if d.is_dir()])
    print(f"📂 Found {len(trial_dirs)} trials to process.")

    for trial_dir in trial_dirs:
        trial_id = trial_dir.name
        print(f"\n{'='*40}\nProcessing: {trial_dir.name}\n{'='*40}")

        for pdf_dir in sorted([d for d in trial_dir.iterdir() if d.is_dir()]):
            pdf_stem = pdf_dir.name
            md_path = pdf_dir / f"{pdf_stem}.md"
            if not md_path.exists():
                continue

            print(f"  📄 {pdf_stem}")

            # Checkpoint: Check for this SPECIFIC file in this SPECIFIC trial
            existing_docs = vectorstore.get(
                where={"$and": [{"pdf_stem": pdf_stem}, {"trial_id": trial_id}]},
                limit=1,
            )

            if existing_docs and len(existing_docs["ids"]) > 0:
                print("    ✅ Already processed. Skipping.")
                continue

            # Extraction
            with open(md_path, "r") as f:
                full_text = f.read()
            smart_ctx = extract_smart_context(full_text)

            print(f"    🧠 Claude extracting metadata...")
            try:
                meta = agent_extract_metadata(extractor_llm, smart_ctx, pdf_stem)
            except Exception as e:
                print(f"    ⚠️ Extraction failed: {e}")
                meta = {}

            meta = run_regex_fallbacks(full_text[:10000], meta)
            meta = {k: v for k, v in meta.items() if v is not None}
            meta.update(
                {
                    "source": str(md_path),
                    "trial_id": trial_dir.name,
                    "pdf_stem": pdf_stem,
                }
            )

            print(
                f"       → {meta.get('trial_phase', '?')} | {meta.get('intervention_drug', '?')} | {meta.get('protocol_id', '?')}"
            )

            # Summarization
            chunks = load_and_chunk_markdown(md_path)
            print(f"    ✂️  Summarizing {len(chunks)} chunks...")

            batch_sum, batch_org = [], []
            for i, chunk in enumerate(chunks):
                chunk.metadata.update(meta)
                chunk.metadata["chunk_index"] = i
                chunk.metadata["chunk_type"] = "text"

                summary = agent_summarize(
                    summarizer_llm, chunk.page_content, "text", meta
                )
                doc_id = str(uuid.uuid4())

                batch_sum.append(
                    Document(
                        page_content=summary,
                        metadata={**chunk.metadata, "doc_id": doc_id},
                    )
                )
                batch_org.append((doc_id, chunk))

                if (i + 1) % 10 == 0:
                    print(f"       ...{i+1}")

            if batch_sum:
                retriever.vectorstore.add_documents(batch_sum)
                retriever.docstore.mset(batch_org)
                print(f"    💾 Saved {len(batch_sum)} chunks.")

            # Images
            images_dir = pdf_dir / "images"
            images = load_images(images_dir)
            if images:
                print(f"    🖼️  Processing {len(images)} images...")
                img_batch_sums, img_batch_orgs = [], []
                for img in images:
                    img_meta = meta.copy()
                    img_meta.update(
                        {"chunk_type": "image", "filename": img["filename"]}
                    )
                    summary = agent_summarize(
                        summarizer_llm, str(img["path"]), "image", img_meta
                    )
                    doc_id = str(uuid.uuid4())

                    img_batch_sums.append(
                        Document(
                            page_content=summary,
                            metadata={**img_meta, "doc_id": doc_id},
                        )
                    )
                    img_batch_orgs.append(
                        (
                            doc_id,
                            Document(
                                page_content=f"Image: {img['filename']}",
                                metadata=img_meta,
                            ),
                        )
                    )

                if img_batch_sums:
                    retriever.vectorstore.add_documents(img_batch_sums)
                    retriever.docstore.mset(img_batch_orgs)
                    print(f"    💾 Saved {len(img_batch_sums)} images.")

    print(f"\n💾 Saving docstore to {DOCSTORE_PATH}")
    with open(DOCSTORE_PATH, "wb") as f:
        pickle.dump(dict(retriever.docstore.store), f)
    print("✅ Done.")


if __name__ == "__main__":
    main()

# #!/usr/bin/env python3
# """
# Advanced Ingestion Script: "The Clinical Researcher"

# Architecture:
# 1. Extraction (The Brain): Uses Claude Sonnet (superior clinical reasoning) to read
#    headers/synopses and extract structured metadata (Phase, Drugs, Endpoints).
# 2. Summarization (The Worker): Uses GPT-5 Mini (cost-effective) to summarize
#    individual chunks (text/tables/images).
# 3. Storage: Embeds via HuggingFace and stores in ChromaDB + Docstore.

# Usage:
#     python scripts/ingest_data_advanced.py
# """
# import os
# import sys
# import re
# import uuid
# import pickle
# import backoff
# from pathlib import Path
# from typing import List, Optional, Dict, Any
# from pydantic import BaseModel, Field

# # LangChain Core
# from langchain_core.documents import Document
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_text_splitters import MarkdownTextSplitter
# from langchain_core.stores import InMemoryStore

# # LangChain Integrations
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_chroma import Chroma
# from langchain_anthropic import ChatAnthropic
# from langchain_openai import ChatOpenAI
# from langchain_classic.retrievers.multi_vector import MultiVectorRetriever

# # Utilities
# from dotenv import load_dotenv

# load_dotenv()

# # --- CONFIGURATION (THE FUTURE STACK) ---
# # If these models don't exist in your API yet, fallback to:
# # EXTRACTION: "claude-3-5-sonnet-20241022"
# # SUMMARY: "gpt-4o-mini"

# EXTRACTION_MODEL_NAME = "claude-sonnet-4-5"
# SUMMARY_MODEL_NAME = "gpt-5-mini"

# # --- CONFIGURATION ---
# # OLD: EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# # NEW:
# EMBEDDING_MODEL = "NeuML/pubmedbert-base-embeddings"
# CHROMA_DIR = Path("data/chroma_db_advanced")
# DOCSTORE_PATH = Path("data/docstore_advanced.pkl")
# MAX_CONTEXT_CHARS = 15000

# # --- METADATA SCHEMA ---
# class ClinicalTrialMetadata(BaseModel):
#     """Schema for extracting structured metadata from clinical trial documents."""
#     trial_id: Optional[str] = Field(None, description="The NCT ID (e.g., NCT01234567).")
#     protocol_id: Optional[str] = Field(None, description="The Sponsor Protocol Number (e.g., 3000-02-005).")
#     trial_phase: Optional[str] = Field(None, description="Phase of the trial (e.g., Phase 1, Phase 3).")
#     indication: Optional[str] = Field(None, description="Primary disease (e.g., NSCLC, Ovarian Cancer).")
#     intervention_drug: Optional[str] = Field(None, description="Main drugs studied.")
#     primary_endpoint: Optional[str] = Field(None, description="Primary outcome (e.g., OS, PFS, ORR, Safety).")
#     sample_size: Optional[str] = Field(None, description="Target enrollment number.")
#     study_design: Optional[str] = Field(None, description="Design type (e.g., Open-Label, Randomized).")
#     document_type: Optional[str] = Field(None, description="Type of document (e.g., Protocol, SAP).")

# # --- HELPERS ---

# def extract_smart_context(full_text: str) -> str:
#     """Extracts Header + Key Sections (Synopsis, Objectives) up to limit."""
#     header_text = full_text[:3000]
#     target_headers = [
#         "PROTOCOL SYNOPSIS", "STUDY SYNOPSIS", "EXECUTIVE SUMMARY",
#         "STUDY OBJECTIVES", "OBJECTIVES AND ENDPOINTS", "PRIMARY ENDPOINT",
#         "STUDY DESIGN", "INVESTIGATIONAL PLAN", "SAMPLE SIZE"
#     ]

#     extracted = []
#     lines = full_text.split('\n')
#     capture = False
#     buffer = []

#     for line in lines[100:]:
#         if any(h in line.upper() for h in target_headers):
#             capture = True
#             buffer.append(f"\n--- SECTION: {line.strip()} ---\n")

#         if capture:
#             buffer.append(line)
#             if len(buffer) > 500: # Limit per section
#                 extracted.extend(buffer)
#                 buffer = []
#                 capture = False

#     return (header_text + "\n".join(extracted))[:MAX_CONTEXT_CHARS]

# def run_regex_fallbacks(text: str, current_metadata: Dict) -> Dict:
#     """Fallback regex for critical IDs."""
#     if not current_metadata.get('trial_id'):
#         match = re.search(r'NCT\d{8}', text, re.IGNORECASE)
#         if match: current_metadata['trial_id'] = match.group(0).upper()

#     if not current_metadata.get('protocol_id'):
#         match = re.search(r'\b\d{4}-\d{2}-\d{3}\b', text)
#         if match: current_metadata['protocol_id'] = match.group(0)

#     return current_metadata

# def load_and_chunk_markdown(md_path: Path) -> List[Document]:
#     with open(md_path, "r", encoding="utf-8") as f:
#         content = f.read()
#     splitter = MarkdownTextSplitter(chunk_size=1024, chunk_overlap=100)
#     return splitter.create_documents([content])

# def load_images(images_dir: Path) -> List[Dict[str, Any]]:
#     if not images_dir.exists(): return []
#     images = []
#     valid = {".jpg", ".jpeg", ".png"}
#     for p in images_dir.iterdir():
#         if p.suffix.lower() in valid and p.stat().st_size > 5000:
#             images.append({"path": p, "filename": p.name})
#     return images

# # --- AI AGENTS ---

# # We use backoff on general Exceptions because LangChain wraps API errors
# @backoff.on_exception(backoff.expo, Exception, max_time=300)
# def agent_extract_metadata(extractor_llm, text_context: str, filename: str) -> Dict:
#     """Uses Claude to understand the clinical document and extract fields."""

#     prompt = ChatPromptTemplate.from_messages([
#         ("system", "You are an expert Clinical Research Coordinator. Extract metadata from this trial document."),
#         ("user", "Filename: {filename}\n\nDocument Text:\n{text}")
#     ])

#     # LangChain's .with_structured_output handles the JSON/Tool schema automatically
#     structured_llm = extractor_llm.with_structured_output(ClinicalTrialMetadata)
#     chain = prompt | structured_llm

#     result = chain.invoke({"filename": filename, "text": text_context})
#     return result.dict()

# @backoff.on_exception(backoff.expo, Exception, max_time=300)
# def agent_summarize(summarizer_llm, content: str, kind: str, meta: Dict) -> str:
#     """Uses GPT to summarize a specific chunk."""
#     context = f"Trial: {meta.get('trial_id')}, Phase: {meta.get('trial_phase')}, Drug: {meta.get('intervention_drug')}"

#     if kind == "table": instr = "Summarize this clinical table. Focus on data."
#     elif kind == "image": instr = "Describe this clinical figure. Focus on labels and trends."
#     else: instr = "Summarize this clinical text chunk."

#     prompt = ChatPromptTemplate.from_messages([
#         ("system", instr),
#         ("user", f"Global Context: {context}\n\nContent:\n{content}")
#     ])

#     chain = prompt | summarizer_llm | (lambda x: x.content)
#     return chain.invoke({})

# # --- MAIN ---

# def main():
#     print(f"🚀 Starting Clinical Ingestion V2 (Hybrid Approach)")

#     # Init Models
#     if not os.getenv("ANTHROPIC_API_KEY"):
#         print("❌ ANTHROPIC_API_KEY missing!")
#         sys.exit(1)

#     extractor_llm = ChatAnthropic(model=EXTRACTION_MODEL_NAME, temperature=0)
#     summarizer_llm = ChatOpenAI(model=SUMMARY_MODEL_NAME, temperature=0)
#     embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

#     # Init Vector Store
#     CHROMA_DIR.mkdir(parents=True, exist_ok=True)
#     vectorstore = Chroma(
#         collection_name="clinical_trials_v2",
#         embedding_function=embeddings,
#         persist_directory=str(CHROMA_DIR)
#     )

#     # Init Docstore (Fixed Loading Logic)
#     docstore = InMemoryStore()
#     if DOCSTORE_PATH.exists():
#         print(f"   📂 Loading existing docstore from {DOCSTORE_PATH}...")
#         with open(DOCSTORE_PATH, "rb") as f:
#             try:
#                 loaded_data = pickle.load(f)
#                 # Check if it's a dict (raw store) or list of tuples
#                 if isinstance(loaded_data, dict):
#                     docstore.mset(list(loaded_data.items()))
#                 else:
#                     print("⚠️ Warning: Docstore file format unknown, starting fresh.")
#             except Exception as e:
#                 print(f"⚠️ Warning: Could not load docstore: {e}")

#     retriever = MultiVectorRetriever(
#         vectorstore=vectorstore,
#         docstore=docstore,
#         id_key="doc_id"
#     )

#     # Load Data
#     output_dir = Path("output")

#     trial_dirs = sorted([d for d in output_dir.iterdir() if d.is_dir()])
#     print(f"📂 Found {len(trial_dirs)} trials.")

#     for trial_dir in trial_dirs:

#         # Look for the main markdown file
#         md_files = list(trial_dir.glob("*.md"))
#         # Also check subdirectories (sometimes marker creates nested folders)
#         if not md_files:
#             md_files = list(trial_dir.glob("**/*.md"))

#         if not md_files:
#             continue

#         # Take the first MD found (usually the protocol)
#         md_path = md_files[0]
#         pdf_stem = md_path.stem

#         print(f"\n{'='*40}\n📄 Processing: {pdf_stem}\n{'='*40}")

#         # Check duplication
#         if len(vectorstore.get(where={"pdf_stem": pdf_stem}, limit=1)["ids"]) > 0:
#             print(f"⏩ Skipping {pdf_stem} (Already Processed)")
#             continue

#         with open(md_path, "r", encoding="utf-8") as f:
#             full_text = f.read()

#         # --- STEP A: SMART EXTRACTION ---
#         print("   🔍 Scanning for Synopsis, Design, and Objectives...")
#         smart_context = extract_smart_sections(full_text)
#         print(f"      -> Context Window: {len(smart_context)} chars")

#         print("   🧠 Claude extraction...")
#         try:
#             meta = agent_extract_metadata(extractor_llm, smart_context, pdf_stem)
#         except Exception as e:
#             print(f"      ⚠️ Extraction Error: {e}")
#             meta = {}

#         # --- STEP B: REGEX SAFETY NET ---
#         meta = run_regex_fallbacks(full_text[:5000], meta)

#         # Clean None values
#         meta = {k: v for k, v in meta.items() if v is not None}
#         meta["pdf_stem"] = pdf_stem
#         meta["source"] = str(md_path)
#         meta["trial_id_folder"] = trial_dir.name # Keep folder name as backup ID

#         print(f"      ✅ Extracted: {meta.get('trial_id', 'N/A')} | {meta.get('trial_phase', 'N/A')} | {meta.get('intervention_drug', 'N/A')}")

#         # --- STEP C: CHUNKING & SUMMARIZATION ---
#         splitter = MarkdownTextSplitter(chunk_size=1000, chunk_overlap=100)
#         chunks = splitter.create_documents([full_text])

#         print(f"   ✂️  Summarizing {len(chunks)} chunks...")
#         batch_vectors = []
#         batch_docs = []

#         for i, chunk in enumerate(chunks):
#             chunk.metadata.update(meta)
#             chunk.metadata["chunk_index"] = i
#             chunk.metadata["chunk_type"] = "text"

#             doc_id = str(uuid.uuid4())

#             # Summarize
#             summary = agent_summarize_chunk(summarizer_llm, chunk.page_content, meta)

#             # Create summary document for Vector Store
#             summary_doc = Document(
#                 page_content=summary,
#                 metadata={**chunk.metadata, "doc_id": doc_id}
#             )
#             batch_vectors.append(summary_doc)

#             # Store original chunk in DocStore
#             batch_docs.append((doc_id, chunk))

#             if i > 0 and i % 20 == 0:
#                 print(f"      ...{i} chunks processed")

#         # Batch Add (Per Document)
#         if batch_vectors:
#             retriever.vectorstore.add_documents(batch_vectors)
#             retriever.docstore.mset(batch_docs)
#             print(f"   💾 Saved {len(batch_vectors)} chunks to DB.")

#     # Persist
#     print(f"\n💾 Saving final docstore to {DOCSTORE_PATH}")
#     with open(DOCSTORE_PATH, "wb") as f:
#         pickle.dump(dict(retriever.docstore.store), f)
#     print("\n✅ Ingestion Complete.")

# if __name__ == "__main__":
#     main()
