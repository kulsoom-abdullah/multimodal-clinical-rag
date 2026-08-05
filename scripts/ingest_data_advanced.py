#!/usr/bin/env python3
"""
Advanced Ingestion Script: "The Clinical Researcher"

Architecture:
1. Extraction (The Brain): Uses Claude Sonnet 4.5 (superior reasoning) to read
   headers/synopses and extract structured metadata (Phase, Drugs, Endpoints).
2. Summarization (The Worker): Uses GPT-4o-mini (cost-effective) to summarize
   individual chunks (text/tables/images).
3. Storage: Embeds via PubMedBERT (Domain Specific) and stores in ChromaDB.

Usage:
    python scripts/ingest_data_advanced.py

!! NEVER RESUME A PARTIAL RUN. Always delete data/chroma_db_advanced and
   data/docstore_advanced.pkl and start over.

   The per-document checkpoint asks "is there already a chunk with this
   pdf_stem and source_dir?" and skips if so. A document's text chunks are
   written before its images are captioned, so a run interrupted between those
   two steps leaves a document the checkpoint considers complete. Resuming
   skips it and its figures are silently missing from the index -- no error,
   no warning, and nothing downstream can tell.

   Making the write transactional would fix this properly; until then,
   clear and rebuild.
"""
import os
import sys
import re
import hashlib
import pickle
from concurrent.futures import ThreadPoolExecutor
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
SUMMARY_MODEL_NAME = "gpt-4o-mini" 

# Embeddings: Switched to PubMedBERT based on Deep Research
EMBEDDING_MODEL = "NeuML/pubmedbert-base-embeddings"

# Concurrent summarization calls. Deliberately conservative: past this, rate-limit
# retries cost more than the added concurrency buys.
SUMMARY_WORKERS = 8

# Paths
CHROMA_DIR = Path("data/chroma_db_advanced")
DOCSTORE_PATH = Path("data/docstore_advanced.pkl")
MAX_CONTEXT_CHARS = 15000


# --- METADATA SCHEMA ---
class ClinicalTrialMetadata(BaseModel):
    """Schema for extracting structured metadata from clinical trial documents."""

    trial_id: Optional[str] = Field(
        None,
        description=(
            "The NCT registration ID of THIS document (e.g., NCT01234567). "
            "For a platform/master protocol, a sub-study document may carry its own "
            "registration that differs from the master record cited on the title page. "
            "Report the ID this document is registered under, not the one it references."
        ),
    )
    protocol_id: Optional[str] = Field(
        None, description="The Sponsor Protocol Number as printed (e.g., 3000-02-005)."
    )
    protocol_number: Optional[str] = Field(
        None,
        description=(
            "The sponsor protocol number WITHOUT any amendment suffix (e.g., '205801' "
            "for '205801/Amendment 08'). This is the stable identity of the study."
        ),
    )
    amendment: Optional[str] = Field(
        None,
        description=(
            "The amendment identifier alone, if the document is an amendment "
            "(e.g., 'Amendment 08'). Null for an original protocol."
        ),
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

    # Derive the split identity from protocol_id when the extractor did not supply it.
    pid = current_metadata.get("protocol_id")
    if pid:
        amend = re.search(r"(Amendment\s*\d+)", pid, re.IGNORECASE)
        if not current_metadata.get("amendment") and amend:
            current_metadata["amendment"] = amend.group(1)
        if not current_metadata.get("protocol_number"):
            current_metadata["protocol_number"] = re.split(
                r"\s*[/,]\s*Amendment", pid, flags=re.IGNORECASE
            )[0].strip()

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
    # Sorted: filesystem order is not stable across machines or reruns, and it
    # would otherwise leak into chunk ordering.
    for p in sorted(images_dir.iterdir()):
        if p.suffix.lower() in valid and p.stat().st_size > 5000:
            images.append({"path": p, "filename": p.name})
    return images


def save_docstore(docstore) -> None:
    """Write the docstore atomically.

    Written after every document rather than once at the end, and via a temp file
    + rename so an interrupted write cannot truncate a good docstore.
    """
    tmp = DOCSTORE_PATH.with_suffix(".pkl.tmp")
    with open(tmp, "wb") as f:
        pickle.dump(dict(docstore.store), f)
    tmp.replace(DOCSTORE_PATH)


def make_doc_id(*parts: Any) -> str:
    """Deterministic parent id.

    Derived from stable content coordinates rather than uuid4 so that re-running
    ingestion over the same corpus reproduces the same ids, which is what makes a
    rebuild comparable to the run before it.
    """
    key = "|".join("" if p is None else str(p) for p in parts)
    return hashlib.sha1(key.encode("utf-8")).hexdigest()


# --- AI AGENTS ---


@backoff.on_exception(backoff.expo, Exception, max_time=300)
def agent_extract_metadata(extractor_llm, text_context: str, filename: str) -> Dict:
    """Uses Claude to understand the clinical document and extract fields."""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert Clinical Research Coordinator. Extract metadata from this trial document. "
                "Identity matters: a master protocol and its sub-studies are separately registered, so the "
                "NCT ID this document belongs to may differ from one it merely cites. Split the protocol "
                "number from its amendment.",
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
    """
    Uses GPT to summarize a specific chunk (Text, Table, or Image).
    handles multimodal inputs correctly.
    """
    
    # Safety check
    if not content or not content.strip():
        return ""

    # 1. Prepare context
    trial_id = meta.get('trial_id', 'Unknown')
    phase = meta.get('trial_phase', 'Unknown')
    drug = meta.get('intervention_drug', 'Unknown')
    protocol_id = meta.get('protocol_id', 'Unknown')
    
    # --- CASE: IMAGE (VISION) ---
    if kind == "image":
        # In image mode, 'content' is the file path
        image_path = Path(content)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        base64_image = encode_image(image_path)

        prompt_text = f"""
        CONTEXT:
        Trial: {trial_id} (Protocol: {protocol_id})
        Phase: {phase}
        Drug: {drug}

        INSTRUCTIONS:
        Analyze this clinical figure.
        1. CLASSIFY: If it's a logo/noise, output "IGNORE_IMAGE".
        2. DESCRIBE: Capture chart type, axis labels, data trends, and key numbers.
        3. REDACTIONS: If present, acknowledge them but describe visible context.
        """

        msg = HumanMessage(
            content=[
                {"type": "text", "text": prompt_text},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
            ]
        )
        # Deliberately NOT wrapped in try/except: this function is @backoff-
        # decorated, and swallowing the exception here returned the error text as
        # the caption, which was then embedded and indexed as if it were a figure
        # description. Letting it raise lets backoff retry, and a genuine failure
        # stops the run instead of quietly poisoning the image index.
        response = summarizer_llm.invoke([msg])
        return response.content

    # --- CASE: TEXT/TABLE ---
    else:
        context_str = f"Trial: {trial_id}, Phase: {phase}, Drug: {drug}"
        
        if kind == "table": 
            instr = "Summarize this clinical table. Focus on data values, row/column headers, and trends."
        else: 
            instr = "Summarize this clinical text chunk. Capture inclusion criteria, dosing, safety signals, or statistical methods."
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", instr),
            ("user", "Global Context: {context}\n\nContent:\n{content}")
        ])
        
        chain = prompt | summarizer_llm | (lambda x: x.content)
        return chain.invoke({"context": context_str, "content": content})

# --- MAIN ---


def main():
    print(f"🚀 Starting ADVANCED Ingestion (PubMedBERT + Claude 4.5)")

    if not os.getenv("ANTHROPIC_API_KEY"):
        print("❌ ANTHROPIC_API_KEY missing!")
        sys.exit(1)

    # Timeouts are required, not optional: @backoff only fires on exceptions, so a
    # socket that stalls without erroring never triggers a retry and the run hangs
    # indefinitely. Observed once at a document boundary, blocked in SSL read.
    extractor_llm = ChatAnthropic(
        model=EXTRACTION_MODEL_NAME, temperature=0, timeout=60, max_retries=3
    )
    summarizer_llm = ChatOpenAI(
        model=SUMMARY_MODEL_NAME, temperature=0, timeout=60, max_retries=3
    )
    # Vision calls run 8-20s on the benchmarked figures, but redacted pages are
    # larger and slower; given its own client so the text path is not slowed to
    # accommodate it.
    vision_llm = ChatOpenAI(
        model=SUMMARY_MODEL_NAME, temperature=0, timeout=120, max_retries=3
    )

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

            # Checkpoint: has this exact source file already been ingested?
            # Matched on source_dir, not trial_id: trial_id is now the value
            # extracted from the document, which legitimately differs from the
            # directory name for a separately-registered sub-study. Matching on
            # trial_id would miss those on a resumed run and ingest them twice.
            existing_docs = vectorstore.get(
                where={"$and": [{"pdf_stem": pdf_stem}, {"source_dir": trial_id}]},
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
            # The folder name is where the PDF was filed, not an authority on identity:
            # a master protocol's sub-studies are separately registered, so the folder and
            # the document's own NCT ID legitimately differ. Keep what was extracted.
            meta.update(
                {
                    "source": str(md_path),
                    "source_dir": trial_dir.name,
                    "pdf_stem": pdf_stem,
                }
            )
            if not meta.get("trial_id"):
                meta["trial_id"] = trial_dir.name

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

            # Summarize concurrently. These calls are pure network I/O and a single
            # stalled request would otherwise block every chunk behind it -- observed
            # throughput swinging 20x when run sequentially. ThreadPoolExecutor.map
            # yields results in INPUT order regardless of completion order, and doc_id
            # is derived from the enumerate index below, never from completion order,
            # so the build stays deterministic.
            done = 0

            def _summarize(chunk):
                nonlocal done
                out = agent_summarize(
                    summarizer_llm, chunk.page_content, "text", meta
                )
                done += 1
                if done % 25 == 0:
                    print(f"       ...{done}/{len(chunks)}", flush=True)
                return out

            with ThreadPoolExecutor(max_workers=SUMMARY_WORKERS) as pool:
                summaries = list(pool.map(_summarize, chunks))

            for i, (chunk, summary) in enumerate(zip(chunks, summaries)):
                doc_id = make_doc_id(
                    meta.get("source_dir"), pdf_stem, "text", i
                )
                batch_sum.append(
                    Document(
                        page_content=summary,
                        metadata={**chunk.metadata, "doc_id": doc_id},
                    )
                )
                batch_org.append((doc_id, chunk))

            if batch_sum:
                # ids= keyed on doc_id so a vector can be located and replaced later;
                # without it Chroma assigns random UUIDs and deletes silently no-op.
                retriever.vectorstore.add_documents(
                    batch_sum, ids=[d.metadata["doc_id"] for d in batch_sum]
                )
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
                        vision_llm, str(img["path"]), "image", img_meta
                    )
                    doc_id = make_doc_id(
                        meta.get("source_dir"), pdf_stem, "image", img["filename"]
                    )

                    img_batch_sums.append(
                        Document(
                            page_content=summary,
                            metadata={**img_meta, "doc_id": doc_id},
                        )
                    )
                    # The caption is both the embedded text and the parent content.
                    # Storing "Image: <filename>" as the parent meant a retrieved
                    # figure carried no description into the answer context.
                    img_batch_orgs.append(
                        (
                            doc_id,
                            Document(
                                page_content=summary,
                                metadata={**img_meta, "doc_id": doc_id},
                            ),
                        )
                    )

                if img_batch_sums:
                    retriever.vectorstore.add_documents(
                        img_batch_sums,
                        ids=[d.metadata["doc_id"] for d in img_batch_sums],
                    )
                    retriever.docstore.mset(img_batch_orgs)
                    print(f"    💾 Saved {len(img_batch_sums)} images.")

            # Persist the docstore after every document. It used to be written once
            # at the very end, so a crash mid-run left an index full of vectors with
            # no parent documents to resolve to -- every query returning nothing.
            # Observed: a run that died on the last document left 5,733 vectors and
            # no docstore at all.
            save_docstore(retriever.docstore)

    save_docstore(retriever.docstore)
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
