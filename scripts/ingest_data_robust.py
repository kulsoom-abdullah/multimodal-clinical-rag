#!/usr/bin/env python3
"""
Ingests extracted clinical trial data into a multimodal RAG system.

This script is robust and idempotent. It saves progress after every
API call, so if it is stopped or crashes, it can be re-run and will
pick up exactly where it left off.

It uses an exponential backoff decorator to handle OpenAI rate limits
aggressively, retrying for up to 5 minutes before failing.
"""
import os
import sys
import pickle
import base64
import uuid
from pathlib import Path
from typing import List, Dict, Any

# Backoff for exponential retry
import backoff
import openai

# LangChain imports
from langchain_text_splitters import MarkdownTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.stores import InMemoryStore
from langchain_core.documents import Document
from langchain_classic.retrievers.multi_vector import MultiVectorRetriever

# OpenAI for summarization
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Configuration constants
BACKOFF_MAX_TIME = 300  # Maximum retry time in seconds
CHUNK_SIZE = 1024  # Size of text chunks
CHUNK_OVERLAP = 100  # Overlap between chunks
MAX_TOKENS = 300  # Maximum tokens for summarization
CHUNK_PROGRESS_INTERVAL = 20  # Show progress every N chunks
IMAGE_PROGRESS_INTERVAL = 5  # Show progress every N images
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "gpt-4o-mini"
COLLECTION_NAME = "clinical_trials"
ID_KEY = "doc_id"


def is_table_chunk(text: str) -> bool:
    """Heuristic to detect if a chunk contains a markdown table.

    Args:
        text: The text content to check for table markers.

    Returns:
        True if the text appears to contain a markdown table, False otherwise.
    """
    lines = text.strip().split('\n')
    table_lines = [line for line in lines if '|' in line]
    return len(table_lines) >= 2


@backoff.on_exception(
    backoff.expo,
    openai.RateLimitError,
    max_time=BACKOFF_MAX_TIME
)
def summarize_text_chunk(
    client: OpenAI,
    text: str,
    chunk_type: str = "text"
) -> str:
    """Summarize a text or table chunk using GPT-4o-mini with exponential backoff.

    Args:
        client: Initialized OpenAI client.
        text: The chunk text to summarize.
        chunk_type: Either "text" or "table".

    Returns:
        Summary string from the LLM.
    """
    if chunk_type == "table":
        system_message = (
            "You are a helpful assistant for a RAG system. Your task is to summarize "
            "the following clinical trial table chunk. This summary will be used to create "
            "a vector embedding for semantic search. Focus *only* on the key data points, "
            "variables, and conclusions in the table. Do not add any outside information."
        )
    else:
        system_message = (
            "You are a helpful assistant for a RAG system. Your task is to summarize "
            "the following clinical trial text chunk. This summary will be used to create "
            "a vector embedding for semantic search. Focus *only* on the key topics, "
            "entities, and conclusions in this chunk. Do not add any outside information."
        )

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": text}
        ],
        max_tokens=MAX_TOKENS,
        temperature=0.0
    )
    return response.choices[0].message.content.strip()


@backoff.on_exception(
    backoff.expo,
    openai.RateLimitError,
    max_time=BACKOFF_MAX_TIME
)
def summarize_image(client: OpenAI, image_path: Path) -> str:
    """Summarize an image using GPT-4o-mini vision API with exponential backoff.

    Args:
        client: Initialized OpenAI client.
        image_path: Path to the image file.

    Returns:
        Summary string describing the image content.
    """
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")

    system_message = (
        "You are a helpful assistant for a RAG system. Your task is to summarize "
        "the following image from a clinical trial document (e.g., a flowchart, graph, or figure). "
        "This summary will be used to create a vector embedding for semantic search. "
        "Describe the key information, data, trends, or processes shown. "
        "If it is a 'junk' image (logo, redaction box), simply state 'Junk image'. "
        "Do not add any outside information."
    )

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": system_message},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}
                }
            ]
        }],
        max_tokens=MAX_TOKENS,
        temperature=0.0
    )
    return response.choices[0].message.content.strip()


def load_and_chunk_markdown(md_path: Path) -> List[Document]:
    """Load a markdown file and chunk it using MarkdownTextSplitter.

    Args:
        md_path: Path to the markdown file.

    Returns:
        List of Document objects with metadata attached.
    """
    with open(md_path, "r", encoding="utf-8") as f:
        content = f.read()

    splitter = MarkdownTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.create_documents([content])

    # Add base metadata
    for chunk in chunks:
        chunk.metadata["source"] = str(md_path)
        chunk.metadata["trial_id"] = md_path.parts[-3]
        chunk.metadata["pdf_stem"] = md_path.parts[-2]
    return chunks


def load_images(images_dir: Path) -> List[Dict[str, Any]]:
    """Load all images from a directory.

    Args:
        images_dir: Path to the directory containing image files.

    Returns:
        List of dictionaries containing image metadata.
    """
    if not images_dir.exists():
        return []

    images = []
    image_extensions = ["*.jpeg", "*.jpg", "*.png", "*.bmp", "*.gif"]

    for ext in image_extensions:
        for img_path in images_dir.glob(ext):
            images.append({
                "path": img_path,
                "trial_id": img_path.parts[-4],
                "pdf_stem": img_path.parts[-3],
                "filename": img_path.name
            })
    return images


def main() -> None:
    """Main entry point for the ingestion script."""
    print("🚀 Starting robust multimodal RAG ingestion with checkpointing...")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not found. Please set it in .env file.")
        sys.exit(1)

    # max_retries=1 because @backoff handles retries
    client = OpenAI(api_key=api_key, max_retries=1)

    print("📦 Loading sentence-transformers embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    print("🗄️  Initializing ChromaDB...")
    chroma_dir = Path("data/chroma_db")
    chroma_dir.mkdir(parents=True, exist_ok=True)
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(chroma_dir)
    )

    # Load or initialize InMemoryStore
    docstore_path = Path("data/docstore.pkl")
    if docstore_path.exists():
        print(f"📂 Loading existing docstore from {docstore_path}...")
        with open(docstore_path, "rb") as f:
            store_dict = pickle.load(f)
        docstore = InMemoryStore(store=store_dict)
    else:
        print("📂 Creating new InMemoryStore...")
        docstore = InMemoryStore()

    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        id_key=ID_KEY
    )

    output_dir = Path("output")
    if not output_dir.exists():
        print(f"❌ Output directory not found: {output_dir}")
        sys.exit(1)

    trial_dirs = sorted([d for d in output_dir.iterdir() if d.is_dir()])
    print(f"📂 Found {len(trial_dirs)} trial directories: "
          f"{[d.name for d in trial_dirs]}\n")

    total_text_chunks = 0
    total_table_chunks = 0
    total_images = 0

    for trial_dir in trial_dirs:
        trial_id = trial_dir.name
        print(f"{'='*60}")
        print(f"Processing Trial: {trial_id}")
        print(f"{'='*60}")

        pdf_dirs = sorted([d for d in trial_dir.iterdir() if d.is_dir()])

        for pdf_dir in pdf_dirs:
            pdf_stem = pdf_dir.name
            print(f"\n  📄 Checking PDF: {pdf_stem}")

            md_path = pdf_dir / f"{pdf_stem}.md"
            if not md_path.exists():
                print(f"    ⚠️  No markdown file found, skipping")
                continue

            # --- Process Text/Table Chunks ---
            chunks = load_and_chunk_markdown(md_path)
            print(f"    ✂️  Checking {len(chunks)} text/table chunks...")
            text_count = 0
            table_count = 0

            for i, chunk in enumerate(chunks):
                chunk_type = (
                    "table" if is_table_chunk(chunk.page_content) else "text"
                )
                chunk.metadata["chunk_type"] = chunk_type
                chunk.metadata["chunk_index"] = i

                # Check if this chunk is already in the DB
                existing = vectorstore.get(
                    where={"$and": [
                        {"pdf_stem": pdf_stem},
                        {"chunk_index": i}
                    ]}
                )
                if len(existing["ids"]) > 0:
                    if chunk_type == "table":
                        table_count += 1
                    else:
                        text_count += 1
                    continue

                # Not found, so we process and save it
                try:
                    summary = summarize_text_chunk(
                        client,
                        chunk.page_content,
                        chunk_type
                    )
                    doc_id = str(uuid.uuid4())

                    summary_doc = Document(
                        page_content=summary,
                        metadata={**chunk.metadata, ID_KEY: doc_id}
                    )

                    retriever.vectorstore.add_documents([summary_doc])
                    retriever.docstore.mset([(doc_id, chunk)])

                    if chunk_type == "table":
                        table_count += 1
                    else:
                        text_count += 1

                    if (i + 1) % CHUNK_PROGRESS_INTERVAL == 0:
                        print(f"      ⏳ Processed {i + 1}/{len(chunks)} chunks...")

                except Exception as e:
                    print(f"\n    ❌ FAILED on chunk {i} of {pdf_stem}: {e}")
                    print("    Stopping to save progress. Re-run script to resume.")
                    # Save docstore before breaking
                    with open(docstore_path, "wb") as f:
                        pickle.dump(dict(retriever.docstore.store), f)
                    return

            total_text_chunks += text_count
            total_table_chunks += table_count
            print(f"    ✅ Checked/Processed {text_count} text, "
                  f"{table_count} table chunks")

            # --- Process Images ---
            images_dir = pdf_dir / "images"
            images = load_images(images_dir)

            if not images:
                continue

            print(f"    🖼️  Checking {len(images)} images...")
            img_count = 0

            for j, img_data in enumerate(images):
                img_metadata = {
                    "source": str(img_data["path"]),
                    "trial_id": img_data["trial_id"],
                    "pdf_stem": img_data["pdf_stem"],
                    "chunk_type": "image",
                    "filename": img_data["filename"]
                }

                # Check if this image is already in the DB
                existing = vectorstore.get(
                    where={"$and": [
                        {"pdf_stem": pdf_stem},
                        {"filename": img_data["filename"]}
                    ]}
                )
                if len(existing["ids"]) > 0:
                    img_count += 1
                    continue

                # Not found, so we process and save it
                try:
                    summary = summarize_image(client, img_data["path"])
                    doc_id = str(uuid.uuid4())

                    summary_doc = Document(
                        page_content=summary,
                        metadata={**img_metadata, ID_KEY: doc_id}
                    )
                    original_img_doc = Document(
                        page_content=f"Image: {img_data['filename']}",
                        metadata=img_metadata
                    )

                    retriever.vectorstore.add_documents([summary_doc])
                    retriever.docstore.mset([(doc_id, original_img_doc)])
                    img_count += 1

                    if (j + 1) % IMAGE_PROGRESS_INTERVAL == 0:
                        print(f"      ⏳ Processed {j + 1}/{len(images)} images...")

                except Exception as e:
                    print(f"\n    ❌ FAILED on image {img_data['filename']}: {e}")
                    print("    Stopping to save progress. Re-run script to resume.")
                    # Save docstore before breaking
                    with open(docstore_path, "wb") as f:
                        pickle.dump(dict(retriever.docstore.store), f)
                    return

            total_images += img_count
            print(f"    ✅ Checked/Processed {img_count} images")

        print(f"\n✅ Finished checking trial: {trial_id}\n")

    # Save the docstore at the very end
    print(f"\n{'='*60}")
    print("💾 All trials checked. Saving final docstore pickle...")
    print(f"{'='*60}")

    print(f"💾 Saving complete docstore to {docstore_path}...")
    with open(docstore_path, "wb") as f:
        pickle.dump(dict(retriever.docstore.store), f)

    print(f"\n{'='*60}")
    print("✅ INGESTION COMPLETE!")
    print(f"{'='*60}")
    print(f"📊 Summary (total items processed this run):")
    print(f"   • Text chunks:      {total_text_chunks}")
    print(f"   • Table chunks:     {total_table_chunks}")
    print(f"   • Images:           {total_images}")
    print(f"\n📁 Output locations:")
    print(f"   • ChromaDB:         {chroma_dir}")
    print(f"   • Docstore:         {docstore_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
