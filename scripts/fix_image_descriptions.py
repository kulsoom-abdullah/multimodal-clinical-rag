import os
import pickle
import base64
import time
from pathlib import Path

# --- LangChain & AI Imports ---
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

# --- CONFIGURATION ---
DOCSTORE_PATH = Path("data/docstore_advanced.pkl")
DB_PATH = Path("data/chroma_db_advanced")
COLLECTION_NAME = "clinical_trials_advanced"
EMBEDDING_MODEL = "NeuML/pubmedbert-base-embeddings"
# VISION_MODEL = "gpt-4o-mini"
VISION_MODEL = "claude-sonnet-4-5"

# Save progress to disk every X images
BATCH_SIZE = 10


def encode_image(image_path):
    """Encodes an image to base64."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def get_redaction_aware_description(image_path, metadata, model):
    try:
        base64_image = encode_image(image_path)

        trial_id = metadata.get("trial_id", "Unknown Trial")
        pdf_name = metadata.get("pdf_stem", "Unknown Document")
        phase = metadata.get("trial_phase", "Unknown Phase")
        drug = metadata.get("intervention_drug", "Investigational Drug")

        prompt_text = f"""
        CONTEXT:
        This image comes from Clinical Trial: {trial_id}
        Document Source: {pdf_name}
        Study Phase: {phase}
        Drug/Intervention: {drug}
        
        INSTRUCTIONS:
        Analyze this image for RAG retrieval. Note that some elements may be redacted.

        STEP 1: SANITY CHECK
        - If the image is a Company Logo, Blank Page, or pure Noise -> Output ONLY: "IGNORE_IMAGE"
        - If the image contains redactions -> Acknowledge them but extract whatever remains visible.

        STEP 2: STRUCTURED DESCRIPTION
        Generate a detailed description including:
        - Image Type (e.g. Kaplan-Meier, Table, Flowchart)
        - Title and Axis Labels (Read these carefully)
        - Visible Data Trends
        - Clinical Relevance based on the context provided above.
        """

        msg = HumanMessage(
            content=[
                {"type": "text", "text": prompt_text},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                },
            ]
        )

        sys_msg = SystemMessage(content="You are a Senior Clinical Data Scientist.")

        # High token limit to prevent cutoffs
        response = model.invoke([sys_msg, msg])
        return response.content.strip()

    except Exception as e:
        print(f"    ⚠️ Error describing {image_path.name}: {e}")
        return None


def save_checkpoint(store_dict, vectorstore, batch_updates, batch_deletes):
    """Saves the current batch to disk immediately."""
    if not batch_updates and not batch_deletes:
        return

    print(f"  💾 Checkpoint: Saving {len(batch_updates)} updates...")

    # 1. Update Pickle (Docstore)
    for did in batch_deletes:
        if did in store_dict:
            del store_dict[did]
    for doc_id, doc in batch_updates:
        store_dict[doc_id] = doc

    with open(DOCSTORE_PATH, "wb") as f:
        pickle.dump(store_dict, f)

    # 2. Update Chroma (VectorStore)
    if batch_deletes:
        try:
            vectorstore.delete(batch_deletes)
        except Exception as e:
            print(f"    ⚠️ Chroma delete error (safe to ignore): {e}")

    if batch_updates:
        ids_to_update = [k[0] for k in batch_updates]
        docs_to_update = [k[1] for k in batch_updates]
        try:
            vectorstore.delete(ids_to_update)
            vectorstore.add_documents(docs_to_update, ids=ids_to_update)
        except Exception as e:
            print(f"    ⚠️ Chroma update error: {e}")

    print("  ✅ Checkpoint Saved.")


def main():
    print("🔧 Starting FORCE Surgical Image Repair...")

    if not DOCSTORE_PATH.exists():
        print("❌ Docstore not found.")
        return

    with open(DOCSTORE_PATH, "rb") as f:
        store_dict = pickle.load(f)

    print("📦 Loading VectorStore...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = Chroma(
        persist_directory=str(DB_PATH),
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )

    # vision_llm = ChatOpenAI(model=VISION_MODEL, max_tokens=1000, request_timeout=30)
    from langchain_anthropic import ChatAnthropic

    vision_llm = ChatAnthropic(model=VISION_MODEL, max_tokens=1000, timeout=30)

    batch_updates = []
    batch_deletes = []

    items = list(store_dict.items())
    total_images = sum(
        1 for _, doc in items if doc.metadata.get("chunk_type") == "image"
    )
    print(f"📂 Found {total_images} images to process.")

    processed_count = 0

    for i, (doc_id, doc) in enumerate(items):
        if doc.metadata.get("chunk_type") == "image":
            content = doc.page_content.strip()

            # --- FORCE MODE: DISABLE SKIP LOGIC ---
            # if len(content) > 60 and not content.startswith("Image:"):
            #    continue
            # --------------------------------------

            # Locate file
            filename = doc.metadata.get("filename")
            trial_id = doc.metadata.get("trial_id")
            pdf_stem = doc.metadata.get("pdf_stem")

            possible_paths = [
                Path(f"output/{trial_id}/{pdf_stem}/images/{filename}"),
                Path(f"output/{trial_id}/{filename}"),
                Path(doc.metadata.get("source", "")),
            ]

            image_path = None
            for p in possible_paths:
                if p.exists() and p.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                    image_path = p
                    break

            if image_path:
                print(
                    f"  📸 Analyzing [{len(batch_updates) + 1}/{BATCH_SIZE}]: {filename}..."
                )

                new_desc = get_redaction_aware_description(
                    image_path, doc.metadata, vision_llm
                )

                if new_desc:
                    if "IGNORE_IMAGE" in new_desc:
                        print(f"     🗑️  Marked as JUNK.")
                        batch_deletes.append(doc_id)
                    else:
                        # Print a small preview to show it's working
                        print(f"     ✅  Salvaged! ({len(new_desc)} chars)")
                        doc.page_content = new_desc
                        batch_updates.append((doc_id, doc))
                else:
                    print("     ⚠️ API returned empty/error.")
            else:
                print(f"     ❌ File missing. Deleting.")
                batch_deletes.append(doc_id)

            # CHECKPOINT TRIGGER
            if len(batch_updates) + len(batch_deletes) >= BATCH_SIZE:
                save_checkpoint(store_dict, vectorstore, batch_updates, batch_deletes)
                batch_updates = []
                batch_deletes = []

    # Final flush
    if batch_updates or batch_deletes:
        save_checkpoint(store_dict, vectorstore, batch_updates, batch_deletes)

    print("\n🏁 Repair Complete.")


if __name__ == "__main__":
    main()
