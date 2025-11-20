#!/usr/bin/env python3
"""
Extracts text, tables, and images from PDFs using the Marker library.

This script is fully automated. It will scan for all trial folders
in 'data/raw/trials/' and process them.

Includes heuristic filtering for junk images (logos, redactions).

Usage:
    python scripts/extract_pdfs.py
"""
import sys
import os
from pathlib import Path
import json
import shutil
import warnings
from PIL import Image  # <-- Make sure this is imported

# Updated imports for marker-pdf >= 0.2.0
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered


def extract_trial_pdf_marker(pdf_path: Path, output_dir: Path, converter):
    """
    Extracts text/tables (as Markdown) and images from a PDF using Marker.
    Saves output to a structured output directory.
    """
    print(f"  Processing: {pdf_path.name}...")

    # This is the directory where we'll save output
    marker_output_dir = output_dir / pdf_path.stem
    if marker_output_dir.exists():
        print(f"    - Output directory already exists, skipping.")
        return

    marker_output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # 1. THE ONLY SLOW STEP: Convert the PDF
        rendered = converter(str(pdf_path))

        # 2. GET TEXT & IMAGES. This is the correct way.
        markdown_text, _, images = text_from_rendered(rendered)

        # 3. Save the Markdown text
        md_path = marker_output_dir / f"{pdf_path.stem}.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(markdown_text)

        # Correctly count tables from metadata
        table_count = 0
        if "page_stats" in rendered.metadata:
            for page in rendered.metadata["page_stats"]:
                if "block_counts" in page:
                    for block_type, count in page["block_counts"]:
                        if block_type == "Table":
                            table_count += count

        print(f"    ✅ Markdown saved: {md_path}")
        print(f"    \tWords: {len(markdown_text.split())}, Tables: {table_count}")

        # 4. Save the Images (with Heuristic Filtering)
        image_dir = marker_output_dir / "images"
        image_dir.mkdir(parents=True, exist_ok=True)

        image_count = 0
        # 'images' is a DICTIONARY {name: image_object}
        for img_name, img_obj in images.items():
            try:
                # img_obj is ALREADY a <PIL.Image.Image> object

                # --- Start Heuristic Filter ---
                # Filter 1: Small images (icons/logos)
                if img_obj.width < 100 or img_obj.height < 50:
                    print(f"    - ℹ️  Filtering junk image (too small): {img_name}")
                    continue

                # Filter 2: Color simplicity (redactions/blank boxes)
                # getcolors() returns None if color count > 256 (a complex image)
                colors = img_obj.getcolors(maxcolors=256)

                if colors:  # It has 256 or fewer colors. Check if it's one solid color.
                    colors.sort(reverse=True)  # Sort by count
                    dominant_color_count = colors[0][0]
                    total_pixels = img_obj.width * img_obj.height

                    # Check if the most dominant color is > 75% of the image
                    if (dominant_color_count / total_pixels) > 0.75:
                        print(
                            f"    - ℹ️  Filtering junk image (color simplicity > 75%): {img_name}"
                        )
                        continue
                # --- End Heuristic Filter ---

                # If it passes all filters, save it
                img_path = image_dir / img_name
                img_obj.save(img_path)
                image_count += 1

            except Exception as img_e:
                print(f"    - ⚠️  Skipping one image. Error saving {img_name}: {img_e}")

        if image_count > 0:  # Use the count of *saved* images
            print(f"    🖼️  Extracted {image_count} good images to: {image_dir}")

        # 5. BONUS: Save the REAL metadata
        meta_path = marker_output_dir / f"{pdf_path.stem}_pagemeta.json"

        real_metadata = rendered.metadata

        with open(meta_path, "w", encoding="utf-8") as f:
            try:
                json.dump(real_metadata, f, indent=2)
                print(f"    ℹ️  Page metadata saved to: {meta_path}")
            except Exception as json_e:
                print(f"      - ⚠️  Error saving metadata JSON: {json_e}")
                f.write(str(real_metadata))

    except Exception as e:
        print(f"    ❌ Error processing {pdf_path.name}: {e}")
        import traceback

        traceback.print_exc()


def main():
    base_trials_dir = Path("data/raw/trials/")
    base_output_dir = Path("output")

    trial_folders = [d for d in base_trials_dir.glob("*") if d.is_dir()]
    if not trial_folders:
        print(f"❌ No trial folders found in {base_trials_dir}")
        sys.exit(1)

    print(f"🔬 Found {len(trial_folders)} trials. Starting Marker extraction...")
    print("Loading Marker models (first time will download models)...")

    warnings.filterwarnings(
        "ignore",
        message="`TableRecEncoderDecoderModel` is not compatible with mps backend.",
    )

    # Initialize converter once (auto-detects MPS on M4 or NVIDIA GPU on RunPod)
    converter = PdfConverter(
        artifact_dict=create_model_dict(),
    )

    print(f"✅ Models loaded. Processing {len(trial_folders)} trials...\n")

    for trial_folder in trial_folders:
        trial_id = trial_folder.name
        output_root = base_output_dir / trial_id
        output_root.mkdir(parents=True, exist_ok=True)

        pdf_files = list(trial_folder.glob("*.pdf"))

        if not pdf_files:
            print(f"--- Skipping Trial: {trial_id} (No PDF files found) ---")
            continue

        print(f"--- Processing Trial: {trial_id} ({len(pdf_files)} PDFs) ---")

        for pdf_path in pdf_files:
            extract_trial_pdf_marker(pdf_path, output_root, converter)

        print(
            f"--- ✅ Finished trial: {trial_id}. Output saved to: {output_root} ---\n"
        )

    print(f"\n========================================================")
    print(f"📊 FINAL EXTRACTION SUMMARY: ALL {len(trial_folders)} TRIALS COMPLETE")
    print(f"💾 All output saved to: {base_output_dir}")
    print(f"========================================================")


if __name__ == "__main__":
    main()
