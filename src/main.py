# Main script for the Stock Photography Assessment Crew
# V2 - Refactored into stages, added metadata embedding

import os
import glob
import json
import numpy as np
from datetime import datetime
from collections import defaultdict
import sys
import io
from typing import Dict, List, Any, Optional
# Removed argparse import

# Attempt to force UTF-8 standard streams
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Configuration
from src.config import (
    INPUT_IMAGE_DIR,
    OUTPUT_REPORTS_DIR,
    OUTPUT_PROCESSED_DIR,
    SIMILARITY_THRESHOLD,
    EXIFTOOL_PATH
    # Removed Shutterstock credential imports
)

# Tool Implementations
from src.tools.metadata_extractor_tool import MetadataExtractorTool
from src.tools.technical_analyzer_tool import BasicTechnicalAnalyzerTool
from src.tools.image_similarity_tool import ImageSimilarityTool
from src.tools.gemini_metadata_generator_tool import GeminiMetadataGeneratorTool # Reverted import
from src.tools.metadata_embedder_tool import MetadataEmbedderTool
# Removed ShutterstockSubmitterTool import

# Model/Validation Imports (Still needed for Gemini Tool validation)
from src.models import StockMetadataOutput
from pydantic import ValidationError

# Other Imports
from sklearn.metrics.pairwise import cosine_similarity
# import litellm # Keep if debugging needed elsewhere, otherwise remove if no CrewAI agents used

# Enable LiteLLM debugging (If needed)
# litellm._turn_on_debug()

# --- Helper Functions ---

def find_image_files(input_dir: str) -> List[str]:
    """Finds supported image files (jpg, jpeg, png, tiff) in the input directory."""
    supported_extensions = [
        "*.jpg", "*.jpeg", "*.png", "*.tiff",
        "*.JPG", "*.JPEG", "*.PNG", "*.TIFF"
    ]
    image_files = []
    for ext in supported_extensions:
        image_files.extend(glob.glob(os.path.join(input_dir, ext)))
    print(f"Found {len(image_files)} image files in {input_dir}.")
    return image_files

def initialize_results(image_files: List[str]) -> Dict[str, Dict[str, Any]]:
    """Creates an initial dictionary to store results for each image."""
    results = {}
    for img_path in image_files:
        filename = os.path.basename(img_path)
        results[filename] = {
            "filename": filename,
            "original_filepath": img_path, # Store original path
            "processed_filepath": None, # Path after embedding
            "status": "Pending",
            "assessment": {
                "metadata": None,
                "quality": None,
                "stock_metadata": None,
            },
            "error_message": None
            # Removed similarity_embedding from final results structure
        }
    return results

# --- Stage Functions ---

def run_stage1_quality_screening(
    all_results: Dict[str, Dict[str, Any]],
    metadata_tool: MetadataExtractorTool,
    quality_tool: BasicTechnicalAnalyzerTool
) -> List[str]:
    """Performs Stage 1: Metadata extraction and technical quality checks."""
    print("\n--- Stage 1: Quality Screening ---")
    provisionally_accepted_files = []
    for filename, data in all_results.items():
        print(f"\nProcessing (Stage 1): {filename}")
        img_path = data["original_filepath"]

        try:
            # Task 1: Extract Metadata
            print(f"  Running Metadata Extraction...")
            metadata_result = metadata_tool.extract_metadata(img_path)
            all_results[filename]["assessment"]["metadata"] = metadata_result

            # Task 2: Assess Technical Quality
            print(f"  Running Technical Quality Assessment...")
            quality_result = quality_tool.perform_analysis(img_path)
            all_results[filename]["assessment"]["quality"] = quality_result

            # Decision Logic
            focus_ok = quality_result.get("focus_assessment") != "Blurry"
            exposure_ok = quality_result.get("exposure_assessment") == "Well-exposed"
            analysis_ok = quality_result.get("focus_assessment") != "Analysis Failed" and \
                          quality_result.get("exposure_assessment") != "Analysis Failed"

            if analysis_ok and focus_ok and exposure_ok:
                all_results[filename]["status"] = "Provisionally Accepted"
                provisionally_accepted_files.append(filename)
                print(f"  Status: Provisionally Accepted")
            elif not analysis_ok:
                 all_results[filename]["status"] = "Rejected - Analysis Failed"
                 all_results[filename]["error_message"] = "Focus or exposure analysis failed."
                 print(f"  Status: Rejected - Analysis Failed")
            else:
                all_results[filename]["status"] = "Rejected - Quality"
                reasons = []
                if not focus_ok: reasons.append(f"Focus: {quality_result.get('focus_assessment')}")
                if not exposure_ok: reasons.append(f"Exposure: {quality_result.get('exposure_assessment')}")
                all_results[filename]["error_message"] = f"Failed quality check. Reasons: {'; '.join(reasons)}"
                print(f"  Status: Rejected - Quality ({'; '.join(reasons)})")

        except Exception as e:
            print(f"  ERROR during Stage 1 processing for {filename}: {e}")
            all_results[filename]["status"] = "Error - Stage 1 Failed"
            all_results[filename]["error_message"] = str(e)
            # Ensure partial results are nulled if error occurred mid-processing
            if "assessment" not in all_results[filename]: all_results[filename]["assessment"] = {}
            if all_results[filename]["assessment"].get("metadata") is None: all_results[filename]["assessment"]["metadata"] = {"exif_data": {}}
            if all_results[filename]["assessment"].get("quality") is None: all_results[filename]["assessment"]["quality"] = {}

    return provisionally_accepted_files


def run_stage2_similarity_analysis(
    all_results: Dict[str, Dict[str, Any]],
    provisionally_accepted_files: List[str],
    similarity_tool: ImageSimilarityTool
) -> List[str]:
    """Performs Stage 2: Similarity analysis and duplicate rejection."""
    print(f"\n--- Stage 2: Similarity Analysis & Curation ({len(provisionally_accepted_files)} images) ---")
    selected_for_processing_files = []

    if not provisionally_accepted_files:
        print("No images passed Stage 1. Skipping Stage 2.")
        return selected_for_processing_files # Return empty list

    # Generate embeddings
    embeddings = {}
    print("  Generating embeddings...")
    for filename in provisionally_accepted_files:
        img_path = all_results[filename]["original_filepath"]
        embedding = similarity_tool.generate_embedding(img_path)
        if embedding is not None:
            embeddings[filename] = embedding
            # We don't store the embedding in the final result anymore
        else:
            print(f"  Warning: Could not generate embedding for {filename}. It will be excluded from similarity check.")
            all_results[filename]["status"] = "Error - Embedding Generation Failed"
            all_results[filename]["error_message"] = "Failed to generate similarity embedding."

    filenames_with_embeddings = list(embeddings.keys())
    if len(filenames_with_embeddings) < 2:
        print("  Not enough images with embeddings to perform similarity check.")
        # Mark all remaining as Selected for Processing
        for filename in filenames_with_embeddings:
             all_results[filename]["status"] = "Selected for Processing"
             selected_for_processing_files.append(filename)
        return selected_for_processing_files

    # Calculate similarity matrix
    print("  Calculating similarity matrix...")
    embedding_list = [embeddings[fname] for fname in filenames_with_embeddings]
    try:
        similarity_matrix = cosine_similarity(np.array(embedding_list))
    except Exception as sim_err:
         print(f"  ERROR: Failed to calculate similarity matrix: {sim_err}. Skipping duplicate rejection.")
         similarity_matrix = None

    if similarity_matrix is None:
         # Mark all as selected if matrix calculation failed
         for filename in filenames_with_embeddings:
              all_results[filename]["status"] = "Selected for Processing"
              selected_for_processing_files.append(filename)
         return selected_for_processing_files

    # Identify duplicate groups
    print(f"  Identifying duplicates (Threshold: {SIMILARITY_THRESHOLD})...")
    processed_indices = set()
    duplicate_groups = []

    for i in range(len(filenames_with_embeddings)):
        if i in processed_indices: continue
        current_group = {filenames_with_embeddings[i]}
        processed_indices.add(i)
        for j in range(i + 1, len(filenames_with_embeddings)):
            if j in processed_indices: continue
            similarity_score = similarity_matrix[i, j]
            if similarity_score >= SIMILARITY_THRESHOLD:
                current_group.add(filenames_with_embeddings[j])
                processed_indices.add(j)
        if len(current_group) > 1:
            duplicate_groups.append(current_group)
            print(f"    Found duplicate group: {current_group}")

    # Select best from duplicates and update status
    selected_count = 0
    rejected_duplicate_count = 0
    processed_in_groups = set()

    for group in duplicate_groups:
        best_image_filename = None
        highest_focus_score = -1.0
        for filename in group:
            processed_in_groups.add(filename)
            focus_score = all_results[filename]["assessment"]["quality"].get("focus_score")
            if focus_score is not None and focus_score > highest_focus_score:
                highest_focus_score = focus_score
                best_image_filename = filename

        if best_image_filename:
            for filename in group:
                if filename == best_image_filename:
                    all_results[filename]["status"] = "Selected for Processing"
                    selected_for_processing_files.append(filename)
                    selected_count += 1
                    print(f"    Selected '{filename}' from group {group} (Focus Score: {highest_focus_score})")
                else:
                    all_results[filename]["status"] = "Rejected - Duplicate"
                    all_results[filename]["error_message"] = f"Duplicate of {best_image_filename}"
                    rejected_duplicate_count += 1
        else:
             print(f"    Warning: Could not determine best image in group {group}. Marking all as duplicates.")
             for filename in group:
                  all_results[filename]["status"] = "Rejected - Duplicate"
                  all_results[filename]["error_message"] = "Could not determine best in group."
                  rejected_duplicate_count += 1

    # Mark images not in any duplicate group as selected
    for filename in filenames_with_embeddings:
        if filename not in processed_in_groups:
            all_results[filename]["status"] = "Selected for Processing"
            selected_for_processing_files.append(filename)
            selected_count += 1

    print(f"  Stage 2 Complete: {selected_count} selected for processing, {rejected_duplicate_count} rejected as duplicates.")
    return selected_for_processing_files


def run_stage3_metadata_generation(
    all_results: Dict[str, Dict[str, Any]],
    selected_files: List[str],
    gemini_tool: GeminiMetadataGeneratorTool # Reverted tool instance variable name
    # Removed llm_provider argument
):
    """Performs Stage 3: Generates stock metadata using the Gemini LLM tool."""
    # Reverted print statement
    print(f"\n--- Stage 3: Metadata Generation ---")
    print(f"{len(selected_files)} images selected for metadata generation.")

    if not selected_files:
        return # Nothing to process

    for filename in selected_files:
        print(f"\nProcessing (Stage 3): {filename}")
        # Check if status is still 'Selected for Processing' before proceeding
        if all_results[filename]["status"] != "Selected for Processing":
             print(f"  Skipping Stage 3 for {filename}, status is '{all_results[filename]['status']}'")
             continue

        img_path = all_results[filename]["original_filepath"]
        # Get location info extracted in Stage 1
        metadata = all_results[filename]["assessment"].get("metadata")
        location_info = metadata.get("location") if metadata else None
        if location_info:
             print(f"  Using location info for prompt: {location_info.get('city', 'N/A')}, {location_info.get('country', 'N/A')}")

        try:
            # Call the reverted tool, passing only location info
            stock_meta_result = gemini_tool.generate_metadata(
                image_path=img_path,
                location_info=location_info
                # Removed llm_provider argument
            )

            if isinstance(stock_meta_result, dict):
                all_results[filename]["assessment"]["stock_metadata"] = stock_meta_result
                # Status will be updated after embedding attempt
                print(f"  Successfully generated metadata via Gemini tool.") # Reverted log
            else:
                # Error message now comes from the tool itself
                print(f"  ERROR from Gemini Tool: {stock_meta_result}") # Reverted log
                all_results[filename]["status"] = "Error - Stage 3 Tool Failed"
                all_results[filename]["error_message"] = stock_meta_result
                all_results[filename]["assessment"]["stock_metadata"] = None

        except Exception as e:
            print(f"  UNEXPECTED ERROR during Stage 3 Tool execution for {filename}: {e}")
            all_results[filename]["status"] = "Error - Stage 3 Unexpected Failure"
            all_results[filename]["error_message"] = f"Unexpected error: {str(e)}"
            all_results[filename]["assessment"]["stock_metadata"] = None


def run_stage3_5_metadata_embedding(
    all_results: Dict[str, Dict[str, Any]],
    selected_files: List[str],
    embedder_tool: MetadataEmbedderTool,
    output_processed_dir: str
):
    """Performs Stage 3.5: Embeds generated metadata into image files."""
    print("\n--- Stage 3.5: Metadata Embedding ---")
    if not selected_files:
        print("No files selected from previous stages. Skipping embedding.")
        return

    for filename in selected_files:
        # Only attempt embedding if metadata generation was successful
        if all_results[filename]["status"] == "Selected for Processing" and \
           all_results[filename]["assessment"].get("stock_metadata"):

            print(f"\nProcessing (Stage 3.5): {filename}")
            original_path = all_results[filename]["original_filepath"]
            metadata_to_embed = all_results[filename]["assessment"]["stock_metadata"]

            try:
                processed_path = embedder_tool.embed_metadata(
                    original_image_path=original_path,
                    output_dir=output_processed_dir,
                    metadata=metadata_to_embed
                )

                if processed_path:
                    all_results[filename]["processed_filepath"] = processed_path
                    all_results[filename]["status"] = "Accepted & Embedded"
                    print(f"  Successfully embedded metadata for {filename}.")
                else:
                    all_results[filename]["status"] = "Error - Metadata Embedding Failed"
                    all_results[filename]["error_message"] = (all_results[filename].get("error_message") or "") + \
                                                             " | Failed to embed metadata using ExifTool."
                    print(f"  Failed to embed metadata for {filename} using ExifTool.")

            except Exception as e:
                 print(f"  UNEXPECTED ERROR during Stage 3.5 execution for {filename}: {e}")
                 all_results[filename]["status"] = "Error - Stage 3.5 Unexpected Failure"
                 all_results[filename]["error_message"] = f"Unexpected embedding error: {str(e)}"
        elif all_results[filename]["status"] == "Selected for Processing":
             # Metadata generation must have failed or been skipped
             print(f"  Skipping embedding for {filename}, metadata not generated successfully.")
             if not all_results[filename].get("error_message"): # Update status if not already an error
                 all_results[filename]["status"] = "Error - Stage 3 Skipped/Failed"
                 all_results[filename]["error_message"] = "Metadata generation failed or skipped."


def run_stage4_reporting(
    all_results: Dict[str, Dict[str, Any]],
    output_reports_dir: str
):
    """Performs Stage 4: Saves individual JSON reports."""
    print("\n--- Stage 4: Saving Reports ---")
    saved_count = 0
    error_count = 0

    # Ensure report directory exists
    os.makedirs(output_reports_dir, exist_ok=True)

    for filename, data in all_results.items():
        # Clean up temporary/internal data before saving report
        report_data = data.copy() # Work on a copy
        report_data.pop("similarity_embedding", None) # Should be gone already, but ensure

        # Determine final status if somehow missed
        if report_data["status"] == "Selected for Processing":
             report_data["status"] = "Error - Stage 3.5 Skipped/Failed"
             report_data["error_message"] = report_data.get("error_message", "Embedding stage skipped or failed.")
        elif report_data["status"] == "Provisionally Accepted":
             report_data["status"] = "Error - Workflow Incomplete"
             report_data["error_message"] = report_data.get("error_message", "Workflow did not complete after Stage 1.")

        output_filename = os.path.splitext(filename)[0] + "_assessment.json"
        output_path = os.path.join(output_reports_dir, output_filename)
        try:
            with open(output_path, 'w', encoding='utf-8') as f: # Ensure UTF-8 for reports
                json.dump(report_data, f, indent=2, ensure_ascii=False)
            saved_count += 1
        except Exception as e:
            print(f"  ERROR saving report for {filename} to {output_path}: {e}")
            error_count += 1

    print(f"\nSaved {saved_count} reports to '{output_reports_dir}'.")
    if error_count > 0:
        print(f"Failed to save {error_count} reports.") # Corrected duplicate print statement


# Removed run_stage5_shutterstock_submission function


def print_summary(all_results: Dict[str, Dict[str, Any]]):
    """Prints a final summary of image statuses."""
    print("\n--- Final Processing Summary ---")
    status_counts = defaultdict(int)
    for filename, data in all_results.items():
        status_counts[data['status']] += 1

    for status, count in sorted(status_counts.items()): # Sort for consistent output
        print(f"- {status}: {count}")

# Removed run_submit_only_mode function

def run_full_pipeline(): # Removed llm_provider parameter
    """Runs the complete image processing pipeline."""
    # Reverted print statement
    print(f"--- Stock Photography Assessment Crew - Full Pipeline Starting ---")
    start_time = datetime.now()

    # Ensure output directories exist
    os.makedirs(OUTPUT_REPORTS_DIR, exist_ok=True)
    os.makedirs(OUTPUT_PROCESSED_DIR, exist_ok=True) # Create processed images dir

    # 1. Find Images
    image_files = find_image_files(INPUT_IMAGE_DIR)
    if not image_files:
        print("No images found in input directory. Exiting.")
        return # Use return instead of exit in main function

    # 2. Initialize Results Structure
    all_results = initialize_results(image_files)

    # 3. Instantiate Tools
    print("\n--- Initializing Tools ---")
    # Removed submitter_tool initialization
    try:
        metadata_tool = MetadataExtractorTool()
        quality_tool = BasicTechnicalAnalyzerTool()
        similarity_tool = ImageSimilarityTool()
        gemini_tool = GeminiMetadataGeneratorTool() # Reverted class name
        # Instantiate embedder tool, catching potential ExifTool not found error
        embedder_tool = MetadataEmbedderTool(exiftool_path=EXIFTOOL_PATH)

        # Removed Submitter Tool instantiation

    except RuntimeError as e:
         print(f"FATAL ERROR during tool initialization: {e}")
         print("Please ensure ExifTool is installed and accessible.") # Removed duplicate print
         return # Stop execution if essential tool fails
    # Removed ValueError catch for submitter credentials
    except Exception as e:
         # Corrected indentation for the general exception block
         print(f"FATAL ERROR during tool initialization: {e}")
         return

    # Ensure AI models are loaded once before batch processing (if applicable)
    # Similarity tool loads lazily, Gemini tool loads per call currently
    # If Gemini performance is an issue, refactor its loading to __init__
    if not similarity_tool._load_model():
         print("FATAL ERROR: Failed to load similarity model. Exiting.")
         return

    # --- Execute Stages ---
    provisionally_accepted = run_stage1_quality_screening(all_results, metadata_tool, quality_tool)
    selected_for_processing = run_stage2_similarity_analysis(all_results, provisionally_accepted, similarity_tool)
    # Call stage 3 without provider
    run_stage3_metadata_generation(all_results, selected_for_processing, gemini_tool)
    run_stage3_5_metadata_embedding(all_results, selected_for_processing, embedder_tool, OUTPUT_PROCESSED_DIR)
    run_stage4_reporting(all_results, OUTPUT_REPORTS_DIR)

    # Removed Stage 5 (Shutterstock Submission) call


    # --- Final Summary ---
    print_summary(all_results)

    end_time = datetime.now()
    print(f"\n--- Workflow Completed in {end_time - start_time} ---")


# --- Main Execution ---

# Removed main() function with argparse

if __name__ == "__main__":
    run_full_pipeline() # Direct call without arguments
