import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Configuration Constants ---

# Similarity threshold for Stage 2 duplicate detection
# Images with cosine similarity >= this value will be considered duplicates.
# Needs tuning based on testing (Task 9).
SIMILARITY_THRESHOLD = 0.95

# Input and Output directories
INPUT_IMAGE_DIR = "images_input"
OUTPUT_REPORTS_DIR = "output_reports"
OUTPUT_PROCESSED_DIR = "images_output_processed" # Directory for images with embedded metadata

# Path to ExifTool executable (can be overridden by environment variable)
EXIFTOOL_PATH = os.getenv("EXIFTOOL_PATH", "exiftool")

# Google API Key (required for GeminiMetadataGeneratorTool)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Add other configurations as needed
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


print("Configuration loaded.")
print(f"Similarity Threshold: {SIMILARITY_THRESHOLD}")
print(f"Input Directory: {INPUT_IMAGE_DIR}")
print(f"Output Reports Directory: {OUTPUT_REPORTS_DIR}")
print(f"Output Processed Images Directory: {OUTPUT_PROCESSED_DIR}")
print(f"ExifTool Path: {EXIFTOOL_PATH}")
