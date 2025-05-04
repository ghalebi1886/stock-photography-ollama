# Stock Photography Processing System Documentation

## Overview

This system automates the process of analyzing, evaluating, and preparing stock photographs for commercial platforms. It provides an end-to-end pipeline that:

1. Screens images for technical quality (focus, exposure)
2. Eliminates duplicate or similar images
3. Generates optimized metadata for stock photo platforms
4. Embeds the metadata into image files
5. Produces detailed reports for each image

The system is designed to help photographers streamline their workflow when preparing images for stock photography sites like Shutterstock, Adobe Stock, and Getty.

## System Architecture

### Core Components

The application is structured as a modular Python system with the following components:

- **Main Script**: `src/main.py` - Orchestrates the entire workflow through four sequential stages
- **Tools**: Independent modules in `src/tools/` that provide specific functionality
- **Models**: Data structure definitions in `src/models.py` 
- **Configuration**: Central configuration in `src/config.py`
- **Agents**: CrewAI agent definitions in `src/agents/` (primarily for reference/future use)

### Directory Structure

```
.
├── .env                        # Environment variables (API keys)
├── images_input/               # Source images directory
├── images_output_processed/    # Output directory for processed images
├── output_reports/             # JSON reports for processed images
├── requirements.txt            # Python dependencies
└── src/
    ├── agents/                 # CrewAI agent definitions
    │   ├── metadata_architect.py
    │   ├── metadata_specialist.py
    │   └── quality_inspector.py
    ├── config.py               # System configuration
    ├── main.py                 # Main workflow script
    ├── models.py               # Data structure definitions
    └── tools/                  # Tool implementations
        ├── gemini_metadata_generator_tool.py
        ├── image_describer_tool.py
        ├── image_loader_tool.py
        ├── image_similarity_tool.py
        ├── metadata_embedder_tool.py
        ├── metadata_extractor_tool.py
        └── technical_analyzer_tool.py
```

## Workflow Stages

### Stage 1: Quality Screening

**Purpose**: Assess basic technical quality of images to filter out unsuitable candidates.

**Tools Used**:
- `MetadataExtractorTool`: Extracts EXIF data (camera model, lens model, ISO, f-number, exposure time)
- `BasicTechnicalAnalyzerTool`: Analyzes focus (sharpness) and exposure quality

**Rejection Criteria**:
- Blurry images (determined by Laplacian variance)
- Poorly exposed images (too dark or too bright)
- Failed analyses

**Output**: A list of "provisionally accepted" images that pass the quality checks.

### Stage 2: Similarity Analysis & Curation

**Purpose**: Identify and eliminate duplicate or highly similar images.

**Tool Used**:
- `ImageSimilarityTool`: Generates embeddings for each image using a CLIP model

**Process**:
1. Generate embeddings for all provisionally accepted images
2. Calculate a similarity matrix using cosine similarity
3. Group images with similarity above the threshold (default: 0.95)
4. For each group of similar images, select the one with the highest focus score
5. Mark rejected duplicates and selected images

**Output**: A curated list of unique images with the best technical quality.

### Stage 3: Metadata Generation

**Purpose**: Generate optimized metadata for stock photo platforms.

**Tool Used**:
- `GeminiMetadataGeneratorTool`: Uses Google's Gemini 2.5 Pro model to analyze images

**Generated Metadata**:
- Title/Description (under 200 characters)
- Keywords (7-50 relevant terms)
- Primary and optional secondary category from a predefined list

**Process**:
1. Load the image and convert to appropriate format
2. Build a prompt with detailed instructions for the Gemini model
3. Process the image with the model to generate metadata
4. Validate the structured output against the `StockMetadataOutput` Pydantic model

### Stage 3.5: Metadata Embedding

**Purpose**: Embed the generated metadata into image files.

**Tool Used**:
- `MetadataEmbedderTool`: Uses ExifTool to write metadata to images

**Process**:
1. Create a copy of the original image
2. Embed title/description into IPTC:Caption-Abstract and XMP:Description fields
3. Embed keywords into IPTC:Keywords and XMP:Subject fields
4. Save the processed image to the output directory

### Stage 4: Reporting

**Purpose**: Generate detailed reports for each processed image.

**Process**:
1. Compile all assessment data for each image
2. Format as structured JSON
3. Save to individual report files in the output directory

## Tools In Detail

### MetadataExtractorTool

**Purpose**: Extract EXIF metadata from image files.

**Key Features**:
- Uses the `exifread` library
- Extracts camera model, lens model, ISO, f-number, exposure time
- Handles missing tags and various edge cases
- Normalizes output format

### BasicTechnicalAnalyzerTool

**Purpose**: Assess image quality in terms of focus and exposure.

**Key Features**:
- Uses OpenCV and NumPy libraries
- Analyzes focus using variance of Laplacian (edge detection)
- Provides focus scores and textual assessment (Sharp, Slightly Soft, Blurry)
- Analyzes exposure based on mean pixel intensity

### ImageSimilarityTool

**Purpose**: Detect similar or duplicate images.

**Key Features**:
- Uses the `sentence-transformers` library with a CLIP model
- Generates embeddings that capture visual content
- Computes cosine similarity between embeddings
- Enables identification of visually similar images

### GeminiMetadataGeneratorTool

**Purpose**: Generate optimized metadata for stock photography.

**Key Features**:
- Uses Google's Gemini 2.5 Pro model via `langchain_google_genai`
- Analyzes image content and generates stock-optimized metadata
- Creates structured output validated against a Pydantic model
- Handles truncation and validation of model outputs

### MetadataEmbedderTool

**Purpose**: Embed metadata into image files.

**Key Features**:
- Uses ExifTool via `pyexiftool` library
- Writes to both IPTC and XMP metadata fields for compatibility
- Creates new copies of images with embedded data
- Preserves original files

### ImageLoaderTool

**Purpose**: Provide consistent image loading across the system.

**Key Features**:
- Uses the Pillow library
- Handles errors gracefully
- Centralizes image loading logic

### ImageDescriberKeywordTool

**Purpose**: Generate text descriptions from images (auxiliary tool).

**Key Features**:
- Uses the BLIP model (Salesforce/blip-image-captioning-large)
- Generates textual descriptions of image content
- Not directly used in the main workflow but available as a utility

## Data Models

### StockMetadataOutput

A Pydantic model that defines the structure for stock photography metadata:

- `title_description`: String, max 200 characters, commercial description
- `keywords`: List of strings, 7-50 relevant keywords
- `category1`: Primary category (from a predefined list)
- `category2`: Optional secondary category (from the same list)

## Configuration

The system configuration is centralized in `src/config.py`:

- `INPUT_IMAGE_DIR`: Location of input images ("images_input")
- `OUTPUT_REPORTS_DIR`: Location for output reports ("output_reports")
- `OUTPUT_PROCESSED_DIR`: Location for processed images ("images_output_processed")
- `SIMILARITY_THRESHOLD`: Threshold for considering images as duplicates (0.95)
- `EXIFTOOL_PATH`: Path to ExifTool executable (from env or default)
- `OLLAMA_MODEL`: Name of the Ollama model to use if `--llm ollama` is specified (e.g., "gemma3:27b"). Defaults to "gemma3:27b" if not set.
- `OLLAMA_BASE_URL`: Base URL for the Ollama server if `--llm ollama` is specified (e.g., "http://localhost:11434"). Defaults to "http://localhost:11434" if not set.
- `GOOGLE_API_KEY`: Required if using the default Gemini provider (`--llm gemini` or no argument).

## Environment Variables

The system uses these environment variables (defined in `.env`):

- `GOOGLE_API_KEY`: Required for Google Gemini API access (when using `--llm gemini`).
- `OLLAMA_MODEL`: Specifies the Ollama model to use (when using `--llm ollama`). Default: "gemma3:27b".
- `OLLAMA_BASE_URL`: Specifies the Ollama server URL (when using `--llm ollama`). Default: "http://localhost:11434".
- `EXIFTOOL_PATH`: Optional, custom path to ExifTool.
- `OPENAI_API_KEY`: Optional, for CrewAI integration (Not used by the core pipeline).
- `OPENROUTER_API_KEY`: Optional, for CrewAI integration (Not used by the core pipeline).

## Dependencies

Key dependencies include:

- **Image Processing**: Pillow, OpenCV, ExifRead
- **AI/ML**: sentence-transformers, transformers, torch, scikit-learn
- **API Integration**: langchain-google-genai, langchain-openai
- **Metadata Handling**: pyexiftool
- **Validation**: Pydantic

## Agent System Architecture (Reference)

The project includes agent definitions using the CrewAI framework, although the current implementation uses the tools directly:

- **MetadataArchitectAgent**: Generates stock-optimized titles/descriptions and keywords
- **MetadataSpecialistAgent**: Extracts technical metadata from images
- **QualityInspectorAgent**: Assesses technical quality and makes acceptance decisions

This architecture could be reactivated for a more agent-driven workflow if desired.

## Execution Flow

1. The main script locates images in the input directory
2. It initializes results tracking for each image
3. It instantiates all required tools
4. It executes each stage sequentially:
   - Stage 1: Quality screening
   - Stage 2: Similarity analysis
   - Stage 3: Metadata generation
   - Stage 3.5: Metadata embedding
   - Stage 4: Report generation
5. It summarizes the results

### Running the Pipeline

**Using Default LLM (Gemini):**
Requires `GOOGLE_API_KEY` set in `.env`.
```bash
python src/main.py
```
or explicitly:
```bash
python src/main.py --llm gemini
```

**Using Ollama:**
Requires Ollama server running locally with the desired model pulled.
Requires `OLLAMA_MODEL` and `OLLAMA_BASE_URL` set in `.env` (or rely on defaults).
```bash
python src/main.py --llm ollama
```

## Error Handling

- Each stage includes comprehensive error handling
- Individual image failures don't affect the overall process
- Tool initializations are checked to ensure essential components are available
- Detailed error messages are included in image reports

## Output Files

### Processed Images

Images that successfully complete the workflow are saved to `images_output_processed/` with embedded metadata.

### Reports

For each processed image, a JSON report is saved to `output_reports/` with the format:

```json
{
  "filename": "image.jpg",
  "original_filepath": "images_input/image.jpg",
  "processed_filepath": "images_output_processed/image.jpg",
  "status": "Accepted & Embedded",
  "assessment": {
    "metadata": {
      "exif_data": {
        "camera_model": "Canon EOS R5",
        "lens_model": "RF24-70mm F2.8 L IS USM",
        "iso": "100",
        "f_number": "f/8.0",
        "exposure_time": "1/250"
      }
    },
    "quality": {
      "focus_score": 156.78,
      "focus_assessment": "Sharp",
      "exposure_assessment": "Well-exposed"
    },
    "stock_metadata": {
      "title_description": "Urban skyline at sunset with dramatic clouds",
      "keywords": ["city", "skyline", "sunset", "urban", "architecture", "building", "dusk", "dramatic", "clouds", "cityscape"],
      "category1": "Buildings/Landmarks",
      "category2": "Nature"
    }
  },
  "error_message": null
}
```

## Status Codes

The system assigns one of these status codes to each image:

- `Pending`: Initial state
- `Provisionally Accepted`: Passed Stage 1 quality checks
- `Selected for Processing`: Passed Stage 2 similarity analysis
- `Accepted & Embedded`: Successfully completed the entire workflow
- `Rejected - Quality`: Failed quality checks (focus or exposure)
- `Rejected - Duplicate`: Identified as a duplicate of another image
- `Error - Stage X Failed`: Failed at a specific stage
- `Error - Analysis Failed`: Technical analysis could not be completed
- `Error - Embedding Failed`: Metadata could not be embedded

## Potential Enhancements

1. Multi-threading for parallel processing of images
2. Web interface for visualizing results and editing metadata
3. Support for video files and additional image formats
4. Cloud storage integration for input/output
5. Fine-tuning of ML models for specific types of stock photography
6. Reactivating the CrewAI agent system for more complex decision making
7. Additional metadata fields for specific stock platforms

## Conclusion

This system provides a comprehensive, automated pipeline for preparing stock photography. It combines computer vision techniques, machine learning, and metadata handling to streamline the often tedious process of preparing images for stock platforms. The modular design allows for easy extension and customization to meet specific needs.
