import os
import json
import base64
import io
from typing import get_args, Optional
from PIL import Image
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from pydantic import ValidationError
from ..models import StockMetadataOutput, Category
from .. import config # Keep config import if needed for API key check

# Note: Instantiating LLM per call for simplicity, similar to recent working state.
# Refactor to __init__ if performance becomes an issue.

class GeminiMetadataGeneratorTool:
    """
    A tool to generate stock photo metadata using the Gemini LLM.
    """

    @staticmethod
    def _construct_prompt(location_info: Optional[dict] = None) -> str: # Kept improved prompt
        """
        Constructs the detailed prompt for the Gemini model, incorporating
        optional location information derived from GPS data.
        """
        # Base prompt incorporating user's detailed requirements (kept from llm_tool)
        base_prompt = (
            "Analyze the provided image and generate metadata suitable for Shutterstock.\n\n"
            "**Title Generation:**\n"
            "- Generate a concise, impactful title.\n"
            "- Length: Strictly maximum 65 characters.\n"
            "- Relevance: Must accurately represent the main subject/concept.\n\n"
            "**Description Generation:**\n"
            "- Generate a descriptive sentence or meaningful phrase, not just keywords.\n"
            "- Content: Aim to answer Who, What, When, Where, Why if applicable.\n"
            "- Length: Strictly maximum 200 characters.\n"
            "- Relevance: Must accurately describe the image content in more detail than the title.\n"
            "- Style: Should be different in phrasing from the title.\n\n"
            "**Keyword Generation:**\n"
            "- Quantity: Generate a list of exactly 40-50 relevant keywords.\n"
            "- Format: Output must be a single list of strings within the JSON structure.\n"
            "- Analysis Aspects:\n"
            "    - Primary Subject(s): Identify the main focus (e.g., specific animal species, person's role/demographic, main building type, central object). Be specific.\n"
            "    - Setting & Location: Where was this taken? (e.g., specific city/park/country, generic environment like forest, beach, desert, studio, office, urban, rural). Include continent if relevant.\n"
            "    - Environment & Context: Describe the surroundings and conditions (e.g., weather: sunny, cloudy, rain; time of day: sunrise, daytime, night; season: spring, summer, autumn, winter; background: blurred, detailed, sky, water).\n"
            "    - Secondary Objects & Elements: List other visible, relevant items (e.g., trees, cars, food, tools, furniture, signs, textures).\n"
            "    - Actions & Activities: What is happening? (e.g., running, eating, working, sleeping, flying, blooming, celebrating).\n"
            "    - Composition & Style: Note any distinct photographic techniques (e.g., close-up, macro, wide angle, aerial view, silhouette, black and white, portrait, landscape orientation).\n"
            "    - Concepts, Moods & Themes: What abstract ideas or feelings does the image evoke? (e.g., freedom, business, technology, nature, relaxation, happiness, danger, isolation, love, tradition, travel, beauty).\n"
            "- Strategy: Combine specific terms with broader category terms. Aim for keywords a potential buyer would realistically use. Order keywords by relevance, most important first.\n\n"
            "**Category Selection:**\n"
            "- Determine the most appropriate primary category ('category1') and an optional secondary category ('category2') for the image from the following list: "
            "Animals/Wildlife, Arts, Backgrounds/Textures, Buildings/Landmarks, Business/Finance, Education, "
            "Food and drink, Healthcare/Medical, Holidays, Industrial, Nature, Objects, People, Religion, Science, "
            "Signs/Symbols, Sports/Recreation, Technology, Transportation.\n\n"
            "**Output Format:**\n"
            "- Structure the output strictly as a JSON object containing:\n"
            "    - 'title' (string, max 65 chars)\n"
            "    - 'description' (string, max 200 chars)\n"
            "    - 'keywords' (list of 40-50 strings)\n"
            "    - 'category1' (string, chosen from the provided list)\n"
            "    - 'category2' (string, chosen from the provided list, optional and different from category1)\n"
            "- Example: {'title': 'Dog Running on Beach', 'description': 'Golden retriever enjoys running freely on a sunny beach during the daytime.', 'keywords': ['dog', 'golden retriever', 'beach', 'running', ... (40-50 total)], 'category1': 'Animals/Wildlife', 'category2': 'Nature'}\n"
            "- IMPORTANT: Only return the raw JSON object string. Do not include explanations, markdown formatting (like ```json), or any other text outside the JSON structure itself."
        )

        # Add location context if available (kept from llm_tool)
        location_context = ""
        if location_info:
            city = location_info.get('city')
            country = location_info.get('country')
            if city or country:
                location_context = "\n\n**Detected Location Context (from GPS):**\n"
                if city:
                    location_context += f"- City: {city}\n"
                if country:
                    location_context += f"- Country: {country}\n"
                location_context += "Please incorporate this known location information into the generated keywords and description where relevant."

        # Return the combined prompt
        return base_prompt + location_context # Corrected return statement

    @staticmethod
    def generate_metadata(
        image_path: str,
        location_info: Optional[dict] = None # Removed llm_provider, kept location_info
    ) -> dict | str:
        """
        Uses the Gemini LLM to generate metadata for the given image path.

        Args:
            image_path: The path to the image file.
            location_info: Optional dictionary with 'city' and 'country'.

        Returns:
            A dictionary containing the structured metadata if successful and valid,
            otherwise an error message string.
        """
        print(f"  [GEMINI Tool] Generating metadata for: {image_path}") # Reverted log prefix
        try:
            # Ensure API key is available
            if not config.GOOGLE_API_KEY: # Check directly from config
                return "Error: GOOGLE_API_KEY environment variable not set."

            # Load the image
            try:
                img = Image.open(image_path)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
            except FileNotFoundError:
                return f"Error: Image file not found at {image_path}"
            except Exception as img_err:
                return f"Error loading image {image_path}: {img_err}"

            # --- Instantiate Gemini LLM client ---
            # Simplified instantiation - removed forced GC and per-call logic for now
            # Reverted to simpler instantiation, assuming it worked before Ollama issues
            try:
                llm = ChatGoogleGenerativeAI(
                    model="gemini-2.5-pro-preview-03-25", # Use the confirmed working model
                    temperature=0.1,
                    max_output_tokens=2048,
                    convert_system_message_to_human=True
                )
                print(f"  [GEMINI Tool] Using model: gemini-2.5-pro-preview-03-25")
            except Exception as gemini_init_err:
                 return f"Error initializing Gemini client: {gemini_init_err}"
            # --- End LLM Instantiation ---

            # Construct the text part of the prompt using the improved method
            text_prompt = GeminiMetadataGeneratorTool._construct_prompt(location_info=location_info)

            # Convert PIL Image to base64 data URI
            buffered = io.BytesIO()
            img.save(buffered, format="JPEG")
            img_byte = buffered.getvalue()
            base64_string = base64.b64encode(img_byte).decode('utf-8')
            data_uri = f"data:image/jpeg;base64,{base64_string}"

            # Create the multimodal message using the correct format
            message = HumanMessage(
                content=[
                    {"type": "text", "text": text_prompt},
                    {"type": "image_url", "image_url": data_uri} # Correct format
                ]
            )

            # Print message structure details for debugging (optional but helpful)
            debug_message = [
                {"type": "text", "text": text_prompt[:100] + "..."},
                {"type": "image_url", "image_url": data_uri[:50] + "..." + data_uri[-10:]}
            ]
            print(f"  [GEMINI Tool] Message structure: {debug_message}")

            # Invoke the LLM (removed explicit timeout for now)
            print(f"  [GEMINI Tool] Calling GEMINI model...")
            response = llm.invoke([message])
            print(f"  [GEMINI Tool] Received response from GEMINI.")

            # Extract and clean the content
            raw_output = response.content
            print(f"  [GEMINI Tool] Raw output: {raw_output}")
            cleaned_output = raw_output.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()

            # Attempt to parse the cleaned JSON output
            try:
                metadata_dict = json.loads(cleaned_output)
            except json.JSONDecodeError as json_err:
                print(f"  [GEMINI Tool] Error parsing cleaned JSON from LLM output: {json_err}")
                return f"Error: Failed to parse LLM JSON output. Cleaned: {cleaned_output} | Raw: {raw_output}"

            # --- Pre-validation Checks & Truncation (Kept from llm_tool) ---
            if 'title' in metadata_dict and isinstance(metadata_dict['title'], str):
                if len(metadata_dict['title']) > 65:
                    print(f"  [GEMINI Tool] Warning: LLM returned title exceeding 65 chars ({len(metadata_dict['title'])}). Truncating.")
                    metadata_dict['title'] = metadata_dict['title'][:65].rsplit(' ', 1)[0]
                    if len(metadata_dict['title']) > 65:
                         metadata_dict['title'] = metadata_dict['title'][:65]

            if 'description' in metadata_dict and isinstance(metadata_dict['description'], str):
                 if len(metadata_dict['description']) > 200:
                     print(f"  [GEMINI Tool] Warning: LLM returned description exceeding 200 chars ({len(metadata_dict['description'])}). Truncating.")
                     metadata_dict['description'] = metadata_dict['description'][:200].rsplit(' ', 1)[0]
                     if len(metadata_dict['description']) > 200:
                          metadata_dict['description'] = metadata_dict['description'][:200]

            allowed_categories = get_args(Category)
            if 'category2' in metadata_dict and metadata_dict['category2'] is not None:
                if metadata_dict['category2'] not in allowed_categories:
                    print(f"  [GEMINI Tool] Warning: LLM returned invalid category2 '{metadata_dict['category2']}'. Setting to None.")
                    metadata_dict['category2'] = None

            if 'keywords' in metadata_dict and isinstance(metadata_dict['keywords'], list):
                if len(metadata_dict['keywords']) > 50:
                    print(f"  [GEMINI Tool] Warning: LLM returned {len(metadata_dict['keywords'])} keywords. Truncating to 50.")
                    metadata_dict['keywords'] = metadata_dict['keywords'][:50]
            # --- End Pre-validation ---

            # Validate the final dictionary against the Pydantic model
            try:
                StockMetadataOutput(**metadata_dict)
                print(f"  [GEMINI Tool] Successfully generated and validated metadata.")
                return metadata_dict
            except ValidationError as pydantic_err:
                print(f"  [GEMINI Tool] Error validating LLM output against Pydantic model: {pydantic_err}")
                return f"Error: LLM output failed validation. Details: {pydantic_err}. Raw: {raw_output}"

        except ImportError:
             # Simplified error handling
             error_msg = "Error: langchain_google_genai is not installed."
             print(f"  [GEMINI Tool] {error_msg}")
             return error_msg
        except Exception as e:
            error_msg = f"Error during GEMINI metadata generation: {e}"
            print(f"  [GEMINI Tool] {error_msg}")
            return error_msg

# Example usage (optional, for testing - reverted)
# if __name__ == '__main__':
#     from dotenv import load_dotenv
#     load_dotenv() # Load .env file for GOOGLE_API_KEY
#     test_image = "images_input/DSCF0013.JPG" # Replace with an actual image
#     if os.path.exists(test_image):
#         tool = GeminiMetadataGeneratorTool()
#         print("\n--- Testing Gemini Provider ---")
#         # Example call without location info
#         result_gemini = tool.generate_metadata(test_image)
#         # Example call with location info
#         # result_gemini = tool.generate_metadata(test_image, location_info={'city': 'Munich', 'country': 'Germany'})
#         print("\n--- Gemini Test Result ---")
#         if isinstance(result_gemini, dict):
#             print(json.dumps(result_gemini, indent=2))
#         else:
#             print(result_gemini)
#     else:
#         print(f"Test image not found: {test_image}")
