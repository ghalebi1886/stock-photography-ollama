from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch # Often needed implicitly by transformers
from .image_loader_tool import ImageLoaderTool # Use our loader

class ImageDescriberKeywordTool:
    """
    Generates a textual description (caption) for an image using a pre-trained model.
    This provides the raw description capability; agents will refine the output.
    """
    # Consider making model names configurable in config.py later.
    MODEL_NAME = "Salesforce/blip-image-captioning-large"
    _processor = None
    _model = None

    @classmethod
    def _load_model_and_processor(cls):
        """Loads the BLIP model and processor if not already loaded."""
        if cls._processor is None or cls._model is None:
            try:
                print(f"Loading BLIP processor and model: {cls.MODEL_NAME}...")
                cls._processor = BlipProcessor.from_pretrained(cls.MODEL_NAME)
                cls._model = BlipForConditionalGeneration.from_pretrained(cls.MODEL_NAME)
                # Check if GPU is available and move model if desired (optional optimization)
                # cls._device = "cuda" if torch.cuda.is_available() else "cpu"
                # cls._model.to(cls._device)
                print("BLIP model and processor loaded successfully.")
            except Exception as e:
                print(f"Error loading BLIP model/processor '{cls.MODEL_NAME}': {e}")
                cls._processor = "loading_failed"
                cls._model = "loading_failed"
        # Return True if loaded, False otherwise
        return cls._model is not None and cls._model != "loading_failed"

    @staticmethod
    def generate_text_from_image(image_path: str, max_length: int = 50, min_length: int = 10) -> str | None:
        """
        Generates a textual description for the image at the specified path.

        Args:
            image_path: The path to the image file.
            max_length: Maximum length for the generated caption.
            min_length: Minimum length for the generated caption.


        Returns:
            A string containing the generated text description, or None if an error occurs.
        """
        if not ImageDescriberKeywordTool._load_model_and_processor():
            print("Cannot generate text, model/processor failed to load.")
            return None

        pil_image = ImageLoaderTool.load_image(image_path)
        if pil_image is None:
            print(f"Skipping text generation for {image_path} due to loading error.")
            return None

        try:
            # Ensure image is in RGB format
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')

            # Prepare image for the model
            inputs = ImageDescriberKeywordTool._processor(pil_image, return_tensors="pt") # .to(cls._device) # Add device if using GPU

            # Generate caption
            # Adjust generation parameters as needed (e.g., num_beams, temperature)
            out = ImageDescriberKeywordTool._model.generate(
                **inputs,
                max_length=max_length,
                min_length=min_length,
                num_beams=4, # Example beam search parameter
                early_stopping=True
            )

            # Decode the generated text
            caption = ImageDescriberKeywordTool._processor.decode(out[0], skip_special_tokens=True)
            print(f"Successfully generated text for: {image_path}")
            return caption.strip()

        except ImportError:
             print("Error: Transformers or Torch not installed correctly.")
             return None
        except Exception as e:
            print(f"Error generating text for {image_path}: {e}")
            return None

# Example usage (optional, for testing)
if __name__ == '__main__':
    # Requires transformers, torch, Pillow
    # Assumes dummy_test_image.png exists
    test_image_path = "dummy_test_image.png"
    # test_image_path = "path/to/your/real_image.jpg" # Replace for more meaningful results

    try:
        import os
        if not os.path.exists(test_image_path) and "dummy" in test_image_path:
             print(f"Warning: {test_image_path} not found. Run image_loader_tool.py first or provide a real image path.")
        else:
            tool = ImageDescriberKeywordTool()
            # Model loading happens on first call
            description = tool.generate_text_from_image(test_image_path)

            if description:
                print(f"\nGenerated Description for {test_image_path}:")
                print(description)
            else:
                print(f"\nCould not generate description for {test_image_path}.")

            # Test non-existent file
            print("\nTesting non-existent file:")
            tool.generate_text_from_image("non_existent_image.jpg")

    except ImportError as imp_err:
        print(f"Import Error during example usage: {imp_err}. Make sure transformers, torch, Pillow are installed.")
    except Exception as e:
        print(f"An error occurred during example usage: {e}")
