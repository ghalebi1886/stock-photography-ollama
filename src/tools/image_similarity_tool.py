from sentence_transformers import SentenceTransformer
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json
from .image_loader_tool import ImageLoaderTool # Use our loader

class ImageSimilarityTool:
    """
    Generates image embeddings using a CLIP model and calculates similarity.
    """
    # Load the CLIP model. This might take time on first run as it downloads the model.
    # Consider making the model name configurable in config.py later.
    MODEL_NAME = 'clip-ViT-B-32'
    _model = None # Class variable to hold the loaded model

    @classmethod
    def _load_model(cls):
        """Loads the SentenceTransformer model if not already loaded."""
        if cls._model is None:
            try:
                print(f"Loading SentenceTransformer model: {cls.MODEL_NAME}...")
                cls._model = SentenceTransformer(cls.MODEL_NAME)
                print("Model loaded successfully.")
            except Exception as e:
                print(f"Error loading SentenceTransformer model '{cls.MODEL_NAME}': {e}")
                # Set model to an indicator that loading failed, to prevent retries
                cls._model = "loading_failed"
        # Return True if model is loaded, False otherwise
        return cls._model is not None and cls._model != "loading_failed"

    @staticmethod
    def generate_embedding(image_path: str) -> np.ndarray | None:
        """
        Generates an embedding for the image at the specified path.

        Args:
            image_path: The path to the image file.

        Returns:
            A NumPy array representing the image embedding, or None if an error occurs.
        """
        if not ImageSimilarityTool._load_model():
             print("Cannot generate embedding, model failed to load.")
             return None

        pil_image = ImageLoaderTool.load_image(image_path)
        if pil_image is None:
            print(f"Skipping embedding generation for {image_path} due to loading error.")
            return None

        try:
            # Ensure image is in RGB format for CLIP model
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')

            # Generate embedding
            embedding = ImageSimilarityTool._model.encode(pil_image)
            print(f"Successfully generated embedding for: {image_path}")
            return embedding
        except Exception as e:
            print(f"Error generating embedding for {image_path}: {e}")
            return None

    @staticmethod
    def calculate_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float | None:
        """
        Calculates the cosine similarity between two embeddings.

        Args:
            embedding1: The first embedding (NumPy array).
            embedding2: The second embedding (NumPy array).

        Returns:
            The cosine similarity score (float between -1 and 1), or None if inputs are invalid.
        """
        if embedding1 is None or embedding2 is None:
            return None
        if not isinstance(embedding1, np.ndarray) or not isinstance(embedding2, np.ndarray):
             print("Error: Embeddings must be NumPy arrays.")
             return None
        if embedding1.ndim == 1: embedding1 = embedding1.reshape(1, -1)
        if embedding2.ndim == 1: embedding2 = embedding2.reshape(1, -1)
        if embedding1.shape[1] != embedding2.shape[1]:
             print("Error: Embeddings must have the same dimension.")
             return None

        try:
            similarity = cosine_similarity(embedding1, embedding2)[0][0]
            return float(similarity) # Ensure it's a standard float
        except Exception as e:
            print(f"Error calculating similarity: {e}")
            return None

# Example usage (optional, for testing)
if __name__ == '__main__':
    # Requires sentence-transformers, torch/tensorflow, Pillow, scikit-learn
    # Assumes dummy_test_image.png exists
    test_image_path1 = "dummy_test_image.png"
    # Create a slightly different dummy image for comparison
    test_image_path2 = "dummy_test_image_blue.png"

    try:
        import os
        if not os.path.exists(test_image_path1) and "dummy" in test_image_path1:
             print(f"Warning: {test_image_path1} not found. Run image_loader_tool.py first.")
        else:
            # Create second dummy image
            try:
                dummy_image_blue = Image.new('RGB', (60, 30), color = 'blue')
                dummy_image_blue.save(test_image_path2)
                print(f"Created {test_image_path2}")
            except Exception as save_e:
                 print(f"Could not create second dummy image: {save_e}")
                 test_image_path2 = None # Prevent further use

            tool = ImageSimilarityTool()

            # Generate embeddings (model loading happens here)
            embedding1 = tool.generate_embedding(test_image_path1)

            if test_image_path2 and os.path.exists(test_image_path2):
                 embedding2 = tool.generate_embedding(test_image_path2)
            else:
                 embedding2 = None


            if embedding1 is not None:
                print(f"\nEmbedding 1 shape: {embedding1.shape}")

                # Test similarity with self
                similarity_self = tool.calculate_similarity(embedding1, embedding1)
                print(f"Similarity ({test_image_path1} vs self): {similarity_self}")

                # Test similarity with second image
                if embedding2 is not None:
                    print(f"Embedding 2 shape: {embedding2.shape}")
                    similarity_other = tool.calculate_similarity(embedding1, embedding2)
                    print(f"Similarity ({test_image_path1} vs {test_image_path2}): {similarity_other}")
                else:
                     print("Could not test similarity with second image.")

            else:
                 print("Could not generate embedding for the first image.")

            # Test non-existent file
            print("\nTesting non-existent file:")
            tool.generate_embedding("non_existent_image.jpg")

    except ImportError as imp_err:
        print(f"Import Error during example usage: {imp_err}. Make sure sentence-transformers, scikit-learn, Pillow, torch/tensorflow are installed.")
    except Exception as e:
        print(f"An error occurred during example usage: {e}")
