from PIL import Image
import io

class ImageLoaderTool:
    """
    A simple tool to load image data using Pillow.
    Handles basic loading and potential errors.
    """

    @staticmethod
    def load_image(image_path: str) -> Image.Image | None:
        """
        Loads an image from the specified path.

        Args:
            image_path: The path to the image file.

        Returns:
            A Pillow Image object if successful, None otherwise.
        """
        try:
            img = Image.open(image_path)
            # Ensure image data is loaded, might help with some file handle issues
            img.load()
            print(f"Successfully loaded image: {image_path}")
            return img
        except FileNotFoundError:
            print(f"Error: Image file not found at {image_path}")
            return None
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None

# Example usage (optional, for testing)
if __name__ == '__main__':
    # Create a dummy image file for testing if needed
    # Note: This requires Pillow to be installed (pip install Pillow)
    try:
        dummy_image = Image.new('RGB', (60, 30), color = 'red')
        dummy_image.save("dummy_test_image.png")
        print("Created dummy_test_image.png")

        loader = ImageLoaderTool()
        loaded_img = loader.load_image("dummy_test_image.png")
        if loaded_img:
            print(f"Dummy image loaded successfully: Size {loaded_img.size}, Mode {loaded_img.mode}")

        # Test non-existent file
        loader.load_image("non_existent_image.png")

    except ImportError:
        print("Pillow is not installed. Cannot run example usage.")
    except Exception as e:
        print(f"An error occurred during example usage: {e}")
