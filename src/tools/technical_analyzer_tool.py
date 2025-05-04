import cv2
import numpy as np
import json
from PIL import Image
from .image_loader_tool import ImageLoaderTool # Use our loader

class BasicTechnicalAnalyzerTool:
    """
    Performs basic technical analysis on an image: focus (blur) and exposure.
    """

    @staticmethod
    def analyze_focus(image: np.ndarray) -> tuple[float | None, str]:
        """
        Analyzes the focus of an image using the variance of the Laplacian.

        Args:
            image: A NumPy array representing the image (grayscale).

        Returns:
            A tuple containing the focus score (Laplacian variance) and a
            textual assessment ('Sharp', 'Slightly Soft', 'Blurry', or 'Analysis Failed').
            Returns (None, 'Analysis Failed') on error.
        """
        # These thresholds are subjective and will likely need tuning (Task 9)
        SHARP_THRESHOLD = 100.0
        SOFT_THRESHOLD = 50.0
        try:
            laplacian_var = cv2.Laplacian(image, cv2.CV_64F).var()
            if laplacian_var > SHARP_THRESHOLD:
                assessment = "Sharp"
            elif laplacian_var > SOFT_THRESHOLD:
                assessment = "Slightly Soft"
            else:
                assessment = "Blurry"
            return round(laplacian_var, 2), assessment
        except Exception as e:
            print(f"Error during focus analysis: {e}")
            return None, "Analysis Failed"

    @staticmethod
    def analyze_exposure(image: np.ndarray) -> str:
        """
        Performs a very basic exposure analysis based on mean pixel intensity.

        Args:
            image: A NumPy array representing the image (grayscale).

        Returns:
            A textual assessment ('Well-exposed', 'Under-exposed', 'Over-exposed',
            or 'Analysis Failed').
        """
        # These thresholds are simplistic and may need refinement (Task 9)
        # Assumes pixel values are in the range [0, 255]
        OVER_EXPOSED_THRESHOLD = 200
        UNDER_EXPOSED_THRESHOLD = 70
        try:
            mean_intensity = np.mean(image)
            if mean_intensity > OVER_EXPOSED_THRESHOLD:
                return "Over-exposed"
            elif mean_intensity < UNDER_EXPOSED_THRESHOLD:
                return "Under-exposed"
            else:
                return "Well-exposed"
        except Exception as e:
            print(f"Error during exposure analysis: {e}")
            return "Analysis Failed"

    @staticmethod
    def perform_analysis(image_path: str) -> dict:
        """
        Performs both focus and exposure analysis on the image.

        Args:
            image_path: The path to the image file.

        Returns:
            A dictionary containing the analysis results in the specified format.
        """
        results = {
            "focus_score": None,
            "focus_assessment": "Analysis Failed",
            "exposure_assessment": "Analysis Failed",
        }

        pil_image = ImageLoaderTool.load_image(image_path)
        if pil_image is None:
            print(f"Skipping technical analysis for {image_path} due to loading error.")
            return results # Return default failure values

        try:
            # Convert PIL image to OpenCV format (Grayscale for analysis)
            # Ensure conversion handles different image modes (e.g., RGBA, P)
            if pil_image.mode == 'RGBA':
                 # Convert RGBA to RGB first, then grayscale
                 rgb_image = pil_image.convert('RGB')
                 open_cv_image = np.array(rgb_image)
                 gray_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2GRAY)
            elif pil_image.mode == 'P':
                 # Convert Palette image to RGB first, then grayscale
                 rgb_image = pil_image.convert('RGB')
                 open_cv_image = np.array(rgb_image)
                 gray_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2GRAY)
            elif pil_image.mode == 'L':
                 # Already grayscale
                 gray_image = np.array(pil_image)
            elif pil_image.mode == 'RGB':
                 open_cv_image = np.array(pil_image)
                 # Convert RGB to BGR for OpenCV if needed, then grayscale
                 # gray_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2GRAY) # Correct conversion
                 # Correct conversion from RGB (Pillow) to Grayscale (OpenCV)
                 gray_image = cv2.cvtColor(open_cv_image[:, :, ::-1], cv2.COLOR_BGR2GRAY) # Convert RGB->BGR then to Gray
            else:
                 print(f"Warning: Unsupported image mode '{pil_image.mode}' for technical analysis of {image_path}. Attempting conversion.")
                 # Attempt conversion to RGB first as a fallback
                 try:
                     rgb_image = pil_image.convert('RGB')
                     open_cv_image = np.array(rgb_image)
                     gray_image = cv2.cvtColor(open_cv_image[:, :, ::-1], cv2.COLOR_BGR2GRAY)
                 except Exception as conv_e:
                     print(f"Error converting image {image_path} to grayscale: {conv_e}")
                     return results # Return default failure values


            # Perform analyses
            focus_score, focus_assessment = BasicTechnicalAnalyzerTool.analyze_focus(gray_image)
            exposure_assessment = BasicTechnicalAnalyzerTool.analyze_exposure(gray_image)

            results["focus_score"] = focus_score
            results["focus_assessment"] = focus_assessment
            results["exposure_assessment"] = exposure_assessment

            print(f"Successfully performed technical analysis for: {image_path}")

        except ImportError:
             print("Error: OpenCV or NumPy not installed. Cannot perform technical analysis.")
             # Keep default failure values in results
        except Exception as e:
            print(f"Error during technical analysis setup or execution for {image_path}: {e}")
            # Keep default failure values in results

        return results

# Example usage (optional, for testing)
if __name__ == '__main__':
    # Requires OpenCV (cv2) and NumPy, plus the ImageLoaderTool
    # Assumes dummy_test_image.png exists from previous steps
    test_image_path = "dummy_test_image.png"
    # test_image_path = "path/to/your/real_image.jpg" # Replace for more meaningful results

    try:
        import os
        if not os.path.exists(test_image_path) and "dummy" in test_image_path:
             print(f"Warning: {test_image_path} not found. Run image_loader_tool.py first or provide a real image path.")
        else:
            analyzer = BasicTechnicalAnalyzerTool()
            analysis_results = analyzer.perform_analysis(test_image_path)
            print("\nTechnical Analysis Results:")
            print(json.dumps(analysis_results, indent=2))

            # Test non-existent file
            print("\nTesting non-existent file:")
            analyzer.perform_analysis("non_existent_image.jpg")

    except ImportError as imp_err:
        print(f"Import Error during example usage: {imp_err}. Make sure OpenCV, NumPy, Pillow are installed.")
    except Exception as e:
        print(f"An error occurred during example usage: {e}")
