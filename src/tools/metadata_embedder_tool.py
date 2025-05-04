import os
import exiftool
import shutil
import subprocess
from typing import Optional, List, Dict, Any

class MetadataEmbedderTool:
    """
    Embeds metadata (title, keywords) into image files using ExifTool via pyexiftool.
    Saves the output to a new file, preserving the original.
    Requires the 'exiftool' command-line utility to be installed.
    """

    def __init__(self, exiftool_path: str = "exiftool"):
        """
        Initializes the tool and checks for ExifTool availability.

        Args:
            exiftool_path: The path to the exiftool executable. Defaults to 'exiftool'
                           assuming it's in the system PATH.

        Raises:
            RuntimeError: If the exiftool executable cannot be found.
        """
        self.exiftool_path = exiftool_path
        if not self._check_exiftool():
            raise RuntimeError(
                f"ExifTool executable not found at '{self.exiftool_path}'. "
                "Please install ExifTool or configure the EXIFTOOL_PATH environment variable."
            )
        print(f"MetadataEmbedderTool initialized. Using ExifTool at: {self.exiftool_path}")

    def _check_exiftool(self) -> bool:
        """Checks if the exiftool command is accessible."""
        try:
            # Use subprocess to check if the command runs without error
            # Capture output to prevent it from printing to console
            subprocess.run([self.exiftool_path, "-ver"], check=True, capture_output=True)
            return True
        except (FileNotFoundError, subprocess.CalledProcessError, Exception) as e:
            print(f"ExifTool check failed: {e}")
            return False

    def embed_metadata(self,
                       original_image_path: str,
                       output_dir: str,
                       metadata: Dict[str, Any]) -> Optional[str]:
        """
        Embeds metadata into a copy of the image file using exiftool directly via subprocess.

        Args:
            original_image_path: Path to the source image file.
            output_dir: Directory where the processed image will be saved.
            metadata: Dictionary containing metadata, expecting keys like
                      'title' (str), 'description' (str), and 'keywords' (List[str]).

        Returns:
            The path to the newly created image file with embedded metadata,
            or None if embedding fails.
        """
        if not os.path.exists(original_image_path):
            print(f"[Embedder Tool] Error: Original image not found: {original_image_path}")
            return None

        filename = os.path.basename(original_image_path)
        output_path = os.path.join(output_dir, filename)

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # First make a copy of the original image to the output directory
        try:
            shutil.copy2(original_image_path, output_path)  # copy2 preserves metadata
            print(f"[Embedder Tool] Copied original file to {output_path}")
        except Exception as copy_err:
            print(f"[Embedder Tool] Error copying file {original_image_path} to {output_path}: {copy_err}")
            return None

        # Check if we have metadata to embed
        if not metadata.get('title') and not metadata.get('description') and not metadata.get('keywords'):
            print(f"[Embedder Tool] Warning: No title, description, or keywords provided to embed for {filename}. File copied without changes.")
            return output_path

        # Build exiftool command arguments
        cmd_args = [self.exiftool_path]

        # Add title metadata (ObjectName and Title)
        if 'title' in metadata and metadata['title']:
            title = metadata['title'].replace('"', '\\"') # Escape quotes
            cmd_args.extend([
                f"-IPTC:ObjectName={title}",
                f"-XMP:Title={title}"
            ])

        # Add description metadata (Caption-Abstract and Description)
        if 'description' in metadata and metadata['description']:
            desc = metadata['description'].replace('"', '\\"') # Escape quotes
            cmd_args.extend([
                f"-IPTC:Caption-Abstract={desc}",
                f"-XMP:Description={desc}"
            ])

        # Add keywords metadata (Keywords and Subject)
        if 'keywords' in metadata and metadata['keywords']:
            for keyword in metadata['keywords']:
                clean_keyword = keyword.strip().replace('"', '\\"')
                if clean_keyword:
                    cmd_args.extend([
                        f"-IPTC:Keywords+={clean_keyword}",
                        f"-XMP:Subject+={clean_keyword}"
                    ])

        # Add options to modify the file in-place and handle character encoding
        cmd_args.extend(["-overwrite_original", "-L", output_path])
        
        print(f"[Embedder Tool] Attempting to embed metadata into: {output_path}")
        
        try:
            # Run exiftool as a subprocess
            process = subprocess.run(
                cmd_args,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True
            )
            
            # Check if the command was successful
            if process.returncode == 0:
                print(f"[Embedder Tool] Successfully embedded metadata in {output_path}")
                print(f"[Embedder Tool] ExifTool output: {process.stdout.strip()}")
                return output_path
            else:
                print(f"[Embedder Tool] ExifTool error: {process.stderr.strip()}")
                return None
                
        except subprocess.SubprocessError as e:
            print(f"[Embedder Tool] Error running ExifTool: {e}")
            return None
        except Exception as e:
            print(f"[Embedder Tool] Unexpected error: {e}")
            return None

# Example usage (optional, for testing)
# if __name__ == '__main__':
#     # Requires pyexiftool and ExifTool installed
#     # Assumes dummy_test_image.png exists from image_loader_tool test
#     test_image = "dummy_test_image.png" # ExifTool handles PNG metadata better than piexif
#     output_dir = "temp_processed_output_pyexiftool"
#     test_metadata = {
#         "title_description": "Test Title: Red Rectangle with Ümlauts (pyexiftool)",
#         "keywords": ["test", "dummy", "red", "rectangle", "illustration", "特殊文字", "pyexiftool"]
#     }

#     if not os.path.exists(test_image):
#         print(f"Test image {test_image} not found. Run image_loader_tool.py first.")
#     else:
#         try:
#             # Ensure the output directory exists for the test
#             os.makedirs(output_dir, exist_ok=True)
#             embedder = MetadataEmbedderTool() # Assumes exiftool is in PATH
#             processed_path = embedder.embed_metadata(test_image, output_dir, test_metadata)

#             if processed_path:
#                 print(f"\n--- Embedder Test Result (pyexiftool) ---")
#                 print(f"Processed image saved to: {processed_path}")
#                 # Verify using exiftool command line:
#                 print("\nVerify using: exiftool -IPTC:Keywords -IPTC:Caption-Abstract -XMP:Subject -XMP:Description temp_processed_output_pyexiftool/dummy_test_image.png")
#             else:
#                 print(f"\n--- Embedder Test Failed (pyexiftool) ---")

#             # Clean up dummy output dir
#             # if os.path.exists(output_dir):
#             #     shutil.rmtree(output_dir)

#         except RuntimeError as e:
#              print(f"Runtime Error during test (likely ExifTool not found): {e}")
#         except ImportError:
#              print("pyexiftool is not installed. Cannot run example.")
#         except Exception as e:
#              print(f"An error occurred during example usage: {e}")
