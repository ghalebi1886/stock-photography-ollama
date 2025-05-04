import exifread
import json
from fractions import Fraction # Import Fraction for coordinate conversion
from typing import Optional

# Attempt to import geopy for reverse geocoding
try:
    from geopy.geocoders import Nominatim
    from geopy.exc import GeocoderTimedOut, GeocoderServiceError
    GEOPY_AVAILABLE = True
except ImportError:
    GEOPY_AVAILABLE = False
    print("Warning: geopy library not found. GPS reverse geocoding will be skipped. Install with: pip install geopy")


class MetadataExtractorTool:
    """
    Extracts specific EXIF metadata from an image file, including GPS coordinates
    and attempts reverse geocoding if geopy is installed.
    """

    @staticmethod
    def _get_tag_value(tags, key, default="Not Available"):
        """Helper to safely get tag value as string."""
        if key in tags:
            return str(tags[key])
        return default

    @staticmethod
    def _dms_to_decimal(degrees, minutes, seconds, direction):
        """Converts DMS (Degrees, Minutes, Seconds) coordinates to decimal degrees."""
        try:
            # Values might be Ratio objects, convert them
            d = float(degrees.num) / float(degrees.den) if hasattr(degrees, 'num') else float(degrees)
            m = float(minutes.num) / float(minutes.den) if hasattr(minutes, 'num') else float(minutes)
            s = float(seconds.num) / float(seconds.den) if hasattr(seconds, 'num') else float(seconds)

            decimal_degrees = d + (m / 60.0) + (s / 3600.0)

            if direction in ['S', 'W']:
                decimal_degrees *= -1
            return decimal_degrees
        except (ValueError, TypeError, ZeroDivisionError, AttributeError) as e:
            print(f"  Warning: Could not convert DMS to decimal: {e}")
            return None

    @staticmethod
    def _get_gps_location(tags) -> Optional[dict]:
        """Extracts GPS coordinates and attempts reverse geocoding."""
        lat_tag = tags.get('GPS GPSLatitude')
        lon_tag = tags.get('GPS GPSLongitude')
        lat_ref_tag = tags.get('GPS GPSLatitudeRef')
        lon_ref_tag = tags.get('GPS GPSLongitudeRef')

        if not all([lat_tag, lon_tag, lat_ref_tag, lon_ref_tag]):
            return None # Not enough GPS info

        try:
            lat_dms = lat_tag.values
            lon_dms = lon_tag.values
            lat_ref = lat_ref_tag.values
            lon_ref = lon_ref_tag.values

            latitude = MetadataExtractorTool._dms_to_decimal(lat_dms[0], lat_dms[1], lat_dms[2], lat_ref)
            longitude = MetadataExtractorTool._dms_to_decimal(lon_dms[0], lon_dms[1], lon_dms[2], lon_ref)

            if latitude is None or longitude is None:
                return None

            location_info = {"latitude": latitude, "longitude": longitude}

            if GEOPY_AVAILABLE:
                try:
                    geolocator = Nominatim(user_agent="stock_photo_crew_tool", timeout=10) # Added timeout
                    # Use language='en' for consistent results if needed
                    location = geolocator.reverse((latitude, longitude), exactly_one=True, language='en')
                    if location and location.address:
                        print(f"  [Geocoder] Found location: {location.address}")
                        # Extract useful parts - adjust based on what Nominatim returns
                        address = location.raw.get('address', {})
                        location_info["address"] = location.address # Full address
                        location_info["city"] = address.get('city', address.get('town', address.get('village')))
                        location_info["state"] = address.get('state')
                        location_info["country"] = address.get('country')
                        location_info["country_code"] = address.get('country_code')
                    else:
                         print("  [Geocoder] No address found for coordinates.")
                except GeocoderTimedOut:
                    print("  [Geocoder] Warning: Reverse geocoding service timed out.")
                except GeocoderServiceError as geo_err:
                     print(f"  [Geocoder] Warning: Reverse geocoding service error: {geo_err}")
                except Exception as geo_ex:
                     print(f"  [Geocoder] Warning: Unexpected error during reverse geocoding: {geo_ex}")

            return location_info

        except (AttributeError, IndexError, TypeError, ValueError) as e:
            print(f"  Warning: Error processing GPS tags: {e}")
            return None


    @staticmethod
    def extract_metadata(image_path: str) -> dict:
        """
        Extracts specified EXIF metadata fields, including GPS location, from the image.

        Args:
            image_path: The path to the image file.

        Returns:
            A dictionary containing the extracted metadata in the specified format.
            Includes a 'location' key if GPS data is found and processed.
        """
        metadata = {
            "exif_data": {
                "camera_model": "Not Available",
                "lens_model": "Not Available",
                "iso": "Not Available",
                "f_number": "Not Available",
                "exposure_time": "Not Available",
            },
            "location": None # Add location field
        }
        try:
            with open(image_path, 'rb') as f:
                # Process all tags, don't stop early
                tags = exifread.process_file(f)

                # Try to get camera model from Make and Model tags
                make = MetadataExtractorTool._get_tag_value(tags, 'Image Make')
                model = MetadataExtractorTool._get_tag_value(tags, 'Image Model')
                if make != "Not Available" and model != "Not Available":
                     # Combine Make and Model if both exist, otherwise use Model if available
                     if make.strip().lower() in model.strip().lower(): # Avoid duplication like "Canon Canon EOS R5"
                         metadata["exif_data"]["camera_model"] = model
                     else:
                         metadata["exif_data"]["camera_model"] = f"{make} {model}"
                elif model != "Not Available":
                     metadata["exif_data"]["camera_model"] = model
                elif make != "Not Available": # Less ideal, but better than nothing
                     metadata["exif_data"]["camera_model"] = make


                metadata["exif_data"]["lens_model"] = MetadataExtractorTool._get_tag_value(tags, 'EXIF LensModel')
                metadata["exif_data"]["iso"] = MetadataExtractorTool._get_tag_value(tags, 'EXIF ISOSpeedRatings')

                # FNumber often needs calculation if represented as a ratio
                f_number_tag = tags.get('EXIF FNumber')
                if f_number_tag and f_number_tag.values:
                    try:
                        # Check if it's a ratio
                        if hasattr(f_number_tag.values[0], 'num') and hasattr(f_number_tag.values[0], 'den'):
                            val = float(f_number_tag.values[0].num) / float(f_number_tag.values[0].den)
                            metadata["exif_data"]["f_number"] = f"f/{val:.1f}" # Format like f/2.8
                        else: # Assume it's a direct value
                             metadata["exif_data"]["f_number"] = f"f/{float(str(f_number_tag)):.1f}"
                    except (ValueError, TypeError, ZeroDivisionError):
                         metadata["exif_data"]["f_number"] = MetadataExtractorTool._get_tag_value(tags, 'EXIF FNumber') # Fallback to string
                else:
                     metadata["exif_data"]["f_number"] = "Not Available"


                metadata["exif_data"]["exposure_time"] = MetadataExtractorTool._get_tag_value(tags, 'EXIF ExposureTime')

                # --- GPS Extraction and Geocoding ---
                gps_location = MetadataExtractorTool._get_gps_location(tags)
                if gps_location:
                    metadata["location"] = gps_location
                    print(f"  Successfully extracted GPS data.")
                    if "address" in gps_location:
                         print(f"  Reverse geocoded to: {gps_location.get('city', 'N/A')}, {gps_location.get('country', 'N/A')}")
                else:
                     print(f"  No usable GPS data found.")
                # --- End GPS ---

            print(f"Successfully processed metadata for: {image_path}")

        except FileNotFoundError:
            print(f"Error: Image file not found at {image_path} for metadata extraction.")
        except Exception as e:
            print(f"Error extracting metadata from {image_path}: {e}")
            # Reset to defaults in case of partial extraction before error
            metadata = {
                "exif_data": {
                    "camera_model": "Not Available",
                    "lens_model": "Not Available",
                    "iso": "Not Available",
                    "f_number": "Not Available",
                    "exposure_time": "Not Available",
                },
                "location": None # Ensure location is reset on error
            }

        return metadata

# Example usage (optional, for testing)
if __name__ == '__main__':
    # Note: Requires an image with EXIF data for meaningful testing.
    # Using the dummy image created by image_loader_tool won't have EXIF.
    # You might need to place a real JPEG/TIFF with EXIF in the root directory
    # or provide a path to one.
    test_image_path = "dummy_test_image.png" # This will likely show "Not Available" for all fields
    # test_image_path = "path/to/your/real_image_with_exif.jpg" # Replace with a real image path

    try:
        # Check if the test image exists from the previous step
        import os
        if not os.path.exists(test_image_path) and "dummy" in test_image_path:
             print(f"Warning: {test_image_path} not found. Run image_loader_tool.py first or provide a real image path.")
        else:
            extractor = MetadataExtractorTool()
            extracted_data = extractor.extract_metadata(test_image_path)
            print("\nExtracted Metadata:")
            print(json.dumps(extracted_data, indent=2))

            # Test non-existent file
            print("\nTesting non-existent file:")
            extractor.extract_metadata("non_existent_image.jpg")

    except ImportError:
        print("exifread is not installed. Cannot run example usage.")
    except Exception as e:
        print(f"An error occurred during example usage: {e}")
