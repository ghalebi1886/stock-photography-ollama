from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Literal

# Define the allowed categories using Literal for strict validation
Category = Literal[
    "Animals/Wildlife", "Arts", "Backgrounds/Textures", "Buildings/Landmarks",
    "Business/Finance", "Education", "Food and drink", "Healthcare/Medical",
    "Holidays", "Industrial", "Nature", "Objects", "People", "Religion",
    "Science", "Signs/Symbols", "Sports/Recreation", "Technology", "Transportation"
]

class StockMetadataOutput(BaseModel):
    """
    Pydantic model for the structured output expected from the metadata generation process.
    """
    title: str = Field(description="Concise, impactful title (max 65 characters).")
    description: str = Field(description="Descriptive sentence or phrase answering Who/What/When/Where/Why (max 200 characters).")
    keywords: List[str] = Field(description="List of 40-50 relevant keywords for stock agencies.", min_length=40, max_length=50) # Updated min_length based on recent prompt change
    category1: Category = Field(description="Primary category chosen from the allowed list.")
    category2: Optional[Category] = Field(default=None, description="Optional secondary category chosen from the allowed list.")

    @field_validator('title')
    @classmethod
    def check_title_length(cls, value):
        """Strictly enforces the 65-character limit for the title."""
        if len(value) > 65:
            raise ValueError(f"Title must be 65 characters or less (was {len(value)})")
        return value

    @field_validator('description')
    @classmethod
    def check_description_length(cls, value):
        """Strictly enforces the 200-character limit for the description."""
        if len(value) > 200:
            raise ValueError(f"Description must be 200 characters or less (was {len(value)})")
        return value

    # Validator for category2 to ensure it's different from category1 if set
    @field_validator('category2')
    @classmethod
    def check_category2_differs_from_category1(cls, v, info):
        if v is not None and 'category1' in info.data and v == info.data['category1']:
            raise ValueError("category2 must be different from category1")
        return v
