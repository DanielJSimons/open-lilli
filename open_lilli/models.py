"""Domain models for the Open Lilli presentation generator."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class SlidePlan(BaseModel):
    """Represents a planned slide with content and layout information."""

    index: int = Field(..., description="Slide index in the presentation")
    slide_type: str = Field(
        ...,
        description="Type of slide (title, content, image, chart, two_column, etc.)"
    )
    title: str = Field(..., description="Main title of the slide")
    bullets: List[str] = Field(
        default_factory=list, description="List of bullet points for the slide"
    )
    image_query: Optional[str] = Field(
        None, description="Search query for finding relevant images"
    )
    chart_data: Optional[Dict[str, Any]] = Field(
        None, description="Data for generating charts/visualizations"
    )
    speaker_notes: Optional[str] = Field(
        None, description="Speaker notes for the slide"
    )
    layout_id: Optional[int] = Field(
        None, description="Template layout index to use for this slide"
    )

    class Config:
        """Pydantic configuration."""
        
        json_schema_extra = {
            "example": {
                "index": 1,
                "slide_type": "content",
                "title": "Market Overview",
                "bullets": [
                    "Market size: $10B and growing",
                    "Key trends: Digital transformation",
                    "Competitive landscape: 3 major players"
                ],
                "image_query": "business market trends",
                "chart_data": None,
                "speaker_notes": "Emphasize the growth opportunity",
                "layout_id": 1
            }
        }


class Outline(BaseModel):
    """Represents the overall structure of a presentation."""

    language: str = Field(
        default="en", description="Language code for the presentation (ISO 639-1)"
    )
    title: str = Field(..., description="Main title of the presentation")
    subtitle: Optional[str] = Field(
        None, description="Subtitle or tagline for the presentation"
    )
    slides: List[SlidePlan] = Field(
        ..., description="List of planned slides in order"
    )
    style_guidance: Optional[str] = Field(
        None, description="Style and tone guidance for content generation"
    )
    target_audience: Optional[str] = Field(
        None, description="Target audience for the presentation"
    )

    @property
    def slide_count(self) -> int:
        """Return the total number of slides."""
        return len(self.slides)

    def get_slide_by_index(self, index: int) -> Optional[SlidePlan]:
        """Get a slide by its index."""
        for slide in self.slides:
            if slide.index == index:
                return slide
        return None

    class Config:
        """Pydantic configuration."""
        
        json_schema_extra = {
            "example": {
                "language": "en",
                "title": "Q4 Business Review",
                "subtitle": "Performance and Strategic Outlook",
                "slides": [
                    {
                        "index": 0,
                        "slide_type": "title",
                        "title": "Q4 Business Review",
                        "bullets": [],
                        "image_query": None,
                        "chart_data": None,
                        "speaker_notes": "Welcome and introduction",
                        "layout_id": 0
                    }
                ],
                "style_guidance": "Professional and data-driven",
                "target_audience": "Executive leadership team"
            }
        }


class GenerationConfig(BaseModel):
    """Configuration for presentation generation."""

    max_slides: int = Field(
        default=20, description="Maximum number of slides to generate"
    )
    max_bullets_per_slide: int = Field(
        default=5, description="Maximum bullet points per slide"
    )
    include_images: bool = Field(
        default=True, description="Whether to include images in slides"
    )
    include_charts: bool = Field(
        default=True, description="Whether to generate charts from data"
    )
    tone: str = Field(
        default="professional", description="Tone for content generation"
    )
    complexity_level: str = Field(
        default="intermediate", 
        description="Complexity level (basic, intermediate, advanced)"
    )

    class Config:
        """Pydantic configuration."""
        
        json_schema_extra = {
            "example": {
                "max_slides": 15,
                "max_bullets_per_slide": 4,
                "include_images": True,
                "include_charts": True,
                "tone": "professional",
                "complexity_level": "intermediate"
            }
        }


class ReviewFeedback(BaseModel):
    """Feedback from the AI reviewer."""

    slide_index: int = Field(..., description="Index of the slide being reviewed")
    severity: str = Field(
        ..., description="Severity level (low, medium, high, critical)"
    )
    category: str = Field(
        ..., description="Category of feedback (content, flow, design, etc.)"
    )
    message: str = Field(..., description="Detailed feedback message")
    suggestion: Optional[str] = Field(
        None, description="Suggested improvement"
    )

    class Config:
        """Pydantic configuration."""
        
        json_schema_extra = {
            "example": {
                "slide_index": 3,
                "severity": "medium",
                "category": "content",
                "message": "This slide has too much text and may be hard to read",
                "suggestion": "Consider breaking into two slides or using more visuals"
            }
        }