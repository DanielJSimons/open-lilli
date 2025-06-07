"""Tests for domain models."""

import pytest
from pydantic import ValidationError

from open_lilli.models import GenerationConfig, Outline, ReviewFeedback, SlidePlan


class TestSlidePlan:
    """Tests for SlidePlan model."""

    def test_slide_plan_creation(self):
        """Test creating a valid SlidePlan."""
        slide = SlidePlan(
            index=1,
            slide_type="content",
            title="Test Slide",
            bullets=["Point 1", "Point 2"],
            image_query="test image",
        )
        
        assert slide.index == 1
        assert slide.slide_type == "content"
        assert slide.title == "Test Slide"
        assert len(slide.bullets) == 2
        assert slide.image_query == "test image"
        assert slide.chart_data is None
        assert slide.speaker_notes is None
        assert slide.layout_id is None

    def test_slide_plan_minimal(self):
        """Test creating SlidePlan with minimal required fields."""
        slide = SlidePlan(
            index=0,
            slide_type="title",
            title="Title Slide"
        )
        
        assert slide.index == 0
        assert slide.slide_type == "title"
        assert slide.title == "Title Slide"
        assert slide.bullets == []

    def test_slide_plan_validation(self):
        """Test SlidePlan validation."""
        with pytest.raises(ValidationError):
            SlidePlan()  # Missing required fields


class TestOutline:
    """Tests for Outline model."""

    def test_outline_creation(self):
        """Test creating a valid Outline."""
        slides = [
            SlidePlan(index=0, slide_type="title", title="Title"),
            SlidePlan(index=1, slide_type="content", title="Content")
        ]
        
        outline = Outline(
            title="Test Presentation",
            slides=slides,
            language="en"
        )
        
        assert outline.title == "Test Presentation"
        assert outline.language == "en"
        assert outline.slide_count == 2
        assert outline.subtitle is None

    def test_outline_slide_count(self):
        """Test slide_count property."""
        slides = [
            SlidePlan(index=i, slide_type="content", title=f"Slide {i}")
            for i in range(5)
        ]
        
        outline = Outline(title="Test", slides=slides)
        assert outline.slide_count == 5

    def test_get_slide_by_index(self):
        """Test getting slide by index."""
        slides = [
            SlidePlan(index=0, slide_type="title", title="Title"),
            SlidePlan(index=1, slide_type="content", title="Content")
        ]
        
        outline = Outline(title="Test", slides=slides)
        
        slide = outline.get_slide_by_index(1)
        assert slide is not None
        assert slide.title == "Content"
        
        missing = outline.get_slide_by_index(99)
        assert missing is None


class TestGenerationConfig:
    """Tests for GenerationConfig model."""

    def test_generation_config_defaults(self):
        """Test GenerationConfig with default values."""
        config = GenerationConfig()
        
        assert config.max_slides == 20
        assert config.max_bullets_per_slide == 5
        assert config.include_images is True
        assert config.include_charts is True
        assert config.tone == "professional"
        assert config.complexity_level == "intermediate"

    def test_generation_config_custom(self):
        """Test GenerationConfig with custom values."""
        config = GenerationConfig(
            max_slides=10,
            tone="casual",
            include_images=False
        )
        
        assert config.max_slides == 10
        assert config.tone == "casual"
        assert config.include_images is False


class TestReviewFeedback:
    """Tests for ReviewFeedback model."""

    def test_review_feedback_creation(self):
        """Test creating ReviewFeedback."""
        feedback = ReviewFeedback(
            slide_index=2,
            severity="high",
            category="content",
            message="Too much text",
            suggestion="Break into multiple slides"
        )
        
        assert feedback.slide_index == 2
        assert feedback.severity == "high"
        assert feedback.category == "content"
        assert feedback.message == "Too much text"
        assert feedback.suggestion == "Break into multiple slides"

    def test_review_feedback_minimal(self):
        """Test ReviewFeedback with minimal fields."""
        feedback = ReviewFeedback(
            slide_index=1,
            severity="low",
            category="design",
            message="Minor formatting issue"
        )
        
        assert feedback.suggestion is None