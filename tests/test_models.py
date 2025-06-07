"""Tests for domain models."""

import pytest
from pydantic import ValidationError

from open_lilli.models import (
    GenerationConfig, Outline, ReviewFeedback, SlidePlan,
    FontInfo, BulletInfo, PlaceholderStyleInfo, TemplateStyle
)


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


class TestFontInfo:
    """Tests for FontInfo model."""

    def test_font_info_creation(self):
        """Test creating a valid FontInfo."""
        font = FontInfo(
            name="Calibri",
            size=14,
            weight="normal",
            color="#000000"
        )
        
        assert font.name == "Calibri"
        assert font.size == 14
        assert font.weight == "normal"
        assert font.color == "#000000"

    def test_font_info_minimal(self):
        """Test FontInfo with minimal required fields."""
        font = FontInfo(name="Arial")
        
        assert font.name == "Arial"
        assert font.size is None
        assert font.weight is None
        assert font.color is None


class TestBulletInfo:
    """Tests for BulletInfo model."""

    def test_bullet_info_creation(self):
        """Test creating a valid BulletInfo."""
        font = FontInfo(name="Calibri", size=12)
        bullet = BulletInfo(
            character="•",
            font=font,
            indent_level=0
        )
        
        assert bullet.character == "•"
        assert bullet.font == font
        assert bullet.indent_level == 0

    def test_bullet_info_minimal(self):
        """Test BulletInfo with minimal fields."""
        bullet = BulletInfo(character="○")
        
        assert bullet.character == "○"
        assert bullet.font is None
        assert bullet.indent_level == 0


class TestPlaceholderStyleInfo:
    """Tests for PlaceholderStyleInfo model."""

    def test_placeholder_style_creation(self):
        """Test creating a valid PlaceholderStyleInfo."""
        font = FontInfo(name="Calibri", size=14)
        bullet = BulletInfo(character="•", font=font)
        
        style = PlaceholderStyleInfo(
            placeholder_type=2,
            type_name="BODY",
            default_font=font,
            bullet_styles=[bullet]
        )
        
        assert style.placeholder_type == 2
        assert style.type_name == "BODY"
        assert style.default_font == font
        assert len(style.bullet_styles) == 1
        assert style.bullet_styles[0] == bullet

    def test_placeholder_style_minimal(self):
        """Test PlaceholderStyleInfo with minimal fields."""
        style = PlaceholderStyleInfo(
            placeholder_type=1,
            type_name="TITLE"
        )
        
        assert style.placeholder_type == 1
        assert style.type_name == "TITLE"
        assert style.default_font is None
        assert style.bullet_styles == []


class TestTemplateStyle:
    """Tests for TemplateStyle model."""

    def test_template_style_creation(self):
        """Test creating a valid TemplateStyle."""
        master_font = FontInfo(name="Calibri", size=12)
        title_font = FontInfo(name="Calibri", size=24, weight="bold")
        body_font = FontInfo(name="Calibri", size=14)
        
        bullet = BulletInfo(character="•", font=body_font)
        
        title_style = PlaceholderStyleInfo(
            placeholder_type=1,
            type_name="TITLE",
            default_font=title_font
        )
        
        body_style = PlaceholderStyleInfo(
            placeholder_type=2,
            type_name="BODY",
            default_font=body_font,
            bullet_styles=[bullet]
        )
        
        template_style = TemplateStyle(
            master_font=master_font,
            placeholder_styles={1: title_style, 2: body_style},
            theme_fonts={"major": "Calibri", "minor": "Calibri"}
        )
        
        assert template_style.master_font == master_font
        assert len(template_style.placeholder_styles) == 2
        assert template_style.placeholder_styles[1] == title_style
        assert template_style.placeholder_styles[2] == body_style
        assert template_style.theme_fonts["major"] == "Calibri"

    def test_template_style_get_font_for_placeholder_type(self):
        """Test getting font for placeholder type."""
        master_font = FontInfo(name="Default", size=12)
        title_font = FontInfo(name="Title Font", size=24)
        
        title_style = PlaceholderStyleInfo(
            placeholder_type=1,
            type_name="TITLE",
            default_font=title_font
        )
        
        template_style = TemplateStyle(
            master_font=master_font,
            placeholder_styles={1: title_style}
        )
        
        # Should return specific font for title
        font = template_style.get_font_for_placeholder_type(1)
        assert font == title_font
        
        # Should return master font for unknown type
        font = template_style.get_font_for_placeholder_type(999)
        assert font == master_font

    def test_template_style_get_bullet_style_for_level(self):
        """Test getting bullet style for level."""
        bullet_font = FontInfo(name="Calibri", size=14)
        bullet0 = BulletInfo(character="•", font=bullet_font, indent_level=0)
        bullet1 = BulletInfo(character="○", font=bullet_font, indent_level=1)
        
        body_style = PlaceholderStyleInfo(
            placeholder_type=2,
            type_name="BODY",
            bullet_styles=[bullet0, bullet1]
        )
        
        template_style = TemplateStyle(
            placeholder_styles={2: body_style}
        )
        
        # Should return correct bullet for level
        bullet = template_style.get_bullet_style_for_level(2, 0)
        assert bullet == bullet0
        
        bullet = template_style.get_bullet_style_for_level(2, 1)
        assert bullet == bullet1
        
        # Should return None for unknown level or type
        bullet = template_style.get_bullet_style_for_level(2, 5)
        assert bullet is None
        
        bullet = template_style.get_bullet_style_for_level(999, 0)
        assert bullet is None

    def test_template_style_get_placeholder_style(self):
        """Test getting placeholder style."""
        title_style = PlaceholderStyleInfo(
            placeholder_type=1,
            type_name="TITLE"
        )
        
        template_style = TemplateStyle(
            placeholder_styles={1: title_style}
        )
        
        # Should return style for known type
        style = template_style.get_placeholder_style(1)
        assert style == title_style
        
        # Should return None for unknown type
        style = template_style.get_placeholder_style(999)
        assert style is None

    def test_template_style_minimal(self):
        """Test TemplateStyle with minimal fields."""
        template_style = TemplateStyle()
        
        assert template_style.master_font is None
        assert template_style.placeholder_styles == {}
        assert template_style.theme_fonts == {}