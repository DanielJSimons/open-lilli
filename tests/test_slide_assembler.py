"""Tests for slide assembler."""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from pptx import Presentation

from open_lilli.models import Outline, SlidePlan
from open_lilli.slide_assembler import SlideAssembler
from open_lilli.template_parser import TemplateParser


class TestSlideAssembler:
    """Tests for SlideAssembler class."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create a basic test template
        self.temp_dir = Path(tempfile.mkdtemp())
        self.template_path = self.temp_dir / "test_template.pptx"
        
        # Create minimal template
        prs = Presentation()
        prs.save(str(self.template_path))
        
        # Mock template parser
        self.mock_template_parser = Mock(spec=TemplateParser)
        self.mock_template_parser.template_path = self.template_path
        self.mock_template_parser.get_layout_index.return_value = 0
        
        # Mock presentation with layouts
        self.mock_template_parser.prs = Mock()
        self.mock_template_parser.prs.slide_layouts = [Mock() for _ in range(3)]
        
        self.assembler = SlideAssembler(self.mock_template_parser)

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def create_test_outline(self) -> Outline:
        """Create a test outline."""
        return Outline(
            title="Test Presentation",
            subtitle="A test presentation",
            slides=[]
        )

    def create_test_slides(self) -> List[SlidePlan]:
        """Create test slides."""
        return [
            SlidePlan(
                index=0,
                slide_type="title",
                title="Presentation Title",
                bullets=[],
                layout_id=0,
                speaker_notes="Welcome to the presentation"
            ),
            SlidePlan(
                index=1,
                slide_type="content",
                title="Content Slide",
                bullets=["Point 1", "Point 2", "Point 3"],
                layout_id=1,
                speaker_notes="Discuss the main points"
            )
        ]

    def test_init(self):
        """Test SlideAssembler initialization."""
        assert self.assembler.template_parser == self.mock_template_parser
        assert self.assembler.max_title_length == 60
        assert self.assembler.max_bullet_length == 120

    def test_assemble_basic(self):
        """Test basic presentation assembly."""
        outline = self.create_test_outline()
        slides = self.create_test_slides()
        output_path = self.temp_dir / "output.pptx"
        
        # Mock the presentation methods we'll use
        with patch('pptx.Presentation') as mock_prs_class:
            mock_prs = Mock()
            mock_prs.slides = Mock()
            mock_prs.slides.__len__ = Mock(return_value=0)  # No existing slides
            mock_prs.slide_layouts = [Mock(), Mock()]
            mock_prs_class.return_value = mock_prs
            
            # Mock slide creation
            mock_slide = Mock()
            mock_prs.slides.add_slide.return_value = mock_slide
            mock_slide.shapes.title = Mock()
            mock_slide.placeholders = []
            
            result_path = self.assembler.assemble(outline, slides, {}, output_path)
            
            assert result_path == output_path
            assert mock_prs.save.called

    def test_add_title(self):
        """Test adding title to slide."""
        mock_slide = Mock()
        mock_title_shape = Mock()
        mock_slide.shapes.title = mock_title_shape
        
        self.assembler._add_title(mock_slide, "Test Title")
        
        assert mock_title_shape.text == "Test Title"

    def test_add_title_too_long(self):
        """Test title truncation when too long."""
        mock_slide = Mock()
        mock_title_shape = Mock()
        mock_slide.shapes.title = mock_title_shape
        
        long_title = "A" * 100  # Longer than max_title_length
        self.assembler._add_title(mock_slide, long_title)
        
        # Should be truncated
        assert len(mock_title_shape.text) <= self.assembler.max_title_length

    def test_add_title_no_placeholder(self):
        """Test handling when no title placeholder exists."""
        mock_slide = Mock()
        mock_slide.shapes.title = None
        
        # Should not raise exception
        self.assembler._add_title(mock_slide, "Test Title")

    def test_add_bullet_content(self):
        """Test adding bullet points to slide."""
        mock_slide = Mock()
        mock_placeholder = Mock()
        mock_placeholder.placeholder_format.type = 2  # BODY type
        mock_slide.placeholders = [mock_placeholder]
        
        mock_text_frame = Mock()
        mock_placeholder.text_frame = mock_text_frame
        mock_paragraph = Mock()
        mock_text_frame.paragraphs = [mock_paragraph]
        mock_text_frame.add_paragraph.return_value = mock_paragraph
        
        bullets = ["Point 1", "Point 2", "Point 3"]
        self.assembler._add_bullet_content(mock_slide, bullets)
        
        # Should have called clear and add_paragraph
        mock_text_frame.clear.assert_called_once()
        assert mock_text_frame.add_paragraph.call_count == 2  # First uses existing paragraph

    def test_add_bullet_content_no_placeholder(self):
        """Test bullet content when no content placeholder exists."""
        mock_slide = Mock()
        mock_slide.placeholders = []
        
        bullets = ["Point 1", "Point 2"]
        
        # Should not raise exception
        self.assembler._add_bullet_content(mock_slide, bullets)

    def test_add_chart_image_with_placeholder(self):
        """Test adding chart image with picture placeholder."""
        mock_slide = Mock()
        mock_placeholder = Mock()
        mock_placeholder.placeholder_format.type = 18  # PICTURE type
        mock_slide.placeholders = [mock_placeholder]
        
        # Create a test image file
        test_image = self.temp_dir / "test_chart.png"
        test_image.write_bytes(b"fake image data")
        
        self.assembler._add_chart_image(mock_slide, str(test_image))
        
        mock_placeholder.insert_picture.assert_called_once_with(str(test_image))

    def test_add_chart_image_no_placeholder(self):
        """Test adding chart image without picture placeholder."""
        mock_slide = Mock()
        mock_slide.placeholders = []
        mock_slide.shapes = Mock()
        mock_slide.part.presentation.slide_width = 10000000  # EMUs
        mock_slide.part.presentation.slide_height = 7500000
        
        # Create a test image file
        test_image = self.temp_dir / "test_chart.png"
        test_image.write_bytes(b"fake image data")
        
        self.assembler._add_chart_image(mock_slide, str(test_image))
        
        mock_slide.shapes.add_picture.assert_called_once()

    def test_add_chart_image_missing_file(self):
        """Test handling of missing chart file."""
        mock_slide = Mock()
        mock_slide.placeholders = []
        
        # Should not raise exception
        self.assembler._add_chart_image(mock_slide, "nonexistent.png")

    def test_add_speaker_notes(self):
        """Test adding speaker notes to slide."""
        mock_slide = Mock()
        mock_notes_slide = Mock()
        mock_text_frame = Mock()
        
        mock_slide.notes_slide = mock_notes_slide
        mock_notes_slide.notes_text_frame = mock_text_frame
        
        notes = "These are speaker notes"
        self.assembler._add_speaker_notes(mock_slide, notes)
        
        assert mock_text_frame.text == notes

    def test_add_fallback_slide(self):
        """Test adding fallback slide when normal creation fails."""
        mock_prs = Mock()
        mock_layout = Mock()
        mock_prs.slide_layouts = [mock_layout]
        mock_slide = Mock()
        mock_prs.slides.add_slide.return_value = mock_slide
        mock_slide.shapes.title = Mock()
        
        slide_plan = SlidePlan(
            index=1,
            slide_type="content",
            title="Test Slide",
            bullets=[]
        )
        
        self.assembler._add_fallback_slide(mock_prs, slide_plan)
        
        assert mock_slide.shapes.title.text == "Test Slide"

    def test_apply_metadata(self):
        """Test applying presentation metadata."""
        mock_prs = Mock()
        mock_core_properties = Mock()
        mock_prs.core_properties = mock_core_properties
        
        outline = Outline(
            title="Test Presentation",
            subtitle="Test Subtitle",
            slides=[]
        )
        
        self.assembler._apply_metadata(mock_prs, outline)
        
        assert mock_core_properties.title == "Test Presentation"
        assert mock_core_properties.subject == "Test Subtitle"
        assert mock_core_properties.author == "Open Lilli AI"

    def test_analyze_slide_placeholders(self):
        """Test slide placeholder analysis."""
        mock_slide = Mock()
        mock_placeholder = Mock()
        mock_placeholder.placeholder_format.type = 2
        mock_placeholder.name = "Content Placeholder"
        mock_placeholder.shape_type = 14
        mock_slide.placeholders = [mock_placeholder]
        
        analysis = self.assembler.analyze_slide_placeholders(mock_slide)
        
        assert analysis["total_placeholders"] == 1
        assert len(analysis["placeholders"]) == 1
        assert analysis["placeholders"][0]["type"] == 2
        assert analysis["placeholders"][0]["name"] == "Content Placeholder"

    def test_get_assembly_statistics(self):
        """Test assembly statistics generation."""
        slides = self.create_test_slides()
        visuals = {
            1: {"chart": "chart.png", "image": "image.jpg"}
        }
        
        stats = self.assembler.get_assembly_statistics(slides, visuals)
        
        assert stats["total_slides"] == 2
        assert stats["slides_with_visuals"] == 1
        assert stats["total_bullets"] == 3
        assert stats["slides_with_notes"] == 2
        assert stats["slide_types"]["title"] == 1
        assert stats["slide_types"]["content"] == 1
        assert stats["visual_types"]["charts"] == 1
        assert stats["visual_types"]["images"] == 1

    def test_validate_slides_before_assembly(self):
        """Test slide validation before assembly."""
        # Create slides with various issues
        slides = [
            SlidePlan(
                index=0,
                slide_type="title",
                title="",  # Missing title
                bullets=[],
                layout_id=0
            ),
            SlidePlan(
                index=1,
                slide_type="content", 
                title="Valid Title",
                bullets=["Point"] * 15,  # Too many bullets
                layout_id=99  # Invalid layout ID
            ),
            SlidePlan(
                index=2,
                slide_type="content",
                title="A" * 100,  # Title too long
                bullets=[],
                layout_id=1
            )
        ]
        
        issues = self.assembler.validate_slides_before_assembly(slides)
        
        assert len(issues) >= 4  # Should find multiple issues
        assert any("no title" in issue.lower() for issue in issues)
        assert any("too many bullets" in issue.lower() for issue in issues)
        assert any("invalid layout id" in issue.lower() for issue in issues)
        assert any("title is too long" in issue.lower() for issue in issues)

    def test_validate_slides_empty_list(self):
        """Test validation with empty slide list."""
        issues = self.assembler.validate_slides_before_assembly([])
        
        assert len(issues) == 1
        assert "No slides provided" in issues[0]

    def test_validate_slides_valid(self):
        """Test validation with valid slides."""
        slides = self.create_test_slides()
        
        issues = self.assembler.validate_slides_before_assembly(slides)
        
        assert len(issues) == 0

    def test_create_slide_from_layout(self):
        """Test creating slide from specific layout."""
        mock_prs = Mock()
        mock_layout = Mock()
        mock_prs.slide_layouts = [mock_layout]
        mock_slide = Mock()
        mock_prs.slides.add_slide.return_value = mock_slide
        mock_slide.shapes.title = Mock()
        
        self.assembler.create_slide_from_layout(
            mock_prs, "content", "Test Title", ["Point 1", "Point 2"]
        )
        
        mock_prs.slides.add_slide.assert_called_once_with(mock_layout)
        assert mock_slide.shapes.title.text == "Test Title"

    def test_assemble_with_visuals(self):
        """Test assembly with visual elements."""
        outline = self.create_test_outline()
        slides = self.create_test_slides()
        
        # Create test visual files
        chart_file = self.temp_dir / "chart.png"
        chart_file.write_bytes(b"fake chart")
        image_file = self.temp_dir / "image.jpg"
        image_file.write_bytes(b"fake image")
        
        visuals = {
            1: {
                "chart": str(chart_file),
                "image": str(image_file)
            }
        }
        
        output_path = self.temp_dir / "output_with_visuals.pptx"
        
        with patch('pptx.Presentation') as mock_prs_class:
            mock_prs = Mock()
            mock_prs.slides = Mock()
            mock_prs.slides.__len__ = Mock(return_value=0)
            mock_prs.slide_layouts = [Mock(), Mock()]
            mock_prs_class.return_value = mock_prs
            
            mock_slide = Mock()
            mock_prs.slides.add_slide.return_value = mock_slide
            mock_slide.shapes.title = Mock()
            mock_slide.placeholders = []
            mock_slide.shapes = Mock()
            mock_slide.part.presentation.slide_width = 10000000
            mock_slide.part.presentation.slide_height = 7500000
            
            result_path = self.assembler.assemble(outline, slides, visuals, output_path)
            
            assert result_path == output_path