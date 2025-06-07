"""Tests for slide planner."""

from unittest.mock import Mock

import pytest

from open_lilli.models import GenerationConfig, Outline, SlidePlan
from open_lilli.slide_planner import SlidePlanner
from open_lilli.template_parser import TemplateParser


class TestSlidePlanner:
    """Tests for SlidePlanner class."""

    def setup_method(self):
        """Set up test fixtures."""
        # Mock template parser
        self.mock_template_parser = Mock(spec=TemplateParser)
        self.mock_template_parser.layout_map = {
            "title": 0,
            "content": 1,
            "image": 2,
            "two_column": 3,
            "section": 4,
            "blank": 5
        }
        self.mock_template_parser.list_available_layouts.return_value = list(
            self.mock_template_parser.layout_map.keys()
        )
        self.mock_template_parser.get_layout_index = lambda layout_type: self.mock_template_parser.layout_map.get(layout_type, 0)
        
        # Mock presentation for validation
        self.mock_template_parser.prs = Mock()
        self.mock_template_parser.prs.slide_layouts = [Mock() for _ in range(6)]
        
        self.planner = SlidePlanner(self.mock_template_parser)

    def create_test_outline(self) -> Outline:
        """Create a test outline."""
        slides = [
            SlidePlan(
                index=0,
                slide_type="title",
                title="Test Presentation",
                bullets=[]
            ),
            SlidePlan(
                index=1,
                slide_type="content",
                title="Overview",
                bullets=["Point 1", "Point 2", "Point 3"]
            ),
            SlidePlan(
                index=2,
                slide_type="image",
                title="Visual Slide",
                bullets=["Key insight"],
                image_query="business growth"
            )
        ]
        
        return Outline(
            title="Test Presentation",
            slides=slides
        )

    def test_init(self):
        """Test SlidePlanner initialization."""
        assert self.planner.template_parser == self.mock_template_parser
        assert isinstance(self.planner.layout_priorities, dict)
        assert "title" in self.planner.layout_priorities
        assert "content" in self.planner.layout_priorities

    def test_plan_slides(self):
        """Test basic slide planning."""
        outline = self.create_test_outline()
        config = GenerationConfig(max_slides=10, max_bullets_per_slide=5)
        
        planned_slides = self.planner.plan_slides(outline, config)
        
        assert len(planned_slides) == 3
        
        # Check that layout IDs are assigned
        for slide in planned_slides:
            assert slide.layout_id is not None
            assert isinstance(slide.layout_id, int)
            assert slide.layout_id >= 0

    def test_select_layout_preferred(self):
        """Test layout selection with preferred layouts."""
        # Title slide should get title layout
        layout_id = self.planner._select_layout("title")
        assert layout_id == 0  # title layout index
        
        # Content slide should get content layout
        layout_id = self.planner._select_layout("content")
        assert layout_id == 1  # content layout index

    def test_select_layout_fallback(self):
        """Test layout selection with fallback."""
        # Remove preferred layout to test fallback
        self.mock_template_parser.layout_map = {"content": 1, "blank": 5}
        self.mock_template_parser.list_available_layouts.return_value = ["content", "blank"]
        
        # Title slide should fallback to content (first in priorities)
        layout_id = self.planner._select_layout("title")
        assert layout_id == 1  # content layout index

    def test_select_layout_no_layouts(self):
        """Test layout selection with no available layouts."""
        self.mock_template_parser.layout_map = {}
        self.mock_template_parser.list_available_layouts.return_value = []
        
        # Should return 0 as last resort
        layout_id = self.planner._select_layout("title")
        assert layout_id == 0

    def test_generate_image_query(self):
        """Test image query generation."""
        # Content slide with business keywords
        slide = SlidePlan(
            index=1,
            slide_type="content",
            title="Market Growth Strategy",
            bullets=["Point 1"]
        )
        
        query = self.planner._generate_image_query(slide)
        assert query is not None
        assert "market" in query.lower() or "growth" in query.lower()

    def test_generate_image_query_skip_types(self):
        """Test that certain slide types skip image generation."""
        # Title slide should not get image query
        slide = SlidePlan(
            index=0,
            slide_type="title",
            title="Presentation Title",
            bullets=[]
        )
        
        query = self.planner._generate_image_query(slide)
        assert query is None

    def test_generate_speaker_notes(self):
        """Test speaker notes generation."""
        # Title slide
        slide = SlidePlan(
            index=0,
            slide_type="title",
            title="Test Title",
            bullets=[]
        )
        
        notes = self.planner._generate_speaker_notes(slide)
        assert "welcome" in notes.lower() or "introduce" in notes.lower()

    def test_split_slide(self):
        """Test slide splitting with too many bullets."""
        slide = SlidePlan(
            index=1,
            slide_type="content",
            title="Long Slide",
            bullets=[f"Point {i}" for i in range(8)]  # 8 bullets
        )
        
        split_slides = self.planner._split_slide(slide, max_bullets=3)
        
        # Should split into 3 slides (3 + 3 + 2 bullets)
        assert len(split_slides) == 3
        assert len(split_slides[0].bullets) == 3
        assert len(split_slides[1].bullets) == 3
        assert len(split_slides[2].bullets) == 2
        
        # Check titles are modified
        assert "Part 1" in split_slides[0].title
        assert "Part 2" in split_slides[1].title
        assert "Part 3" in split_slides[2].title

    def test_split_slide_no_split_needed(self):
        """Test that slides with few bullets are not split."""
        slide = SlidePlan(
            index=1,
            slide_type="content",
            title="Short Slide",
            bullets=["Point 1", "Point 2"]
        )
        
        split_slides = self.planner._split_slide(slide, max_bullets=5)
        
        # Should return original slide unchanged
        assert len(split_slides) == 1
        assert split_slides[0] == slide

    def test_optimize_slide_sequence_splitting(self):
        """Test slide sequence optimization with splitting."""
        slides = [
            SlidePlan(
                index=0,
                slide_type="content",
                title="Long Slide",
                bullets=[f"Point {i}" for i in range(8)]  # Too many bullets
            )
        ]
        
        config = GenerationConfig(max_bullets_per_slide=3)
        
        optimized = self.planner._optimize_slide_sequence(slides, config)
        
        # Should have split into multiple slides
        assert len(optimized) > 1
        
        # Check indices are sequential
        for i, slide in enumerate(optimized):
            assert slide.index == i

    def test_optimize_slide_sequence_max_slides(self):
        """Test slide sequence optimization with slide limit."""
        slides = [
            SlidePlan(
                index=i,
                slide_type="content",
                title=f"Slide {i}",
                bullets=["Point 1"]
            )
            for i in range(10)
        ]
        
        config = GenerationConfig(max_slides=5)
        
        optimized = self.planner._optimize_slide_sequence(slides, config)
        
        # Should be limited to 5 slides
        assert len(optimized) == 5

    def test_validate_slide_plan_success(self):
        """Test successful slide plan validation."""
        slides = [
            SlidePlan(
                index=0,
                slide_type="title",
                title="Title",
                bullets=[],
                layout_id=0
            ),
            SlidePlan(
                index=1,
                slide_type="content",
                title="Content",
                bullets=["Point 1"],
                layout_id=1
            )
        ]
        
        config = GenerationConfig(max_slides=10)
        
        # Should not raise any exception
        self.planner._validate_slide_plan(slides, config)

    def test_validate_slide_plan_no_slides(self):
        """Test validation with no slides."""
        config = GenerationConfig()
        
        with pytest.raises(ValueError, match="No slides in plan"):
            self.planner._validate_slide_plan([], config)

    def test_validate_slide_plan_too_many_slides(self):
        """Test validation with too many slides."""
        slides = [
            SlidePlan(
                index=i,
                slide_type="content",
                title=f"Slide {i}",
                bullets=[],
                layout_id=0
            )
            for i in range(10)
        ]
        
        config = GenerationConfig(max_slides=5)
        
        with pytest.raises(ValueError, match="Too many slides"):
            self.planner._validate_slide_plan(slides, config)

    def test_validate_slide_plan_no_layout(self):
        """Test validation with missing layout ID."""
        slides = [
            SlidePlan(
                index=0,
                slide_type="title",
                title="Title",
                bullets=[],
                layout_id=None  # Missing layout
            )
        ]
        
        config = GenerationConfig()
        
        with pytest.raises(ValueError, match="has no layout assigned"):
            self.planner._validate_slide_plan(slides, config)

    def test_validate_slide_plan_invalid_layout(self):
        """Test validation with invalid layout ID."""
        slides = [
            SlidePlan(
                index=0,
                slide_type="title",
                title="Title",
                bullets=[],
                layout_id=99  # Invalid layout ID
            )
        ]
        
        config = GenerationConfig()
        
        with pytest.raises(ValueError, match="invalid layout ID"):
            self.planner._validate_slide_plan(slides, config)

    def test_analyze_layout_usage(self):
        """Test layout usage analysis."""
        slides = [
            SlidePlan(index=0, slide_type="title", title="Title", bullets=[], layout_id=0),
            SlidePlan(index=1, slide_type="content", title="Content 1", bullets=[], layout_id=1),
            SlidePlan(index=2, slide_type="content", title="Content 2", bullets=[], layout_id=1),
        ]
        
        usage = self.planner.analyze_layout_usage(slides)
        
        assert usage["title"] == 1
        assert usage["content"] == 2

    def test_get_planning_summary(self):
        """Test planning summary generation."""
        slides = [
            SlidePlan(
                index=0,
                slide_type="title",
                title="Title",
                bullets=[],
                layout_id=0
            ),
            SlidePlan(
                index=1,
                slide_type="content",
                title="Content",
                bullets=["Point 1", "Point 2"],
                layout_id=1,
                image_query="test image"
            ),
            SlidePlan(
                index=2,
                slide_type="chart",
                title="Chart",
                bullets=["Point 1"],
                layout_id=1,
                chart_data={"x": [1, 2], "y": [3, 4]}
            )
        ]
        
        summary = self.planner.get_planning_summary(slides)
        
        assert summary["total_slides"] == 3
        assert summary["slide_types"]["title"] == 1
        assert summary["slide_types"]["content"] == 1
        assert summary["slide_types"]["chart"] == 1
        assert summary["visuals"]["images"] == 1
        assert summary["visuals"]["charts"] == 1
        assert summary["content"]["total_bullets"] == 3
        assert summary["content"]["avg_bullets_per_slide"] == 1.0

    def test_plan_individual_slide(self):
        """Test individual slide planning."""
        slide = SlidePlan(
            index=1,
            slide_type="content",
            title="Test Slide",
            bullets=["Point 1", "Point 2"]
        )
        
        config = GenerationConfig(include_images=True, include_charts=True)
        
        planned = self.planner._plan_individual_slide(slide, config)
        
        assert planned.layout_id is not None
        assert planned.speaker_notes is not None
        # Should have generated image query for content slide
        assert planned.image_query is not None