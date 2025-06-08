"""Tests for slide planner."""

from unittest.mock import Mock, MagicMock

import pytest

from open_lilli.models import GenerationConfig, Outline, SlidePlan, DesignPattern
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

    def test_t90_bullet_truncate_replaced_with_slide_split(self):
        """Test T-90: Replace Bullet-Truncate with Slide-Split."""
        # Create slide with 7 bullets when limit is 5
        slide = SlidePlan(
            index=1,
            slide_type="content",
            title="Long Slide",
            bullets=[f"Point {i}" for i in range(1, 8)]  # 7 bullets
        )
        
        config = GenerationConfig(max_bullets_per_slide=5)
        
        # Plan individual slide - should mark for splitting, not truncate
        planned = self.planner._plan_individual_slide(slide, config)
        
        # Verify all bullets are preserved (no truncation)
        assert len(planned.bullets) == 7, "All bullets should be preserved"
        
        # Verify slide is marked for splitting
        assert planned.needs_splitting is True, "Slide should be marked for splitting"
        
        # Now test the optimization sequence which should perform the split
        outline = Outline(title="Test", slides=[planned])
        final_slides = self.planner.plan_slides(outline, config)
        
        # Should result in split slides with zero content loss
        total_bullets = sum(len(s.bullets) for s in final_slides)
        assert total_bullets == 7, "Zero content loss - all bullets should be preserved after splitting"
        
        # Should have more than 1 slide after splitting
        assert len(final_slides) > 1, "Should have split into multiple slides"


    # --- Tests for Design Pattern Influence ---

    @pytest.fixture
    def planner_with_design_pattern(self, request):
        """Fixture to create SlidePlanner with a specific DesignPattern."""
        design_pattern_params = getattr(request, "param", {})
        design_pattern = DesignPattern(**design_pattern_params) if design_pattern_params else None

        # Mock TemplateParser and OpenAI client as they are dependencies
        mock_tp = Mock(spec=TemplateParser)
        mock_tp.layout_map = {
            "title": 0, "content": 1, "section": 2, "image": 3, "two_column": 4, "blank": 5
        }
        mock_tp.list_available_layouts.return_value = list(mock_tp.layout_map.keys())
        mock_tp.get_layout_index = lambda lt: mock_tp.layout_map.get(lt, 0)
        mock_tp.get_layout_type_by_id = lambda idx: next((k for k, v in mock_tp.layout_map.items() if v == idx), f"layout_{idx}")

        # Mock presentation for validation step in planner
        mock_tp.prs = Mock()
        mock_tp.prs.slide_layouts = [Mock() for _ in range(len(mock_tp.layout_map))]

        # Mock ContentFitAnalyzer and SmartContentFitter as they are used internally
        # and might require template_style or openai_client
        mock_openai_client = MagicMock() # Using MagicMock for more flexibility if methods are called

        planner = SlidePlanner(
            template_parser=mock_tp,
            openai_client=mock_openai_client, # Pass the mock client
            design_pattern=design_pattern
        )

        # Mock internal components that might be complex to set up otherwise
        planner.content_fit_analyzer = MagicMock()
        planner.smart_fitter = MagicMock()

        # Default behavior for smart_fitter.rebalance to return slides as is
        planner.smart_fitter.rebalance.side_effect = lambda s, ts: s

        # Default behavior for content_fit_analyzer.optimize_slide_content
        def mock_optimize_slide_content(slide, template_style):
            from open_lilli.models import ContentFitResult, ContentDensityAnalysis
            density_analysis = ContentDensityAnalysis(
                total_characters=len(slide.title) + sum(len(b) for b in slide.get_bullet_texts()),
                estimated_lines=5, # Dummy value
                placeholder_capacity=500, # Dummy value
                density_ratio=0.5, # Dummy value
                requires_action=False,
                recommended_action="none"
            )
            return ContentFitResult(
                slide_index=slide.index,
                density_analysis=density_analysis,
                final_action="none",
                modified_slide_plan=slide # Return the slide as is
            )
        planner.content_fit_analyzer.optimize_slide_content.side_effect = mock_optimize_slide_content

        # Mock split_slide_content to return the slide in a list by default
        planner.content_fit_analyzer.split_slide_content.side_effect = lambda s: [s]
        planner.content_fit_analyzer.get_optimization_summary.return_value = {
            "slides_requiring_action":0, "total_slides":0, "slides_split":0, "slides_font_adjusted":0
        }


        return planner

    @pytest.mark.parametrize("planner_with_design_pattern", [{"name": "minimalist", "primary_intent": "readability"}], indirect=True)
    def test_minimalist_design_reduces_max_bullets(self, planner_with_design_pattern: SlidePlanner):
        """Test that 'minimalist' design reduces max_bullets_per_slide and causes splitting."""
        planner = planner_with_design_pattern
        outline = Outline(
            title="Test Outline",
            slides=[
                SlidePlan(index=0, slide_type="title", title="Title Slide", bullets=[]),
                SlidePlan(index=1, slide_type="content", title="Content Slide 1", bullets=["B1", "B2", "B3", "B4"]) # 4 bullets
            ]
        )
        original_config = GenerationConfig(max_bullets_per_slide=5, include_images=False, include_charts=False)

        # With minimalist design, max_bullets should be 5 - 2 = 3. So, slide with 4 bullets should be marked for splitting.

        # Mock the _split_slide method to check it's called as expected due to reduced bullet limit
        # or check needs_splitting flag

        planned_slides = planner.plan_slides(outline, original_config)

        # Effective max_bullets should be 3. Slide 1 has 4 bullets.
        # The _optimize_slide_sequence method should handle the splitting.
        # We need to ensure that the logic inside _plan_individual_slide sets needs_splitting=True

        slide1_planned = next(s for s in planned_slides if s.title == "Content Slide 1 (Part 1)" or s.title == "Content Slide 1")

        # Check if the original config was modified
        assert original_config.max_bullets_per_slide == 5

        # Check if the slide was marked for splitting or actually split
        # If it was split, there will be more slides than in the original outline for that content.
        content_slides = [s for s in planned_slides if "Content Slide 1" in s.title]
        assert len(content_slides) > 1, "Slide should have been split due to reduced bullet limit from minimalist design."
        assert content_slides[0].bullets == ["B1", "B2", "B3"]
        assert content_slides[1].bullets == ["B4"]


    @pytest.mark.parametrize("planner_with_design_pattern", [{"name": "standard"}], indirect=True)
    def test_standard_design_uses_default_max_bullets(self, planner_with_design_pattern: SlidePlanner):
        """Test that 'standard' design uses the default max_bullets_per_slide."""
        planner = planner_with_design_pattern
        outline = Outline(
            title="Test Outline",
            slides=[
                SlidePlan(index=0, slide_type="title", title="Title Slide", bullets=[]),
                SlidePlan(index=1, slide_type="content", title="Content Slide 1", bullets=["B1", "B2", "B3", "B4"]) # 4 bullets
            ]
        )
        original_config = GenerationConfig(max_bullets_per_slide=5, include_images=False, include_charts=False)

        planned_slides = planner.plan_slides(outline, original_config)

        # Check if the original config was modified
        assert original_config.max_bullets_per_slide == 5

        # Slide should not be split as 4 bullets <= 5 (default limit)
        content_slides = [s for s in planned_slides if "Content Slide 1" in s.title]
        assert len(content_slides) == 1, "Slide should not have been split with standard design."
        assert content_slides[0].bullets == ["B1", "B2", "B3", "B4"]
        assert not getattr(content_slides[0], 'needs_splitting', False)


    @pytest.mark.parametrize("planner_with_design_pattern", [{"name": "vibrant", "primary_intent": "visual_impact"}], indirect=True)
    def test_other_design_intent_uses_default_max_bullets(self, planner_with_design_pattern: SlidePlanner):
        """Test that non-minimalist/readability designs use default max_bullets_per_slide."""
        planner = planner_with_design_pattern
        outline = Outline(
            title="Test Outline",
            slides=[
                SlidePlan(index=0, slide_type="title", title="Title Slide", bullets=[]),
                SlidePlan(index=1, slide_type="content", title="Content Slide 1", bullets=["B1", "B2", "B3", "B4"]) # 4 bullets
            ]
        )
        original_config = GenerationConfig(max_bullets_per_slide=5, include_images=False, include_charts=False)

        planned_slides = planner.plan_slides(outline, original_config)

        assert original_config.max_bullets_per_slide == 5
        content_slides = [s for s in planned_slides if "Content Slide 1" in s.title]
        assert len(content_slides) == 1, "Slide should not have been split."
        assert not getattr(content_slides[0], 'needs_splitting', False)

    def test_plan_slides_no_design_pattern(self, planner_with_design_pattern: SlidePlanner):
        """Test plan_slides when no design_pattern is provided to SlidePlanner."""
        # This test will use the planner_with_design_pattern fixture,
        # but the default for "param" in the fixture is {} which results in design_pattern=None
        planner = planner_with_design_pattern # design_pattern will be None here
        assert planner.design_pattern is None

        outline = Outline(
            title="Test Outline",
            slides=[
                SlidePlan(index=0, slide_type="title", title="Title Slide", bullets=[]),
                SlidePlan(index=1, slide_type="content", title="Content Slide 1", bullets=["B1", "B2", "B3", "B4"]) # 4 bullets
            ]
        )
        original_config = GenerationConfig(max_bullets_per_slide=5, include_images=False, include_charts=False)

        planned_slides = planner.plan_slides(outline, original_config)

        assert original_config.max_bullets_per_slide == 5
        content_slides = [s for s in planned_slides if "Content Slide 1" in s.title]
        assert len(content_slides) == 1, "Slide should not have been split when no design pattern is active."
        assert not getattr(content_slides[0], 'needs_splitting', False)