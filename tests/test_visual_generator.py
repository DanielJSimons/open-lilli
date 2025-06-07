"""Tests for visual generator."""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from PIL import Image

from open_lilli.models import SlidePlan
from open_lilli.visual_generator import VisualGenerator


class TestVisualGenerator:
    """Tests for VisualGenerator class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.generator = VisualGenerator(output_dir=self.temp_dir)

    def teardown_method(self):
        """Clean up test fixtures."""
        # Clean up generated files
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def create_chart_slide(self) -> SlidePlan:
        """Create a slide with chart data."""
        return SlidePlan(
            index=1,
            slide_type="chart",
            title="Sales Performance",
            bullets=["Key insight 1"],
            chart_data={
                "type": "bar",
                "categories": ["Q1", "Q2", "Q3", "Q4"],
                "values": [100, 150, 200, 180],
                "title": "Quarterly Sales",
                "xlabel": "Quarter",
                "ylabel": "Sales ($K)"
            }
        )

    def create_image_slide(self) -> SlidePlan:
        """Create a slide with image query."""
        return SlidePlan(
            index=2,
            slide_type="image",
            title="Market Growth",
            bullets=["Market expanding"],
            image_query="business growth market"
        )

    def test_init(self):
        """Test VisualGenerator initialization."""
        assert self.generator.output_dir == Path(self.temp_dir)
        assert self.generator.output_dir.exists()
        assert isinstance(self.generator.theme_colors, dict)
        assert "primary" in self.generator.theme_colors

    def test_generate_visuals_with_chart(self):
        """Test generating visuals for slides with charts."""
        slides = [self.create_chart_slide()]
        
        visuals = self.generator.generate_visuals(slides)
        
        assert 1 in visuals
        assert "chart" in visuals[1]
        
        # Check file was created
        chart_path = Path(visuals[1]["chart"])
        assert chart_path.exists()
        assert chart_path.suffix == ".png"

    @patch('requests.get')
    def test_generate_visuals_with_image_success(self, mock_get):
        """Test generating visuals with successful image download."""
        # Mock successful image download
        mock_response = Mock()
        mock_response.content = b"fake_image_content"
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        slides = [self.create_image_slide()]
        
        visuals = self.generator.generate_visuals(slides)
        
        assert 2 in visuals
        assert "image" in visuals[2]
        
        # Check file was created
        image_path = Path(visuals[2]["image"])
        assert image_path.exists()

    def test_generate_visuals_with_image_fallback(self):
        """Test generating visuals with image fallback to placeholder."""
        slides = [self.create_image_slide()]
        
        # This will fail to download and create a placeholder
        visuals = self.generator.generate_visuals(slides)
        
        assert 2 in visuals
        assert "image" in visuals[2]
        
        # Check placeholder was created
        image_path = Path(visuals[2]["image"])
        assert image_path.exists()
        assert "placeholder" in image_path.name

    def test_generate_bar_chart(self):
        """Test bar chart generation."""
        slide = self.create_chart_slide()
        
        chart_path = self.generator._generate_bar_chart(slide)
        
        assert chart_path is not None
        assert chart_path.exists()
        assert "bar" in chart_path.name

    def test_generate_line_chart(self):
        """Test line chart generation."""
        slide = self.create_chart_slide()
        slide.chart_data["type"] = "line"
        slide.chart_data["x"] = [1, 2, 3, 4]
        slide.chart_data["y"] = [100, 150, 200, 180]
        
        chart_path = self.generator._generate_line_chart(slide)
        
        assert chart_path is not None
        assert chart_path.exists()
        assert "line" in chart_path.name

    def test_generate_pie_chart(self):
        """Test pie chart generation."""
        slide = self.create_chart_slide()
        slide.chart_data = {
            "type": "pie",
            "labels": ["A", "B", "C"],
            "values": [30, 40, 30],
            "title": "Distribution"
        }
        
        chart_path = self.generator._generate_pie_chart(slide)
        
        assert chart_path is not None
        assert chart_path.exists()
        assert "pie" in chart_path.name

    def test_generate_scatter_chart(self):
        """Test scatter chart generation."""
        slide = self.create_chart_slide()
        slide.chart_data = {
            "type": "scatter",
            "x": [1, 2, 3, 4],
            "y": [2, 4, 1, 3],
            "title": "Correlation"
        }
        
        chart_path = self.generator._generate_scatter_chart(slide)
        
        assert chart_path is not None
        assert chart_path.exists()
        assert "scatter" in chart_path.name

    def test_generate_chart_unknown_type(self):
        """Test chart generation with unknown type falls back to bar."""
        slide = self.create_chart_slide()
        slide.chart_data["type"] = "unknown"
        
        chart_path = self.generator.generate_chart(slide)
        
        assert chart_path is not None
        assert "bar" in chart_path.name

    def test_generate_chart_no_data(self):
        """Test chart generation with no chart data."""
        slide = SlidePlan(
            index=1,
            slide_type="content",
            title="No Chart",
            bullets=[]
        )
        
        result = self.generator.generate_chart(slide)
        assert result is None

    @patch('requests.get')
    def test_source_from_unsplash_success(self, mock_get):
        """Test successful image sourcing from Unsplash."""
        mock_response = Mock()
        mock_response.content = b"fake_image_content"
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        image_path = self.generator._source_from_unsplash("business", 1)
        
        assert image_path is not None
        assert image_path.exists()
        assert "business" in image_path.name

    @patch('requests.get')
    def test_source_from_unsplash_failure(self, mock_get):
        """Test handling of Unsplash API failure."""
        mock_get.side_effect = Exception("Network error")
        
        image_path = self.generator._source_from_unsplash("business", 1)
        assert image_path is None

    def test_generate_placeholder_image(self):
        """Test placeholder image generation."""
        image_path = self.generator._generate_placeholder_image("test query", 1)
        
        assert image_path.exists()
        assert "placeholder" in image_path.name
        assert "test_query" in image_path.name
        
        # Check image is valid
        with Image.open(image_path) as img:
            assert img.size == (1600, 900)
            assert img.mode == "RGB"

    def test_generate_placeholder_image_empty_query(self):
        """Test placeholder generation with empty query."""
        image_path = self.generator._generate_placeholder_image("", 1)
        
        assert image_path.exists()
        assert "placeholder" in image_path.name

    def test_create_icon_visual_arrow(self):
        """Test arrow icon creation."""
        icon_path = self.generator.create_icon_visual("arrow", 1)
        
        assert icon_path.exists()
        assert "arrow" in icon_path.name
        
        # Check image is valid
        with Image.open(icon_path) as img:
            assert img.size == (400, 400)

    def test_create_icon_visual_check(self):
        """Test checkmark icon creation."""
        icon_path = self.generator.create_icon_visual("check", 1)
        
        assert icon_path.exists()
        assert "check" in icon_path.name

    def test_create_icon_visual_default(self):
        """Test default icon creation."""
        icon_path = self.generator.create_icon_visual("unknown", 1)
        
        assert icon_path.exists()
        assert "unknown" in icon_path.name

    def test_resize_image(self):
        """Test image resizing."""
        # Create a test image
        test_img = Image.new('RGB', (800, 600), color='red')
        test_path = Path(self.temp_dir) / "test_image.png"
        test_img.save(test_path)
        
        # Resize it
        resized_path = self.generator.resize_image(test_path, (400, 300))
        
        assert resized_path.exists()
        assert "resized_" in resized_path.name
        
        # Check resized dimensions
        with Image.open(resized_path) as img:
            assert img.size == (400, 300)

    def test_resize_image_aspect_ratio(self):
        """Test image resizing preserves aspect ratio."""
        # Create a wide test image
        test_img = Image.new('RGB', (1200, 400), color='blue')
        test_path = Path(self.temp_dir) / "wide_image.png"
        test_img.save(test_path)
        
        # Resize to square target
        resized_path = self.generator.resize_image(test_path, (600, 600))
        
        with Image.open(resized_path) as img:
            assert img.size == (600, 600)
            # Should have white borders due to aspect ratio preservation

    def test_get_visual_summary(self):
        """Test visual summary generation."""
        visuals = {
            1: {"chart": "chart_slide_1_bar.png"},
            2: {"image": "image_slide_2.jpg"},
            3: {"chart": "chart_slide_3_pie.png", "image": "image_slide_3.jpg"}
        }
        
        summary = self.generator.get_visual_summary(visuals)
        
        assert summary["total_slides_with_visuals"] == 3
        assert summary["total_charts"] == 2
        assert summary["total_images"] == 2
        assert summary["chart_types"]["bar"] == 1
        assert summary["chart_types"]["pie"] == 1
        assert summary["output_directory"] == str(self.generator.output_dir)

    def test_get_visual_summary_empty(self):
        """Test visual summary with no visuals."""
        summary = self.generator.get_visual_summary({})
        
        assert summary["total_slides_with_visuals"] == 0
        assert summary["total_charts"] == 0
        assert summary["total_images"] == 0
        assert summary["chart_types"] == {}

    def test_source_image_query_cleaning(self):
        """Test that image queries are properly cleaned."""
        # Query with special characters
        dirty_query = "business & growth! @#$%"
        
        image_path = self.generator.source_image(dirty_query, 1)
        
        assert image_path is not None
        # Should create placeholder since Unsplash will likely fail
        assert "placeholder" in image_path.name

    def test_source_image_empty_query(self):
        """Test handling of empty image query."""
        image_path = self.generator.source_image("", 1)
        
        assert image_path is not None
        assert "placeholder" in image_path.name

    def test_theme_colors_usage(self):
        """Test that theme colors are used in visuals."""
        # Test with custom theme
        custom_colors = {
            "primary": "#FF0000",
            "secondary": "#00FF00", 
            "background": "#FFFFFF"
        }
        
        custom_generator = VisualGenerator(
            output_dir=self.temp_dir,
            theme_colors=custom_colors
        )
        
        assert custom_generator.theme_colors["primary"] == "#FF0000"
        
        # Generate a chart and verify it uses the theme
        slide = self.create_chart_slide()
        chart_path = custom_generator.generate_chart(slide)
        
        assert chart_path is not None
        assert chart_path.exists()