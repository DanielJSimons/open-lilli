"""Tests for visual generator."""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call, ANY # ensure 'call' and 'ANY' is imported

import pytest
from PIL import Image

from open_lilli.models import SlidePlan, VisualExcellenceConfig, NativeChartData, ChartType, AssetLibraryConfig # Add AssetLibraryConfig
from open_lilli.visual_generator import VisualGenerator
from open_lilli.corporate_asset_library import CorporateAssetLibrary # For mocking


# Mock for PNG generation methods
MOCK_PNG_PATH = Path("mock_chart.png")

# Data for new chart types
AREA_CHART_DATA = {"type": "area", "title": "Monthly Growth", "categories": ["Jan", "Feb", "Mar"], "series": [{"name": "Growth A", "values": [10, 15, 12]}, {"name": "Growth B", "values": [8, 10, 14]}]}
DOUGHNUT_CHART_DATA = {"type": "doughnut", "title": "Market Share", "labels": ["Alpha", "Beta", "Gamma"], "values": [50, 30, 20]}


@pytest.fixture
def visual_generator_native_enabled(tmp_path):
    config = VisualExcellenceConfig(enable_native_charts=True, enable_process_flows=False, enable_asset_library=False)
    # Mock dependencies if NativeChartBuilder is instantiated within VisualGenerator's init
    with patch('open_lilli.visual_generator.NativeChartBuilder') as MockNativeChartBuilder:
        vg = VisualGenerator(output_dir=tmp_path, visual_config=config)
        vg.native_chart_builder = MockNativeChartBuilder() # Ensure it has the mock
        return vg

@pytest.fixture
def visual_generator_native_disabled(tmp_path):
    config = VisualExcellenceConfig(enable_native_charts=False, enable_process_flows=False, enable_asset_library=False)
    vg = VisualGenerator(output_dir=tmp_path, visual_config=config)
    # No need to mock NativeChartBuilder if it's not expected to be used or checked here
    return vg

# Test data for generative AI
GEN_AI_CONFIG = AssetLibraryConfig(
    generative_ai_provider="dalle3",
    generative_ai_api_key="fake_openai_key",
    generative_ai_model="dall-e-3"
)
STABLE_DIFFUSION_CONFIG = AssetLibraryConfig(
    generative_ai_provider="stablediffusion",
    generative_ai_api_key="fake_sd_key",
    generative_ai_model="sd-xl-1.0"
)


class TestVisualGenerator:
    """Tests for VisualGenerator class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        # Default generator for most tests
        self.generator = VisualGenerator(output_dir=self.temp_dir)

        # Generator with GenAI configured
        gen_ai_visual_config = VisualExcellenceConfig(
            enable_generative_ai=True,
            asset_library=GEN_AI_CONFIG
        )
        self.gen_ai_generator = VisualGenerator(
            output_dir=self.temp_dir,
            visual_config=gen_ai_visual_config
        )
        # Ensure corporate_asset_library is also initialized if source_image checks it
        # based on visual_config.asset_library being present
        if gen_ai_visual_config.asset_library:
             self.gen_ai_generator.corporate_asset_library = CorporateAssetLibrary(gen_ai_visual_config.asset_library)


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

    def test_generate_area_chart(self):
        """Test area chart generation."""
        slide = SlidePlan(index=3, slide_type="chart", title="Area Chart Test", chart_data=AREA_CHART_DATA.copy()) # Use .copy() if data is modified in tested method

        chart_path = self.generator._generate_area_chart(slide)

        assert chart_path is not None
        assert chart_path.exists()
        assert "chart_slide_3_area.png" in chart_path.name
        try:
            with Image.open(chart_path) as img:
                assert img.format == "PNG"
        except Exception as e:
            pytest.fail(f"Failed to open generated area chart: {e}")

    def test_generate_doughnut_chart(self):
        """Test doughnut chart generation."""
        slide = SlidePlan(index=4, slide_type="chart", title="Doughnut Chart Test", chart_data=DOUGHNUT_CHART_DATA.copy()) # Use .copy()

        chart_path = self.generator._generate_doughnut_chart(slide)

        assert chart_path is not None
        assert chart_path.exists()
        assert "chart_slide_4_doughnut.png" in chart_path.name
        try:
            with Image.open(chart_path) as img:
                assert img.format == "PNG"
        except Exception as e:
            pytest.fail(f"Failed to open generated doughnut chart: {e}")


@patch('open_lilli.visual_generator.VisualGenerator._generate_bar_chart', return_value=MOCK_PNG_PATH)
def test_native_bar_chart_when_enabled(mock_generate_bar, visual_generator_native_enabled):
    vg = visual_generator_native_enabled
    slide = SlidePlan(index=0, slide_type="chart", title="Test", chart_data={"type": "bar", "categories": ["A"], "values": [1]})
    visuals = vg.generate_visuals([slide])
    assert visuals[0] == {"native_chart": "pending"}
    mock_generate_bar.assert_not_called()

@patch('open_lilli.visual_generator.VisualGenerator._generate_line_chart', return_value=MOCK_PNG_PATH)
def test_native_line_chart_when_enabled(mock_generate_line, visual_generator_native_enabled):
    vg = visual_generator_native_enabled
    slide = SlidePlan(index=0, slide_type="chart", title="Test", chart_data={"type": "line", "x": [1], "y": [1]})
    visuals = vg.generate_visuals([slide])
    assert visuals[0] == {"native_chart": "pending"}
    mock_generate_line.assert_not_called()

@patch('open_lilli.visual_generator.VisualGenerator._generate_bar_chart', return_value=MOCK_PNG_PATH) # Assuming column uses _generate_bar_chart or similar
def test_native_column_chart_when_enabled(mock_generate_bar, visual_generator_native_enabled): # RENAMED from mock_generate_column
    vg = visual_generator_native_enabled
    slide = SlidePlan(index=0, slide_type="chart", title="Test", chart_data={"type": "column", "categories": ["A"], "values": [1]})
    visuals = vg.generate_visuals([slide])
    assert visuals[0] == {"native_chart": "pending"}
    mock_generate_bar.assert_not_called() # Or assert specific type if _generate_column_chart exists

@patch('open_lilli.visual_generator.VisualGenerator._generate_pie_chart', return_value=MOCK_PNG_PATH)
def test_png_fallback_for_other_types_when_native_enabled(mock_generate_pie, visual_generator_native_enabled):
    vg = visual_generator_native_enabled
    slide = SlidePlan(index=0, slide_type="chart", title="Test", chart_data={"type": "pie", "labels": ["L"], "values": [1]})
    visuals = vg.generate_visuals([slide])
    assert visuals[0] == {"chart": str(MOCK_PNG_PATH)}
    mock_generate_pie.assert_called_once()

@patch('open_lilli.visual_generator.VisualGenerator._generate_bar_chart', return_value=MOCK_PNG_PATH)
def test_png_fallback_when_native_disabled(mock_generate_bar, visual_generator_native_disabled):
    vg = visual_generator_native_disabled
    slide = SlidePlan(index=0, slide_type="chart", title="Test", chart_data={"type": "bar", "categories": ["A"], "values": [1]})
    visuals = vg.generate_visuals([slide])
    assert visuals[0] == {"chart": str(MOCK_PNG_PATH)}
    mock_generate_bar.assert_called_once()

@patch('open_lilli.visual_generator.VisualGenerator._generate_pie_chart', return_value=MOCK_PNG_PATH)
def test_explicit_native_request_honored(mock_generate_pie, visual_generator_native_enabled):
    vg = visual_generator_native_enabled
    slide = SlidePlan(index=0, slide_type="chart", title="Test", chart_data={"type": "pie", "native_chart": "pending", "labels": ["L"], "values": [1]})
    visuals = vg.generate_visuals([slide])
    assert visuals[0] == {"native_chart": "pending"}
    mock_generate_pie.assert_not_called()

@patch('open_lilli.visual_generator.VisualGenerator._generate_bar_chart', return_value=MOCK_PNG_PATH) # Assuming NativeChartData might be bar
def test_native_chart_data_instance_handled(mock_generate_bar, visual_generator_native_enabled):
    vg = visual_generator_native_enabled
    chart_data_obj = NativeChartData(chart_type=ChartType.BAR, title="Native Obj", categories=["A"], series=[{"name": "S1", "values": [1]}])
    slide = SlidePlan(index=0, slide_type="chart", title="Test", chart_data=chart_data_obj)
    visuals = vg.generate_visuals([slide])
    assert visuals[0] == {"native_chart": "pending"}
    mock_generate_bar.assert_not_called()

# Test for when chart_data is not a dict or NativeChartData (should fallback or log error, ensure it doesn't crash)
@patch('open_lilli.visual_generator.VisualGenerator.generate_chart', return_value=MOCK_PNG_PATH) # Mock the generic generate_chart
def test_invalid_chart_data_type_fallback(mock_generate_chart_png, visual_generator_native_enabled):
    vg = visual_generator_native_enabled
    slide = SlidePlan(index=0, slide_type="chart", title="Test", chart_data="this is not a dict") # Invalid chart_data
    # This call should not raise an unhandled exception due to type error
    # The implementation in VisualGenerator's generate_visuals logs an error and continues.
    # We expect no visuals to be generated for the chart part or a fallback to PNG if generate_chart can handle it.
    # Based on current VisualGenerator, an error is logged, and no chart visual is added to 'visuals'.
    visuals = vg.generate_visuals([slide])
    assert 0 not in visuals or ("chart" not in visuals[0] and "native_chart" not in visuals[0])
    # Depending on implementation, generate_chart might be called or not.
    # If it's called, it should handle the non-dict data gracefully.
    # If the new logic in generate_visuals catches this before calling generate_chart, then it won't be called.
    # The current VisualGenerator change has a try-except that logs error, so generate_chart (PNG) won't be called.
    mock_generate_chart_png.assert_not_called()

# --- Area Chart ---
@patch('open_lilli.visual_generator.VisualGenerator._generate_area_chart', return_value=Path("mock_area_chart.png"))
def test_native_area_chart_when_enabled(mock_generate_area_png, visual_generator_native_enabled):
    vg = visual_generator_native_enabled
    # Use a simplified version of AREA_CHART_DATA for this test if full data isn't needed for flagging logic
    slide_area_data = {"type": "area", "categories": ["X1", "X2"], "series": [{"name": "S1", "values": [1,2]}]}
    slide = SlidePlan(index=0, slide_type="chart", title="Test Area", chart_data=slide_area_data)
    visuals = vg.generate_visuals([slide])
    assert visuals[0] == {"native_chart": "pending"}
    mock_generate_area_png.assert_not_called()

@patch('open_lilli.visual_generator.VisualGenerator._generate_area_chart', return_value=MOCK_PNG_PATH)
def test_png_area_chart_when_native_disabled(mock_generate_area_png, visual_generator_native_disabled):
    vg = visual_generator_native_disabled
    slide_area_data = {"type": "area", "categories": ["X1", "X2"], "series": [{"name": "S1", "values": [1,2]}]}
    slide = SlidePlan(index=0, slide_type="chart", title="Test Area PNG", chart_data=slide_area_data)
    visuals = vg.generate_visuals([slide])
    assert visuals[0] == {"chart": str(MOCK_PNG_PATH)}
    mock_generate_area_png.assert_called_once()

# --- Doughnut Chart ---
@patch('open_lilli.visual_generator.VisualGenerator._generate_doughnut_chart', return_value=Path("mock_doughnut_chart.png"))
def test_native_doughnut_chart_when_enabled(mock_generate_doughnut_png, visual_generator_native_enabled):
    vg = visual_generator_native_enabled
    slide_doughnut_data = {"type": "doughnut", "labels": ["L1", "L2"], "values": [1,2]}
    slide = SlidePlan(index=0, slide_type="chart", title="Test Doughnut", chart_data=slide_doughnut_data)
    visuals = vg.generate_visuals([slide])
    assert visuals[0] == {"native_chart": "pending"}
    mock_generate_doughnut_png.assert_not_called()

@patch('open_lilli.visual_generator.VisualGenerator._generate_doughnut_chart', return_value=MOCK_PNG_PATH)
def test_png_doughnut_chart_when_native_disabled(mock_generate_doughnut_png, visual_generator_native_disabled):
    vg = visual_generator_native_disabled
    slide_doughnut_data = {"type": "doughnut", "labels": ["L1", "L2"], "values": [1,2]}
    slide = SlidePlan(index=0, slide_type="chart", title="Test Doughnut PNG", chart_data=slide_doughnut_data)
    visuals = vg.generate_visuals([slide])
    assert visuals[0] == {"chart": str(MOCK_PNG_PATH)}
    mock_generate_doughnut_png.assert_called_once()


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

# --- Tests for T-72 (Generative AI Provider) ---

@patch('openai.Image.create')
@patch('requests.get')
@patch('open_lilli.visual_generator.Image.open') # Mock PIL.Image.open via its import location
def test_source_from_generative_ai_dalle3_success(self, mock_pil_image_open, mock_requests_get, mock_openai_create):
    """Test _source_from_generative_ai with DALL·E 3 successfully generates and saves an image."""
    # Setup mocks
    mock_openai_create.return_value = MagicMock(data=[MagicMock(url="https://fake.openai.com/image.png")])

    mock_image_content = b"fake_image_bytes"
    mock_requests_response = MagicMock()
    mock_requests_response.content = mock_image_content
    mock_requests_response.raise_for_status = MagicMock()
    mock_requests_get.return_value = mock_requests_response

    mock_pil_img_instance = MagicMock()
    mock_pil_image_open.return_value = mock_pil_img_instance # To simulate saving

    slide_plan = SlidePlan(
        index=0,
        slide_type="content",
        title="AI Benefits",
        bullets=["Faster processing", "New insights"],
        image_query="abstract concept of AI"
    )
    
    # Use the generator configured for GenAI
    vg = self.gen_ai_generator
    
    # Directly call the method to test its isolated behavior
    # This requires visual_config and asset_library to be set up for generative_ai_provider
    if not vg.visual_config or not vg.visual_config.asset_library:
        pytest.fail("VisualGenerator not configured correctly for GenAI test.")

    vg.visual_config.asset_library.generative_ai_provider = "dalle3"
    vg.visual_config.asset_library.generative_ai_api_key = "test_dalle_key"
    vg.visual_config.asset_library.generative_ai_model = "dall-e-3"


    with patch('builtins.open', mock_open()) as mock_file_write:
        result_path = vg._source_from_generative_ai(
            query="abstract concept of AI",
            slide_index=0,
            slide_context=slide_plan,
            palette=vg.theme_colors
        )

    assert result_path is not None
    assert result_path.name.startswith("gen_image_slide_0_abstract_concept_of_ai")
    assert result_path.suffix == ".png"

    mock_openai_create.assert_called_once()
    args_create, kwargs_create = mock_openai_create.call_args

    # Assert prompt construction
    self.assertIn("abstract concept of AI", kwargs_create['prompt'])
    self.assertIn("Slide title: 'AI Benefits'", kwargs_create['prompt'])
    self.assertIn("Slide bullets: Faster processing; New insights", kwargs_create['prompt'])
    self.assertIn("photorealistic", kwargs_create['prompt'])
    # Check a primary color from default theme_colors is in prompt
    self.assertIn(f"primary color {vg.theme_colors['primary']}", kwargs_create['prompt'])


    assert kwargs_create['model'] == "dall-e-3"
    assert openai.api_key == "test_dalle_key" # Check API key was set

    mock_requests_get.assert_called_once_with("https://fake.openai.com/image.png", timeout=30)
    mock_file_write.assert_called_once_with(result_path, 'wb')
    mock_file_write().write.assert_called_once_with(mock_image_content)

@patch.object(VisualGenerator, '_generate_placeholder_image')
def test_source_from_generative_ai_stablediffusion_placeholder(self, mock_gen_placeholder):
    """Test _source_from_generative_ai with StableDiffusion uses placeholder."""

    sd_visual_config = VisualExcellenceConfig(
        enable_generative_ai=True, # ensure this is True for the source_image path
        asset_library=STABLE_DIFFUSION_CONFIG
    )
    vg = VisualGenerator(output_dir=self.temp_dir, visual_config=sd_visual_config)
    if sd_visual_config.asset_library: # ensure CAL is init if source_image checks it
        vg.corporate_asset_library = CorporateAssetLibrary(sd_visual_config.asset_library)


    mock_gen_placeholder.return_value = Path(self.temp_dir) / "sd_placeholder.png"

    slide_plan = SlidePlan(index=1, slide_type="content", title="SD Test", image_query="diffusion art")

    with self.assertLogs(logger='open_lilli.visual_generator', level='WARNING') as log_watcher:
        result_path = vg._source_from_generative_ai(
            query="diffusion art",
            slide_index=1,
            slide_context=slide_plan,
            palette=vg.theme_colors
        )

    assert result_path == mock_gen_placeholder.return_value
    mock_gen_placeholder.assert_called_once_with("StableDiffusion: diffusion art", 1)
    self.assertTrue(any("Stable Diffusion provider selected, but implementation is a placeholder" in message for message in log_watcher.output))

@patch.object(VisualGenerator, '_source_from_generative_ai')
@patch.object(VisualGenerator, '_source_from_unsplash')
def test_source_image_gen_ai_success_skips_unsplash(self, mock_unsplash, mock_gen_ai):
    """Test source_image uses GenAI result and skips Unsplash."""
    mock_gen_ai.return_value = Path(self.temp_dir) / "gen_ai_image.png"

    vg = self.gen_ai_generator # Uses dalle3 config by default in setup
    slide_plan = SlidePlan(index=0, slide_type="content", title="Test", image_query="test query")

    result = vg.source_image(query="test query", slide_index=0, slide_plan=slide_plan)

    assert result == mock_gen_ai.return_value
    mock_gen_ai.assert_called_once_with("test query", 0, slide_plan, vg.theme_colors)
    mock_unsplash.assert_not_called()

@patch.object(VisualGenerator, '_source_from_generative_ai')
@patch.object(VisualGenerator, '_source_from_unsplash')
@patch.object(VisualGenerator, '_generate_placeholder_image')
def test_source_image_gen_ai_fails_calls_unsplash(self, mock_placeholder, mock_unsplash, mock_gen_ai):
    """Test source_image tries Unsplash if GenAI fails."""
    mock_gen_ai.return_value = None # GenAI fails
    mock_unsplash.return_value = Path(self.temp_dir) / "unsplash_image.jpg"

    vg = self.gen_ai_generator
    slide_plan = SlidePlan(index=0, slide_type="content", title="Test", image_query="test query")

    result = vg.source_image(query="test query", slide_index=0, slide_plan=slide_plan)

    assert result == mock_unsplash.return_value
    mock_gen_ai.assert_called_once()
    mock_unsplash.assert_called_once_with("test query", 0)
    mock_placeholder.assert_not_called()

@patch.object(VisualGenerator, '_source_from_generative_ai')
@patch.object(VisualGenerator, '_source_from_unsplash')
@patch.object(VisualGenerator, '_generate_placeholder_image')
def test_source_image_all_fail_calls_placeholder(self, mock_placeholder, mock_unsplash, mock_gen_ai):
    """Test source_image uses placeholder if GenAI and Unsplash fail."""
    mock_gen_ai.return_value = None
    mock_unsplash.return_value = None
    mock_placeholder.return_value = Path(self.temp_dir) / "placeholder.png"

    vg = self.gen_ai_generator
    slide_plan = SlidePlan(index=0, slide_type="content", title="Test", image_query="test query")

    result = vg.source_image(query="test query", slide_index=0, slide_plan=slide_plan)

    assert result == mock_placeholder.return_value
    mock_gen_ai.assert_called_once()
    mock_unsplash.assert_called_once()
    mock_placeholder.assert_called_once_with("test query", 0)


# --- Tests for T-73 Integration (target_aspect_ratio in generate_visuals) ---
@patch('open_lilli.visual_generator.CorporateAssetLibrary') # Mock the CAL class itself
def test_generate_visuals_passes_target_aspect_ratio_to_cal(self, MockCorporateAssetLibrary):
    """Test generate_visuals calculates and passes target_aspect_ratio to CAL."""

    # Setup mock CAL instance and its method
    mock_cal_instance = MockCorporateAssetLibrary.return_value
    mock_cal_instance.get_brand_approved_image.return_value = Path(self.temp_dir) / "cal_image.png"

    # Setup mock template_parser
    mock_template_parser = MagicMock()
    mock_layout = MagicMock()
    mock_placeholder_shape = MagicMock()
    mock_placeholder_shape.width = 1600 # EMU or consistent unit
    mock_placeholder_shape.height = 900  # EMU or consistent unit
    # Assuming the heuristic for identifying picture placeholders
    # For example, by checking a 'type_name' attribute added by the parser
    mock_placeholder_shape.type_name = "PICTURE"

    mock_layout.placeholders = [mock_placeholder_shape]
    mock_template_parser.get_layout.return_value = mock_layout

    # Configure VisualGenerator with this mock parser and CAL
    config_with_cal = VisualExcellenceConfig(
        enable_asset_library=True,
        asset_library=AssetLibraryConfig(dam_api_url="http://fake.dam") # Needs some config for CAL to be init'd
    )
    vg = VisualGenerator(
        output_dir=self.temp_dir,
        template_parser=mock_template_parser,
        visual_config=config_with_cal
    )
    # Manually assign the mocked CAL instance if it's not done via constructor due to patching class
    vg.corporate_asset_library = mock_cal_instance


    slide = SlidePlan(
        index=0,
        slide_type="image",
        title="CAL Test",
        image_query="corporate image",
        layout_id=1 # Important for triggering aspect ratio logic
    )
    vg.generate_visuals([slide])

    expected_aspect_ratio = 1600 / 900
    mock_cal_instance.get_brand_approved_image.assert_called_once_with(
        query="corporate image",
        slide_index=0,
        orientation=None,
        dominant_color=None,
        tags=None,
        target_aspect_ratio=expected_aspect_ratio
    )

@patch('open_lilli.visual_generator.CorporateAssetLibrary')
def test_generate_visuals_no_aspect_ratio_if_no_parser_or_layout(self, MockCorporateAssetLibrary):
    """Test target_aspect_ratio is None if no parser or layout_id."""
    mock_cal_instance = MockCorporateAssetLibrary.return_value
    mock_cal_instance.get_brand_approved_image.return_value = Path(self.temp_dir) / "cal_image_no_ratio.png"

    config_with_cal = VisualExcellenceConfig(
        enable_asset_library=True,
        asset_library=AssetLibraryConfig(dam_api_url="http://fake.dam")
    )
    # Case 1: No template_parser
    vg_no_parser = VisualGenerator(output_dir=self.temp_dir, template_parser=None, visual_config=config_with_cal)
    vg_no_parser.corporate_asset_library = mock_cal_instance

    slide_no_parser = SlidePlan(index=0, slide_type="image", title="Test", image_query="q", layout_id=1)
    vg_no_parser.generate_visuals([slide_no_parser])

    mock_cal_instance.get_brand_approved_image.assert_called_with(
        query="q", slide_index=0, orientation=None, dominant_color=None, tags=None, target_aspect_ratio=None
    )
    mock_cal_instance.reset_mock()

    # Case 2: No layout_id
    mock_template_parser = MagicMock() # Has parser, but slide has no layout_id
    vg_no_layout_id = VisualGenerator(output_dir=self.temp_dir, template_parser=mock_template_parser, visual_config=config_with_cal)
    vg_no_layout_id.corporate_asset_library = mock_cal_instance

    slide_no_layout_id = SlidePlan(index=1, slide_type="image", title="Test", image_query="q2", layout_id=None)
    vg_no_layout_id.generate_visuals([slide_no_layout_id])

    mock_cal_instance.get_brand_approved_image.assert_called_with(
        query="q2", slide_index=1, orientation=None, dominant_color=None, tags=None, target_aspect_ratio=None
    )


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
