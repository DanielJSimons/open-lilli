"""Tests for CLI module."""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from click.testing import CliRunner

from open_lilli.cli import cli


class TestCLI:
    """Tests for CLI functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
        self.temp_dir = Path(tempfile.mkdtemp())
        
        # Create test files
        self.test_content = self.temp_dir / "content.txt"
        self.test_content.write_text("This is test content for the presentation.")
        
        self.test_template = self.temp_dir / "template.pptx"
        # Create a minimal PPTX file (simplified for testing)
        self.test_template.write_bytes(b"fake pptx content")

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_cli_version(self):
        """Test CLI version option."""
        result = self.runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "version" in result.output.lower()

    def test_cli_help(self):
        """Test CLI help."""
        result = self.runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "Open Lilli" in result.output
        assert "generate" in result.output

    def test_generate_help(self):
        """Test generate command help."""
        result = self.runner.invoke(cli, ["generate", "--help"])
        assert result.exit_code == 0
        assert "--template" in result.output
        assert "--input" in result.output
        assert "--output" in result.output

    def test_generate_missing_api_key(self):
        """Test generate command with missing API key."""
        with patch.dict("os.environ", {}, clear=True):
            result = self.runner.invoke(cli, [
                "generate",
                "--template", str(self.test_template),
                "--input", str(self.test_content),
                "--output", str(self.temp_dir / "output.pptx")
            ])
            
            assert result.exit_code == 1
            assert "OPENAI_API_KEY" in result.output

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test_key"})
    @patch("open_lilli.cli.OpenAI")
    @patch("open_lilli.cli.TemplateParser")
    @patch("open_lilli.cli.ContentProcessor")
    def test_generate_basic_success(self, mock_content_processor, mock_template_parser, mock_openai):
        """Test successful basic generation."""
        # Mock components
        mock_client = Mock()
        mock_openai.return_value = mock_client
        
        mock_processor = Mock()
        mock_processor.extract_text.return_value = "Test content"
        mock_content_processor.return_value = mock_processor
        
        mock_parser = Mock()
        mock_parser.get_template_info.return_value = {
            "total_layouts": 5,
            "available_layout_types": ["title", "content"],
            "layout_mapping": {"title": 0, "content": 1},
            "slide_dimensions": {"width_inches": 10, "height_inches": 7.5},
            "theme_colors": {"primary": "#000000"}
        }
        mock_parser.palette = {"primary": "#000000"}
        mock_template_parser.return_value = mock_parser
        
        # Mock all the pipeline components
        with patch("open_lilli.cli.OutlineGenerator") as mock_outline_gen, \
             patch("open_lilli.cli.SlidePlanner") as mock_slide_planner, \
             patch("open_lilli.cli.ContentGenerator") as mock_content_gen, \
             patch("open_lilli.cli.VisualGenerator") as mock_visual_gen, \
             patch("open_lilli.cli.Reviewer") as mock_reviewer, \
             patch("open_lilli.cli.SlideAssembler") as mock_assembler:
            
            # Mock outline generator
            mock_outline = Mock()
            mock_outline.slide_count = 3
            mock_outline.style_guidance = "Professional"
            mock_outline_gen.return_value.generate_outline.return_value = mock_outline
            
            # Mock slide planner
            mock_slides = [Mock(), Mock(), Mock()]
            mock_slide_planner.return_value.plan_slides.return_value = mock_slides
            mock_slide_planner.return_value.get_planning_summary.return_value = {}
            
            # Mock content generator
            mock_content_gen.return_value.generate_content.return_value = mock_slides
            mock_content_gen.return_value.get_content_statistics.return_value = {
                "total_bullets": 5,
                "total_words": 100
            }
            
            # Mock visual generator
            mock_visual_gen.return_value.generate_visuals.return_value = {}
            mock_visual_gen.return_value.get_visual_summary.return_value = {
                "total_charts": 1,
                "total_images": 2
            }
            
            # Mock reviewer
            mock_reviewer.return_value.review_presentation.return_value = []
            mock_reviewer.return_value.get_review_summary.return_value = {
                "overall_score": 8.5
            }
            
            # Mock assembler
            output_path = self.temp_dir / "output.pptx"
            mock_assembler.return_value.validate_slides_before_assembly.return_value = []
            mock_assembler.return_value.assemble.return_value = output_path
            mock_assembler.return_value.get_assembly_statistics.return_value = {
                "total_slides": 3
            }
            
            result = self.runner.invoke(cli, [
                "generate",
                "--template", str(self.test_template),
                "--input", str(self.test_content),
                "--output", str(output_path),
                "--slides", "5",
                "--lang", "en"
            ])
            
            # Should succeed
            assert result.exit_code == 0
            assert "Generation Complete" in result.output

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test_key"})
    @patch("open_lilli.cli.OpenAI")
    def test_generate_openai_error(self, mock_openai):
        """Test generate command with OpenAI initialization error."""
        mock_openai.side_effect = Exception("API Error")
        
        result = self.runner.invoke(cli, [
            "generate",
            "--template", str(self.test_template),
            "--input", str(self.test_content)
        ])
        
        assert result.exit_code == 1
        assert "Error initializing OpenAI" in result.output

    def test_generate_missing_template(self):
        """Test generate command with missing template file."""
        result = self.runner.invoke(cli, [
            "generate",
            "--template", "nonexistent.pptx",
            "--input", str(self.test_content)
        ])
        
        assert result.exit_code == 2  # Click file not found error

    def test_generate_missing_input(self):
        """Test generate command with missing input file."""
        result = self.runner.invoke(cli, [
            "generate",
            "--template", str(self.test_template),
            "--input", "nonexistent.txt"
        ])
        
        assert result.exit_code == 2  # Click file not found error

    def test_generate_with_options(self):
        """Test generate command with various options."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test_key"}):
            result = self.runner.invoke(cli, [
                "generate", "--help"
            ])
            
            assert "--template" in result.output
            assert "--lang" in result.output
            assert "--slides" in result.output
            assert "--tone" in result.output
            assert "--complexity" in result.output
            assert "--no-images" in result.output
            assert "--no-charts" in result.output
            assert "--review" in result.output

    def test_analyze_template_success(self):
        """Test template analysis command."""
        with patch("open_lilli.cli.TemplateParser") as mock_parser:
            mock_template_parser = Mock()
            mock_template_parser.get_template_info.return_value = {
                "template_path": str(self.test_template),
                "total_layouts": 3,
                "available_layout_types": ["title", "content", "blank"],
                "layout_mapping": {"title": 0, "content": 1, "blank": 2},
                "slide_dimensions": {"width_inches": 10.0, "height_inches": 7.5},
                "theme_colors": {"primary": "#000000", "secondary": "#FFFFFF"}
            }
            mock_template_parser.analyze_layout_placeholders.return_value = {
                "total_placeholders": 2,
                "has_title": True,
                "has_content": True,
                "has_image": False,
                "has_chart": False
            }
            mock_parser.return_value = mock_template_parser
            
            result = self.runner.invoke(cli, [
                "analyze-template", str(self.test_template)
            ])
            
            assert result.exit_code == 0
            assert "Analyzing template" in result.output
            assert "Available Layouts" in result.output
            assert "Theme Colors" in result.output

    def test_analyze_template_error(self):
        """Test template analysis with error."""
        with patch("open_lilli.cli.TemplateParser") as mock_parser:
            mock_parser.side_effect = Exception("Template error")
            
            result = self.runner.invoke(cli, [
                "analyze-template", str(self.test_template)
            ])
            
            assert result.exit_code == 1
            assert "Error analyzing template" in result.output

    def test_review_command(self):
        """Test review command (placeholder implementation)."""
        result = self.runner.invoke(cli, [
            "review", str(self.test_template)
        ])
        
        assert result.exit_code == 0
        assert "Direct .pptx review not yet implemented" in result.output

    def test_setup_command(self):
        """Test setup command."""
        with self.runner.isolated_filesystem():
            result = self.runner.invoke(cli, ["setup"])
            
            assert result.exit_code == 0
            assert "Open Lilli Setup" in result.output
            assert "Created .env file" in result.output or "Found existing .env file" in result.output

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test_key"})
    @patch("open_lilli.cli.OpenAI")
    def test_setup_with_api_test(self, mock_openai):
        """Test setup command with API test."""
        mock_client = Mock()
        mock_response = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        with self.runner.isolated_filesystem():
            result = self.runner.invoke(cli, ["setup"])
            
            assert result.exit_code == 0
            assert "OpenAI API connection successful" in result.output

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test_key"})
    @patch("open_lilli.cli.OpenAI")
    def test_setup_api_test_failure(self, mock_openai):
        """Test setup command with API test failure."""
        mock_openai.side_effect = Exception("API Error")
        
        with self.runner.isolated_filesystem():
            result = self.runner.invoke(cli, ["setup"])
            
            assert result.exit_code == 0  # Setup continues even if API test fails
            assert "OpenAI API test failed" in result.output

    def test_generate_verbose_option(self):
        """Test generate command with verbose option."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test_key"}):
            with patch("open_lilli.cli.OpenAI"):
                # This will fail early, but we can test that verbose flag is recognized
                result = self.runner.invoke(cli, [
                    "generate",
                    "--template", str(self.test_template),
                    "--input", str(self.test_content),
                    "--verbose"
                ])
                
                # The command should recognize the verbose flag
                # (actual behavior depends on mocked components)
                assert "--verbose" not in result.output  # Flag consumed, not shown as unknown

    def test_generate_no_review_option(self):
        """Test generate command with review disabled."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test_key"}):
            result = self.runner.invoke(cli, [
                "generate", "--help"
            ])
            
            assert "--review/--no-review" in result.output

    def test_main_function(self):
        """Test main function entry point."""
        from open_lilli.cli import main
        
        # Test that main function exists and is callable
        assert callable(main)

    def test_tone_and_complexity_choices(self):
        """Test that tone and complexity have proper choices."""
        result = self.runner.invoke(cli, ["generate", "--help"])
        
        assert result.exit_code == 0
        # Check tone choices
        assert "professional" in result.output
        assert "casual" in result.output
        assert "formal" in result.output
        assert "friendly" in result.output
        
        # Check complexity choices
        assert "basic" in result.output
        assert "intermediate" in result.output
        assert "advanced" in result.output