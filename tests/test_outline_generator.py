"""Tests for outline generator."""

import json
from unittest.mock import Mock, patch

import pytest
from openai import OpenAI

from open_lilli.models import GenerationConfig, Outline
from open_lilli.outline_generator import OutlineGenerator


class TestOutlineGenerator:
    """Tests for OutlineGenerator class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_client = Mock(spec=OpenAI)
        self.generator = OutlineGenerator(self.mock_client)

    def test_init(self):
        """Test OutlineGenerator initialization."""
        assert self.generator.client == self.mock_client
        assert self.generator.model == "gpt-4"
        assert self.generator.temperature == 0.3
        assert self.generator.max_retries == 3

    def test_generate_outline_empty_text(self):
        """Test handling of empty input text."""
        with pytest.raises(ValueError, match="Input text cannot be empty"):
            self.generator.generate_outline("")

    def test_generate_outline_success(self):
        """Test successful outline generation."""
        # Mock response data
        mock_response_data = {
            "language": "en",
            "title": "Test Presentation",
            "subtitle": "A test presentation",
            "slides": [
                {
                    "index": 0,
                    "slide_type": "title",
                    "title": "Test Title",
                    "bullets": [],
                    "image_query": None,
                    "chart_data": None,
                    "speaker_notes": "Introduction"
                },
                {
                    "index": 1,
                    "slide_type": "content",
                    "title": "Content Slide",
                    "bullets": ["Point 1", "Point 2"],
                    "image_query": "business meeting",
                    "chart_data": None,
                    "speaker_notes": "Main content"
                }
            ],
            "style_guidance": "Professional",
            "target_audience": "Business executives"
        }

        # Mock OpenAI response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = json.dumps(mock_response_data)
        
        self.mock_client.chat.completions.create.return_value = mock_response

        # Test the generation
        result = self.generator.generate_outline("Test content for presentation")

        # Verify the result
        assert isinstance(result, Outline)
        assert result.title == "Test Presentation"
        assert result.slide_count == 2
        assert result.language == "en"

        # Verify OpenAI was called correctly
        self.mock_client.chat.completions.create.assert_called_once()
        call_args = self.mock_client.chat.completions.create.call_args
        assert call_args[1]["model"] == "gpt-4"
        assert call_args[1]["temperature"] == 0.3
        assert call_args[1]["response_format"] == {"type": "json_object"}

    def test_build_outline_prompt(self):
        """Test prompt building with different configurations."""
        config = GenerationConfig(max_slides=10, tone="casual")
        
        prompt = self.generator._build_outline_prompt(
            "Test content", config, "Custom Title", "es"
        )
        
        assert "Maximum 10 slides" in prompt
        assert "Tone: casual" in prompt
        assert "Generate the presentation in Spanish" in prompt
        assert "Test content" in prompt

    def test_validate_outline_structure_valid(self):
        """Test validation of valid outline structure."""
        valid_data = {
            "language": "en",
            "title": "Test",
            "slides": [
                {
                    "index": 0,
                    "slide_type": "title",
                    "title": "Title"
                }
            ]
        }
        
        # Should not raise any exception
        self.generator._validate_outline_structure(valid_data)

    def test_validate_outline_structure_missing_fields(self):
        """Test validation with missing required fields."""
        # Missing 'title'
        invalid_data = {
            "language": "en",
            "slides": []
        }
        
        with pytest.raises(ValueError, match="Missing required field: title"):
            self.generator._validate_outline_structure(invalid_data)

    def test_validate_outline_structure_invalid_slides(self):
        """Test validation with invalid slides structure."""
        # slides is not a list
        invalid_data = {
            "language": "en",
            "title": "Test",
            "slides": "not a list"
        }
        
        with pytest.raises(ValueError, match="Slides must be a list"):
            self.generator._validate_outline_structure(invalid_data)

    def test_validate_outline_structure_empty_slides(self):
        """Test validation with empty slides list."""
        invalid_data = {
            "language": "en",
            "title": "Test",
            "slides": []
        }
        
        with pytest.raises(ValueError, match="At least one slide is required"):
            self.generator._validate_outline_structure(invalid_data)

    def test_validate_outline_structure_invalid_slide(self):
        """Test validation with invalid slide structure."""
        invalid_data = {
            "language": "en",
            "title": "Test",
            "slides": [
                {
                    "index": 0,
                    # Missing 'slide_type' and 'title'
                }
            ]
        }
        
        with pytest.raises(ValueError, match="Slide 0 missing required field"):
            self.generator._validate_outline_structure(invalid_data)

    @patch('time.sleep')  # Mock sleep to speed up tests
    def test_call_openai_with_retries_rate_limit(self, mock_sleep):
        """Test retry logic for rate limit errors."""
        from openai import RateLimitError
        
        # Mock rate limit error on first call, success on second
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = '{"test": "data"}'
        
        self.mock_client.chat.completions.create.side_effect = [
            RateLimitError("Rate limited", response=Mock(), body=Mock()),
            mock_response
        ]
        
        result = self.generator._call_openai_with_retries("test prompt")
        
        assert result == {"test": "data"}
        assert self.mock_client.chat.completions.create.call_count == 2
        mock_sleep.assert_called_once()

    def test_call_openai_with_retries_max_retries_exceeded(self):
        """Test behavior when max retries are exceeded."""
        from openai import RateLimitError
        
        # Always return rate limit error
        self.mock_client.chat.completions.create.side_effect = RateLimitError(
            "Rate limited", response=Mock(), body=Mock()
        )
        
        with pytest.raises(RateLimitError):
            self.generator._call_openai_with_retries("test prompt")
        
        assert self.mock_client.chat.completions.create.call_count == 3

    def test_call_openai_with_retries_json_parse_error(self):
        """Test handling of invalid JSON response."""
        # Mock response with invalid JSON
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "invalid json {"
        
        self.mock_client.chat.completions.create.return_value = mock_response
        
        with pytest.raises(ValueError, match="Invalid JSON response"):
            self.generator._call_openai_with_retries("test prompt")

    def test_refine_outline(self):
        """Test outline refinement functionality."""
        # Create initial outline
        initial_outline = Outline(
            title="Initial Title",
            slides=[
                {
                    "index": 0,
                    "slide_type": "title",
                    "title": "Initial",
                    "bullets": [],
                    "image_query": None,
                    "chart_data": None
                }
            ]
        )

        # Mock refined response
        refined_data = {
            "language": "en",
            "title": "Refined Title",
            "slides": [
                {
                    "index": 0,
                    "slide_type": "title",
                    "title": "Refined",
                    "bullets": [],
                    "image_query": None,
                    "chart_data": None
                }
            ]
        }

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = json.dumps(refined_data)
        
        self.mock_client.chat.completions.create.return_value = mock_response

        # Test refinement
        result = self.generator.refine_outline(initial_outline, "Make it better")

        assert isinstance(result, Outline)
        assert result.title == "Refined Title"
        
        # Verify the feedback was included in the prompt
        call_args = self.mock_client.chat.completions.create.call_args
        prompt = call_args[1]["messages"][1]["content"]
        assert "Make it better" in prompt
        assert "Initial Title" in prompt

    @pytest.mark.asyncio
    async def test_generate_outline_async(self):
        """Test asynchronous outline generation."""
        from unittest.mock import AsyncMock

        async_client = AsyncMock()
        generator = OutlineGenerator(async_client)

        mock_response_data = {
            "language": "en",
            "title": "Async Presentation",
            "slides": [{"index": 0, "slide_type": "title", "title": "Hello"}],
        }
        mock_resp = Mock()
        mock_resp.choices = [Mock()]
        mock_resp.choices[0].message.content = json.dumps(mock_response_data)
        async_client.chat.completions.create.return_value = mock_resp

        outline = await generator.generate_outline_async("test")

        assert outline.title == "Async Presentation"
        async_client.chat.completions.create.assert_called_once()
