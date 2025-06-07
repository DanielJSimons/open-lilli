"""Tests for content generator."""

import json
from unittest.mock import Mock

import pytest
from openai import OpenAI

from open_lilli.content_generator import ContentGenerator
from open_lilli.models import GenerationConfig, SlidePlan


class TestContentGenerator:
    """Tests for ContentGenerator class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_client = Mock(spec=OpenAI)
        self.generator = ContentGenerator(self.mock_client)

    def create_test_slide(self) -> SlidePlan:
        """Create a test slide."""
        return SlidePlan(
            index=1,
            slide_type="content",
            title="Original Title",
            bullets=["Original bullet 1", "Original bullet 2"],
            speaker_notes="Original notes"
        )

    def test_init(self):
        """Test ContentGenerator initialization."""
        assert self.generator.client == self.mock_client
        assert self.generator.model == "gpt-4"
        assert self.generator.temperature == 0.3

    def test_generate_content_success(self):
        """Test successful content generation."""
        slides = [self.create_test_slide()]
        
        # Mock API response
        mock_response_data = {
            "title": "Enhanced Title",
            "bullets": ["Enhanced bullet 1", "Enhanced bullet 2", "Enhanced bullet 3"],
            "speaker_notes": "Enhanced speaker notes"
        }
        
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = json.dumps(mock_response_data)
        
        self.mock_client.chat.completions.create.return_value = mock_response
        
        # Test generation
        result = self.generator.generate_content(slides)
        
        assert len(result) == 1
        enhanced_slide = result[0]
        assert enhanced_slide.title == "Enhanced Title"
        assert len(enhanced_slide.bullets) == 3
        assert enhanced_slide.speaker_notes == "Enhanced speaker notes"

    def test_generate_content_skip_title_slide(self):
        """Test that title slides are skipped when appropriate."""
        title_slide = SlidePlan(
            index=0,
            slide_type="title",
            title="Presentation Title",
            bullets=[]
        )
        
        result = self.generator.generate_content([title_slide])
        
        # Should return unchanged title slide
        assert len(result) == 1
        assert result[0].title == "Presentation Title"
        
        # OpenAI should not have been called
        self.mock_client.chat.completions.create.assert_not_called()

    def test_generate_slide_content_api_failure(self):
        """Test handling of API failures."""
        slide = self.create_test_slide()
        
        # Mock API failure
        self.mock_client.chat.completions.create.side_effect = Exception("API Error")
        
        # Should return original slide on failure
        result = self.generator._generate_slide_content(
            slide, GenerationConfig(), None, "en"
        )
        
        assert result == slide

    def test_build_content_prompt(self):
        """Test content prompt building."""
        slide = self.create_test_slide()
        config = GenerationConfig(tone="casual", complexity_level="basic")
        
        prompt = self.generator._build_content_prompt(
            slide, config, "Be engaging", "es"
        )
        
        assert "Generate content in Spanish" in prompt
        assert "Tone: casual" in prompt
        assert "complexity_level: basic" in prompt
        assert "Be engaging" in prompt
        assert "Original Title" in prompt

    def test_apply_generated_content(self):
        """Test applying generated content to slide."""
        slide = self.create_test_slide()
        
        response_data = {
            "title": "New Title",
            "bullets": ["New bullet 1", "New bullet 2"],
            "speaker_notes": "New notes"
        }
        
        result = self.generator._apply_generated_content(slide, response_data)
        
        assert result.title == "New Title"
        assert result.bullets == ["New bullet 1", "New bullet 2"]
        assert result.speaker_notes == "New notes"
        assert result.index == slide.index  # Unchanged fields preserved

    def test_apply_generated_content_partial(self):
        """Test applying partial generated content."""
        slide = self.create_test_slide()
        
        # Only title provided
        response_data = {
            "title": "New Title"
        }
        
        result = self.generator._apply_generated_content(slide, response_data)
        
        assert result.title == "New Title"
        assert result.bullets == slide.bullets  # Unchanged
        assert result.speaker_notes == slide.speaker_notes  # Unchanged

    def test_apply_generated_content_empty_values(self):
        """Test handling of empty values in response."""
        slide = self.create_test_slide()
        
        response_data = {
            "title": "",  # Empty title
            "bullets": ["", "Valid bullet", ""],  # Mixed empty/valid
            "speaker_notes": "   "  # Whitespace only
        }
        
        result = self.generator._apply_generated_content(slide, response_data)
        
        assert result.title == slide.title  # Unchanged due to empty title
        assert result.bullets == ["Valid bullet"]  # Only valid bullet kept
        assert result.speaker_notes == slide.speaker_notes  # Unchanged due to whitespace

    def test_generate_speaker_notes(self):
        """Test speaker notes generation."""
        slide = self.create_test_slide()
        
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Generated speaker notes"
        
        self.mock_client.chat.completions.create.return_value = mock_response
        
        result = self.generator.generate_speaker_notes(
            slide, "Presentation context", "en"
        )
        
        assert result == "Generated speaker notes"
        
        # Verify API was called with correct parameters
        call_args = self.mock_client.chat.completions.create.call_args
        assert "Presentation context" in call_args[1]["messages"][1]["content"]

    def test_generate_speaker_notes_api_failure(self):
        """Test speaker notes generation with API failure."""
        slide = self.create_test_slide()
        
        self.mock_client.chat.completions.create.side_effect = Exception("API Error")
        
        result = self.generator.generate_speaker_notes(slide)
        
        # Should return fallback notes
        assert "Original Title" in result

    def test_refine_content(self):
        """Test content refinement."""
        slide = self.create_test_slide()
        
        mock_response_data = {
            "title": "Refined Title",
            "bullets": ["Refined bullet"],
            "speaker_notes": "Refined notes"
        }
        
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = json.dumps(mock_response_data)
        
        self.mock_client.chat.completions.create.return_value = mock_response
        
        result = self.generator.refine_content(
            slide, "Make it more concise", "en"
        )
        
        assert result.title == "Refined Title"
        assert result.bullets == ["Refined bullet"]
        
        # Verify feedback was included in prompt
        call_args = self.mock_client.chat.completions.create.call_args
        assert "Make it more concise" in call_args[1]["messages"][1]["content"]

    def test_batch_generate_content(self):
        """Test batch content generation."""
        slides = [self.create_test_slide() for _ in range(3)]
        
        mock_response_data = {
            "title": "Enhanced Title",
            "bullets": ["Enhanced bullet"],
            "speaker_notes": "Enhanced notes"
        }
        
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = json.dumps(mock_response_data)
        
        self.mock_client.chat.completions.create.return_value = mock_response
        
        result = self.generator.batch_generate_content(
            slides, batch_size=2
        )
        
        assert len(result) == 3
        
        # Should have made API calls for each slide
        assert self.mock_client.chat.completions.create.call_count == 3

    def test_get_content_statistics(self):
        """Test content statistics calculation."""
        slides = [
            SlidePlan(
                index=0,
                slide_type="title",
                title="Title Slide",
                bullets=[],
                speaker_notes="Notes"
            ),
            SlidePlan(
                index=1,
                slide_type="content",
                title="Content Slide",
                bullets=["Bullet one", "Bullet two"],
                speaker_notes=None
            )
        ]
        
        stats = self.generator.get_content_statistics(slides)
        
        assert stats["total_slides"] == 2
        assert stats["total_bullets"] == 2
        assert stats["avg_bullets_per_slide"] == 1.0
        assert stats["slides_with_speaker_notes"] == 1
        assert stats["slide_types"]["title"] == 1
        assert stats["slide_types"]["content"] == 1
        assert stats["total_words"] == 6  # "Title Slide" + "Content Slide" + "Bullet one" + "Bullet two"

    def test_get_content_statistics_empty(self):
        """Test content statistics with empty slide list."""
        stats = self.generator.get_content_statistics([])
        
        assert stats["total_slides"] == 0
        assert stats["total_bullets"] == 0
        assert stats["avg_bullets_per_slide"] == 0
        assert stats["avg_words_per_slide"] == 0

    def test_call_openai_with_retries_success(self):
        """Test successful OpenAI API call."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = '{"test": "data"}'
        
        self.mock_client.chat.completions.create.return_value = mock_response
        
        result = self.generator._call_openai_with_retries("test prompt")
        
        assert result == {"test": "data"}

    def test_call_openai_with_retries_json_error(self):
        """Test handling of invalid JSON response."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "invalid json"
        
        self.mock_client.chat.completions.create.return_value = mock_response
        
        with pytest.raises(ValueError, match="Invalid JSON response"):
            self.generator._call_openai_with_retries("test prompt")

    def test_call_openai_with_retries_empty_response(self):
        """Test handling of empty response."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = None
        
        self.mock_client.chat.completions.create.return_value = mock_response
        
        with pytest.raises(ValueError, match="Empty response from OpenAI"):
            self.generator._call_openai_with_retries("test prompt")