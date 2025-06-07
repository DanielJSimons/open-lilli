"""Tests for T-39: Content Style Context Integration."""

import pytest
from unittest.mock import Mock, patch

from open_lilli.content_generator import ContentGenerator
from open_lilli.template_parser import TemplateParser
from open_lilli.models import SlidePlan, GenerationConfig, FontInfo, TemplateStyle


class TestContentStyleContext:
    """Test content generator with template style context integration."""
    
    def test_template_style_context_integration(self):
        """Test that template style context is included in content generation."""
        
        # Mock OpenAI client
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = '{"title": "Enhanced Title", "bullets": ["Enhanced bullet"]}'
        mock_client.chat.completions.create.return_value = mock_response
        
        # Create mock template parser with style information
        mock_template_parser = Mock(spec=TemplateParser)
        mock_template_parser.palette = {
            'dk1': '#000000',
            'lt1': '#FFFFFF', 
            'acc1': '#5B9BD5',
            'acc2': '#70AD47'
        }
        
        # Mock template style
        mock_font = FontInfo(name="Calibri", size=18, weight="normal", color="#000000")
        mock_template_style = TemplateStyle(
            master_font=mock_font,
            theme_fonts={"major": "Calibri", "minor": "Calibri"}
        )
        mock_template_parser.template_style = mock_template_style
        
        # Create content generator with template parser
        generator = ContentGenerator(
            client=mock_client,
            template_parser=mock_template_parser
        )
        
        # Create test slide
        slide = SlidePlan(
            index=1,
            slide_type="content",
            title="Test Title",
            bullets=["Test bullet"]
        )
        
        # Generate content
        config = GenerationConfig()
        enhanced_slides = generator.generate_content([slide], config, "Corporate style")
        
        # Verify OpenAI was called
        assert mock_client.chat.completions.create.called
        
        # Get the actual prompt that was sent
        call_args = mock_client.chat.completions.create.call_args
        messages = call_args[1]['messages']
        user_message = messages[1]['content']
        
        # Verify template style context is included
        assert "Template style requirements:" in user_message
        assert "Primary colors:" in user_message
        assert "dk1: #000000" in user_message
        assert "Accent colors:" in user_message
        assert "acc1: #5B9BD5" in user_message
        assert "Primary font: Calibri (18pt)" in user_message
        assert "Heading font: Calibri" in user_message
        assert "Body font: Calibri" in user_message
        assert "Brand voice:" in user_message
        assert "corporate brand voice" in user_message
    
    def test_template_style_context_without_parser(self):
        """Test content generation works without template parser."""
        
        # Mock OpenAI client
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = '{"title": "Enhanced Title", "bullets": ["Enhanced bullet"]}'
        mock_client.chat.completions.create.return_value = mock_response
        
        # Create content generator without template parser
        generator = ContentGenerator(client=mock_client)
        
        # Create test slide
        slide = SlidePlan(
            index=1,
            slide_type="content", 
            title="Test Title",
            bullets=["Test bullet"]
        )
        
        # Generate content
        config = GenerationConfig()
        enhanced_slides = generator.generate_content([slide], config)
        
        # Verify it works without template context
        assert len(enhanced_slides) == 1
        assert enhanced_slides[0].title == "Enhanced Title"
        
        # Get the prompt and verify no template context is included
        call_args = mock_client.chat.completions.create.call_args
        messages = call_args[1]['messages']
        user_message = messages[1]['content']
        
        assert "Template style requirements:" not in user_message
        assert "Primary colors:" not in user_message
        assert "Brand voice:" not in user_message
    
    def test_build_template_style_context_empty_parser(self):
        """Test building style context with empty template parser."""
        
        mock_client = Mock()
        generator = ContentGenerator(client=mock_client)
        
        # Should return empty string when no template parser
        context = generator._build_template_style_context()
        assert context == ""
    
    def test_build_template_style_context_partial_data(self):
        """Test building style context with partial template data."""
        
        mock_client = Mock()
        mock_template_parser = Mock(spec=TemplateParser)
        
        # Only provide palette, no template style
        mock_template_parser.palette = {
            'dk1': '#1F497D',
            'acc1': '#4F81BD',
            'acc2': '#9CBB58'
        }
        # No template_style attribute
        
        generator = ContentGenerator(
            client=mock_client,
            template_parser=mock_template_parser
        )
        
        context = generator._build_template_style_context()
        
        # Should include color information but not font information
        assert "Primary colors: dk1: #1F497D" in context
        assert "Accent colors: acc1: #4F81BD, acc2: #9CBB58" in context
        assert "Brand voice:" in context
        assert "Primary font:" not in context
        assert "Heading font:" not in context
    
    def test_build_template_style_context_colors_only(self):
        """Test building style context with only color information."""
        
        mock_client = Mock()
        mock_template_parser = Mock(spec=TemplateParser)
        mock_template_parser.palette = {
            'lt1': '#FFFFFF',
            'acc3': '#8064A2'
        }
        
        generator = ContentGenerator(
            client=mock_client,
            template_parser=mock_template_parser
        )
        
        context = generator._build_template_style_context()
        
        assert "Primary colors: lt1: #FFFFFF" in context
        assert "Accent colors: acc3: #8064A2" in context
        assert "Brand voice:" in context
    
    def test_regression_llm_stub_includes_tone_guidelines(self):
        """Regression test: verify LLM stub output includes requested tone guidelines."""
        
        # This test simulates the acceptance criteria for T-39
        mock_client = Mock()
        mock_template_parser = Mock(spec=TemplateParser)
        mock_template_parser.palette = {'acc1': '#5B9BD5', 'acc2': '#70AD47'}
        mock_template_parser.template_style = TemplateStyle(
            master_font=FontInfo(name="Arial", size=12),
            theme_fonts={"major": "Arial", "minor": "Arial"}
        )
        
        generator = ContentGenerator(
            client=mock_client,
            template_parser=mock_template_parser
        )
        
        # Build the style context
        style_context = generator._build_template_style_context()
        
        # Verify the output includes requested tone guidelines
        assert "Brand voice:" in style_context
        assert "corporate brand voice" in style_context.lower()
        assert "template's visual identity" in style_context
        assert "professional styling" in style_context
        
        # Verify color and font information is included  
        assert "acc1: #5B9BD5" in style_context
        assert "acc2: #70AD47" in style_context
        assert "Primary font: Arial (12pt)" in style_context
        assert "Heading font: Arial" in style_context
        assert "Body font: Arial" in style_context


if __name__ == "__main__":
    pytest.main([__file__])