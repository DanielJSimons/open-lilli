"""Tests for template parser."""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from pptx import Presentation

from open_lilli.template_parser import TemplateParser


class TestTemplateParser:
    """Tests for TemplateParser class."""

    def create_test_template(self) -> str:
        """Create a minimal test template."""
        prs = Presentation()
        
        # Create a temporary file
        temp_file = tempfile.NamedTemporaryFile(suffix='.pptx', delete=False)
        temp_path = temp_file.name
        temp_file.close()
        
        # Save the presentation
        prs.save(temp_path)
        return temp_path

    def test_init_valid_template(self):
        """Test initialization with valid template."""
        template_path = self.create_test_template()
        
        try:
            parser = TemplateParser(template_path)
            
            assert parser.template_path == Path(template_path)
            assert parser.prs is not None
            assert isinstance(parser.layout_map, dict)
            assert isinstance(parser.palette, dict)
            
        finally:
            Path(template_path).unlink()

    def test_init_file_not_found(self):
        """Test initialization with non-existent file."""
        with pytest.raises(FileNotFoundError):
            TemplateParser("nonexistent.pptx")

    def test_init_invalid_extension(self):
        """Test initialization with invalid file extension."""
        with pytest.raises(ValueError, match="Template must be a .pptx file"):
            TemplateParser("template.txt")

    def test_init_invalid_file(self):
        """Test initialization with invalid PowerPoint file."""
        # Create a text file with .pptx extension
        with tempfile.NamedTemporaryFile(suffix='.pptx', mode='w', delete=False) as f:
            f.write("This is not a PowerPoint file")
            temp_path = f.name
        
        try:
            with pytest.raises(ValueError, match="Failed to load template"):
                TemplateParser(temp_path)
        finally:
            Path(temp_path).unlink()

    def test_classify_layout(self):
        """Test layout classification logic."""
        template_path = self.create_test_template()
        
        try:
            parser = TemplateParser(template_path)
            
            # Test with actual layouts from the template
            for i, layout in enumerate(parser.prs.slide_layouts):
                layout_type = parser._classify_layout(layout, i)
                assert isinstance(layout_type, str)
                assert len(layout_type) > 0
                
        finally:
            Path(template_path).unlink()

    def test_ensure_basic_layouts(self):
        """Test that basic layouts are ensured."""
        template_path = self.create_test_template()
        
        try:
            parser = TemplateParser(template_path)
            
            # These should always be present after _ensure_basic_layouts
            essential_layouts = ["title", "content", "section", "blank"]
            
            for layout_type in essential_layouts:
                assert layout_type in parser.layout_map
                assert isinstance(parser.layout_map[layout_type], int)
                
        finally:
            Path(template_path).unlink()

    def test_get_layout_valid(self):
        """Test getting a valid layout."""
        template_path = self.create_test_template()
        
        try:
            parser = TemplateParser(template_path)
            
            # Should be able to get any layout in the map
            for layout_type in parser.layout_map:
                layout = parser.get_layout(layout_type)
                assert layout is not None
                
        finally:
            Path(template_path).unlink()

    def test_get_layout_invalid(self):
        """Test getting an invalid layout."""
        template_path = self.create_test_template()
        
        try:
            parser = TemplateParser(template_path)
            
            with pytest.raises(ValueError, match="Layout type 'nonexistent' not found"):
                parser.get_layout("nonexistent")
                
        finally:
            Path(template_path).unlink()

    def test_get_layout_index(self):
        """Test getting layout index."""
        template_path = self.create_test_template()
        
        try:
            parser = TemplateParser(template_path)
            
            # Valid layout type
            for layout_type in parser.layout_map:
                index = parser.get_layout_index(layout_type)
                assert isinstance(index, int)
                assert index >= 0
            
            # Invalid layout type should return default
            index = parser.get_layout_index("nonexistent")
            assert isinstance(index, int)
            assert index >= 0
            
        finally:
            Path(template_path).unlink()

    def test_list_available_layouts(self):
        """Test listing available layouts."""
        template_path = self.create_test_template()
        
        try:
            parser = TemplateParser(template_path)
            
            layouts = parser.list_available_layouts()
            assert isinstance(layouts, list)
            assert len(layouts) > 0
            
            # Should match the layout_map keys
            assert set(layouts) == set(parser.layout_map.keys())
            
        finally:
            Path(template_path).unlink()

    def test_analyze_layout_placeholders(self):
        """Test placeholder analysis."""
        template_path = self.create_test_template()
        
        try:
            parser = TemplateParser(template_path)
            
            # Test with first available layout
            layout_type = list(parser.layout_map.keys())[0]
            analysis = parser.analyze_layout_placeholders(layout_type)
            
            assert isinstance(analysis, dict)
            assert "total_placeholders" in analysis
            assert "placeholder_details" in analysis
            assert "has_title" in analysis
            assert "has_content" in analysis
            assert "has_image" in analysis
            assert "has_chart" in analysis
            
            assert isinstance(analysis["total_placeholders"], int)
            assert isinstance(analysis["placeholder_details"], list)
            assert isinstance(analysis["has_title"], bool)
            
        finally:
            Path(template_path).unlink()

    def test_get_theme_color(self):
        """Test theme color retrieval."""
        template_path = self.create_test_template()
        
        try:
            parser = TemplateParser(template_path)
            
            # Should return a color for any name in palette
            for color_name in parser.palette:
                color = parser.get_theme_color(color_name)
                assert isinstance(color, str)
                assert color.startswith("#")
                assert len(color) == 7  # Hex color format
            
            # Should return default for unknown color
            color = parser.get_theme_color("unknown_color")
            assert color == "#000000"
            
        finally:
            Path(template_path).unlink()

    def test_get_slide_size(self):
        """Test slide size retrieval."""
        template_path = self.create_test_template()
        
        try:
            parser = TemplateParser(template_path)
            
            width, height = parser.get_slide_size()
            assert isinstance(width, int)
            assert isinstance(height, int)
            assert width > 0
            assert height > 0
            
        finally:
            Path(template_path).unlink()

    def test_get_template_info(self):
        """Test comprehensive template info."""
        template_path = self.create_test_template()
        
        try:
            parser = TemplateParser(template_path)
            
            info = parser.get_template_info()
            
            assert isinstance(info, dict)
            assert "template_path" in info
            assert "total_layouts" in info
            assert "available_layout_types" in info
            assert "layout_mapping" in info
            assert "slide_dimensions" in info
            assert "theme_colors" in info
            
            # Verify slide dimensions structure
            dimensions = info["slide_dimensions"]
            assert "width" in dimensions
            assert "height" in dimensions
            assert "width_inches" in dimensions
            assert "height_inches" in dimensions
            
        finally:
            Path(template_path).unlink()

    def test_placeholder_type_name(self):
        """Test placeholder type name conversion."""
        template_path = self.create_test_template()
        
        try:
            parser = TemplateParser(template_path)
            
            # Test known types
            assert parser._placeholder_type_name(1) == "TITLE"
            assert parser._placeholder_type_name(2) == "BODY"
            assert parser._placeholder_type_name(18) == "PICTURE"
            
            # Test unknown type
            assert parser._placeholder_type_name(999) == "TYPE_999"
            
        finally:
            Path(template_path).unlink()

    def test_get_layout_info(self):
        """Test layout info generation."""
        template_path = self.create_test_template()
        
        try:
            parser = TemplateParser(template_path)
            
            # Test with first layout
            layout = parser.prs.slide_layouts[0]
            info = parser._get_layout_info(layout)
            
            assert isinstance(info, str)
            assert info.startswith("[")
            assert info.endswith("]")
            
        finally:
            Path(template_path).unlink()