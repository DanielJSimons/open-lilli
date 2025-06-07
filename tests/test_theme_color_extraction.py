"""Tests for theme color extraction functionality."""

import tempfile
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import Mock, patch, mock_open

import pytest

from open_lilli.template_parser import TemplateParser


class TestThemeColorExtraction:
    """Tests for theme color extraction from PowerPoint templates."""

    def create_mock_theme_xml(self) -> str:
        """Create a mock theme1.xml with standard Office theme colors."""
        return '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<a:theme xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" name="Office Theme">
    <a:themeElements>
        <a:clrScheme name="Office">
            <a:dk1>
                <a:sysClr val="windowText" lastClr="000000"/>
            </a:dk1>
            <a:lt1>
                <a:sysClr val="window" lastClr="FFFFFF"/>
            </a:lt1>
            <a:dk2>
                <a:srgbClr val="44546A"/>
            </a:dk2>
            <a:lt2>
                <a:srgbClr val="E7E6E6"/>
            </a:lt2>
            <a:accent1>
                <a:srgbClr val="5B9BD5"/>
            </a:accent1>
            <a:accent2>
                <a:srgbClr val="70AD47"/>
            </a:accent2>
            <a:accent3>
                <a:srgbClr val="A5A5A5"/>
            </a:accent3>
            <a:accent4>
                <a:srgbClr val="FFC000"/>
            </a:accent4>
            <a:accent5>
                <a:srgbClr val="4472C4"/>
            </a:accent5>
            <a:accent6>
                <a:srgbClr val="C5504B"/>
            </a:accent6>
        </a:clrScheme>
    </a:themeElements>
</a:theme>'''

    def create_mock_theme_xml_with_preset_colors(self) -> str:
        """Create a mock theme1.xml with preset colors."""
        return '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<a:theme xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" name="Custom Theme">
    <a:themeElements>
        <a:clrScheme name="Custom">
            <a:dk1>
                <a:prstClr val="black"/>
            </a:dk1>
            <a:lt1>
                <a:prstClr val="white"/>
            </a:lt1>
            <a:accent1>
                <a:prstClr val="darkBlue"/>
            </a:accent1>
            <a:accent2>
                <a:prstClr val="darkGreen"/>
            </a:accent2>
            <a:accent3>
                <a:prstClr val="darkRed"/>
            </a:accent3>
            <a:accent4>
                <a:prstClr val="yellow"/>
            </a:accent4>
            <a:accent5>
                <a:prstClr val="blue"/>
            </a:accent5>
            <a:accent6>
                <a:prstClr val="red"/>
            </a:accent6>
        </a:clrScheme>
    </a:themeElements>
</a:theme>'''

    def create_mock_pptx_with_theme(self, theme_xml: str) -> str:
        """Create a mock PPTX file with theme XML."""
        # Create a temporary zip file that mimics a PPTX structure
        temp_file = tempfile.NamedTemporaryFile(suffix='.pptx', delete=False)
        temp_path = temp_file.name
        temp_file.close()
        
        with zipfile.ZipFile(temp_path, 'w') as zip_file:
            # Add minimal PPTX structure
            zip_file.writestr('[Content_Types].xml', '<?xml version="1.0"?><Types/>')
            zip_file.writestr('_rels/.rels', '<?xml version="1.0"?><Relationships/>')
            zip_file.writestr('ppt/presentation.xml', '<?xml version="1.0"?><presentation/>')
            zip_file.writestr('ppt/theme/theme1.xml', theme_xml)
        
        return temp_path

    @patch('open_lilli.template_parser.Presentation')
    def test_get_theme_colors_standard_theme(self, mock_presentation):
        """Test extracting colors from standard Office theme."""
        # Setup mock
        mock_prs = Mock()
        mock_prs.slide_layouts = []
        mock_presentation.return_value = mock_prs
        
        # Create mock PPTX with theme
        theme_xml = self.create_mock_theme_xml()
        template_path = self.create_mock_pptx_with_theme(theme_xml)
        
        try:
            parser = TemplateParser(template_path)
            colors = parser.get_theme_colors()
            
            # Verify expected colors are extracted
            assert colors['dk1'] == '#000000'  # Black from sysClr
            assert colors['lt1'] == '#FFFFFF'  # White from sysClr
            assert colors['acc1'] == '#5B9BD5'  # Blue from srgbClr
            assert colors['acc2'] == '#70AD47'  # Green from srgbClr
            assert colors['acc3'] == '#A5A5A5'  # Gray from srgbClr
            assert colors['acc4'] == '#FFC000'  # Orange from srgbClr
            assert colors['acc5'] == '#4472C4'  # Dark blue from srgbClr
            assert colors['acc6'] == '#C5504B'  # Red from srgbClr
            
        finally:
            Path(template_path).unlink()

    @patch('open_lilli.template_parser.Presentation')
    def test_get_theme_colors_preset_colors(self, mock_presentation):
        """Test extracting preset colors."""
        # Setup mock
        mock_prs = Mock()
        mock_prs.slide_layouts = []
        mock_presentation.return_value = mock_prs
        
        # Create mock PPTX with preset colors
        theme_xml = self.create_mock_theme_xml_with_preset_colors()
        template_path = self.create_mock_pptx_with_theme(theme_xml)
        
        try:
            parser = TemplateParser(template_path)
            colors = parser.get_theme_colors()
            
            # Verify preset colors are mapped correctly
            assert colors['dk1'] == '#000000'  # black
            assert colors['lt1'] == '#FFFFFF'  # white
            assert colors['acc1'] == '#000080'  # darkBlue
            assert colors['acc2'] == '#008000'  # darkGreen
            assert colors['acc3'] == '#800000'  # darkRed
            assert colors['acc4'] == '#FFFF00'  # yellow
            assert colors['acc5'] == '#0000FF'  # blue
            assert colors['acc6'] == '#FF0000'  # red
            
        finally:
            Path(template_path).unlink()

    @patch('open_lilli.template_parser.Presentation')
    def test_get_theme_colors_missing_file(self, mock_presentation):
        """Test behavior when theme file is missing."""
        # Setup mock
        mock_prs = Mock()
        mock_prs.slide_layouts = []
        mock_presentation.return_value = mock_prs
        
        # Create mock PPTX without theme file
        temp_file = tempfile.NamedTemporaryFile(suffix='.pptx', delete=False)
        temp_path = temp_file.name
        temp_file.close()
        
        with zipfile.ZipFile(temp_path, 'w') as zip_file:
            # Add minimal PPTX structure without theme
            zip_file.writestr('[Content_Types].xml', '<?xml version="1.0"?><Types/>')
            zip_file.writestr('_rels/.rels', '<?xml version="1.0"?><Relationships/>')
            zip_file.writestr('ppt/presentation.xml', '<?xml version="1.0"?><presentation/>')
        
        try:
            parser = TemplateParser(temp_path)
            colors = parser.get_theme_colors()
            
            # Should return empty dict when theme file is missing
            assert colors == {}
            
        finally:
            Path(temp_path).unlink()

    @patch('open_lilli.template_parser.Presentation')
    def test_get_theme_colors_invalid_xml(self, mock_presentation):
        """Test behavior with invalid XML."""
        # Setup mock
        mock_prs = Mock()
        mock_prs.slide_layouts = []
        mock_presentation.return_value = mock_prs
        
        # Create mock PPTX with invalid theme XML
        invalid_xml = "This is not valid XML"
        template_path = self.create_mock_pptx_with_theme(invalid_xml)
        
        try:
            parser = TemplateParser(template_path)
            colors = parser.get_theme_colors()
            
            # Should return empty dict when XML is invalid
            assert colors == {}
            
        finally:
            Path(template_path).unlink()

    def test_extract_color_value_srgb(self):
        """Test extracting sRGB color values."""
        template_path = "dummy.pptx"  # Won't be used in this test
        
        with patch('open_lilli.template_parser.Presentation'):
            parser = TemplateParser.__new__(TemplateParser)  # Create without __init__
            
            # Create test XML element
            xml_str = '<a:accent1 xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"><a:srgbClr val="5B9BD5"/></a:accent1>'
            elem = ET.fromstring(xml_str)
            
            namespaces = {'a': 'http://schemas.openxmlformats.org/drawingml/2006/main'}
            
            result = parser._extract_color_value(elem, namespaces)
            assert result == '#5B9BD5'

    def test_extract_color_value_sysclr(self):
        """Test extracting system color values."""
        template_path = "dummy.pptx"  # Won't be used in this test
        
        with patch('open_lilli.template_parser.Presentation'):
            parser = TemplateParser.__new__(TemplateParser)  # Create without __init__
            
            # Create test XML element
            xml_str = '<a:dk1 xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"><a:sysClr val="windowText" lastClr="000000"/></a:dk1>'
            elem = ET.fromstring(xml_str)
            
            namespaces = {'a': 'http://schemas.openxmlformats.org/drawingml/2006/main'}
            
            result = parser._extract_color_value(elem, namespaces)
            assert result == '#000000'

    def test_extract_color_value_preset(self):
        """Test extracting preset color values."""
        template_path = "dummy.pptx"  # Won't be used in this test
        
        with patch('open_lilli.template_parser.Presentation'):
            parser = TemplateParser.__new__(TemplateParser)  # Create without __init__
            
            # Create test XML element
            xml_str = '<a:accent1 xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"><a:prstClr val="darkBlue"/></a:accent1>'
            elem = ET.fromstring(xml_str)
            
            namespaces = {'a': 'http://schemas.openxmlformats.org/drawingml/2006/main'}
            
            result = parser._extract_color_value(elem, namespaces)
            assert result == '#000080'

    def test_extract_color_value_unknown_preset(self):
        """Test extracting unknown preset color values."""
        template_path = "dummy.pptx"  # Won't be used in this test
        
        with patch('open_lilli.template_parser.Presentation'):
            parser = TemplateParser.__new__(TemplateParser)  # Create without __init__
            
            # Create test XML element with unknown preset
            xml_str = '<a:accent1 xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"><a:prstClr val="unknownColor"/></a:accent1>'
            elem = ET.fromstring(xml_str)
            
            namespaces = {'a': 'http://schemas.openxmlformats.org/drawingml/2006/main'}
            
            result = parser._extract_color_value(elem, namespaces)
            assert result == '#000000'  # Should default to black

    def test_extract_color_value_no_color_info(self):
        """Test extracting from element with no color information."""
        template_path = "dummy.pptx"  # Won't be used in this test
        
        with patch('open_lilli.template_parser.Presentation'):
            parser = TemplateParser.__new__(TemplateParser)  # Create without __init__
            
            # Create test XML element without color info
            xml_str = '<a:accent1 xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"></a:accent1>'
            elem = ET.fromstring(xml_str)
            
            namespaces = {'a': 'http://schemas.openxmlformats.org/drawingml/2006/main'}
            
            result = parser._extract_color_value(elem, namespaces)
            assert result is None

    @patch('open_lilli.template_parser.Presentation')
    def test_extract_theme_colors_integration(self, mock_presentation):
        """Test the integration of theme color extraction with template parser."""
        # Setup mock
        mock_prs = Mock()
        mock_prs.slide_layouts = []
        mock_presentation.return_value = mock_prs
        
        # Create mock PPTX with theme
        theme_xml = self.create_mock_theme_xml()
        template_path = self.create_mock_pptx_with_theme(theme_xml)
        
        try:
            parser = TemplateParser(template_path)
            
            # Verify the palette is populated with extracted colors
            assert 'dk1' in parser.palette
            assert 'lt1' in parser.palette
            assert 'acc1' in parser.palette
            assert 'acc2' in parser.palette
            assert 'acc3' in parser.palette
            assert 'acc4' in parser.palette
            assert 'acc5' in parser.palette
            assert 'acc6' in parser.palette
            
            # Verify get_theme_color method works
            assert parser.get_theme_color('dk1') == '#000000'
            assert parser.get_theme_color('lt1') == '#FFFFFF'
            assert parser.get_theme_color('acc1') == '#5B9BD5'
            
            # Verify template info includes theme colors
            info = parser.get_template_info()
            assert 'theme_colors' in info
            assert info['theme_colors']['dk1'] == '#000000'
            
        finally:
            Path(template_path).unlink()

    @patch('open_lilli.template_parser.Presentation')
    def test_fallback_to_default_colors(self, mock_presentation):
        """Test fallback to default colors when extraction fails."""
        # Setup mock
        mock_prs = Mock()
        mock_prs.slide_layouts = []
        mock_presentation.return_value = mock_prs
        
        # Create invalid template path
        template_path = "/nonexistent/template.pptx"
        
        with patch.object(TemplateParser, 'get_theme_colors', side_effect=Exception("Failed")):
            with patch('pathlib.Path.exists', return_value=True):
                parser = TemplateParser(template_path)
                
                # Should have default colors
                assert 'dk1' in parser.palette
                assert 'lt1' in parser.palette
                assert 'acc1' in parser.palette
                assert parser.palette['dk1'] == '#000000'
                assert parser.palette['lt1'] == '#FFFFFF'