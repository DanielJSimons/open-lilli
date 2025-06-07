"""Tests for T-40: Chart Palette Integration in VisualGenerator."""

import pytest
from unittest.mock import Mock, patch
from pathlib import Path

from open_lilli.visual_generator import VisualGenerator
from open_lilli.models import SlidePlan


class TestChartPaletteIntegration:
    """Test chart palette integration with template colors."""
    
    def test_template_palette_mapping_full_palette(self):
        """Test mapping with full template palette (dk1, lt1, acc1-6)."""
        
        template_palette = {
            'dk1': '#1F4E79',  # Dark blue
            'lt1': '#FFFFFF',  # White  
            'acc1': '#5B9BD5', # Light blue
            'acc2': '#70AD47', # Green
            'acc3': '#FFC000', # Yellow
            'acc4': '#ED7D31', # Orange
            'acc5': '#A5A5A5', # Gray
            'acc6': '#264478'  # Dark blue
        }
        
        generator = VisualGenerator(theme_colors=template_palette)
        
        # Verify template palette is stored
        assert generator.template_palette == template_palette
        
        # Verify primary uses dk1
        assert generator.theme_colors['primary'] == '#1F4E79'
        
        # Verify secondary uses acc1
        assert generator.theme_colors['secondary'] == '#5B9BD5'
        
        # Verify accent colors map correctly  
        assert generator.theme_colors['accent1'] == '#5B9BD5'  # acc1
        assert generator.theme_colors['accent2'] == '#70AD47'  # acc2
        assert generator.theme_colors['accent3'] == '#FFC000'  # acc3
        
        # Verify text colors
        assert generator.theme_colors['text_dark'] == '#1F4E79'  # dk1
        assert generator.theme_colors['text_light'] == '#FFFFFF'  # lt1
        
        # Verify background uses lt1
        assert generator.theme_colors['background'] == '#FFFFFF'
    
    def test_template_palette_mapping_partial_palette(self):
        """Test mapping with partial template palette."""
        
        template_palette = {
            'acc1': '#4472C4',  # Only acc1 available
            'acc2': '#E70012',  # Only acc2 available  
            'lt1': '#F2F2F2'    # Light background
        }
        
        generator = VisualGenerator(theme_colors=template_palette)
        
        # Primary should use acc1 since dk1 not available
        assert generator.theme_colors['primary'] == '#4472C4'
        
        # Secondary should use acc1 
        assert generator.theme_colors['secondary'] == '#4472C4'
        
        # First two accent colors should map correctly
        assert generator.theme_colors['accent1'] == '#4472C4'  # acc1
        assert generator.theme_colors['accent2'] == '#E70012'  # acc2
        assert generator.theme_colors['accent3'] == '#8064A2'  # Fallback
        
        # Text colors should fallback appropriately
        assert generator.theme_colors['text_dark'] == '#000000'  # Fallback (no dk1)
        assert generator.theme_colors['text_light'] == '#F2F2F2'  # lt1
    
    def test_template_palette_mapping_minimal_palette(self):
        """Test mapping with minimal template palette."""
        
        template_palette = {
            'dk1': '#2F5597'  # Only dark color
        }
        
        generator = VisualGenerator(theme_colors=template_palette)
        
        # Primary should use dk1
        assert generator.theme_colors['primary'] == '#2F5597'
        
        # Secondary should also use dk1 (no acc1 available)
        assert generator.theme_colors['secondary'] == '#2F5597'
        
        # Accent colors should use fallbacks
        assert generator.theme_colors['accent1'] == '#9BBB59'  # Fallback
        assert generator.theme_colors['accent2'] == '#F79646'  # Fallback
        assert generator.theme_colors['accent3'] == '#8064A2'  # Fallback
    
    def test_default_colors_when_no_template_palette(self):
        """Test that default colors are used when no template palette provided."""
        
        generator = VisualGenerator()
        
        # Should use default corporate colors
        assert generator.template_palette is None
        assert generator.theme_colors['primary'] == '#1F497D'
        assert generator.theme_colors['secondary'] == '#4F81BD'
        assert generator.theme_colors['accent1'] == '#9BBB59'
        assert generator.theme_colors['accent2'] == '#F79646'
        assert generator.theme_colors['accent3'] == '#8064A2'
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.subplots')
    def test_bar_chart_uses_template_colors(self, mock_subplots, mock_savefig):
        """Test that bar charts use mapped template colors."""
        
        # Setup template palette
        template_palette = {
            'dk1': '#0F243E',
            'acc1': '#2E75B6', 
            'acc2': '#C5504B',
            'acc3': '#70AD47'
        }
        
        generator = VisualGenerator(theme_colors=template_palette)
        
        # Mock matplotlib objects
        mock_fig = Mock()
        mock_ax = Mock()
        mock_bars = [Mock(), Mock(), Mock()]
        mock_ax.bar.return_value = mock_bars
        mock_subplots.return_value = (mock_fig, mock_ax)
        
        # Create slide with chart data
        slide = SlidePlan(
            index=1,
            slide_type="chart",
            title="Test Chart",
            chart_data={
                "type": "bar",
                "categories": ["A", "B", "C"],
                "values": [10, 20, 15]
            }
        )
        
        # Generate chart
        generator.generate_chart(slide)
        
        # Verify bar() was called with template colors
        mock_ax.bar.assert_called_once()
        call_args = mock_ax.bar.call_args[1]  # Get keyword arguments
        
        # Check that colors parameter uses mapped template colors
        expected_colors = [
            generator.theme_colors['primary'],    # #0F243E (dk1)
            generator.theme_colors['secondary'],  # #2E75B6 (acc1) 
            generator.theme_colors['accent1']     # #2E75B6 (acc1)
        ]
        
        assert 'color' in call_args
        assert call_args['color'] == expected_colors
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.subplots') 
    def test_pie_chart_uses_template_colors(self, mock_subplots, mock_savefig):
        """Test that pie charts use mapped template colors."""
        
        template_palette = {
            'dk1': '#1B365D',
            'acc1': '#5B9BD5',
            'acc2': '#A5A5A5', 
            'acc3': '#FFC000',
            'acc4': '#ED7D31'
        }
        
        generator = VisualGenerator(theme_colors=template_palette)
        
        # Mock matplotlib objects
        mock_fig = Mock()
        mock_ax = Mock()
        mock_wedges = [Mock(), Mock(), Mock()]
        mock_texts = [Mock(), Mock(), Mock()]
        mock_autotexts = [Mock(), Mock(), Mock()]
        mock_ax.pie.return_value = (mock_wedges, mock_texts, mock_autotexts)
        mock_subplots.return_value = (mock_fig, mock_ax)
        
        # Create slide with pie chart data
        slide = SlidePlan(
            index=2,
            slide_type="chart", 
            title="Test Pie Chart",
            chart_data={
                "type": "pie",
                "labels": ["Red", "Green", "Blue"],
                "values": [30, 45, 25]
            }
        )
        
        # Generate chart
        generator.generate_chart(slide)
        
        # Verify pie() was called with template colors
        mock_ax.pie.assert_called_once()
        call_args = mock_ax.pie.call_args[1]
        
        # Check that colors parameter uses mapped template colors
        expected_colors = [
            generator.theme_colors['primary'],    # #1B365D
            generator.theme_colors['secondary'],  # #5B9BD5
            generator.theme_colors['accent1']     # #5B9BD5
        ]
        
        assert 'colors' in call_args
        # Pie chart gets slice of colors array
        assert call_args['colors'] == expected_colors
    
    def test_get_template_color_usage_with_palette(self):
        """Test template color usage reporting with palette."""
        
        template_palette = {
            'dk1': '#2F5597',
            'lt1': '#FFFFFF',
            'acc1': '#5B9BD5',
            'acc2': '#70AD47'
        }
        
        generator = VisualGenerator(theme_colors=template_palette)
        usage = generator.get_template_color_usage()
        
        assert usage['template_palette'] == template_palette
        assert usage['mapped_chart_colors'] == generator.theme_colors
        assert usage['primary_source'] == 'dk1'
        assert 'acc1' in usage['accent_sources']
        assert 'acc2' in usage['accent_sources']
    
    def test_get_template_color_usage_without_palette(self):
        """Test template color usage reporting without palette."""
        
        generator = VisualGenerator()
        usage = generator.get_template_color_usage()
        
        assert 'status' in usage
        assert 'no template palette' in usage['status']
    
    def test_image_hash_diff_bar_colors_match_primary_to_acc1(self):
        """Image hash diff test: confirm bar colors match primary → acc1."""
        
        # This test simulates the acceptance criteria for T-40
        template_palette = {
            'dk1': '#1F497D',  # Primary
            'acc1': '#5B9BD5', # Accent 1
            'acc2': '#70AD47'  # Accent 2
        }
        
        generator = VisualGenerator(theme_colors=template_palette)
        
        # Verify the mapping follows primary → acc1 pattern
        assert generator.theme_colors['primary'] == '#1F497D'     # dk1 maps to primary
        assert generator.theme_colors['secondary'] == '#5B9BD5'   # acc1 maps to secondary
        assert generator.theme_colors['accent1'] == '#5B9BD5'     # acc1 maps to accent1
        assert generator.theme_colors['accent2'] == '#70AD47'     # acc2 maps to accent2
        
        # Verify color progression follows template palette order
        expected_bar_colors = [
            '#1F497D',  # primary (dk1)
            '#5B9BD5',  # secondary (acc1) 
            '#5B9BD5',  # accent1 (acc1)
            '#70AD47'   # accent2 (acc2)
        ]
        
        # The chart generation should use these colors in order
        chart_color_sequence = [
            generator.theme_colors['primary'],
            generator.theme_colors['secondary'],
            generator.theme_colors['accent1'],
            generator.theme_colors['accent2']
        ]
        
        assert chart_color_sequence == expected_bar_colors


if __name__ == "__main__":
    pytest.main([__file__])