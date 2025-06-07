"""Template parser for analyzing PowerPoint templates."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from pptx import Presentation
from pptx.slide import SlideLayout
from pptx.enum.shapes import MSO_SHAPE_TYPE

logger = logging.getLogger(__name__)


class TemplateParser:
    """Analyzes PowerPoint templates and catalogues slide layouts."""

    def __init__(self, template_path: str):
        """
        Initialize the template parser.
        
        Args:
            template_path: Path to the .pptx template file
            
        Raises:
            FileNotFoundError: If template file doesn't exist
            ValueError: If template file is invalid
        """
        self.template_path = Path(template_path)
        
        if not self.template_path.exists():
            raise FileNotFoundError(f"Template file not found: {template_path}")
        
        if not self.template_path.suffix.lower() == '.pptx':
            raise ValueError(f"Template must be a .pptx file, got: {self.template_path.suffix}")
        
        logger.info(f"Loading template: {self.template_path}")
        
        try:
            self.prs = Presentation(str(self.template_path))
            logger.info(f"Template loaded successfully with {len(self.prs.slide_layouts)} layouts")
        except Exception as e:
            raise ValueError(f"Failed to load template: {e}")
        
        # Analyze the template
        self.layout_map = self._index_layouts()
        self.palette = self._extract_theme_colors()
        
        logger.info(f"Template analysis complete. Layout map: {self.layout_map}")

    def _index_layouts(self) -> Dict[str, int]:
        """
        Analyze slide layouts and create a mapping from semantic names to indices.
        
        Returns:
            Dictionary mapping layout types to layout indices
        """
        layout_map = {}
        
        for i, layout in enumerate(self.prs.slide_layouts):
            layout_type = self._classify_layout(layout, i)
            
            # Store the first occurrence of each type
            if layout_type not in layout_map:
                layout_map[layout_type] = i
                logger.debug(f"Layout {i}: {layout_type} - {self._get_layout_info(layout)}")
        
        # Ensure we have at least basic layouts
        self._ensure_basic_layouts(layout_map)
        
        return layout_map

    def _classify_layout(self, layout: SlideLayout, index: int) -> str:
        """
        Classify a layout based on its placeholders and structure.
        
        Args:
            layout: SlideLayout to classify
            index: Layout index for fallback naming
            
        Returns:
            Semantic name for the layout
        """
        placeholders = layout.placeholders
        placeholder_types = [ph.placeholder_format.type for ph in placeholders]
        
        # Count different types of placeholders
        title_count = sum(1 for t in placeholder_types if t in (1, 13))  # TITLE, CENTERED_TITLE
        content_count = sum(1 for t in placeholder_types if t in (2, 7))  # BODY, OBJECT
        subtitle_count = sum(1 for t in placeholder_types if t == 3)  # SUBTITLE
        picture_count = sum(1 for t in placeholder_types if t == 18)  # PICTURE
        
        total_placeholders = len(placeholders)
        
        logger.debug(f"Layout {index}: {total_placeholders} placeholders - "
                    f"title:{title_count}, content:{content_count}, "
                    f"subtitle:{subtitle_count}, picture:{picture_count}")
        
        # Classification logic
        if title_count >= 1 and subtitle_count >= 1 and total_placeholders <= 3:
            return "title"
        elif title_count >= 1 and content_count >= 2:
            return "two_column"
        elif title_count >= 1 and picture_count >= 1 and content_count >= 1:
            return "image_content"
        elif title_count >= 1 and picture_count >= 1:
            return "image"
        elif title_count >= 1 and content_count == 1:
            return "content"
        elif title_count >= 1 and content_count == 0 and total_placeholders <= 2:
            return "section"
        elif total_placeholders == 0:
            return "blank"
        else:
            # Fallback to generic naming
            return f"layout_{index}"

    def _ensure_basic_layouts(self, layout_map: Dict[str, int]) -> None:
        """Ensure we have mappings for essential layout types."""
        essentials = {
            "title": "title",
            "content": "content", 
            "section": "section",
            "blank": "blank"
        }
        
        for layout_type, fallback in essentials.items():
            if layout_type not in layout_map:
                # Try to find a suitable fallback
                if fallback in layout_map:
                    layout_map[layout_type] = layout_map[fallback]
                elif layout_map:
                    # Use the first available layout as last resort
                    layout_map[layout_type] = list(layout_map.values())[0]
                    logger.warning(f"No {layout_type} layout found, using layout {layout_map[layout_type]} as fallback")

    def _get_layout_info(self, layout: SlideLayout) -> str:
        """Get human-readable info about a layout."""
        placeholders = layout.placeholders
        ph_info = []
        
        for ph in placeholders:
            ph_type = ph.placeholder_format.type
            type_name = self._placeholder_type_name(ph_type)
            ph_info.append(f"{type_name}({ph_type})")
        
        return f"[{', '.join(ph_info)}]"

    def _placeholder_type_name(self, ph_type: int) -> str:
        """Convert placeholder type number to readable name."""
        type_names = {
            1: "TITLE",
            2: "BODY", 
            3: "SUBTITLE",
            7: "OBJECT",
            13: "CENTERED_TITLE",
            18: "PICTURE",
            14: "CHART",
            15: "TABLE",
            16: "CLIP_ART",
            17: "MEDIA_CLIP"
        }
        return type_names.get(ph_type, f"TYPE_{ph_type}")

    def _extract_theme_colors(self) -> Dict[str, str]:
        """
        Extract theme colors from the presentation.
        
        Returns:
            Dictionary of color names to hex values
        """
        # This is a simplified version - full implementation would parse theme XML
        # For now, return some common corporate colors as defaults
        default_palette = {
            "primary": "#1F497D",      # Dark blue
            "secondary": "#4F81BD",    # Medium blue  
            "accent1": "#9BBB59",      # Green
            "accent2": "#F79646",      # Orange
            "accent3": "#8064A2",      # Purple
            "accent4": "#4BACC6",      # Light blue
            "text_dark": "#000000",    # Black
            "text_light": "#FFFFFF",   # White
            "background": "#FFFFFF",   # White
            "neutral": "#808080"       # Gray
        }
        
        # TODO: Implement actual theme color extraction from XML
        # This would involve parsing the theme part of the PPTX file
        
        logger.debug(f"Using default color palette: {list(default_palette.keys())}")
        return default_palette

    def get_layout(self, slide_type: str) -> SlideLayout:
        """
        Get a slide layout by semantic type.
        
        Args:
            slide_type: Semantic layout type (title, content, image, etc.)
            
        Returns:
            SlideLayout object
            
        Raises:
            ValueError: If layout type is not found
        """
        if slide_type not in self.layout_map:
            available = list(self.layout_map.keys())
            raise ValueError(f"Layout type '{slide_type}' not found. Available: {available}")
        
        layout_index = self.layout_map[slide_type]
        return self.prs.slide_layouts[layout_index]

    def get_layout_index(self, slide_type: str) -> int:
        """
        Get layout index by semantic type.
        
        Args:
            slide_type: Semantic layout type
            
        Returns:
            Layout index
        """
        if slide_type not in self.layout_map:
            # Return a sensible default
            return self.layout_map.get("content", 0)
        
        return self.layout_map[slide_type]

    def list_available_layouts(self) -> List[str]:
        """
        Get list of available layout types.
        
        Returns:
            List of layout type names
        """
        return list(self.layout_map.keys())

    def analyze_layout_placeholders(self, slide_type: str) -> Dict[str, any]:
        """
        Analyze the placeholders in a specific layout.
        
        Args:
            slide_type: Layout type to analyze
            
        Returns:
            Dictionary with placeholder analysis
        """
        layout = self.get_layout(slide_type)
        placeholders = layout.placeholders
        
        analysis = {
            "total_placeholders": len(placeholders),
            "placeholder_details": [],
            "has_title": False,
            "has_content": False,
            "has_image": False,
            "has_chart": False
        }
        
        for i, ph in enumerate(placeholders):
            ph_type = ph.placeholder_format.type
            type_name = self._placeholder_type_name(ph_type)
            
            ph_detail = {
                "index": i,
                "type": ph_type,
                "type_name": type_name,
                "position": {
                    "left": ph.left,
                    "top": ph.top,
                    "width": ph.width,
                    "height": ph.height
                }
            }
            
            analysis["placeholder_details"].append(ph_detail)
            
            # Set flags based on placeholder types
            if ph_type in (1, 13):  # TITLE types
                analysis["has_title"] = True
            elif ph_type in (2, 7):  # BODY, OBJECT types
                analysis["has_content"] = True
            elif ph_type == 18:  # PICTURE
                analysis["has_image"] = True
            elif ph_type == 14:  # CHART
                analysis["has_chart"] = True
        
        return analysis

    def get_theme_color(self, color_name: str) -> str:
        """
        Get a theme color by name.
        
        Args:
            color_name: Name of the color (primary, secondary, etc.)
            
        Returns:
            Hex color code
        """
        return self.palette.get(color_name, "#000000")

    def get_slide_size(self) -> Tuple[int, int]:
        """
        Get the slide dimensions.
        
        Returns:
            Tuple of (width, height) in EMUs (English Metric Units)
        """
        return (self.prs.slide_width, self.prs.slide_height)

    def get_template_info(self) -> Dict[str, any]:
        """
        Get comprehensive information about the template.
        
        Returns:
            Dictionary with template analysis
        """
        width, height = self.get_slide_size()
        
        info = {
            "template_path": str(self.template_path),
            "total_layouts": len(self.prs.slide_layouts),
            "available_layout_types": self.list_available_layouts(),
            "layout_mapping": self.layout_map.copy(),
            "slide_dimensions": {
                "width": width,
                "height": height,
                "width_inches": width / 914400,  # Convert EMU to inches
                "height_inches": height / 914400
            },
            "theme_colors": self.palette.copy()
        }
        
        return info