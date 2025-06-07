"""Template parser for analyzing PowerPoint templates."""

import logging
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from pptx import Presentation
from pptx.slide import SlideLayout
from pptx.enum.shapes import MSO_SHAPE_TYPE

from .models import TemplateStyle, FontInfo, BulletInfo, PlaceholderStyleInfo

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
        self.template_style = self._extract_template_style()
        
        logger.info(f"Template analysis complete. Layout map: {self.layout_map}")
        logger.info(f"Template style extracted with {len(self.template_style.placeholder_styles)} placeholder styles")

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
        try:
            # Extract theme colors from theme1.xml
            theme_colors = self.get_theme_colors()
            if theme_colors:
                logger.debug(f"Extracted theme colors: {list(theme_colors.keys())}")
                return theme_colors
        except Exception as e:
            logger.warning(f"Failed to extract theme colors from XML: {e}")
        
        # Fallback to default colors if extraction fails
        default_palette = {
            "dk1": "#000000",      # Dark 1 (usually black)
            "lt1": "#FFFFFF",      # Light 1 (usually white)
            "acc1": "#1F497D",     # Accent 1 (dark blue)
            "acc2": "#4F81BD",     # Accent 2 (medium blue)
            "acc3": "#9BBB59",     # Accent 3 (green)
            "acc4": "#F79646",     # Accent 4 (orange)
            "acc5": "#8064A2",     # Accent 5 (purple)
            "acc6": "#4BACC6",     # Accent 6 (light blue)
        }
        
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
            "theme_colors": self.palette.copy(),
            "template_style": {
                "placeholder_styles_count": len(self.template_style.placeholder_styles),
                "theme_fonts": self.template_style.theme_fonts.copy(),
                "has_master_font": self.template_style.master_font is not None,
                "placeholder_types_with_styles": list(self.template_style.placeholder_styles.keys())
            }
        }
        
        return info

    def get_theme_colors(self) -> Dict[str, str]:
        """
        Extract theme colors from ppt/theme/theme1.xml.
        
        Returns:
            Dictionary mapping color names (dk1, lt1, acc1-6) to hex RGB values
        """
        try:
            with zipfile.ZipFile(self.template_path, 'r') as pptx_zip:
                # Read the theme XML file
                theme_xml = pptx_zip.read('ppt/theme/theme1.xml')
                root = ET.fromstring(theme_xml)
                
                # Define namespaces used in the theme XML
                namespaces = {
                    'a': 'http://schemas.openxmlformats.org/drawingml/2006/main'
                }
                
                # Extract colors from the color scheme
                colors = {}
                color_scheme = root.find('.//a:clrScheme', namespaces)
                
                if color_scheme is not None:
                    # Map the standard color names to their XML elements
                    color_mappings = {
                        'dk1': 'a:dk1',
                        'lt1': 'a:lt1', 
                        'acc1': 'a:accent1',
                        'acc2': 'a:accent2',
                        'acc3': 'a:accent3',
                        'acc4': 'a:accent4',
                        'acc5': 'a:accent5',
                        'acc6': 'a:accent6'
                    }
                    
                    for color_name, xml_path in color_mappings.items():
                        color_elem = color_scheme.find(xml_path, namespaces)
                        if color_elem is not None:
                            hex_color = self._extract_color_value(color_elem, namespaces)
                            if hex_color:
                                colors[color_name] = hex_color
                
                return colors
                
        except (zipfile.BadZipFile, FileNotFoundError, ET.ParseError, KeyError) as e:
            logger.debug(f"Could not extract theme colors: {e}")
            return {}
    
    def _extract_color_value(self, color_elem: ET.Element, namespaces: Dict[str, str]) -> Optional[str]:
        """
        Extract hex color value from a color element.
        
        Args:
            color_elem: XML element containing color information
            namespaces: XML namespaces dictionary
            
        Returns:
            Hex color string or None if extraction fails
        """
        try:
            # Look for srgbClr (explicit RGB color)
            srgb_color = color_elem.find('.//a:srgbClr', namespaces)
            if srgb_color is not None:
                val = srgb_color.get('val')
                if val and len(val) == 6:
                    return f"#{val.upper()}"
            
            # Look for sysClr (system color) 
            sys_color = color_elem.find('.//a:sysClr', namespaces)
            if sys_color is not None:
                last_clr = sys_color.get('lastClr')
                if last_clr and len(last_clr) == 6:
                    return f"#{last_clr.upper()}"
            
            # Look for prstClr (preset color)
            preset_color = color_elem.find('.//a:prstClr', namespaces)
            if preset_color is not None:
                val = preset_color.get('val')
                # Convert common preset colors to hex
                preset_to_hex = {
                    'black': '#000000',
                    'white': '#FFFFFF',
                    'red': '#FF0000',
                    'green': '#008000',
                    'blue': '#0000FF',
                    'yellow': '#FFFF00',
                    'cyan': '#00FFFF',
                    'magenta': '#FF00FF',
                    'gray': '#808080',
                    'darkBlue': '#000080',
                    'darkGreen': '#008000',
                    'darkRed': '#800000'
                }
                return preset_to_hex.get(val, '#000000')
                
        except Exception as e:
            logger.debug(f"Error extracting color value: {e}")
            
        return None

    def _extract_template_style(self) -> TemplateStyle:
        """
        Extract comprehensive style information from the template.
        
        Returns:
            TemplateStyle object with font and bullet hierarchy information
        """
        try:
            # Extract theme fonts
            theme_fonts = self._extract_theme_fonts()
            
            # Extract master font information
            master_font = self._extract_master_font()
            
            # Extract placeholder-specific styles
            placeholder_styles = self._extract_placeholder_styles()
            
            template_style = TemplateStyle(
                master_font=master_font,
                placeholder_styles=placeholder_styles,
                theme_fonts=theme_fonts
            )
            
            logger.debug(f"Extracted template style with {len(placeholder_styles)} placeholder styles")
            return template_style
            
        except Exception as e:
            logger.warning(f"Failed to extract template style: {e}")
            # Return minimal style object
            return TemplateStyle()

    def _extract_theme_fonts(self) -> Dict[str, str]:
        """
        Extract theme font definitions from the presentation.
        
        Returns:
            Dictionary mapping font roles (major, minor) to font names
        """
        try:
            with zipfile.ZipFile(self.template_path, 'r') as pptx_zip:
                theme_xml = pptx_zip.read('ppt/theme/theme1.xml')
                root = ET.fromstring(theme_xml)
                
                namespaces = {
                    'a': 'http://schemas.openxmlformats.org/drawingml/2006/main'
                }
                
                fonts = {}
                font_scheme = root.find('.//a:fontScheme', namespaces)
                
                if font_scheme is not None:
                    # Extract major font (typically used for headings)
                    major_font = font_scheme.find('.//a:majorFont/a:latin', namespaces)
                    if major_font is not None:
                        fonts['major'] = major_font.get('typeface', 'Calibri')
                    
                    # Extract minor font (typically used for body text)  
                    minor_font = font_scheme.find('.//a:minorFont/a:latin', namespaces)
                    if minor_font is not None:
                        fonts['minor'] = minor_font.get('typeface', 'Calibri')
                
                return fonts
                
        except Exception as e:
            logger.debug(f"Could not extract theme fonts: {e}")
            return {'major': 'Calibri', 'minor': 'Calibri'}

    def _extract_master_font(self) -> Optional[FontInfo]:
        """
        Extract default font information from the slide master.
        
        Returns:
            FontInfo object or None if extraction fails
        """
        try:
            with zipfile.ZipFile(self.template_path, 'r') as pptx_zip:
                # Try to read slide master XML
                master_xml = pptx_zip.read('ppt/slideMasters/slideMaster1.xml')
                root = ET.fromstring(master_xml)
                
                namespaces = {
                    'a': 'http://schemas.openxmlformats.org/drawingml/2006/main',
                    'p': 'http://schemas.openxmlformats.org/presentationml/2006/main'
                }
                
                # Look for default font in master text styles
                default_font = root.find('.//a:defRPr/a:latin', namespaces)
                if default_font is not None:
                    font_name = default_font.get('typeface', 'Calibri')
                    
                    # Look for font size in the same element
                    rpr = default_font.getparent()
                    font_size = None
                    if rpr is not None:
                        size_attr = rpr.get('sz')
                        if size_attr:
                            # Font size is in hundredths of a point
                            font_size = int(size_attr) // 100
                    
                    return FontInfo(
                        name=font_name,
                        size=font_size,
                        weight="normal",
                        color="#000000"
                    )
                
        except Exception as e:
            logger.debug(f"Could not extract master font: {e}")
            
        # Return default font if extraction fails
        return FontInfo(
            name="Calibri",
            size=12,
            weight="normal", 
            color="#000000"
        )

    def _extract_placeholder_styles(self) -> Dict[int, PlaceholderStyleInfo]:
        """
        Extract style information for each placeholder type from layouts.
        
        Returns:
            Dictionary mapping placeholder types to their style information
        """
        placeholder_styles = {}
        
        # Process each layout to find placeholder-specific styles
        for layout in self.prs.slide_layouts:
            try:
                layout_styles = self._extract_layout_placeholder_styles(layout)
                
                # Merge layout styles into main dictionary
                for ph_type, style_info in layout_styles.items():
                    if ph_type not in placeholder_styles:
                        placeholder_styles[ph_type] = style_info
                    else:
                        # If we already have this placeholder type, merge bullet styles
                        existing = placeholder_styles[ph_type]
                        for bullet_style in style_info.bullet_styles:
                            if not any(b.indent_level == bullet_style.indent_level for b in existing.bullet_styles):
                                existing.bullet_styles.append(bullet_style)
                        
            except Exception as e:
                logger.debug(f"Failed to extract styles from layout: {e}")
                continue
        
        return placeholder_styles

    def _extract_layout_placeholder_styles(self, layout: SlideLayout) -> Dict[int, PlaceholderStyleInfo]:
        """
        Extract placeholder styles from a specific layout.
        
        Args:
            layout: SlideLayout to analyze
            
        Returns:
            Dictionary mapping placeholder types to style information
        """
        styles = {}
        
        for placeholder in layout.placeholders:
            ph_type = placeholder.placeholder_format.type
            type_name = self._placeholder_type_name(ph_type)
            
            try:
                # Extract font information for this placeholder
                font_info = self._extract_placeholder_font(placeholder)
                
                # Extract bullet styles for this placeholder
                bullet_styles = self._extract_placeholder_bullets(placeholder)
                
                styles[ph_type] = PlaceholderStyleInfo(
                    placeholder_type=ph_type,
                    type_name=type_name,
                    default_font=font_info,
                    bullet_styles=bullet_styles
                )
                
            except Exception as e:
                logger.debug(f"Failed to extract style for placeholder type {ph_type}: {e}")
                continue
        
        return styles

    def _extract_placeholder_font(self, placeholder) -> Optional[FontInfo]:
        """
        Extract font information from a placeholder.
        
        Args:
            placeholder: Placeholder shape to analyze
            
        Returns:
            FontInfo object or None if extraction fails
        """
        try:
            # Try to get font information from placeholder's text frame
            if hasattr(placeholder, 'text_frame') and placeholder.text_frame:
                text_frame = placeholder.text_frame
                
                # Look at the first paragraph's font properties
                if len(text_frame.paragraphs) > 0:
                    paragraph = text_frame.paragraphs[0]
                    
                    if hasattr(paragraph, 'font'):
                        font = paragraph.font
                        
                        # Extract font properties
                        font_name = getattr(font, 'name', None) or 'Calibri'
                        font_size = getattr(font, 'size', None)
                        if font_size:
                            font_size = font_size.pt
                        
                        font_bold = getattr(font, 'bold', None)
                        font_weight = "bold" if font_bold else "normal"
                        
                        font_color = "#000000"  # Default color
                        if hasattr(font, 'color') and font.color:
                            try:
                                # Try to get RGB color
                                rgb = font.color.rgb
                                if rgb:
                                    font_color = f"#{rgb}"
                            except:
                                pass
                        
                        return FontInfo(
                            name=font_name,
                            size=int(font_size) if font_size else None,
                            weight=font_weight,
                            color=font_color
                        )
                        
        except Exception as e:
            logger.debug(f"Could not extract placeholder font: {e}")
        
        return None

    def _extract_placeholder_bullets(self, placeholder) -> List[BulletInfo]:
        """
        Extract bullet style information from a placeholder.
        
        Args:
            placeholder: Placeholder shape to analyze
            
        Returns:
            List of BulletInfo objects for different indentation levels
        """
        bullet_styles = []
        
        try:
            if hasattr(placeholder, 'text_frame') and placeholder.text_frame:
                text_frame = placeholder.text_frame
                
                # Analyze each paragraph level (up to 5 levels typical in PowerPoint)
                for level in range(5):
                    try:
                        # Try to get bullet character for this level
                        bullet_char = self._get_bullet_character_for_level(text_frame, level)
                        
                        if bullet_char:
                            # Get font for this level
                            level_font = self._get_font_for_level(text_frame, level)
                            
                            bullet_info = BulletInfo(
                                character=bullet_char,
                                font=level_font,
                                indent_level=level
                            )
                            bullet_styles.append(bullet_info)
                            
                    except Exception as e:
                        logger.debug(f"Could not extract bullet for level {level}: {e}")
                        continue
                        
        except Exception as e:
            logger.debug(f"Could not extract bullets from placeholder: {e}")
        
        return bullet_styles

    def _get_bullet_character_for_level(self, text_frame, level: int) -> Optional[str]:
        """
        Get the bullet character for a specific indentation level.
        
        Args:
            text_frame: Text frame to analyze
            level: Indentation level (0-based)
            
        Returns:
            Bullet character string or None
        """
        try:
            # Default bullet characters for each level
            default_bullets = ["•", "○", "▪", "‒", "►"]
            
            # Return default bullet for the level
            if level < len(default_bullets):
                return default_bullets[level]
            else:
                return "•"  # Default to filled circle
                
        except Exception:
            return "•"  # Fallback

    def _get_font_for_level(self, text_frame, level: int) -> Optional[FontInfo]:
        """
        Get font information for a specific bullet level.
        
        Args:
            text_frame: Text frame to analyze  
            level: Indentation level (0-based)
            
        Returns:
            FontInfo object or None
        """
        try:
            # For now, return the same font for all levels
            # This could be enhanced to extract level-specific fonts
            if len(text_frame.paragraphs) > 0:
                paragraph = text_frame.paragraphs[0]
                if hasattr(paragraph, 'font'):
                    font = paragraph.font
                    
                    font_name = getattr(font, 'name', None) or 'Calibri'
                    font_size = getattr(font, 'size', None)
                    if font_size:
                        font_size = font_size.pt
                    
                    return FontInfo(
                        name=font_name,
                        size=int(font_size) if font_size else 14,
                        weight="normal",
                        color="#000000"
                    )
                    
        except Exception:
            pass
        
        # Return default font
        return FontInfo(
            name="Calibri",
            size=14,
            weight="normal", 
            color="#000000"
        )

    def get_template_style(self) -> TemplateStyle:
        """
        Get the extracted template style information.
        
        Returns:
            TemplateStyle object with font and bullet hierarchy
        """
        return self.template_style

    def get_font_for_placeholder_type(self, placeholder_type: int) -> Optional[FontInfo]:
        """
        Get font information for a specific placeholder type.
        
        Args:
            placeholder_type: PowerPoint placeholder type number
            
        Returns:
            FontInfo object or None if not found
        """
        return self.template_style.get_font_for_placeholder_type(placeholder_type)

    def get_bullet_style_for_level(self, placeholder_type: int, level: int = 0) -> Optional[BulletInfo]:
        """
        Get bullet style for a specific placeholder type and indentation level.
        
        Args:
            placeholder_type: PowerPoint placeholder type number
            level: Indentation level (0-based)
            
        Returns:
            BulletInfo object or None if not found
        """
        return self.template_style.get_bullet_style_for_level(placeholder_type, level)