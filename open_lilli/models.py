"""Domain models for the Open Lilli presentation generator."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union
from enum import Enum

from pydantic import BaseModel, Field
try:
    from open_lilli.visual_proofreader import DesignIssueType
except ImportError: # Handle potential circular import during initialization or if file doesn't exist yet
    class DesignIssueType(str, Enum): # Fallback definition
        """
        Fallback Enum for DesignIssueType.
        Defines the types of design issues that VisualProofreader can identify.
        This fallback is used if `open_lilli.visual_proofreader.DesignIssueType` cannot be imported,
        preventing circular dependencies during initialization while still providing type hinting.
        """
        CAPITALIZATION = "capitalization"
        FORMATTING = "formatting"
        CONSISTENCY = "consistency"
        ALIGNMENT = "alignment"
        SPACING = "spacing"
        TYPOGRAPHY = "typography"
        COLOR = "color"
        HIERARCHY = "hierarchy"


class BulletItem(BaseModel):
    """Represents a bullet point with hierarchical level support."""
    
    text: str = Field(..., description="Text content of the bullet point")
    level: int = Field(default=0, description="Indentation level (0=top level, 1=sub-bullet, etc.)")
    
    class Config:
        """Pydantic configuration."""
        
        json_schema_extra = {
            "example": {
                "text": "Main bullet point",
                "level": 0
            }
        }


class TextOverflowConfig(BaseModel):
    """Configuration for handling text overflow in placeholders."""

    enable_bullet_splitting: bool = Field(
        default=True,
        description="Enable splitting of bullet lists across multiple slides if they overflow."
    )
    max_lines_per_placeholder: int = Field(
        default=15,
        description="Maximum number of lines allowed in a single placeholder before overflow handling (e.g., splitting) is triggered. This can be adjusted based on typical template font sizes and placeholder dimensions."
    )
    split_slide_title_suffix: str = Field(
        default="(Cont.)",
        description="Suffix to append to the title of a slide that has been created as a result of splitting a previous slide's content due to overflow."
    )
    enable_auto_fit_text: bool = Field(default=False, description="Enable auto-fitting of text to its shape. When True, text_frame.auto_size will be set to MSO_AUTO_SIZE.TEXT_TO_FIT_SHAPE.")

    class Config:
        """Pydantic configuration."""

        json_schema_extra = {
            "example": {
                "enable_bullet_splitting": True,
                "max_lines_per_placeholder": 15,
                "split_slide_title_suffix": "(Cont.)",
                "enable_auto_fit_text": False
            }
        }


class QualityGates(BaseModel):
    """Configuration for quality gate thresholds in presentation review."""

    max_bullets_per_slide: int = Field(
        default=7,
        description="Maximum number of bullet points allowed per slide"
    )
    max_readability_grade: float = Field(
        default=9.0,
        description="Maximum readability grade level (higher is more complex)"
    )
    max_style_errors: int = Field(
        default=0,
        description="Maximum number of style errors allowed"
    )
    min_overall_score: float = Field(
        default=7.0,
        description="Minimum overall quality score required (0-10 scale)"
    )
    min_contrast_ratio: float = Field(
        default=4.5,
        description="DEPRECATED: Minimum acceptable contrast ratio for text (WCAG AA for normal text). Use min_apca_lc_for_body_text instead."
    )
    min_apca_lc_for_body_text: float = Field(
        default=45.0,
        description="Minimum acceptable absolute APCA Lc value for body text. Scores below this threshold are typically flagged."
    )
    enable_alignment_check: bool = Field(
        default=True,
        description="Enable check for shape alignment within slide margins."
    )
    enable_overset_text_check: bool = Field(
        default=True,
        description="Enable check for overset or hidden text in shapes."
    )
    slide_margin_top_inches: float = Field(
        default=0.5,
        description="Top margin of the slide in inches."
    )
    slide_margin_bottom_inches: float = Field(
        default=0.5,
        description="Bottom margin of the slide in inches."
    )
    slide_margin_left_inches: float = Field(
        default=0.5,
        description="Left margin of the slide in inches."
    )
    slide_margin_right_inches: float = Field(
        default=0.5,
        description="Right margin of the slide in inches."
    )

    class Config:
        """Pydantic configuration."""

        json_schema_extra = {
            "example": {
                "max_bullets_per_slide": 7,
                "max_readability_grade": 9.0,
                "max_style_errors": 0,
                "min_overall_score": 7.0,
                "min_contrast_ratio": 4.5, # Kept for now for backward compatibility examples if any
                "min_apca_lc_for_body_text": 45.0,
                "enable_alignment_check": True,
                "enable_overset_text_check": True,
                "slide_margin_top_inches": 0.5,
                "slide_margin_bottom_inches": 0.5,
                "slide_margin_left_inches": 0.5,
                "slide_margin_right_inches": 0.5
            }
        }


class SlidePlan(BaseModel):
    """Represents a planned slide with content and layout information."""

    index: int = Field(..., description="Slide index in the presentation")
    slide_type: str = Field(
        ...,
        description="Type of slide (title, content, image, chart, two_column, etc.)"
    )
    title: str = Field(..., description="Main title of the slide")
    bullets: List[str] = Field(
        default_factory=list, description="List of bullet points for the slide (legacy format)"
    )
    bullet_hierarchy: Optional[List[BulletItem]] = Field(
        None, description="Hierarchical bullet structure with levels (T-100)"
    )
    image_query: Optional[str] = Field(
        None, description="Search query for finding relevant images"
    )
    chart_data: Optional[Dict[str, Any]] = Field(
        None, description="Data for generating charts/visualizations"
    )
    speaker_notes: Optional[str] = Field(
        None, description="Speaker notes for the slide"
    )
    layout_id: Optional[int] = Field(
        None, description="Template layout index to use for this slide"
    )
    summarized_by_llm: bool = Field(
        default=False, description="Indicates if the slide content was summarized by an LLM"
    )
    needs_splitting: bool = Field(
        default=False, description="Indicates if the slide needs to be split due to excessive content"
    )
    image_alt_text: Optional[str] = Field(
        None, description="Alternative text for the slide image, for accessibility."
    )

    def get_effective_bullets(self) -> List[BulletItem]:
        """
        Get the effective bullet structure, converting legacy bullets if needed.
        
        Returns:
            List of BulletItem objects with hierarchy information
        """
        if self.bullet_hierarchy is not None:
            return self.bullet_hierarchy
        
        # Convert legacy bullets to BulletItem format
        return [BulletItem(text=bullet, level=0) for bullet in self.bullets]
    
    def get_bullet_texts(self) -> List[str]:
        """
        Get bullet text content as flat list for backward compatibility.
        
        Returns:
            List of bullet text strings
        """
        if self.bullet_hierarchy is not None:
            return [bullet.text for bullet in self.bullet_hierarchy]
        
        return self.bullets

    class Config:
        """Pydantic configuration."""
        
        json_schema_extra = {
            "example": {
                "index": 1,
                "slide_type": "content",
                "title": "Market Overview",
                "bullets": [
                    "Market size: $10B and growing",
                    "Key trends: Digital transformation",
                    "Competitive landscape: 3 major players"
                ],
                "image_query": "business market trends",
                "chart_data": None,
                "speaker_notes": "Emphasize the growth opportunity",
                "layout_id": 1,
                "summarized_by_llm": False,
                "image_alt_text": "A bar chart showing market growth trends over the last 5 years."
            }
        }


class Outline(BaseModel):
    """Represents the overall structure of a presentation."""

    language: str = Field(
        default="en", description="Language code for the presentation (ISO 639-1)"
    )
    title: str = Field(..., description="Main title of the presentation")
    subtitle: Optional[str] = Field(
        None, description="Subtitle or tagline for the presentation"
    )
    slides: List[SlidePlan] = Field(
        ..., description="List of planned slides in order"
    )
    style_guidance: Optional[str] = Field(
        None, description="Style and tone guidance for content generation"
    )
    target_audience: Optional[str] = Field(
        None, description="Target audience for the presentation"
    )

    @property
    def slide_count(self) -> int:
        """Return the total number of slides."""
        return len(self.slides)

    def get_slide_by_index(self, index: int) -> Optional[SlidePlan]:
        """Get a slide by its index."""
        for slide in self.slides:
            if slide.index == index:
                return slide
        return None

    class Config:
        """Pydantic configuration."""
        
        json_schema_extra = {
            "example": {
                "language": "en",
                "title": "Q4 Business Review",
                "subtitle": "Performance and Strategic Outlook",
                "slides": [
                    {
                        "index": 0,
                        "slide_type": "title",
                        "title": "Q4 Business Review",
                        "bullets": [],
                        "image_query": None,
                        "chart_data": None,
                        "speaker_notes": "Welcome and introduction",
                        "layout_id": 0
                    }
                ],
                "style_guidance": "Professional and data-driven",
                "target_audience": "Executive leadership team"
            }
        }


class GenerationConfig(BaseModel):
    """Configuration for presentation generation."""

    max_slides: int = Field(
        default=20, description="Maximum number of slides to generate"
    )
    max_bullets_per_slide: int = Field(
        default=5, description="Maximum bullet points per slide"
    )
    include_images: bool = Field(
        default=True, description="Whether to include images in slides"
    )
    include_charts: bool = Field(
        default=True, description="Whether to generate charts from data"
    )
    tone: str = Field(
        default="professional", description="Tone for content generation"
    )
    complexity_level: str = Field(
        default="intermediate", 
        description="Complexity level (basic, intermediate, advanced)"
    )
    max_iterations: int = Field(
        default=3, 
        description="Maximum number of refinement iterations for auto-refine"
    )

    class Config:
        """Pydantic configuration."""
        
        json_schema_extra = {
            "example": {
                "max_slides": 15,
                "max_bullets_per_slide": 4,
                "include_images": True,
                "include_charts": True,
                "tone": "professional",
                "complexity_level": "intermediate",
                "max_iterations": 3
            }
        }


class ReviewFeedback(BaseModel):
    """
    Standardized model for feedback items from various validation and review processes.
    This includes rule-based style checks (e.g., font, alignment, placeholder population)
    and LLM-based visual proofreading results.
    """

    slide_index: int = Field(..., description="0-based index of the slide to which this feedback applies. Can be -1 for presentation-level feedback or errors not specific to a single slide.")
    severity: str = Field(
        ..., description="Severity level of the identified issue (e.g., 'low', 'medium', 'high', 'critical'). Helps prioritize fixes."
    )
    category: str = Field(
        ..., description="Broad category of the feedback (e.g., 'placeholder_population', 'text_overflow', 'alignment', 'accessibility', 'design', 'style_validation_font', 'visual_proofreader_error')."
    )
    message: str = Field(..., description="Detailed message describing the specific issue or feedback point.")
    suggestion: Optional[str] = Field(
        None, description="An optional suggested improvement, fix, or action to take to address the issue."
    )

    class Config:
        """Pydantic configuration."""
        
        json_schema_extra = {
            "example": {
                "slide_index": 3,
                "severity": "medium",
                "category": "content",
                "message": "This slide has too much text and may be hard to read",
                "suggestion": "Consider breaking into two slides or using more visuals"
            }
        }


class FontInfo(BaseModel):
    """Font information extracted from template."""
    
    name: str = Field(..., description="Font family name")
    size: Optional[float] = Field(None, description="Font size in points")  # Changed to float
    weight: Optional[str] = Field(None, description="Font weight (normal, bold)")
    color: Optional[str] = Field(None, description="Font color in hex format")
    
    class Config:
        """Pydantic configuration."""
        
        json_schema_extra = {
            "example": {
                "name": "Calibri",
                "size": 18.0,  # Example updated to float
                "weight": "normal",
                "color": "#000000"
            }
        }


class BulletInfo(BaseModel):
    """Bullet point formatting information."""
    
    character: str = Field(..., description="Bullet character")
    font: Optional[FontInfo] = Field(None, description="Font information for bullet")
    indent_level: int = Field(default=0, description="Indentation level (0-based)")
    
    class Config:
        """Pydantic configuration."""
        
        json_schema_extra = {
            "example": {
                "character": "•",
                "font": {
                    "name": "Calibri",
                    "size": 14.0,  # Example updated to float
                    "weight": "normal",
                    "color": "#000000"
                },
                "indent_level": 0
            }
        }


class PlaceholderStyleInfo(BaseModel):
    """Style information for a specific placeholder type."""
    
    placeholder_type: int = Field(..., description="PowerPoint placeholder type number")
    type_name: str = Field(..., description="Human-readable placeholder type name")
    default_font: Optional[FontInfo] = Field(None, description="Default font for this placeholder")
    bullet_styles: List[BulletInfo] = Field(
        default_factory=list, 
        description="Bullet styles for different indentation levels"
    )
    fill_color: Optional[str] = Field(None, description="Fill color of the placeholder shape, in hex format")
    
    class Config:
        """Pydantic configuration."""
        
        json_schema_extra = {
            "example": {
                "placeholder_type": 2,
                "type_name": "BODY",
                "default_font": {
                    "name": "Calibri",
                    "size": 14.0,  # Example updated to float
                    "weight": "normal",
                    "color": "#000000"
                },
                "bullet_styles": [
                    {
                        "character": "•",
                        "font": {
                            "name": "Calibri",
                            "size": 14.0,  # Example updated to float
                            "weight": "normal",
                            "color": "#000000"
                        },
                        "indent_level": 0
                    }
                ],
                "fill_color": "#FFFFFF"
            }
        }


class StyleValidationConfig(BaseModel):
    """Configuration for style validation during presentation assembly."""
    
    enabled: bool = Field(default=True, description="Enable style validation")
    mode: str = Field(
        default="lenient", 
        description="Validation mode: 'strict', 'lenient', or 'disabled'"
    )
    font_size_tolerance: int = Field(
        default=2, 
        description="Allowed font size deviation in points"
    )
    color_tolerance: float = Field(
        default=0.1,
        description="Allowed color deviation (0.0-1.0)"
    )
    enforce_font_name: bool = Field(
        default=True,
        description="Enforce exact font name matching"
    )
    enforce_font_weight: bool = Field(
        default=True,
        description="Enforce font weight (bold/normal) matching"
    )
    enforce_bullet_characters: bool = Field(
        default=True,
        description="Enforce bullet character consistency"
    )
    enforce_color_compliance: bool = Field(
        default=True,
        description="Enforce color compliance with theme"
    )
    allow_empty_placeholders: bool = Field(
        default=False,
        description="Allow placeholders to remain empty"
    )
    check_title_placeholders: bool = Field(
        default=True,
        description="Validate title placeholder styles"
    )
    check_content_placeholders: bool = Field(
        default=True,
        description="Validate content placeholder styles"
    )
    check_bullet_styles: bool = Field(
        default=True,
        description="Validate bullet point styles"
    )
    check_placeholder_population: bool = Field(
        default=True, description="Ensure required placeholders are populated"
    )
    check_text_overflow: bool = Field(
        default=True, description="Check for text overflowing its container"
    )
    check_alignment: bool = Field(
        default=True, description="Verify proper alignment of elements"
    )
    check_alt_text_accessibility: bool = Field(
        default=True, description="Perform rule-based accessibility check for image alt text (verifies presence and non-generic content of shape names for pictures)."
    )
    autofix_text_overflow: bool = Field(
        default=True, description="Enable automatic attempts to fix text overflow by reducing font size or truncating title text."
    )
    autofix_alignment: bool = Field(
        default=True, description="Enable automatic attempts to fix shape alignment issues by moving or resizing shapes to fit within defined slide margins."
    )
    quality_gates_config: Optional[QualityGates] = Field(
        default_factory=QualityGates,
        description="Configuration for quality gates, including slide margins which are crucial for alignment checks."
    )
    text_overflow_config: Optional[TextOverflowConfig] = Field(
        default=None,
        description="Configuration for text overflow handling, such as splitting bullets or slides."
    )

    # Visual Proofreader specific settings
    enable_visual_proofreader: bool = Field(
        default=False,
        description="Enable LLM-based visual proofreading for broader design consistency, formatting, and other visual aspects beyond rule-based checks."
    )
    visual_proofreader_focus_areas: Optional[List[DesignIssueType]] = Field(
        default=None,
        description="Optional list of specific `DesignIssueType` areas (e.g., 'CAPITALIZATION', 'ALIGNMENT') for the VisualProofreader to concentrate on. If None, the proofreader typically uses its own default set of focus areas."
    )
    visual_proofreader_model: str = Field(
        default="gpt-4.1",
        description="Specifies the LLM model to be used by the VisualProofreader (e.g., 'gpt-4', 'gpt-4-turbo-preview'). Model choice can impact quality, speed, and cost."
    )
    visual_proofreader_temperature: float = Field(
        default=0.1,
        ge=0.0, le=2.0,
        description="Controls the randomness of the LLM's output for VisualProofreader. Lower values (e.g., 0.1) produce more deterministic and consistent results, suitable for validation tasks."
    )
    visual_proofreader_enable_corrections: bool = Field(
        default=True,
        description="Allows the VisualProofreader to suggest textual corrections for detected issues, where applicable (e.g., suggesting a correctly capitalized title)."
    )
    
    class Config:
        """Pydantic configuration."""
        
        json_schema_extra = {
            "example": {
                "enabled": True,
                "mode": "strict",
                "font_size_tolerance": 2,
                "color_tolerance": 0.1,
                "enforce_font_name": True,
                "enforce_font_weight": True,
                "enforce_bullet_characters": True,
                "enforce_color_compliance": True,
                "allow_empty_placeholders": False,
                "check_title_placeholders": True,
                "check_content_placeholders": True,
                "check_bullet_styles": True,
                "check_placeholder_population": True,
                "check_text_overflow": True,
                "check_alignment": True,
                "check_alt_text_accessibility": True, # Renamed
                "autofix_text_overflow": True,
                "autofix_alignment": True,
                "quality_gates_config": {
                    "slide_margin_top_inches": 0.5,
                    "slide_margin_bottom_inches": 0.5,
                    "slide_margin_left_inches": 0.5,
                    "slide_margin_right_inches": 0.5,
                    "enable_alignment_check": True
                },
                "text_overflow_config": {
                    "enable_bullet_splitting": True,
                    "max_lines_per_placeholder": 15,
                    "split_slide_title_suffix": "(Cont.)",
                    "enable_auto_fit_text": False
                },
                "enable_visual_proofreader": True,
                "visual_proofreader_focus_areas": ["capitalization", "consistency"],
                "visual_proofreader_model": "gpt-4-turbo-preview",
                "visual_proofreader_temperature": 0.2,
                "visual_proofreader_enable_corrections": True
            }
        }


class TemplateStyle(BaseModel):
    """Complete style information extracted from a PowerPoint template."""
    
    master_font: Optional[FontInfo] = Field(
        None, description="Default font from slide master"
    )
    placeholder_styles: Dict[int, PlaceholderStyleInfo] = Field(
        default_factory=dict,
        description="Style information for each placeholder type"
    )
    theme_fonts: Dict[str, str] = Field(
        default_factory=dict,
        description="Theme font definitions (major, minor fonts)"
    )
    language_specific_fonts: Dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of language codes to specific font names to use for that language."
    )
    
    def get_font_for_placeholder_type(self, placeholder_type: int) -> Optional[FontInfo]:
        """
        Get font information for a specific placeholder type.
        
        Args:
            placeholder_type: PowerPoint placeholder type number
            
        Returns:
            FontInfo object or None if not found
        """
        placeholder_style = self.placeholder_styles.get(placeholder_type)
        if placeholder_style and placeholder_style.default_font:
            return placeholder_style.default_font
        return self.master_font
    
    def get_bullet_style_for_level(self, placeholder_type: int, level: int = 0) -> Optional[BulletInfo]:
        """
        Get bullet style for a specific placeholder type and indentation level.
        
        Args:
            placeholder_type: PowerPoint placeholder type number
            level: Indentation level (0-based)
            
        Returns:
            BulletInfo object or None if not found
        """
        placeholder_style = self.placeholder_styles.get(placeholder_type)
        if placeholder_style:
            for bullet_style in placeholder_style.bullet_styles:
                if bullet_style.indent_level == level:
                    return bullet_style
        return None
    
    def get_placeholder_style(self, placeholder_type: int) -> Optional[PlaceholderStyleInfo]:
        """
        Get complete style information for a placeholder type.
        
        Args:
            placeholder_type: PowerPoint placeholder type number
            
        Returns:
            PlaceholderStyleInfo object or None if not found
        """
        return self.placeholder_styles.get(placeholder_type)
    
    class Config:
        """Pydantic configuration."""
        
        json_schema_extra = {
            "example": {
                "master_font": {
                    "name": "Calibri",
                    "size": 12.0,  # Example updated to float
                    "weight": "normal",
                    "color": "#000000"
                },
                "placeholder_styles": {
                    1: {
                        "placeholder_type": 1,
                        "type_name": "TITLE",
                        "default_font": {
                            "name": "Calibri",
                            "size": 24.0,  # Example updated to float
                            "weight": "bold",
                            "color": "#1F497D"
                        },
                        "bullet_styles": [],
                        "fill_color": "#EEEEEE"
                    },
                    2: {
                        "placeholder_type": 2,
                        "type_name": "BODY",
                        "default_font": {
                            "name": "Calibri",
                            "size": 14.0,  # Example updated to float
                            "weight": "normal",
                            "color": "#000000"
                        },
                        "bullet_styles": [
                            {
                                "character": "•",
                                "font": {
                                    "name": "Calibri",
                                    "size": 14.0,  # Example updated to float
                                    "weight": "normal",
                                    "color": "#000000"
                                },
                                "indent_level": 0
                            }
                        ],
                        "fill_color": "#FFFFFF"
                    }
                },
                "theme_fonts": {
                    "major": "Calibri",
                    "minor": "Calibri"
                },
                "language_specific_fonts": {
                    "ar": "Arial Unicode MS",
                    "he": "David Libre",
                    "fa": "Tahoma"
                }
            }
        }


class QualityGates(BaseModel):
    """Configuration for quality gate thresholds in presentation review."""
    
    max_bullets_per_slide: int = Field(
        default=7, 
        description="Maximum number of bullet points allowed per slide"
    )
    max_readability_grade: float = Field(
        default=9.0,
        description="Maximum readability grade level (higher is more complex)"
    )
    max_style_errors: int = Field(
        default=0,
        description="Maximum number of style errors allowed"
    )
    min_overall_score: float = Field(
        default=7.0,
        description="Minimum overall quality score required (0-10 scale)"
    )
    min_contrast_ratio: float = Field(
        default=4.5,
        description="DEPRECATED: Minimum acceptable contrast ratio for text (WCAG AA for normal text). Use min_apca_lc_for_body_text instead."
    )
    min_apca_lc_for_body_text: float = Field(
        default=45.0,
        description="Minimum acceptable absolute APCA Lc value for body text. Scores below this threshold are typically flagged."
    )
    enable_alignment_check: bool = Field(
        default=True,
        description="Enable check for shape alignment within slide margins."
    )
    enable_overset_text_check: bool = Field(
        default=True,
        description="Enable check for overset or hidden text in shapes."
    )
    slide_margin_top_inches: float = Field(
        default=0.5,
        description="Top margin of the slide in inches."
    )
    slide_margin_bottom_inches: float = Field(
        default=0.5,
        description="Bottom margin of the slide in inches."
    )
    slide_margin_left_inches: float = Field(
        default=0.5,
        description="Left margin of the slide in inches."
    )
    slide_margin_right_inches: float = Field(
        default=0.5,
        description="Right margin of the slide in inches."
    )
    
    class Config:
        """Pydantic configuration."""
        
        json_schema_extra = {
            "example": {
                "max_bullets_per_slide": 7,
                "max_readability_grade": 9.0,
                "max_style_errors": 0,
                "min_overall_score": 7.0,
                "min_contrast_ratio": 4.5, # Kept for now for backward compatibility examples if any
                "min_apca_lc_for_body_text": 45.0,
                "enable_alignment_check": True,
                "enable_overset_text_check": True,
                "slide_margin_top_inches": 0.5,
                "slide_margin_bottom_inches": 0.5,
                "slide_margin_left_inches": 0.5,
                "slide_margin_right_inches": 0.5
            }
        }


class QualityGateResult(BaseModel):
    """Result of quality gate evaluation for a presentation."""
    
    status: str = Field(
        ..., 
        description="Overall status: 'pass' or 'needs_fix'"
    )
    gate_results: Dict[str, bool] = Field(
        ...,
        description="Results for each individual quality gate (True = passed, e.g., 'bullet_count', 'readability', 'contrast_check')"
    )
    violations: List[str] = Field(
        default_factory=list,
        description="List of specific quality gate violations"
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="List of improvement recommendations"
    )
    metrics: Dict[str, float] = Field(
        default_factory=dict,
        description="Quantitative metrics measured (e.g., 'avg_readability_grade', 'min_contrast_ratio_found')"
    )
    
    @property
    def passed_gates(self) -> int:
        """Return number of quality gates that passed."""
        return sum(1 for passed in self.gate_results.values() if passed)
    
    @property
    def total_gates(self) -> int:
        """Return total number of quality gates evaluated."""
        return len(self.gate_results)
    
    @property
    def pass_rate(self) -> float:
        """Return percentage of gates that passed."""
        if self.total_gates == 0:
            return 0.0
        return (self.passed_gates / self.total_gates) * 100.0
    
    class Config:
        """Pydantic configuration."""
        
        json_schema_extra = {
            "example": {
                "status": "needs_fix",
                "gate_results": {
                    "bullet_count": True,
                    "readability": False,
                    "style_errors": True,
                    "overall_score": False,
                    "contrast_check": False,
                    "alignment_check": True,
                    "overset_text_check": True,
                },
                "violations": [
                    "Slide 2 has readability grade 11.2 (exceeds limit of 9.0)",
                    "Overall score 6.5 is below minimum threshold of 7.0",
                    "Slide 3 (Body Placeholder): APCA Lc is 35.0. Recommended minimum absolute Lc is 45.0 for body text.",
                    "Shape 'Logo' is outside the right slide margin.",
                    "Shape 'TextBox 1' on slide 2 appears to have overset text."
                ],
                "recommendations": [
                    "Simplify language on Slide 2 to improve readability",
                    "Address critical and high severity feedback to improve overall score",
                    "Improve text contrast on Slide 3 for better accessibility. Aim for absolute APCA Lc of 45.0 or higher.",
                    "Adjust position of 'Logo' to be within slide margins.",
                    "Check 'TextBox 1' on slide 2 for text overflow and adjust size or content."
                ],
                "metrics": {
                    "max_bullets_found": 5,
                    "avg_readability_grade": 8.7,
                    "max_readability_grade": 11.2,
                    "style_error_count": 0,
                    "overall_score": 6.5,
                    "min_apca_lc_found": -20.5, # Example value, can be negative
                    "avg_apca_lc_found": 45.0,  # Example value
                    "max_apca_lc_found": 75.8,  # Example value
                    "min_abs_apca_lc_found": 20.5, # Example value
                    "avg_abs_apca_lc_found": 50.0,   # Example value
                    "misaligned_shapes_count": 0,
                    "overset_text_shapes_count": 0
                }
            }
        }


class SlideEmbedding(BaseModel):
    """Represents a slide's text embedding and metadata for ML layout recommendation."""
    
    slide_id: str = Field(..., description="Unique identifier for the slide")
    title: str = Field(..., description="Slide title")
    content_text: str = Field(..., description="Combined text content (title + bullets)")
    slide_type: str = Field(..., description="Actual slide type/layout used")
    layout_id: int = Field(..., description="Template layout ID used")
    embedding: List[float] = Field(..., description="Text embedding vector")
    bullet_count: int = Field(default=0, description="Number of bullet points")
    has_image: bool = Field(default=False, description="Whether slide has images")
    has_chart: bool = Field(default=False, description="Whether slide has charts")
    source_file: Optional[str] = Field(None, description="Original presentation file")
    created_at: str = Field(..., description="Timestamp when embedding was created")
    
    class Config:
        """Pydantic configuration."""
        
        json_schema_extra = {
            "example": {
                "slide_id": "pres1_slide3",
                "title": "Market Analysis Comparison",
                "content_text": "Market Analysis Comparison: Q1 vs Q2 revenue, Customer acquisition costs, Market share growth",
                "slide_type": "two_column",
                "layout_id": 3,
                "embedding": [0.1, -0.2, 0.3],  # Truncated for example
                "bullet_count": 3,
                "has_image": False,
                "has_chart": True,
                "source_file": "business_review.pptx",
                "created_at": "2024-01-15T10:30:00Z"
            }
        }


class LayoutRecommendation(BaseModel):
    """Recommendation from ML layout recommender system."""
    
    slide_type: str = Field(..., description="Recommended slide type/layout")
    layout_id: int = Field(..., description="Recommended template layout ID")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score (0.0-1.0)")
    reasoning: str = Field(..., description="Explanation for the recommendation")
    similar_slides: List[str] = Field(
        default_factory=list, 
        description="IDs of similar slides used for recommendation"
    )
    fallback_used: bool = Field(
        default=False, 
        description="Whether rule-based fallback was used"
    )
    
    class Config:
        """Pydantic configuration."""
        
        json_schema_extra = {
            "example": {
                "slide_type": "two_column",
                "layout_id": 3,
                "confidence": 0.85,
                "reasoning": "Based on comparison keywords and historical 'vs' patterns",
                "similar_slides": ["pres1_slide3", "pres2_slide7"],
                "fallback_used": False
            }
        }


class VectorStoreConfig(BaseModel):
    """Configuration for the vector store system."""
    
    embedding_model: str = Field(
        default="text-embedding-3-small", 
        description="OpenAI embedding model to use"
    )
    vector_dimension: int = Field(
        default=1536, 
        description="Dimension of embedding vectors"
    )
    similarity_threshold: float = Field(
        default=0.7, 
        ge=0.0, le=1.0,
        description="Minimum similarity threshold for recommendations"
    )
    confidence_threshold: float = Field(
        default=0.6, 
        ge=0.0, le=1.0,
        description="Minimum confidence to override rule-based layout"
    )
    max_neighbors: int = Field(
        default=5, 
        description="Maximum number of nearest neighbors to consider"
    )
    vector_store_path: str = Field(
        default="layouts.vec", 
        description="Path to vector store file"
    )
    default_layout_mappings: Dict[str, int] = Field(
        default_factory=lambda: {
            "title": 0,
            "content": 1,
            "two_column": 3,
            "image": 5,
            "chart": 1,
            "section": 2,
            "blank": 6,
            "image_content": 4,
            "content_dense": 7,
            "three_column": 8,
            "comparison": 9
        },
        description="Default layout type to ID mappings, used as a fallback."
    )
    
    class Config:
        """Pydantic configuration."""
        
        json_schema_extra = {
            "example": {
                "embedding_model": "text-embedding-3-small",
                "vector_dimension": 1536,
                "similarity_threshold": 0.7,
                "confidence_threshold": 0.6,
                "max_neighbors": 5,
                "vector_store_path": "layouts.vec",
                "default_layout_mappings": {
                    "title": 0,
                    "content": 1,
                    "two_column": 3,
                    "image": 5,
                    "chart": 1,
                    "section": 2,
                    "blank": 6,
                    "image_content": 4,
                    "content_dense": 7,
                    "three_column": 8,
                    "comparison": 9
                }
            }
        }


class IngestionResult(BaseModel):
    """Result of ingesting slides into the vector store."""
    
    total_slides_processed: int = Field(..., description="Total number of slides processed")
    successful_embeddings: int = Field(..., description="Number of successful embeddings created")
    failed_embeddings: int = Field(default=0, description="Number of failed embeddings")
    unique_layouts_found: int = Field(..., description="Number of unique layout types found")
    vector_store_size: int = Field(..., description="Total size of vector store after ingestion")
    processing_time_seconds: float = Field(..., description="Total processing time")
    errors: List[str] = Field(
        default_factory=list, 
        description="List of error messages encountered"
    )
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate of embeddings."""
        if self.total_slides_processed == 0:
            return 0.0
        return (self.successful_embeddings / self.total_slides_processed) * 100.0
    
    class Config:
        """Pydantic configuration."""
        
        json_schema_extra = {
            "example": {
                "total_slides_processed": 1000,
                "successful_embeddings": 985,
                "failed_embeddings": 15,
                "unique_layouts_found": 8,
                "vector_store_size": 2543,
                "processing_time_seconds": 120.5,
                "errors": ["Failed to extract text from slide 15", "Invalid layout detected in slide 42"]
            }
        }


class ContentDensityAnalysis(BaseModel):
    """Analysis of content density for a slide."""
    
    total_characters: int = Field(..., description="Total character count in slide content")
    estimated_lines: int = Field(..., description="Estimated number of lines needed")
    placeholder_capacity: int = Field(..., description="Estimated capacity of target placeholder")
    density_ratio: float = Field(..., description="Ratio of content to capacity (>1.0 = overflow)")
    requires_action: bool = Field(..., description="Whether content adjustment is needed")
    recommended_action: str = Field(..., description="Recommended action to take")
    
    @property
    def is_overflow(self) -> bool:
        """Check if content overflows the available space."""
        return self.density_ratio > 1.0
    
    @property
    def overflow_severity(self) -> str:
        """Get severity level of overflow."""
        if self.density_ratio <= 1.0:
            return "none"
        elif self.density_ratio <= 1.2:
            return "mild"
        elif self.density_ratio <= 1.5:
            return "moderate"
        else:
            return "severe"
    
    class Config:
        """Pydantic configuration."""
        
        json_schema_extra = {
            "example": {
                "total_characters": 850,
                "estimated_lines": 12,
                "placeholder_capacity": 600,
                "density_ratio": 1.42,
                "requires_action": True,
                "recommended_action": "split_slide"
            }
        }


class FontAdjustment(BaseModel):
    """Font size adjustment recommendation."""
    
    original_size: float = Field(..., description="Original font size in points") # Changed to float
    recommended_size: float = Field(..., description="Recommended font size in points") # Changed to float
    adjustment_points: float = Field(..., description="Size change in points (negative = smaller)") # Changed to float
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in adjustment")
    reasoning: str = Field(..., description="Explanation for the adjustment")
    safe_bounds: bool = Field(default=True, description="Whether adjustment is within safe bounds")
    
    @property
    def size_change_pct(self) -> float:
        """Calculate percentage change in font size."""
        return (self.adjustment_points / self.original_size) * 100
    
    class Config:
        """Pydantic configuration."""
        
        json_schema_extra = {
            "example": {
                "original_size": 18.0, # Example updated to float
                "recommended_size": 16.0, # Example updated to float
                "adjustment_points": -2.0, # Example updated to float
                "confidence": 0.85,
                "reasoning": "Mild overflow detected, reducing font size within safe bounds",
                "safe_bounds": True
            }
        }


class ContentFitResult(BaseModel):
    """Result of content fit analysis and adjustments."""
    
    slide_index: int = Field(..., description="Index of the slide analyzed")
    density_analysis: ContentDensityAnalysis = Field(..., description="Content density analysis")
    font_adjustment: Optional[FontAdjustment] = Field(None, description="Font adjustment if applied")
    split_performed: bool = Field(default=False, description="Whether slide was split")
    split_count: int = Field(default=1, description="Number of slides after splitting")
    final_action: str = Field(..., description="Final action taken")
    modified_slide_plan: Optional[SlidePlan] = Field(None, description="The slide plan after modifications like summarization, if any.")
    
    class Config:
        """Pydantic configuration."""
        
        json_schema_extra = {
            "example": {
                "slide_index": 3,
                "density_analysis": {
                    "total_characters": 850,
                    "estimated_lines": 12,
                    "placeholder_capacity": 600,
                    "density_ratio": 1.42,
                    "requires_action": True,
                    "recommended_action": "split_slide"
                },
                "font_adjustment": None,
                "split_performed": True,
                "split_count": 2,
                "final_action": "split_slide",
                "modified_slide_plan": None # Or an example SlidePlan if applicable
            }
        }


class ContentFitConfig(BaseModel):
    """Configuration for content fit analysis and adjustments."""
    
    characters_per_line: int = Field(default=50, description="Estimated characters per line")
    lines_per_placeholder: int = Field(default=8, description="Estimated lines per content placeholder")
    min_font_size: int = Field(default=12, description="Minimum allowed font size")
    max_font_size: int = Field(default=24, description="Maximum allowed font size")
    font_adjustment_limit: int = Field(default=3, description="Maximum font size adjustment in points")
    split_threshold: float = Field(default=1.5, description="Density ratio threshold for splitting")  # Adjusted default
    font_tune_threshold: float = Field(default=1.1, description="Density ratio threshold for font tuning")
    rewrite_threshold: float = Field(default=1.3, description="Density ratio threshold for attempting content rewrite")
    proportional_shrink_factor: float = Field(default=0.9, description="Factor by which to shrink font size proportionally (e.g., 0.9 for a 10% reduction attempt).")
    max_proportional_shrink_cap_factor: float = Field(default=0.85, description="Maximum shrink allowed, as a factor of the template's original body font size (e.g., 0.85 for 85% of original).")
    
    class Config:
        """Pydantic configuration."""
        
        json_schema_extra = {
            "example": {
                "characters_per_line": 50,
                "lines_per_placeholder": 8,
                "min_font_size": 12,
                "max_font_size": 24,
                "font_adjustment_limit": 3,
                "split_threshold": 1.5,
                "font_tune_threshold": 1.1,
                "rewrite_threshold": 1.3,
                "proportional_shrink_factor": 0.9,
                "max_proportional_shrink_cap_factor": 0.85
            }
        }


# Phase 4: Visual & Data Excellence Models

class ChartType(str, Enum):
    """Enumeration of supported native chart types."""
    
    BAR = "bar"
    COLUMN = "column"
    LINE = "line"
    PIE = "pie"
    SCATTER = "scatter"
    AREA = "area"
    DOUGHNUT = "doughnut"


class NativeChartData(BaseModel):
    """Configuration for native PowerPoint chart generation."""
    
    chart_type: ChartType = Field(..., description="Type of chart to create")
    title: str = Field(..., description="Chart title")
    categories: List[str] = Field(..., description="Category labels for x-axis")
    series: List[Dict[str, Union[str, List[float]]]] = Field(
        ..., 
        description="Data series with name and values"
    )
    x_axis_title: Optional[str] = Field(None, description="X-axis title")
    y_axis_title: Optional[str] = Field(None, description="Y-axis title")
    has_legend: bool = Field(default=True, description="Whether to show legend")
    has_data_labels: bool = Field(default=False, description="Whether to show data labels")
    use_template_colors: bool = Field(default=True, description="Use template color palette")
    
    class Config:
        """Pydantic configuration."""
        
        json_schema_extra = {
            "example": {
                "chart_type": "bar",
                "title": "Revenue by Quarter",
                "categories": ["Q1", "Q2", "Q3", "Q4"],
                "series": [
                    {"name": "2023", "values": [100, 120, 135, 150]},
                    {"name": "2024", "values": [110, 140, 160, 180]}
                ],
                "x_axis_title": "Quarter",
                "y_axis_title": "Revenue ($M)",
                "has_legend": True,
                "has_data_labels": True,
                "use_template_colors": True
            }
        }


class ProcessFlowType(str, Enum):
    """Types of process flow diagrams."""
    
    SEQUENTIAL = "sequential"
    BRANCHING = "branching"
    CIRCULAR = "circular"
    SWIMLANE = "swimlane"


class ProcessFlowStep(BaseModel):
    """Individual step in a process flow."""
    
    id: str = Field(..., description="Unique identifier for the step")
    label: str = Field(..., description="Step label/text")
    step_type: str = Field(default="process", description="Type: process, decision, start, end")
    connections: List[str] = Field(default_factory=list, description="Connected step IDs")
    position: Optional[Dict[str, float]] = Field(None, description="Optional x,y position")
    
    class Config:
        """Pydantic configuration."""
        
        json_schema_extra = {
            "example": {
                "id": "step1",
                "label": "Start Process",
                "step_type": "start",
                "connections": ["step2"],
                "position": {"x": 100, "y": 50}
            }
        }


class ProcessFlowConfig(BaseModel):
    """Configuration for process flow diagram generation."""
    
    flow_type: ProcessFlowType = Field(..., description="Type of process flow")
    title: str = Field(..., description="Flow diagram title")
    steps: List[ProcessFlowStep] = Field(..., description="Process steps")
    orientation: str = Field(default="horizontal", description="horizontal or vertical")
    use_template_colors: bool = Field(default=True, description="Use template colors")
    show_step_numbers: bool = Field(default=True, description="Show step numbers")
    
    class Config:
        """Pydantic configuration."""
        
        json_schema_extra = {
            "example": {
                "flow_type": "sequential",
                "title": "Order Processing Flow",
                "steps": [
                    {"id": "start", "label": "Order Received", "step_type": "start", "connections": ["validate"]},
                    {"id": "validate", "label": "Validate Order", "step_type": "process", "connections": ["approve"]},
                    {"id": "approve", "label": "Approve Payment", "step_type": "process", "connections": ["ship"]},
                    {"id": "ship", "label": "Ship Product", "step_type": "process", "connections": ["end"]},
                    {"id": "end", "label": "Order Complete", "step_type": "end", "connections": []}
                ],
                "orientation": "horizontal",
                "use_template_colors": True,
                "show_step_numbers": True
            }
        }


class AssetLibraryConfig(BaseModel):
    """Configuration for corporate asset library integration."""
    
    dam_api_url: Optional[str] = Field(None, description="Digital Asset Management API URL")
    api_key: Optional[str] = Field(None, description="API key for asset library")
    brand_guidelines_strict: bool = Field(default=False, description="Strict brand compliance mode")
    fallback_to_external: bool = Field(default=True, description="Allow external sources as fallback")
    preferred_asset_types: List[str] = Field(
        default_factory=lambda: ["icon", "photo", "logo"],
        description="Preferred asset types"
    )
    max_asset_size_mb: int = Field(default=10, description="Maximum asset size in MB")
    generative_ai_provider: Optional[str] = Field(None, description="Generative AI provider (e.g., 'dalle3', 'stablediffusion')")
    generative_ai_api_key: Optional[str] = Field(None, description="API key for the generative AI provider")
    generative_ai_model: Optional[str] = Field(None, description="Specific model name for the generative AI provider, if applicable")
    
    class Config:
        """Pydantic configuration."""
        
        json_schema_extra = {
            "example": {
                "dam_api_url": "https://api.company.com/assets",
                "api_key": "your-api-key-here",
                "brand_guidelines_strict": True,
                "fallback_to_external": False,
                "preferred_asset_types": ["icon", "photo", "logo"],
                "max_asset_size_mb": 10,
                "generative_ai_provider": "dalle3",
                "generative_ai_api_key": "your-generative-ai-api-key",
                "generative_ai_model": "dall-e-3"
            }
        }


class VisualExcellenceConfig(BaseModel):
    """Combined configuration for Phase 4 visual excellence features."""
    
    enable_native_charts: bool = Field(default=True, description="Enable native PowerPoint charts")
    enable_process_flows: bool = Field(default=True, description="Enable process flow diagrams")
    enable_asset_library: bool = Field(default=False, description="Enable corporate asset library")
    asset_library: Optional[AssetLibraryConfig] = Field(None, description="Asset library configuration")
    mermaid_to_svg: bool = Field(default=True, description="Convert Mermaid to SVG")
    svg_color_rewriting: bool = Field(default=True, description="Rewrite SVG colors to match template")
    enable_generative_ai: bool = Field(default=False, description="Enable generative AI for image sourcing")
    
    class Config:
        """Pydantic configuration."""
        
        json_schema_extra = {
            "example": {
                "enable_native_charts": True,
                "enable_process_flows": True,
                "enable_asset_library": False,
                "asset_library": None,
                "mermaid_to_svg": True,
                "svg_color_rewriting": True,
                "enable_generative_ai": False
            }
        }


class TemplateCompatibilityReport(BaseModel):
    """Report on the compatibility of a PowerPoint template."""

    issues: List[str] = Field(default_factory=list, description="General issues found with the template")
    suggestions: List[str] = Field(default_factory=list, description="Suggestions for improving template compatibility")
    missing_placeholders: List[str] = Field(
        default_factory=list, description="List of common placeholder types missing from the template"
    )
    color_scheme_warnings: List[str] = Field(
        default_factory=list, description="Issues found in the theme's color palette"
    )
    contrast_issues: List[str] = Field(
        default_factory=list, description="Specific color pairs with insufficient contrast"
    )
    passed_all_checks: bool = Field(
        default=True, description="True if the template passed all critical compatibility checks"
    )

    class Config:
        """Pydantic configuration."""

        json_schema_extra = {
            "example": {
                "issues": ["Template usability is limited due to missing common placeholders."],
                "suggestions": ["Add a 'content' placeholder to most slide layouts.", "Ensure 'dk1' and 'lt1' theme colors have good contrast."],
                "missing_placeholders": ["content", "image"],
                "color_scheme_warnings": ["Theme color 'dk1' is not defined or is identical to 'lt1'."],
                "contrast_issues": ["Color 'dk1' (#808080) and 'lt1' (#FFFFFF) have a contrast ratio of 2.1:1, which is too low for text."],
                "passed_all_checks": False
            }
        }


class DesignPattern(BaseModel):
    """Describes the inferred design pattern of a template."""

    name: str = Field(
        default="standard",
        description="Overall name of the design pattern (e.g., 'minimalist', 'standard', 'vibrant', 'data-heavy')"
    )
    font_scale_ratio: float = Field(
        default=1.8,
        description="Ratio of title font size to body font size (e.g., title_font / body_font). Default is typical for standard designs."
    )
    color_complexity_score: float = Field(
        default=0.5,
        description="Score from 0.0 (very simple, dk1/lt1 mainly) to 1.0 (complex, many distinct accent colors used) based on palette."
    )
    layout_density_preference: str = Field(
        default="medium",
        description="Preferred layout density: 'low', 'medium', or 'high'."
    )
    primary_intent: str = Field(
        default="balanced",
        description="Inferred primary intent of the design (e.g., 'readability', 'visual_impact', 'information_density', 'balanced')."
    )

    class Config:
        """Pydantic configuration."""

        json_schema_extra = {
            "example": {
                "name": "vibrant",
                "font_scale_ratio": 2.0,
                "color_complexity_score": 0.8,
                "layout_density_preference": "medium",
                "primary_intent": "visual_impact"
            }
        }