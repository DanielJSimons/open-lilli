"""Slide assembler for building PowerPoint presentations."""

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Union

from pptx import Presentation
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR, MSO_AUTO_SIZE
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE_TYPE, PP_PLACEHOLDER

from .models import Outline, SlidePlan, StyleValidationConfig, FontInfo, BulletInfo, FontAdjustment, NativeChartData, BulletItem
from .template_parser import TemplateParser
from .exceptions import StyleError, ValidationConfigError

logger = logging.getLogger(__name__)

RTL_LANGUAGES = ["ar", "he", "fa"]

class SlideAssembler:
    """Assembles PowerPoint presentations from templates and generated content."""

    def __init__(
        self, 
        template_parser: TemplateParser,
        validation_config: Optional[StyleValidationConfig] = None
    ):
        """
        Initialize the slide assembler.
        
        Args:
            template_parser: TemplateParser instance for template access
            validation_config: Style validation configuration
        """
        self.template_parser = template_parser
        self.max_title_length = 60
        self.max_bullet_length = 120
        self.validation_config = validation_config or StyleValidationConfig()
        
        logger.info("SlideAssembler initialized")
        logger.debug(f"Style validation: {self.validation_config.enabled}, mode: {self.validation_config.mode}")

    def assemble(
        self,
        outline: Outline,
        slides: List[SlidePlan],
        visuals: Optional[Dict[int, Dict[str, str]]] = None,
        output_path: Union[str, Path] = "output.pptx"
    ) -> Path:
        """
        Assemble a complete PowerPoint presentation.
        
        Args:
            outline: Presentation outline with metadata
            slides: List of slide plans with content
            visuals: Dictionary mapping slide indices to visual file paths
            output_path: Path to save the output presentation
            
        Returns:
            Path to the assembled presentation file
        """
        output_path = Path(output_path)
        visuals = visuals or {}
        language = outline.language # Get language from outline
        
        logger.info(f"Assembling presentation with {len(slides)} slides in language: {language}")
        logger.info(f"Output path: {output_path}")
        
        # Create new presentation from template
        prs = Presentation(str(self.template_parser.template_path))
        
        # Clear any existing slides from the template
        # Iterate backwards to safely delete
        for i in range(len(prs.slides) - 1, -1, -1):
            r_id = prs.slides._sldIdLst[i].rId
            prs.part.drop_rel(r_id)
            del prs.slides._sldIdLst[i]
        
        # Add slides
        for slide_plan in slides:
            try:
                self._add_slide(prs, slide_plan, visuals.get(slide_plan.index, {}), language) # Pass language
                logger.debug(f"Added slide {slide_plan.index}: {slide_plan.title}")
            except Exception as e:
                logger.error(f"Failed to add slide {slide_plan.index}: {e}")
                # Add a basic slide as fallback
                self._add_fallback_slide(prs, slide_plan)
        
        # Apply presentation metadata
        self._apply_metadata(prs, outline)
        
        # Validate presentation style before saving
        self.validate_presentation_style(prs)
        
        # Save presentation
        prs.save(str(output_path))
        
        logger.info(f"Successfully assembled presentation: {output_path}")
        return output_path

    def patch_existing_presentation(
        self,
        input_pptx: Path,
        updated_slides: List[SlidePlan],
        target_indices: List[int],
        language: str, # Add language parameter
        visuals: Optional[Dict[int, Dict[str, str]]] = None,
        output_path: Union[str, Path] = "updated.pptx"
    ) -> Path:
        """
        Patch specific slides in an existing presentation.
        
        Args:
            input_pptx: Path to existing presentation
            updated_slides: List of all slide plans (with some updated)
            target_indices: List of slide indices that were updated
            language: Language code for the presentation
            visuals: Dictionary mapping slide indices to visual file paths
            output_path: Path to save the patched presentation
            
        Returns:
            Path to the patched presentation file
        """
        output_path = Path(output_path)
        visuals = visuals or {}
        
        logger.info(f"Patching presentation with {len(target_indices)} updated slides in language: {language}")
        logger.info(f"Target indices: {target_indices}")
        logger.info(f"Output path: {output_path}")
        
        # Load existing presentation
        prs = Presentation(str(input_pptx))
        
        # Verify slide count matches
        if len(prs.slides) != len(updated_slides):
            logger.warning(
                f"Slide count mismatch: existing {len(prs.slides)}, "
                f"updated {len(updated_slides)}"
            )
        
        # Replace only the target slides
        for target_idx in target_indices:
            if target_idx >= len(prs.slides):
                logger.error(f"Target index {target_idx} exceeds slide count {len(prs.slides)}")
                continue
                
            if target_idx >= len(updated_slides):
                logger.error(f"Target index {target_idx} exceeds updated slides count {len(updated_slides)}")
                continue
            
            try:
                # Get the slide plan for this index
                slide_plan = updated_slides[target_idx]
                
                # Replace the slide at this index
                self._replace_slide_at_index(
                    prs, target_idx, slide_plan, visuals.get(target_idx, {}), language # Pass language
                )
                
                logger.debug(f"Replaced slide {target_idx}: {slide_plan.title}")
                
            except Exception as e:
                logger.error(f"Failed to replace slide {target_idx}: {e}")
        
        # Validate patched presentation style
        self.validate_presentation_style(prs)
        
        # Save patched presentation
        prs.save(str(output_path))
        
        logger.info(f"Successfully patched presentation: {output_path}")
        return output_path

    def _replace_slide_at_index(
        self,
        prs: Presentation,
        slide_idx: int,
        slide_plan: SlidePlan,
        slide_visuals: Dict[str, str],
        language: str # Add language parameter
    ) -> None:
        """
        Replace a slide at a specific index with new content.
        
        Args:
            prs: PowerPoint presentation object
            slide_idx: Index of slide to replace
            slide_plan: New slide plan with content
            slide_visuals: Visual assets for the slide
            language: Language code for the presentation
        """
        # Get the existing slide
        existing_slide = prs.slides[slide_idx]
        
        # Get the layout that matches our slide plan
        layout_index = slide_plan.layout_id or 0
        if layout_index >= len(prs.slide_layouts):
            logger.warning(f"Layout index {layout_index} out of range, using existing layout")
            layout = existing_slide.slide_layout
        else:
            layout = prs.slide_layouts[layout_index]
        
        # Clear existing content while preserving slide structure
        self._clear_slide_content(existing_slide)
        
        # Apply new layout if different
        if layout != existing_slide.slide_layout:
            # Note: Changing layout is complex in python-pptx
            # For now, we'll work with the existing layout
            logger.debug(f"Layout change requested but using existing layout for slide {slide_idx}")
        
        # Add new content
        self._add_title(existing_slide, slide_plan.title, language) # Pass language
        
        # Add content based on slide type
        if slide_plan.slide_type == "title":
            self._add_title_slide_content(existing_slide, slide_plan, language) # Pass language
        elif slide_plan.bullet_hierarchy is not None or slide_plan.bullets:
            # T-100: Use hierarchical bullet content if available, otherwise fall back to legacy
            if slide_plan.bullet_hierarchy is not None:
                self._add_hierarchical_bullet_content(existing_slide, slide_plan, language)
            else:
                self._add_bullet_content(existing_slide, slide_plan.bullets, language) # Pass language
        
        # Add visuals
        if "native_chart" in slide_visuals and slide_plan.chart_data:
            self._add_native_chart(existing_slide, slide_plan.chart_data)
        elif "chart" in slide_visuals:
            chart_alt_text = None
            if slide_plan.chart_data:
                if isinstance(slide_plan.chart_data, NativeChartData) and slide_plan.chart_data.title:
                    chart_alt_text = slide_plan.chart_data.title
                elif isinstance(slide_plan.chart_data, dict) and slide_plan.chart_data.get("title"):
                    chart_alt_text = slide_plan.chart_data.get("title")
            if not chart_alt_text:
                chart_alt_text = slide_plan.image_alt_text or "Chart" # Fallback
            self._add_chart_image(existing_slide, slide_visuals["chart"], alt_text=chart_alt_text)
        
        if "process_flow" in slide_visuals:
            # Assuming process flow is like an image, use image_alt_text or a generic one
            self._add_process_flow(existing_slide, slide_visuals["process_flow"], alt_text=slide_plan.image_alt_text or "Process flow diagram")
        
        if "image" in slide_visuals:
            self._add_image(existing_slide, slide_visuals["image"], alt_text=slide_plan.image_alt_text)
        
        # Update speaker notes
        if slide_plan.speaker_notes:
            self._add_speaker_notes(existing_slide, slide_plan.speaker_notes)

    def _clear_slide_content(self, slide) -> None:
        """
        Clear content from a slide while preserving structure.
        
        Args:
            slide: PowerPoint slide object to clear
        """
        try:
            # Clear text content from placeholders
            for placeholder in slide.placeholders:
                if hasattr(placeholder, 'text_frame') and placeholder.text_frame:
                    placeholder.text_frame.clear()
                elif hasattr(placeholder, 'text'):
                    placeholder.text = ""
            
            # Remove non-placeholder shapes (like added images/charts)
            shapes_to_remove = []
            for shape in slide.shapes:
                # Only remove shapes that are not placeholders
                if not hasattr(shape, 'placeholder_format'):
                    shapes_to_remove.append(shape)
            
            # Remove collected shapes
            for shape in shapes_to_remove:
                try:
                    slide.shapes._spTree.remove(shape._element)
                except Exception as e:
                    logger.debug(f"Could not remove shape: {e}")
            
            # Clear speaker notes
            if slide.has_notes_slide:
                notes_slide = slide.notes_slide
                if hasattr(notes_slide, 'notes_text_frame'):
                    notes_slide.notes_text_frame.clear()
            
            logger.debug("Cleared slide content")
            
        except Exception as e:
            logger.error(f"Failed to clear slide content: {e}")

    def _add_slide(
        self,
        prs: Presentation,
        slide_plan: SlidePlan,
        slide_visuals: Dict[str, str],
        language: str # Add language parameter
    ) -> None:
        """Add a single slide to the presentation."""
        
        # Get the appropriate layout
        layout_index = slide_plan.layout_id or 0
        if layout_index >= len(prs.slide_layouts):
            logger.warning(f"Layout index {layout_index} out of range, using 0")
            layout_index = 0
        
        layout = prs.slide_layouts[layout_index]
        slide = prs.slides.add_slide(layout)
        
        # Log if content was summarized by LLM
        if hasattr(slide_plan, 'summarized_by_llm') and slide_plan.summarized_by_llm:
            logger.info(f"Slide {slide_plan.index} content was summarized by LLM.")

        # Add title
        self._add_title(slide, slide_plan.title, language) # Pass language
        
        # Add content based on slide type
        if slide_plan.slide_type == "title":
            # Assuming _add_title_slide_content also needs language for RTL subtitle
            self._add_title_slide_content(slide, slide_plan, language) # Pass language
        elif slide_plan.bullet_hierarchy is not None or slide_plan.bullets:
            # T-100: Use hierarchical bullet content if available, otherwise fall back to legacy
            if slide_plan.bullet_hierarchy is not None:
                self._add_hierarchical_bullet_content(slide, slide_plan, language)
            else:
                self._add_bullet_content(slide, slide_plan.bullets, language) # Pass language
        
        # Add visuals
        if "native_chart" in slide_visuals and slide_plan.chart_data:
            self._add_native_chart(slide, slide_plan.chart_data)
        elif "chart" in slide_visuals:
            chart_alt_text = None
            if slide_plan.chart_data:
                if isinstance(slide_plan.chart_data, NativeChartData) and slide_plan.chart_data.title:
                    chart_alt_text = slide_plan.chart_data.title
                elif isinstance(slide_plan.chart_data, dict) and slide_plan.chart_data.get("title"):
                    chart_alt_text = slide_plan.chart_data.get("title")
            if not chart_alt_text: # Fallback if no specific chart title
                chart_alt_text = slide_plan.image_alt_text or "Chart"
            self._add_chart_image(slide, slide_visuals["chart"], alt_text=chart_alt_text)
        
        if "process_flow" in slide_visuals:
            # Assuming process flow is like an image, use image_alt_text or a generic one
            self._add_process_flow(slide, slide_visuals["process_flow"], alt_text=slide_plan.image_alt_text or "Process flow diagram")
        
        if "image" in slide_visuals:
            self._add_image(slide, slide_visuals["image"], alt_text=slide_plan.image_alt_text)
        
        # Check for RTL language and skip font adjustment if necessary
        if language.lower() in RTL_LANGUAGES:
            logger.info(f"Slide {slide_plan.index} is in RTL language ({language}), skipping font adjustment.")
            # Clear any existing font adjustment to prevent it from being applied
            if hasattr(slide_plan, '__dict__') and 'font_adjustment' in slide_plan.__dict__:
                slide_plan.__dict__['font_adjustment'] = None

        # Apply font adjustments if needed (will be skipped if cleared above)
        self._apply_font_adjustments(slide, slide_plan)
        
        # Add speaker notes
        if slide_plan.speaker_notes:
            self._add_speaker_notes(slide, slide_plan.speaker_notes)
        
        # T-93: Remove/hide any empty placeholders to avoid style warnings
        hidden_count = self._remove_empty_placeholders_from_slide(slide)
        if hidden_count > 0:
            logger.debug(f"Hidden {hidden_count} empty placeholders on slide {slide_plan.index}")

    def _add_title(self, slide, title: str, language: str) -> None: # Add language
        """Add title to slide."""
        try:
            if hasattr(slide.shapes, 'title') and slide.shapes.title:
                title_shape = slide.shapes.title
                
                # Truncate title if too long
                if len(title) > self.max_title_length:
                    title = title[:self.max_title_length - 3] + "..."
                
                title_shape.text = title
                
                # Apply basic formatting
                if title_shape.text_frame.paragraphs:
                    paragraph = title_shape.text_frame.paragraphs[0]
                    if paragraph.runs:
                        run = paragraph.runs[0]
                        run.font.bold = True
                        
                        # Proactive font selection
                        specific_font_name = self.template_parser.template_style.language_specific_fonts.get(language.lower())
                        if specific_font_name:
                            run.font.name = specific_font_name
                            logger.debug(f"Applied language-specific font '{specific_font_name}' for title in language '{language.lower()}'")

                # RTL alignment for title
                if language.lower() in RTL_LANGUAGES:
                    if title_shape.text_frame and title_shape.text_frame.paragraphs:
                        for para in title_shape.text_frame.paragraphs:
                            para.alignment = PP_ALIGN.RIGHT

                logger.debug(f"Added title: {title}")
            else:
                logger.warning("No title placeholder found in slide")
                
        except Exception as e:
            logger.error(f"Failed to add title '{title}': {e}")

    def _add_title_slide_content(self, slide, slide_plan: SlidePlan, language: str) -> None: # Add language
        """Add content specific to title slides."""
        try:
            # Try to find subtitle placeholder
            subtitle_placeholder = None
            for placeholder in slide.placeholders:
                if placeholder.placeholder_format.type == 3:  # SUBTITLE
                    subtitle_placeholder = placeholder
                    break
            
            if subtitle_placeholder:
                subtitle_text = ""
                if slide_plan.bullets: # Assuming bullets might be used as subtitle parts
                    subtitle_text = " • ".join(slide_plan.bullets[:2])
                elif hasattr(slide_plan, 'subtitle') and slide_plan.subtitle:
                    subtitle_text = slide_plan.subtitle
                
                if subtitle_text:
                    subtitle_placeholder.text = subtitle_text
                    # Apply RTL alignment for subtitle if needed
                    if language.lower() in RTL_LANGUAGES:
                        if subtitle_placeholder.text_frame and subtitle_placeholder.text_frame.paragraphs:
                            for para in subtitle_placeholder.text_frame.paragraphs:
                                para.alignment = PP_ALIGN.RIGHT
                    logger.debug(f"Added subtitle: {subtitle_text}")
                else:
                    # T-93: Hide empty subtitle placeholder to avoid style warning
                    self._hide_empty_placeholder(subtitle_placeholder, "subtitle")
                    logger.debug("Hidden empty subtitle placeholder to avoid style warning")
                    
        except Exception as e:
            logger.error(f"Failed to add title slide content: {e}")

    def _add_bullet_content(self, slide, bullets: List[str], language: str) -> None: # Add language
        """Add bullet points to slide."""
        try:
            # T-94: Find all BODY placeholders for two-column distribution
            body_placeholders = []
            
            for placeholder in slide.placeholders:
                ph_type = placeholder.placeholder_format.type
                if ph_type in (2, 7):  # BODY or OBJECT
                    body_placeholders.append(placeholder)
            
            if not body_placeholders:
                logger.warning("No content placeholder found, looking for any text placeholder")
                # Fallback: look for any placeholder we can use
                for placeholder in slide.placeholders:
                    if hasattr(placeholder, 'text_frame'):
                        body_placeholders.append(placeholder)
                        break
            
            if not body_placeholders:
                logger.warning("No suitable placeholder found for bullet content")
                return
            
            # T-94: Two-column bullet distribution when ≥2 BODY placeholders
            if len(body_placeholders) >= 2 and len(bullets) > 1:
                self._distribute_bullets_across_columns(body_placeholders, bullets, language)
            else:
                # Standard single-column bullet distribution
                self._add_bullets_to_placeholder(body_placeholders[0], bullets, language)
                
        except Exception as e:
            logger.error(f"Failed to add bullet content: {e}")

    def _add_hierarchical_bullet_content(self, slide, slide_plan: SlidePlan, language: str) -> None:
        """
        Add hierarchical bullet content to slide (T-100).
        
        Args:
            slide: PowerPoint slide object
            slide_plan: SlidePlan with hierarchical bullet structure
            language: Language code for text formatting
        """
        try:
            # Get effective bullets (hierarchical or legacy)
            bullet_items = slide_plan.get_effective_bullets()
            
            if not bullet_items:
                return
            
            # Find BODY placeholders
            body_placeholders = []
            
            for placeholder in slide.placeholders:
                ph_type = placeholder.placeholder_format.type
                if ph_type in (2, 7):  # BODY or OBJECT
                    body_placeholders.append(placeholder)
            
            if not body_placeholders:
                logger.warning("No content placeholder found, looking for any text placeholder")
                # Fallback: look for any placeholder we can use
                for placeholder in slide.placeholders:
                    if hasattr(placeholder, 'text_frame'):
                        body_placeholders.append(placeholder)
                        break
            
            if not body_placeholders:
                logger.warning("No suitable placeholder found for hierarchical bullet content")
                return
            
            # For hierarchical bullets, use single placeholder to maintain structure
            # Multi-column distribution would break hierarchy, so we use the first placeholder
            self._add_hierarchical_bullets_to_placeholder(body_placeholders[0], bullet_items, language)
            logger.debug(f"Added {len(bullet_items)} hierarchical bullets to slide")
                
        except Exception as e:
            logger.error(f"Failed to add hierarchical bullet content: {e}")

    def _distribute_bullets_across_columns(
        self, 
        body_placeholders: List, 
        bullets: List[str], 
        language: str
    ) -> None:
        """
        Distribute bullets evenly across multiple BODY placeholders (T-94).
        
        Args:
            body_placeholders: List of BODY placeholder objects
            bullets: List of bullet text strings
            language: Language code for text formatting
        """
        num_columns = min(len(body_placeholders), 2)  # Limit to 2 columns for now
        bullets_per_column = len(bullets) // num_columns
        extra_bullets = len(bullets) % num_columns
        
        start_idx = 0
        for col_idx in range(num_columns):
            # Distribute extra bullets to first columns
            bullets_in_this_column = bullets_per_column + (1 if col_idx < extra_bullets else 0)
            end_idx = start_idx + bullets_in_this_column
            
            column_bullets = bullets[start_idx:end_idx]
            
            if column_bullets:  # Only add if there are bullets for this column
                self._add_bullets_to_placeholder(body_placeholders[col_idx], column_bullets, language)
                logger.debug(f"Added {len(column_bullets)} bullets to column {col_idx + 1}")
            
            start_idx = end_idx
        
        logger.info(f"Distributed {len(bullets)} bullets across {num_columns} columns")

    def _add_bullets_to_placeholder(
        self, 
        placeholder, 
        bullets: List[str], 
        language: str
    ) -> None:
        """
        Add bullets to a specific placeholder (legacy format).
        
        Args:
            placeholder: Placeholder object to add bullets to
            bullets: List of bullet text strings
            language: Language code for text formatting
        """
        # Convert to BulletItem format for unified handling
        bullet_items = [BulletItem(text=bullet, level=0) for bullet in bullets]
        self._add_hierarchical_bullets_to_placeholder(placeholder, bullet_items, language)

    def _add_hierarchical_bullets_to_placeholder(
        self, 
        placeholder, 
        bullet_items: List[BulletItem], 
        language: str
    ) -> None:
        """
        Add hierarchical bullets to a specific placeholder (T-100).
        
        Args:
            placeholder: Placeholder object to add bullets to
            bullet_items: List of BulletItem objects with hierarchy information
            language: Language code for text formatting
        """
        if not hasattr(placeholder, 'text_frame'):
            logger.warning("Placeholder does not have text_frame attribute")
            return
        
        text_frame = placeholder.text_frame
        text_frame.clear()  # Clear existing content
        
        # Add bullets with hierarchy support
        for i, bullet_item in enumerate(bullet_items):
            bullet_text = bullet_item.text
            bullet_level = bullet_item.level
            
            # Truncate if too long
            if len(bullet_text) > self.max_bullet_length:
                bullet_text = bullet_text[:self.max_bullet_length - 3] + "..."
            
            if i == 0:
                # Use the first paragraph
                p = text_frame.paragraphs[0]
            else:
                # Add new paragraphs for subsequent bullets
                p = text_frame.add_paragraph()
            
            p.text = bullet_text
            p.level = min(bullet_level, 4)  # PowerPoint supports levels 0-4
            
            # Apply formatting (font size will be adjusted later if needed)
            if p.runs:
                run = p.runs[0]
                
                # Adjust font size based on hierarchy level (T-100)
                base_size = 18
                size_reduction = bullet_level * 2  # Reduce by 2pt per level
                adjusted_size = max(12, base_size - size_reduction)  # Minimum 12pt
                run.font.size = Pt(adjusted_size)

                # Proactive font selection for bullets
                specific_font_name = self.template_parser.template_style.language_specific_fonts.get(language.lower())
                if specific_font_name:
                    run.font.name = specific_font_name
                    logger.debug(f"Applied language-specific font '{specific_font_name}' for bullets in language '{language.lower()}'")

                logger.debug(f"Applied bullet level {bullet_level} with font size {adjusted_size}pt")

        # RTL alignment for bullets
        if language.lower() in RTL_LANGUAGES:
            for para in text_frame.paragraphs:
                if para.text and para.text.strip(): # Only align if paragraph has text
                    para.alignment = PP_ALIGN.RIGHT

    def _add_chart_image(self, slide, chart_path: str, alt_text: Optional[str] = None) -> None:
        """Add chart image to slide."""
        try:
            chart_path = Path(chart_path)
            if not chart_path.exists():
                logger.error(f"Chart file not found: {chart_path}")
                return
            
            # Try to find picture placeholder first
            picture_placeholder = None
            for placeholder in slide.placeholders:
                if placeholder.placeholder_format.type == 18:  # PICTURE
                    picture_placeholder = placeholder
                    break
            
            if picture_placeholder:
                # Use placeholder
                picture_shape = picture_placeholder.insert_picture(str(chart_path))
                if alt_text:
                    picture_shape.name = alt_text
                logger.debug(f"Inserted chart into picture placeholder: {chart_path}")
            else:
                # Add as free-floating image
                # Position in bottom right area
                slide_width = slide.part.presentation.slide_width
                slide_height = slide.part.presentation.slide_height
                
                img_width = Inches(4)
                img_height = Inches(3)
                left = slide_width - img_width - Inches(0.5)
                top = slide_height - img_height - Inches(0.5)
                
                picture_shape = slide.shapes.add_picture(str(chart_path), left, top, img_width, img_height)
                if alt_text:
                    picture_shape.name = alt_text
                logger.debug(f"Added chart as floating image: {chart_path}")
                
        except Exception as e:
            logger.error(f"Failed to add chart image '{chart_path}': {e}")

    def _add_image(self, slide, image_path: str, alt_text: Optional[str] = None) -> None:
        """Add image to slide."""
        try:
            image_path = Path(image_path)
            if not image_path.exists():
                logger.error(f"Image file not found: {image_path}")
                return
            
            # Try to find picture placeholder first
            picture_placeholder = None
            for placeholder in slide.placeholders:
                if placeholder.placeholder_format.type == 18:  # PICTURE
                    picture_placeholder = placeholder
                    break
            
            if picture_placeholder:
                # Use placeholder
                picture_shape = picture_placeholder.insert_picture(str(image_path))
                if alt_text:
                    picture_shape.name = alt_text
                logger.debug(f"Inserted image into picture placeholder: {image_path}")
            else:
                # Add as free-floating image
                # Position in upper right area if no chart is present
                slide_width = slide.part.presentation.slide_width
                
                img_width = Inches(3)
                img_height = Inches(2)
                left = slide_width - img_width - Inches(0.5)
                top = Inches(1)
                
                picture_shape = slide.shapes.add_picture(str(image_path), left, top, img_width, img_height)
                if alt_text:
                    picture_shape.name = alt_text
                logger.debug(f"Added image as floating image: {image_path}")
                
        except Exception as e:
            logger.error(f"Failed to add image '{image_path}': {e}")

    def _add_native_chart(self, slide, chart_data) -> None:
        """Add native PowerPoint chart to slide."""
        try:
            # Import here to avoid circular imports
            from .native_chart_builder import NativeChartBuilder
            
            # Convert chart data to NativeChartData if needed
            if isinstance(chart_data, dict):
                chart_builder = NativeChartBuilder(self.template_parser)
                native_chart = chart_builder.convert_legacy_chart_data(chart_data)
                
                if native_chart:
                    # Try to use chart placeholder first
                    chart_placeholder = self._find_chart_placeholder(slide)
                    if chart_placeholder:
                        chart_builder.create_chart_in_placeholder(slide, native_chart, chart_placeholder)
                    else:
                        chart_builder.create_native_chart(slide, native_chart)
                    
                    logger.debug(f"Added native chart: {native_chart.title}")
                else:
                    logger.warning("Failed to convert chart data to native format")
            
            elif isinstance(chart_data, NativeChartData):
                chart_builder = NativeChartBuilder(self.template_parser)
                chart_builder.create_native_chart(slide, chart_data)
                logger.debug(f"Added native chart: {chart_data.title}")
            
            else:
                logger.warning(f"Unsupported chart data type: {type(chart_data)}")
                
        except Exception as e:
            logger.error(f"Failed to add native chart: {e}")

    def _add_process_flow(self, slide, flow_path: str, alt_text: Optional[str] = None) -> None:
        """Add process flow SVG to slide."""
        try:
            flow_path = Path(flow_path)
            if not flow_path.exists():
                logger.error(f"Process flow file not found: {flow_path}")
                return
            
            # Convert SVG to image for insertion
            # For now, just add as image - could be enhanced to embed SVG directly
            self._add_image(slide, flow_path, alt_text=alt_text) # Pass alt_text
            logger.debug(f"Added process flow: {flow_path}")
            
        except Exception as e:
            logger.error(f"Failed to add process flow '{flow_path}': {e}")

    def _find_chart_placeholder(self, slide) -> Optional[object]:
        """Find a chart placeholder on the slide."""
        try:
            for placeholder in slide.placeholders:
                if (hasattr(placeholder, 'placeholder_format') and 
                    placeholder.placeholder_format.type == 12):  # CHART placeholder
                    return placeholder
                
                # Check for content placeholder that can hold charts
                if (hasattr(placeholder, 'placeholder_format') and 
                    placeholder.placeholder_format.type in [7]):  # OBJECT
                    return placeholder
            
        except Exception as e:
            logger.debug(f"Error finding chart placeholder: {e}")
        
        return None

    def _add_speaker_notes(self, slide, notes: str) -> None:
        """Add speaker notes to slide."""
        try:
            if hasattr(slide, 'notes_slide'):
                notes_slide = slide.notes_slide
                if hasattr(notes_slide, 'notes_text_frame'):
                    text_frame = notes_slide.notes_text_frame
                    text_frame.text = notes
                    logger.debug(f"Added speaker notes: {notes[:50]}...")
                    
        except Exception as e:
            logger.error(f"Failed to add speaker notes: {e}")

    def _add_fallback_slide(self, prs: Presentation, slide_plan: SlidePlan) -> None:
        """Add a basic fallback slide when normal slide creation fails."""
        try:
            # Use the first available layout
            layout = prs.slide_layouts[0]
            slide = prs.slides.add_slide(layout)
            
            # Just add the title
            if hasattr(slide.shapes, 'title'):
                slide.shapes.title.text = slide_plan.title or f"Slide {slide_plan.index + 1}"
            
            logger.warning(f"Added fallback slide for slide {slide_plan.index}")
            
        except Exception as e:
            logger.error(f"Failed to add fallback slide: {e}")

    def _apply_metadata(self, prs: Presentation, outline: Outline) -> None:
        """Apply presentation metadata."""
        try:
            # Set presentation properties
            if hasattr(prs.core_properties, 'title'):
                prs.core_properties.title = outline.title
            
            if hasattr(prs.core_properties, 'subject') and outline.subtitle:
                prs.core_properties.subject = outline.subtitle
            
            if hasattr(prs.core_properties, 'author'):
                prs.core_properties.author = "Open Lilli AI"
            
            logger.debug("Applied presentation metadata")
            
        except Exception as e:
            logger.error(f"Failed to apply metadata: {e}")

    def create_slide_from_layout(
        self,
        prs: Presentation,
        layout_name: str,
        title: str,
            content: Optional[List[str]] = None,
            language: str = "en" # Add language, default to "en" for existing calls
    ) -> None:
        """
        Create a slide using a specific layout by name.
        
        Args:
            prs: Presentation object
            layout_name: Name of the layout to use
            title: Slide title
            content: Optional list of content items
            language: Language code for text alignment
        """
        try:
            layout_index = self.template_parser.get_layout_index(layout_name)
            layout = prs.slide_layouts[layout_index]
            slide = prs.slides.add_slide(layout)
            
            # Add title
            # Assuming this internal method might not need language if it's simple title setting
            # or if it calls the main _add_title, it should pass language.
            # For now, let's assume it's a simple set, or we need to trace its callers.
            # Let's assume _add_title needs language.
            if hasattr(slide.shapes, 'title'):
                 self._add_title(slide, title, language) # Pass language
            
            # Add content if provided
            if content:
                self._add_bullet_content(slide, content, language) # Pass language
            
            logger.debug(f"Created slide with layout '{layout_name}': {title}")
            
        except Exception as e:
            logger.error(f"Failed to create slide with layout '{layout_name}': {e}")

    def analyze_slide_placeholders(self, slide) -> Dict[str, any]:
        """
        Analyze placeholders in a slide for debugging.
        
        Args:
            slide: Slide object to analyze
            
        Returns:
            Dictionary with placeholder information
        """
        placeholders_info = {
            "total_placeholders": len(slide.placeholders),
            "placeholders": []
        }
        
        for i, placeholder in enumerate(slide.placeholders):
            ph_info = {
                "index": i,
                "type": placeholder.placeholder_format.type,
                "has_text_frame": hasattr(placeholder, 'text_frame'),
                "shape_type": placeholder.shape_type if hasattr(placeholder, 'shape_type') else None
            }
            
            # Try to get placeholder name
            try:
                ph_info["name"] = placeholder.name
            except:
                ph_info["name"] = None
            
            placeholders_info["placeholders"].append(ph_info)
        
        return placeholders_info

    def get_assembly_statistics(
        self,
        slides: List[SlidePlan],
        visuals: Dict[int, Dict[str, str]]
    ) -> Dict[str, any]:
        """
        Get statistics about the assembly process.
        
        Args:
            slides: List of slides to assemble
            visuals: Dictionary of visuals
            
        Returns:
            Assembly statistics
        """
        stats = {
            "total_slides": len(slides),
            "slides_with_visuals": len([s for s in slides if s.index in visuals]),
            "total_bullets": sum(len(s.bullets) for s in slides),
            "slides_with_notes": len([s for s in slides if s.speaker_notes]),
            "slide_types": {},
            "layout_usage": {},
            "visual_types": {"charts": 0, "images": 0}
        }
        
        # Count slide types and layout usage
        for slide in slides:
            stats["slide_types"][slide.slide_type] = stats["slide_types"].get(slide.slide_type, 0) + 1
            
            layout_id = slide.layout_id or 0
            stats["layout_usage"][layout_id] = stats["layout_usage"].get(layout_id, 0) + 1
        
        # Count visual types
        for slide_visuals in visuals.values():
            if "chart" in slide_visuals:
                stats["visual_types"]["charts"] += 1
            if "image" in slide_visuals:
                stats["visual_types"]["images"] += 1
        
        return stats

    def validate_slides_before_assembly(self, slides: List[SlidePlan]) -> List[str]:
        """
        Validate slides before assembly and return list of issues.
        
        Args:
            slides: List of slides to validate
            
        Returns:
            List of validation issues
        """
        issues = []
        
        if not slides:
            issues.append("No slides provided")
            return issues
        
        # Check for missing titles
        for slide in slides:
            if not slide.title or not slide.title.strip():
                issues.append(f"Slide {slide.index} has no title")
        
        # Check for layout IDs
        max_layout_index = len(self.template_parser.prs.slide_layouts) - 1
        for slide in slides:
            if slide.layout_id is not None and slide.layout_id > max_layout_index:
                issues.append(f"Slide {slide.index} has invalid layout ID: {slide.layout_id}")
        
        # Check for excessive content
        for slide in slides:
            if len(slide.bullets) > 10:
                issues.append(f"Slide {slide.index} has too many bullets: {len(slide.bullets)}")
            
            if slide.title and len(slide.title) > self.max_title_length:
                issues.append(f"Slide {slide.index} title is too long: {len(slide.title)} chars")
        
        return issues

    def validate_presentation_style(self, prs: Presentation, language: Optional[str] = "en") -> None: # Add language
        """
        Validate the entire presentation against template style rules.
        
        Args:
            prs: Presentation object to validate
            language: Language code for the presentation (for font validation)
            
        Raises:
            StyleError: If style violations are found and validation is enabled
        """
        # Ensure language is available, default to "en" if not passed (e.g. from patch_existing_presentation if not updated)
        # However, assemble method should always pass it from outline.language.
        # For patch_existing_presentation, it's now a required parameter.
        # So, language should always be valid.

        if not self.validation_config.enabled or self.validation_config.mode == "disabled":
            logger.debug("Style validation disabled")
            return
        
        logger.info(f"Starting presentation style validation for language: {language}")
        violations = []
        
        for slide_index, slide in enumerate(prs.slides):
            try:
                slide_violations = self._validate_slide_style(slide, slide_index, language) # Pass language
                violations.extend(slide_violations)
            except Exception as e:
                logger.warning(f"Failed to validate slide {slide_index}: {e}")
                if self.validation_config.mode == "strict":
                    violations.append({
                        'type': 'validation_error',
                        'description': f"Failed to validate slide {slide_index}: {e}",
                        'slide_index': slide_index
                    })
        
        if violations:
            if self.validation_config.mode == "strict":
                error_msg = f"Style validation failed with {len(violations)} violations"
                style_error = StyleError(error_msg, violations=violations)
                logger.error(f"Style validation failed: {style_error}")
                raise style_error
            elif self.validation_config.mode == "lenient":
                logger.warning(f"Style validation found {len(violations)} violations (lenient mode)")
                for violation in violations[:5]:  # Log first 5 violations
                    logger.warning(f"  {violation.get('type', 'Unknown')}: {violation.get('description', 'No description')}")
                if len(violations) > 5:
                    logger.warning(f"  ... and {len(violations) - 5} more violations")
        else:
            logger.info("Style validation passed successfully")

    def _validate_slide_style(self, slide, slide_index: int, language: str) -> List[Dict[str, any]]: # Add language
        """
        Validate style for a single slide.
        
        Args:
            slide: Slide object to validate
            slide_index: Index of the slide
            language: Language code for the presentation
            
        Returns:
            List of violation dictionaries
        """
        violations = []
        
        for shape in slide.shapes:
            try:
                shape_violations = self._validate_shape_style(shape, slide, slide_index, language) # Pass language and slide
                violations.extend(shape_violations)
            except Exception as e:
                logger.debug(f"Failed to validate shape in slide {slide_index}: {e}")
                continue
        
        # Add placeholder population check
        population_violations = self._check_slide_placeholder_population(slide, slide_index)
        violations.extend(population_violations)

        return violations

    def _autofix_shape_alignment(self, shape, slide, slide_index: int) -> bool:
        """
        Attempts to automatically fix shape alignment issues against slide margins.
        Args:
            shape: The shape with alignment violations.
            slide: The slide object.
            slide_index: Index of the slide.
        Returns:
            True if any fix was attempted, False otherwise.
            Note: This simple version doesn't guarantee all violations are fixed,
                  it just attempts common corrections. The caller should re-check.
        """
        # autofix_alignment is checked by the caller (_check_shape_alignment_against_margins)
        # before this method is called.

        qg_config = self.validation_config.quality_gates_config
        # qg_config and qg_config.enable_alignment_check are also checked by caller.

        slide_width_emu = slide.parent.slide_width
        slide_height_emu = slide.parent.slide_height
        margin_top_emu = Inches(qg_config.slide_margin_top_inches)
        margin_bottom_emu = Inches(qg_config.slide_margin_bottom_inches)
        margin_left_emu = Inches(qg_config.slide_margin_left_inches)
        margin_right_emu = Inches(qg_config.slide_margin_right_inches)

        safe_left = margin_left_emu
        safe_top = margin_top_emu
        safe_right = slide_width_emu - margin_right_emu
        safe_bottom = slide_height_emu - margin_bottom_emu

        shape_name_desc = shape.name if shape.name else f"Shape ID {shape.shape_id}"
        if hasattr(shape, 'placeholder_format') and hasattr(shape.placeholder_format, 'type'):
            try:
                ph_type = shape.placeholder_format.type
                ph_type_name = self._get_placeholder_type_name(ph_type.value if hasattr(ph_type, 'value') else int(ph_type))
                shape_name_desc = f"{ph_type_name} (Idx: {shape.placeholder_format.idx}, Name: {shape.name or 'Unnamed'})"
            except Exception: pass

        action_taken = False

        # Fix left margin violations by moving
        if shape.left < safe_left:
            original_left = shape.left
            shape.left = safe_left
            logger.info(f"Autofix: Moved shape '{shape_name_desc}' on slide {slide_index} from L:{original_left:.0f} to L:{shape.left:.0f} (due to left margin).")
            action_taken = True

        # Fix top margin violations by moving
        if shape.top < safe_top:
            original_top = shape.top
            shape.top = safe_top
            logger.info(f"Autofix: Moved shape '{shape_name_desc}' on slide {slide_index} from T:{original_top:.0f} to T:{shape.top:.0f} (due to top margin).")
            action_taken = True

        # Fix right margin violations (try moving first, then resizing width)
        if (shape.left + shape.width) > safe_right:
            overflow_amount = (shape.left + shape.width) - safe_right

            new_left_candidate = shape.left - overflow_amount
            # Ensure shape has a positive width before attempting to move/resize
            if shape.width > 0 : # Check current width is positive
                if new_left_candidate >= safe_left :
                    shape.left = new_left_candidate
                    logger.info(f"Autofix: Moved shape '{shape_name_desc}' on slide {slide_index} left to fit right margin. New L:{shape.left:.0f}, R_Edge:{(shape.left + shape.width):.0f}.")
                    action_taken = True
                else:
                    new_width = shape.width - overflow_amount
                    if new_width > Inches(0.1):
                        shape.width = new_width
                        logger.info(f"Autofix: Resized width of shape '{shape_name_desc}' on slide {slide_index} to fit right margin. New W:{shape.width:.0f}, R_Edge:{(shape.left + shape.width):.0f}.")
                        action_taken = True
                    else:
                        logger.warning(f"Autofix: Failed to fit shape '{shape_name_desc}' on slide {slide_index} within right margin without making width too small.")
            else:
                 logger.warning(f"Autofix: Shape '{shape_name_desc}' on slide {slide_index} has zero or negative width, cannot fix right margin.")


        # Fix bottom margin violations (try moving first, then resizing height)
        if (shape.top + shape.height) > safe_bottom:
            overflow_amount = (shape.top + shape.height) - safe_bottom

            new_top_candidate = shape.top - overflow_amount
            if shape.height > 0 : # Check current height
                if new_top_candidate >= safe_top:
                    shape.top = new_top_candidate
                    logger.info(f"Autofix: Moved shape '{shape_name_desc}' on slide {slide_index} up to fit bottom margin. New T:{shape.top:.0f}, B_Edge:{(shape.top + shape.height):.0f}.")
                    action_taken = True
                else:
                    new_height = shape.height - overflow_amount
                    if new_height > Inches(0.1):
                        shape.height = new_height
                        logger.info(f"Autofix: Resized height of shape '{shape_name_desc}' on slide {slide_index} to fit bottom margin. New H:{shape.height:.0f}, B_Edge:{(shape.top + shape.height):.0f}.")
                        action_taken = True
                    else:
                        logger.warning(f"Autofix: Failed to fit shape '{shape_name_desc}' on slide {slide_index} within bottom margin without making height too small.")
            else:
                logger.warning(f"Autofix: Shape '{shape_name_desc}' on slide {slide_index} has zero or negative height, cannot fix bottom margin.")

        return action_taken

    def _check_shape_alignment_against_margins(self, shape, slide, slide_index: int, language: str) -> List[Dict[str, any]]:
        """
        Checks if a shape is within the defined slide margins.
        Args:
            shape: The shape to check.
            slide: The slide object the shape belongs to.
            slide_index: Index of the slide.
            language: Language (passed for consistency, not directly used in this check).
        Returns:
            A list of alignment violation dictionaries.
        """
        # Initial check for violations
        initial_violations = []
        if not self.validation_config.check_alignment:
            return initial_violations

        qg_config = self.validation_config.quality_gates_config
        if not qg_config or not qg_config.enable_alignment_check:
            logger.debug("Alignment check skipped as it's not enabled in QualityGatesConfig or QualityGatesConfig is missing.")
            return initial_violations

        if not slide.parent or not hasattr(slide.parent, 'slide_width') or not hasattr(slide.parent, 'slide_height'):
            logger.warning("Cannot access presentation-level slide dimensions for alignment check. Skipping.")
            return initial_violations

        slide_width_emu = slide.parent.slide_width
        slide_height_emu = slide.parent.slide_height
        margin_top_emu = Inches(qg_config.slide_margin_top_inches)
        margin_bottom_emu = Inches(qg_config.slide_margin_bottom_inches)
        margin_left_emu = Inches(qg_config.slide_margin_left_inches)
        margin_right_emu = Inches(qg_config.slide_margin_right_inches)

        safe_left = margin_left_emu
        safe_top = margin_top_emu
        safe_right = slide_width_emu - margin_right_emu
        safe_bottom = slide_height_emu - margin_bottom_emu

        shape_name_desc_orig = shape.name if shape.name else f"Shape ID {shape.shape_id}" # Keep original for logs if autofix changes name (unlikely)
        current_shape_name_desc = shape_name_desc_orig
        if hasattr(shape, 'placeholder_format') and hasattr(shape.placeholder_format, 'type'):
            try:
                ph_type = shape.placeholder_format.type
                ph_type_name = self._get_placeholder_type_name(ph_type.value if hasattr(ph_type, 'value') else int(ph_type))
                current_shape_name_desc = f"{ph_type_name} (Idx: {shape.placeholder_format.idx}, Name: {shape.name or 'Unnamed'})"
            except Exception:
                pass

        if shape.width == 0 or shape.height == 0:
            logger.debug(f"Shape '{current_shape_name_desc}' has zero width or height, skipping alignment check.")
            return initial_violations

        # Perform initial violation detection
        if shape.left < safe_left:
            initial_violations.append({
                'type': 'alignment_violation', 'subtype': 'left_margin',
                'description': f"Shape '{current_shape_name_desc}' extends beyond the left slide margin.",
                'slide_index': slide_index, 'shape_name': current_shape_name_desc,
                'details': f"Shape left: {shape.left:.0f} EMU, Safe left: {safe_left:.0f} EMU."})
        if shape.top < safe_top:
            initial_violations.append({
                'type': 'alignment_violation', 'subtype': 'top_margin',
                'description': f"Shape '{current_shape_name_desc}' extends beyond the top slide margin.",
                'slide_index': slide_index, 'shape_name': current_shape_name_desc,
                'details': f"Shape top: {shape.top:.0f} EMU, Safe top: {safe_top:.0f} EMU."})
        if (shape.left + shape.width) > safe_right:
            initial_violations.append({
                'type': 'alignment_violation', 'subtype': 'right_margin',
                'description': f"Shape '{current_shape_name_desc}' extends beyond the right slide margin.",
                'slide_index': slide_index, 'shape_name': current_shape_name_desc,
                'details': f"Shape right edge: {(shape.left + shape.width):.0f} EMU, Safe right: {safe_right:.0f} EMU."})
        if (shape.top + shape.height) > safe_bottom:
            initial_violations.append({
                'type': 'alignment_violation', 'subtype': 'bottom_margin',
                'description': f"Shape '{current_shape_name_desc}' extends beyond the bottom slide margin.",
                'slide_index': slide_index, 'shape_name': current_shape_name_desc,
                'details': f"Shape bottom edge: {(shape.top + shape.height):.0f} EMU, Safe bottom: {safe_bottom:.0f} EMU."})

        # Attempt autofix if violations are found and autofix is enabled
        if initial_violations and self.validation_config.autofix_alignment:
            logger.debug(f"Alignment violations found for '{current_shape_name_desc}' on slide {slide_index}. Attempting autofix.")
            action_taken = self._autofix_shape_alignment(shape, slide, slide_index) # Pass only necessary args

            if action_taken:
                logger.info(f"Autofix attempted for '{current_shape_name_desc}' on slide {slide_index}. Re-checking alignment.")
                # Re-check violations on the potentially modified shape
                final_violations = []
                # Update shape_name_desc in case autofix changed something (though unlikely for current autofix)
                current_shape_name_desc_after_fix = shape.name if shape.name else f"Shape ID {shape.shape_id}"
                if hasattr(shape, 'placeholder_format') and hasattr(shape.placeholder_format, 'type'):
                    try:
                        ph_type = shape.placeholder_format.type
                        ph_type_name = self._get_placeholder_type_name(ph_type.value if hasattr(ph_type, 'value') else int(ph_type))
                        current_shape_name_desc_after_fix = f"{ph_type_name} (Idx: {shape.placeholder_format.idx}, Name: {shape.name or 'Unnamed'})"
                    except Exception: pass

                if shape.left < safe_left:
                    final_violations.append({
                        'type': 'alignment_violation', 'subtype': 'left_margin',
                        'description': f"Shape '{current_shape_name_desc_after_fix}' still extends beyond the left slide margin after autofix.",
                        'slide_index': slide_index, 'shape_name': current_shape_name_desc_after_fix,
                        'details': f"Shape left: {shape.left:.0f} EMU, Safe left: {safe_left:.0f} EMU."})
                if shape.top < safe_top:
                    final_violations.append({
                        'type': 'alignment_violation', 'subtype': 'top_margin',
                        'description': f"Shape '{current_shape_name_desc_after_fix}' still extends beyond the top slide margin after autofix.",
                        'slide_index': slide_index, 'shape_name': current_shape_name_desc_after_fix,
                        'details': f"Shape top: {shape.top:.0f} EMU, Safe top: {safe_top:.0f} EMU."})
                if (shape.left + shape.width) > safe_right:
                    final_violations.append({
                        'type': 'alignment_violation', 'subtype': 'right_margin',
                        'description': f"Shape '{current_shape_name_desc_after_fix}' still extends beyond the right slide margin after autofix.",
                        'slide_index': slide_index, 'shape_name': current_shape_name_desc_after_fix,
                        'details': f"Shape right edge: {(shape.left + shape.width):.0f} EMU, Safe right: {safe_right:.0f} EMU."})
                if (shape.top + shape.height) > safe_bottom:
                    final_violations.append({
                        'type': 'alignment_violation', 'subtype': 'bottom_margin',
                        'description': f"Shape '{current_shape_name_desc_after_fix}' still extends beyond the bottom slide margin after autofix.",
                        'slide_index': slide_index, 'shape_name': current_shape_name_desc_after_fix,
                        'details': f"Shape bottom edge: {(shape.top + shape.height):.0f} EMU, Safe bottom: {safe_bottom:.0f} EMU."})

                if not final_violations:
                    logger.info(f"All alignment issues appear fixed for '{current_shape_name_desc_after_fix}' on slide {slide_index}.")
                else:
                    logger.warning(f"Autofix did not resolve all alignment issues for '{current_shape_name_desc_after_fix}' on slide {slide_index}.")
                return final_violations
            else: # Autofix was enabled but took no action (e.g. shape already too small)
                logger.debug(f"Autofix enabled but no specific action taken for '{current_shape_name_desc}' on slide {slide_index}. Reporting original violations.")
                return initial_violations

        return initial_violations # Return initial violations if no autofix or autofix disabled

    def _estimate_text_lines(self, text_frame, shape_width_emu) -> int:
        """
        Estimates the number of lines text will occupy in a text_frame.
        This is a simplified heuristic.
        Args:
            text_frame: The TextFrame object.
            shape_width_emu: The width of the shape in EMUs. (Currently not fully utilized)
        Returns:
            Estimated number of lines.
        """
        if not text_frame.text.strip():
            return 0

        # This would ideally come from ContentFitConfig or similar
        avg_chars_per_line_estimate = 50

        text = text_frame.text

        # Estimate lines based on text length divided by average chars per line for each segment split by newline
        estimated_lines_from_length = 0
        if avg_chars_per_line_estimate > 0:
            estimated_lines_from_length = sum(
                # Calculate lines for each segment: (length + N-1) // N for ceiling division
                (len(line_segment) + avg_chars_per_line_estimate - 1) // avg_chars_per_line_estimate
                for line_segment in text.split('\n')
            )
        else: # Avoid division by zero if estimate is bad
            estimated_lines_from_length = text.count('\n') + 1 # Fallback: count explicit newlines + 1

        # If text is "Line1\nLine2", split produces ["Line1", "Line2"].
        # Loop runs twice. len("Line1")//50 (say 1), len("Line2")//50 (say 1). Sum = 2.
        # If text is "Very long line without newlines...", loop runs once.
        # This calculation correctly sums lines from wrapped segments and adds them up.

        return estimated_lines_from_length

    def _autofix_shape_text_overflow(self, shape, text_frame, slide_index: int, language: str) -> bool:
        """
        Attempts to automatically fix text overflow in a shape by reducing font size.
        Args:
            shape: The shape with overflow.
            text_frame: The TextFrame of the shape.
            slide_index: Index of the slide.
            language: Language (currently unused here but good for consistency).
        Returns:
            True if overflow was fixed, False otherwise.
        """
        # This method is called when self.validation_config.autofix_text_overflow is already true.

        logger.debug(f"Attempting to autofix text overflow in shape '{shape.name or shape.shape_id}' on slide {slide_index}")

        min_font_size_pt = 10  # Fallback, ideally from ContentFitConfig.min_font_size
        font_shrink_step_pt = 1 # How much to shrink font by each iteration
        max_fix_attempts = 5    # Max iterations for font shrinking

        original_auto_size = text_frame.auto_size
        if text_frame.auto_size != MSO_AUTO_SIZE.NONE: # Only change if not already NONE
            text_frame.auto_size = MSO_AUTO_SIZE.NONE

        fixed = False
        for attempt in range(max_fix_attempts):
            # Re-check overflow by directly evaluating conditions, not full _check_text_frame_overflow
            # This avoids recursive calls and simplifies logic here.
            shape_height_emu = shape.height
            margin_top_emu = text_frame.margin_top if text_frame.margin_top is not None else 0
            margin_bottom_emu = text_frame.margin_bottom if text_frame.margin_bottom is not None else 0
            available_text_height_emu = shape_height_emu - margin_top_emu - margin_bottom_emu

            estimated_lines = self._estimate_text_lines(text_frame, shape.width)
            if estimated_lines == 0 : # No text or bad estimate
                 fixed = True # Assume fixed if no lines/text
                 break

            avg_font_size_pt = 18.0
            try:
                if text_frame.paragraphs and text_frame.paragraphs[0].runs:
                    first_run_font_size = text_frame.paragraphs[0].runs[0].font.size
                    if first_run_font_size:
                        avg_font_size_pt = float(first_run_font_size.pt)
            except (AttributeError, IndexError): pass

            estimated_line_height_emu = avg_font_size_pt * 12700 * 1.2
            if estimated_line_height_emu == 0: estimated_line_height_emu = 12700 * 1.2
            required_height_emu = estimated_lines * estimated_line_height_emu

            if required_height_emu <= available_text_height_emu:
                fixed = True
                logger.info(f"Overflow fixed for shape '{shape.name or shape.shape_id}' on slide {slide_index} by font size reduction (attempt {attempt+1}).")
                break

            # If still overflowing, try to reduce font size further
            fonts_shrunk_in_step = False
            for para in text_frame.paragraphs:
                for run in para.runs:
                    if run.font.size:
                        current_size_pt = run.font.size.pt
                        if current_size_pt > min_font_size_pt:
                            new_size_pt = max(min_font_size_pt, current_size_pt - font_shrink_step_pt)
                            if new_size_pt < current_size_pt:
                                run.font.size = Pt(new_size_pt)
                                fonts_shrunk_in_step = True

            if not fonts_shrunk_in_step:
                logger.debug(f"Could not shrink fonts further for shape '{shape.name or shape.shape_id}' on slide {slide_index}.")
                break

        if original_auto_size is not None and text_frame.auto_size != original_auto_size :
            text_frame.auto_size = original_auto_size

        if fixed:
            return True
        else:
            is_title_placeholder = False
            if hasattr(shape, 'placeholder_format') and shape.placeholder_format.type:
                 try:
                    is_title_placeholder = shape.placeholder_format.type == PP_PLACEHOLDER.TITLE
                 except AttributeError: # .type might not be an enum directly if it's an unknown int
                    pass


            if is_title_placeholder and len(text_frame.text) > self.max_title_length :
                 original_text = text_frame.text
                 # Basic truncation
                 # Find the last space within max_title_length - 3 for cleaner cut
                 safe_truncate_pos = (original_text[:self.max_title_length - 3]).rfind(' ')
                 if safe_truncate_pos == -1 or safe_truncate_pos < self.max_title_length // 2 : # if no space or too short
                     truncated_text = original_text[:self.max_title_length - 3] + "..."
                 else:
                     truncated_text = original_text[:safe_truncate_pos] + "..."

                 # Preserve first paragraph's formatting for the new truncated text
                 first_para_font = None
                 if text_frame.paragraphs and text_frame.paragraphs[0].runs:
                     first_run = text_frame.paragraphs[0].runs[0]
                     first_para_font = {
                         "name": first_run.font.name,
                         "size": first_run.font.size,
                         "bold": first_run.font.bold,
                         "italic": first_run.font.italic,
                         "color": first_run.font.color.rgb if first_run.font.color and first_run.font.color.rgb else None
                     }

                 text_frame.clear()
                 p = text_frame.paragraphs[0] if text_frame.paragraphs else text_frame.add_paragraph()
                 run = p.add_run()
                 run.text = truncated_text

                 if first_para_font:
                     if first_para_font["name"]: run.font.name = first_para_font["name"]
                     if first_para_font["size"]: run.font.size = first_para_font["size"]
                     if first_para_font["bold"] is not None: run.font.bold = first_para_font["bold"]
                     if first_para_font["italic"] is not None: run.font.italic = first_para_font["italic"]
                     if first_para_font["color"]: run.font.color.rgb = first_para_font["color"]

                 logger.info(f"Overflow in title shape '{shape.name or shape.shape_id}' fixed by truncation.")
                 fixed = True

            if not fixed:
                 logger.warning(f"Failed to autofix text overflow for shape '{shape.name or shape.shape_id}' on slide {slide_index}.")
            return fixed

    def _check_text_frame_overflow(self, shape, slide_index: int, language: str) -> List[Dict[str, any]]:
        """
        Checks a single shape for text overflow.
        Args:
            shape: The shape to check.
            slide_index: Index of the slide.
            language: Language of the presentation.
        Returns:
            A list of violation dictionaries if overflow is detected.
        """
        violations = []
        if not self.validation_config.check_text_overflow or not hasattr(shape, 'text_frame') or not shape.text_frame:
            return violations

        text_frame = shape.text_frame
        # text_frame.auto_size can be:
        # MSO_AUTO_SIZE.NONE (0): No auto-sizing. Text can overflow. (This is what we want to detect)
        # MSO_AUTO_SIZE.SHAPE_TO_FIT_TEXT (1): Shape resizes to fit text. (Less likely to overflow visually)
        # MSO_AUTO_SIZE.TEXT_TO_FIT_SHAPE (2): Text font size shrinks to fit shape. (Overflow is 'hidden' by shrinking)

        # For this check, we are most interested in cases where auto_size is NONE or TEXT_TO_FIT_SHAPE
        # (where text might become too small, but that's a different check - here we estimate raw overflow).

        shape_height_emu = shape.height
        margin_top_emu = text_frame.margin_top if text_frame.margin_top is not None else 0 # Ensure EMUs
        margin_bottom_emu = text_frame.margin_bottom if text_frame.margin_bottom is not None else 0 # Ensure EMUs
        available_text_height_emu = shape_height_emu - margin_top_emu - margin_bottom_emu

        ph_type_name = "Shape" # Default
        placeholder_idx_str = f"Shape ID {shape.shape_id}"
        if hasattr(shape, 'placeholder_format') and shape.placeholder_format.type: # Check type before accessing value
            try:
                # Ensure placeholder_format.type is an enum before accessing .value
                current_ph_type = shape.placeholder_format.type
                if hasattr(current_ph_type, 'value'):
                    ph_type_name = self._get_placeholder_type_name(current_ph_type.value)
                else: # Is an int, direct use might be risky if not a known value
                    ph_type_name = self._get_placeholder_type_name(int(current_ph_type))
                placeholder_idx_str = f"Placeholder Type: {ph_type_name}, Index: {shape.placeholder_format.idx}"
            except Exception as e:
                logger.debug(f"Error getting placeholder type name for overflow check: {e}")
                ph_type_name = f"Placeholder Idx {shape.placeholder_format.idx}" # Fallback


        if available_text_height_emu <= 0:
            if text_frame.text.strip():
                violations.append({
                    'type': 'text_overflow',
                    'description': f"Shape '{shape.name if shape.name else placeholder_idx_str}' is too small for any text, but contains text.",
                    'slide_index': slide_index,
                    'shape_name': shape.name if shape.name else placeholder_idx_str,
                    'details': 'Shape height insufficient for margins.'
                })
            return violations

        estimated_lines = self._estimate_text_lines(text_frame, shape.width)

        if estimated_lines == 0 and not text_frame.text.strip():
            return violations # No text, no overflow

        if estimated_lines == 0 and text_frame.text.strip():
            logger.warning(f"Text overflow estimator returned 0 lines for non-empty text in shape {shape.name or placeholder_idx_str} on slide {slide_index}")
            # Potentially add a specific violation for estimator failure if desired
            return violations

        avg_font_size_pt = 18.0 # Default
        try:
            if text_frame.paragraphs and text_frame.paragraphs[0].runs:
                # Attempt to get font size from the first run of the first paragraph
                first_run_font_size = text_frame.paragraphs[0].runs[0].font.size
                if first_run_font_size: # Check if size is not None
                    avg_font_size_pt = float(first_run_font_size.pt)
        except (AttributeError, IndexError): # No size set, no runs, or no paragraphs
            pass # Use default

        # EMUs per point: 1 point = 12700 EMUs
        # Line height factor (e.g., 1.2 for 120% line spacing)
        estimated_line_height_emu = avg_font_size_pt * 12700 * 1.2
        if estimated_line_height_emu == 0:
             estimated_line_height_emu = 12700 * 1.2

        required_height_emu = estimated_lines * estimated_line_height_emu

        if required_height_emu > available_text_height_emu:
            if self.validation_config.autofix_text_overflow:
                was_fixed = self._autofix_shape_text_overflow(shape, text_frame, slide_index, language)
                if was_fixed:
                    logger.info(f"Text overflow in shape '{shape.name or placeholder_idx_str}' on slide {slide_index} was auto-fixed.")
                    # No violation is added if fixed
                else:
                    # Add violation if autofix failed
                    violations.append({
                        'type': 'text_overflow',
                        'description': f"Estimated text overflows '{shape.name if shape.name else placeholder_idx_str}' (autofix failed).",
                        'slide_index': slide_index,
                        'shape_name': shape.name if shape.name else placeholder_idx_str,
                        'estimated_lines': estimated_lines,
                        'available_height_emu': f"{available_text_height_emu:.0f}",
                        'required_height_emu': f"{required_height_emu:.0f}",
                        'placeholder_type_name': ph_type_name if hasattr(shape, 'placeholder_format') and shape.placeholder_format.type else "N/A"
                    })
            else:
                # Autofix disabled, just add violation
                violations.append({
                    'type': 'text_overflow',
                    'description': f"Estimated text overflows '{shape.name if shape.name else placeholder_idx_str}'.",
                    'slide_index': slide_index,
                    'shape_name': shape.name if shape.name else placeholder_idx_str,
                    'estimated_lines': estimated_lines,
                    'available_height_emu': f"{available_text_height_emu:.0f}",
                    'required_height_emu': f"{required_height_emu:.0f}",
                    'placeholder_type_name': ph_type_name if hasattr(shape, 'placeholder_format') and shape.placeholder_format.type else "N/A"
                })

        return violations

    def _check_slide_placeholder_population(self, slide, slide_index: int) -> List[Dict[str, any]]:
        violations = []
        if not self.validation_config.check_placeholder_population:
            return violations

        # Define which placeholder types are considered essential for population
        # Using PP_PLACEHOLDER enum values directly
        required_placeholder_types = {
            PP_PLACEHOLDER.TITLE.value: "Title",
            PP_PLACEHOLDER.BODY.value: "Body Content"
        }
        # PP_PLACEHOLDER.SUBTITLE (3) could also be considered, especially for title slides.

        has_populated_title = False
        # For body, we check if a body placeholder *exists* and is empty.
        # We don't want to penalize layouts that legitimately don't have a body placeholder.
        body_placeholder_exists = False
        body_placeholder_is_populated = False


        for ph in slide.placeholders:
            # Use ph.placeholder_format.type for standard placeholder types
            ph_type_val = ph.placeholder_format.type.value # .value to get the integer

            is_empty = True
            # placeholder_name = self._get_placeholder_type_name(ph_type_val) # For logging/reporting

            if hasattr(ph, 'text_frame') and ph.text_frame and ph.text_frame.text.strip():
                is_empty = False
            elif ph.shape_type == MSO_SHAPE_TYPE.PICTURE:
                # Check if it's a picture placeholder that has a picture
                # Accessing ph.image can raise an exception if no image is present.
                try:
                    if ph.image: # placeholder.image exists if a picture has been inserted
                        is_empty = False
                except ValueError: # Handles case where there's no image
                    pass
            # Add checks for other content types like charts, tables if necessary
            # e.g., if ph.has_chart or ph.has_table

            if ph_type_val == PP_PLACEHOLDER.TITLE.value: # Title placeholder
                if not is_empty:
                    has_populated_title = True
            elif ph_type_val == PP_PLACEHOLDER.BODY.value: # Body placeholder
                body_placeholder_exists = True
                if not is_empty:
                    body_placeholder_is_populated = True

        # After checking all placeholders, report violations
        if PP_PLACEHOLDER.TITLE.value in required_placeholder_types and not has_populated_title:
            violations.append({
                'type': 'missing_required_placeholder_content',
                'description': f"Required '{required_placeholder_types[PP_PLACEHOLDER.TITLE.value]}' placeholder is empty or missing.",
                'slide_index': slide_index,
                'placeholder_type_name': required_placeholder_types[PP_PLACEHOLDER.TITLE.value],
                'placeholder_idx': "N/A (slide-level check)"
            })

        if PP_PLACEHOLDER.BODY.value in required_placeholder_types and body_placeholder_exists and not body_placeholder_is_populated:
            violations.append({
                'type': 'empty_required_placeholder',
                'description': f"Required '{required_placeholder_types[PP_PLACEHOLDER.BODY.value]}' placeholder is present but empty.",
                'slide_index': slide_index,
                'placeholder_type_name': required_placeholder_types[PP_PLACEHOLDER.BODY.value],
                # Could try to get ph.placeholder_format.idx if needed, but might be complex here
                'placeholder_idx': "N/A (specific body placeholder idx could be found)"
            })

        return violations

    def _validate_shape_style(self, shape, slide, slide_index: int, language: str) -> List[Dict[str, any]]:
        """
        Validate style for a single shape.
        
        Args:
            shape: Shape object to validate
            slide: Slide object containing the shape
            slide_index: Index of the slide containing this shape
            language: Language code for the presentation
            
        Returns:
            List of violation dictionaries
        """
        violations = []
        
        # Alignment check (applies to all shapes)
        alignment_violations = self._check_shape_alignment_against_margins(shape, slide, slide_index, language)
        violations.extend(alignment_violations)

        # Only validate text-containing shapes for further text-related checks
        if not hasattr(shape, 'text_frame') or not shape.text_frame:
            return violations # Return here if no text frame, but alignment violations are kept
        
        text_frame = shape.text_frame
        
        # Get placeholder information if available
        placeholder_type = None
        if hasattr(shape, 'placeholder_format') and shape.placeholder_format: # Check if placeholder_format exists
            if hasattr(shape.placeholder_format, 'type'): # Check if type attribute exists
                 placeholder_type = shape.placeholder_format.type
        
        # Validate each paragraph
        for para_index, paragraph in enumerate(text_frame.paragraphs):
            para_violations = self._validate_paragraph_style(
                paragraph, placeholder_type, slide_index, para_index, language # Pass language
            )
            violations.extend(para_violations)
        
        # Check for empty placeholders
        if not self.validation_config.allow_empty_placeholders:
             # Ensure placeholder_type was successfully retrieved before using it
            if not text_frame.text.strip() and placeholder_type is not None:
                violations.append({
                    'type': 'empty_placeholder',
                    'description': f"Empty placeholder of type {self._get_placeholder_type_name(placeholder_type.value if hasattr(placeholder_type, 'value') else int(placeholder_type))}",
                    'slide_index': slide_index,
                    'placeholder_type': self._get_placeholder_type_name(placeholder_type.value if hasattr(placeholder_type, 'value') else int(placeholder_type))
                })
        
        # Check for text overflow
        # This check is done after paragraph/run validation as it's a shape-level property
        overflow_violations = self._check_text_frame_overflow(shape, slide_index, language) # Pass language
        violations.extend(overflow_violations)

        # NEW: Image Accessibility Check (runs for MSO_SHAPE_TYPE.PICTURE)
        if hasattr(shape, 'shape_type') and shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
            accessibility_violations = self._check_image_accessibility(shape, slide_index)
            violations.extend(accessibility_violations)

        return violations

    def _check_image_accessibility(self, shape, slide_index: int) -> List[Dict[str, any]]:
        """
        Checks if an image shape has meaningful alt text (stored in shape.name).
        Args:
            shape: The image shape to check (must be MSO_SHAPE_TYPE.PICTURE).
            slide_index: Index of the slide.
        Returns:
            A list of accessibility violation dictionaries.
        """
        violations = []
        if not self.validation_config.check_accessibility:
            return violations

        # Generic names that suggest missing meaningful alt text.
        # This list can be expanded. Case-insensitive check.
        generic_names = [
            "picture", "image", "chart", "graph",
            "process flow", "diagram", "flowchart", "img", "pic", "graphic"
        ] # Added a few more common ones

        shape_name = shape.name
        is_missing_alt_text = False
        description = "" # Initialize description

        if not shape_name or not shape_name.strip():
            is_missing_alt_text = True
            description = f"Image shape (ID: {shape.shape_id}) is missing alt text (name property is empty)."
        else:
            lower_shape_name = shape_name.lower()
            # Check if the entire name is just a generic term (e.g. "picture", "image")
            if lower_shape_name in generic_names:
                 is_missing_alt_text = True
                 description = f"Image '{shape_name}' has alt text that is too generic ('{shape_name}') and may not be descriptive."
            # Check for default patterns like "Picture 1", "Image 23", etc.
            # This regex matches common default names like "Picture 1", "Image 23", "Chart 5" etc.
            # Also matches if a generic name is followed by numbers, e.g. "Image 123"
            elif re.match(r"^(" + "|".join(generic_names) + r")(\s+\d+)?$", lower_shape_name, re.IGNORECASE):
                is_missing_alt_text = True
                description = f"Image '{shape_name}' has generic alt text ('{shape_name}') that follows a default pattern and may not be descriptive."

        if is_missing_alt_text:
            violations.append({
                'type': 'accessibility_violation',
                'subtype': 'missing_alt_text',
                'description': description,
                'slide_index': slide_index,
                'shape_name': shape_name if shape_name and shape_name.strip() else f"Shape ID {shape.shape_id}",
                'details': "Ensure images have descriptive alt text set in their 'name' property (accessible via Selection Pane in PowerPoint)."
            })
        return violations

    def _validate_paragraph_style(
        self, 
        paragraph, 
        placeholder_type: Optional[int], 
        slide_index: int, 
        para_index: int,
        language: str # Add language
    ) -> List[Dict[str, any]]:
        """
        Validate style for a single paragraph.
        
        Args:
            paragraph: Paragraph object to validate
            placeholder_type: Type of placeholder containing this paragraph
            slide_index: Index of the slide
            para_index: Index of the paragraph
            language: Language code for the presentation
            
        Returns:
            List of violation dictionaries
        """
        violations = []
        
        # Skip empty paragraphs
        if not paragraph.text.strip():
            return violations
        
        # Get expected style for this placeholder type
        expected_font = self._get_expected_font(placeholder_type, paragraph.level, language) # Pass language
        expected_bullet = self._get_expected_bullet(placeholder_type, paragraph.level)
        
        # Validate each run in the paragraph
        for run_index, run in enumerate(paragraph.runs):
            run_violations = self._validate_run_style(
                run, expected_font, slide_index, para_index, run_index, language # Pass language
            )
            violations.extend(run_violations)
        
        # Validate bullet style if this is a bulleted paragraph
        if expected_bullet and self.validation_config.check_bullet_styles:
            bullet_violations = self._validate_bullet_style(
                paragraph, expected_bullet, slide_index, para_index
            )
            violations.extend(bullet_violations)
        
        return violations

    def _validate_run_style(
        self, 
        run, 
        expected_font: Optional[FontInfo], 
        slide_index: int, 
        para_index: int, 
        run_index: int,
        language: str # Add language
    ) -> List[Dict[str, any]]:
        """
        Validate style for a single text run.
        
        Args:
            run: Run object to validate
            expected_font: Expected font information
            slide_index: Index of the slide
            para_index: Index of the paragraph
            run_index: Index of the run
            language: Language code for the presentation
            
        Returns:
            List of violation dictionaries
        """
        violations = []
        
        if not expected_font:
            return violations
        
        font = run.font
        location = f"slide {slide_index}, paragraph {para_index}, run {run_index}"
        
        # Validate font name
        if (self.validation_config.enforce_font_name and 
            expected_font.name and 
            font.name and 
            font.name != expected_font.name):

            # Check if the actual font.name is a valid language-specific override
            is_valid_override = False
            if language and self.template_parser and self.template_parser.template_style:
                specific_font_name = self.template_parser.template_style.language_specific_fonts.get(language.lower())
                if specific_font_name and font.name == specific_font_name:
                    is_valid_override = True

            if not is_valid_override:
                violations.append({
                    'type': 'font_name',
                    'description': f"Font name mismatch at {location}",
                'expected': expected_font.name,
                'actual': font.name,
                'slide_index': slide_index
            })
        
        # Validate font size
        if expected_font.size and font.size:
            expected_size = expected_font.size
            actual_size = font.size.pt
            size_diff = abs(actual_size - expected_size)
            
            if size_diff > self.validation_config.font_size_tolerance:
                violations.append({
                    'type': 'font_size',
                    'description': f"Font size mismatch at {location}",
                    'expected': expected_size,
                    'actual': actual_size,
                    'tolerance': self.validation_config.font_size_tolerance,
                    'slide_index': slide_index
                })
        
        # Validate font weight/bold
        if (self.validation_config.enforce_font_weight and 
            expected_font.weight):
            expected_bold = expected_font.weight == "bold"
            actual_bold = font.bold is True
            
            if expected_bold != actual_bold:
                violations.append({
                    'type': 'font_weight',
                    'description': f"Font weight mismatch at {location}",
                    'expected': expected_font.weight,
                    'actual': "bold" if actual_bold else "normal",
                    'slide_index': slide_index
                })
        
        # Validate font color
        if (self.validation_config.enforce_color_compliance and 
            expected_font.color and 
            font.color):
            color_violations = self._validate_color(
                font.color, expected_font.color, f"font color at {location}", slide_index
            )
            violations.extend(color_violations)
        
        return violations

    def _validate_bullet_style(
        self, 
        paragraph, 
        expected_bullet: BulletInfo, 
        slide_index: int, 
        para_index: int
    ) -> List[Dict[str, any]]:
        """
        Validate bullet point style.
        
        Args:
            paragraph: Paragraph object to validate
            expected_bullet: Expected bullet information
            slide_index: Index of the slide
            para_index: Index of the paragraph
            
        Returns:
            List of violation dictionaries
        """
        violations = []
        
        if not self.validation_config.enforce_bullet_characters:
            return violations
        
        # This is a simplified check - in reality, extracting bullet characters
        # from python-pptx is complex. For now, we'll just validate that
        # paragraphs with level > 0 have appropriate indentation
        location = f"slide {slide_index}, paragraph {para_index}"
        
        if paragraph.level != expected_bullet.indent_level:
            violations.append({
                'type': 'bullet_level',
                'description': f"Bullet indentation level mismatch at {location}",
                'expected': expected_bullet.indent_level,
                'actual': paragraph.level,
                'slide_index': slide_index
            })
        
        return violations

    def _validate_color(
        self, 
        actual_color, 
        expected_color_hex: str, 
        location: str, 
        slide_index: int
    ) -> List[Dict[str, any]]:
        """
        Validate color against expected color with tolerance.
        
        Args:
            actual_color: Color object from python-pptx
            expected_color_hex: Expected color in hex format
            location: Description of where the color is used
            slide_index: Index of the slide
            
        Returns:
            List of violation dictionaries
        """
        violations = []
        
        try:
            # Get actual color as hex
            actual_hex = self._color_to_hex(actual_color)
            if not actual_hex:
                return violations
            
            # Compare colors with tolerance
            if not self._colors_match(actual_hex, expected_color_hex):
                violations.append({
                    'type': 'color',
                    'description': f"Color mismatch at {location}",
                    'expected': expected_color_hex,
                    'actual': actual_hex,
                    'tolerance': self.validation_config.color_tolerance,
                    'slide_index': slide_index
                })
        
        except Exception as e:
            logger.debug(f"Failed to validate color at {location}: {e}")
        
        return violations

    def _color_to_hex(self, color) -> Optional[str]:
        """
        Convert python-pptx color to hex string.
        
        Args:
            color: Color object from python-pptx
            
        Returns:
            Hex color string or None if conversion fails
        """
        try:
            if hasattr(color, 'rgb') and color.rgb:
                rgb = color.rgb
                return f"#{rgb:06X}"
        except Exception:
            pass
        
        return None

    def _colors_match(self, color1_hex: str, color2_hex: str) -> bool:
        """
        Check if two hex colors match within tolerance.
        
        Args:
            color1_hex: First color in hex format
            color2_hex: Second color in hex format
            
        Returns:
            True if colors match within tolerance
        """
        if color1_hex.lower() == color2_hex.lower():
            return True
        
        # If tolerance is 0, require exact match
        if self.validation_config.color_tolerance == 0:
            return False
        
        try:
            # Parse hex colors to RGB
            r1, g1, b1 = self._hex_to_rgb(color1_hex)
            r2, g2, b2 = self._hex_to_rgb(color2_hex)
            
            # Calculate color difference as percentage of maximum possible difference
            max_diff = 255 * 3  # Maximum possible RGB difference
            actual_diff = abs(r1 - r2) + abs(g1 - g2) + abs(b1 - b2)
            
            return (actual_diff / max_diff) <= self.validation_config.color_tolerance
            
        except Exception:
            return False

    def _hex_to_rgb(self, hex_color: str) -> tuple:
        """
        Convert hex color to RGB tuple.
        
        Args:
            hex_color: Color in hex format (e.g., "#FF0000")
            
        Returns:
            RGB tuple (r, g, b)
        """
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

    def _get_expected_font(self, placeholder_type: Optional[int], level: int = 0, language: Optional[str] = None) -> Optional[FontInfo]: # Add language
        """
        Get expected font for a placeholder type and indentation level.
        
        Args:
            placeholder_type: PowerPoint placeholder type number
            level: Indentation level
            language: Language code for considering specific fonts
            
        Returns:
            Expected FontInfo or None
        """
        base_font: Optional[FontInfo] = None
        if not placeholder_type:
            base_font = self.template_parser.template_style.master_font
        else:
            # For bullet points, get font from bullet style
            bullet_info = self.template_parser.get_bullet_style_for_level(placeholder_type, level)
            if bullet_info and bullet_info.font:
                base_font = bullet_info.font
            else:
                # Fall back to placeholder default font
                base_font = self.template_parser.get_font_for_placeholder_type(placeholder_type)

        if language and self.template_parser and self.template_parser.template_style:
            specific_font_name = self.template_parser.template_style.language_specific_fonts.get(language.lower())
            if specific_font_name and base_font:
                # Create a new FontInfo object with the name overridden
                # Assuming FontInfo is a Pydantic model, create a new one or copy and update.
                overridden_font = base_font.model_copy() # Pydantic v2 (use .copy(deep=True) for v1)
                overridden_font.name = specific_font_name
                return overridden_font

        return base_font # Original expected font

    def _get_expected_bullet(self, placeholder_type: Optional[int], level: int = 0) -> Optional[BulletInfo]:
        """
        Get expected bullet style for a placeholder type and indentation level.
        
        Args:
            placeholder_type: PowerPoint placeholder type number
            level: Indentation level
            
        Returns:
            Expected BulletInfo or None
        """
        if not placeholder_type:
            return None
        
        return self.template_parser.get_bullet_style_for_level(placeholder_type, level)

    def _apply_font_adjustments(self, slide, slide_plan: SlidePlan) -> None:
        """
        Apply font adjustments from content fit analysis.
        
        Args:
            slide: PowerPoint slide object
            slide_plan: SlidePlan with potential font adjustments
        """
        # Check if slide has font adjustment metadata
        font_adjustment = None
        if hasattr(slide_plan, '__dict__') and 'font_adjustment' in slide_plan.__dict__:
            font_adjustment = slide_plan.__dict__['font_adjustment']
        
        if not font_adjustment:
            return  # No adjustments needed
        
        logger.debug(f"Applying font adjustment to slide {slide_plan.index}: "
                    f"{font_adjustment.original_size}pt → {font_adjustment.recommended_size}pt")
        
        try:
            # Apply font adjustment to all text content
            for shape in slide.shapes:
                if hasattr(shape, 'text_frame') and shape.text_frame:
                    self._adjust_text_frame_font(shape.text_frame, font_adjustment)
                elif hasattr(shape, 'text') and shape.text:
                    # For simple text shapes
                    if hasattr(shape, 'text_frame') and shape.text_frame.paragraphs:
                        for paragraph in shape.text_frame.paragraphs:
                            for run in paragraph.runs:
                                if run.font.size and run.font.size.pt == font_adjustment.original_size:
                                    run.font.size = Pt(font_adjustment.recommended_size)
        
        except Exception as e:
            logger.error(f"Failed to apply font adjustments to slide {slide_plan.index}: {e}")

    def _adjust_text_frame_font(self, text_frame, font_adjustment: FontAdjustment) -> None:
        """
        Adjust font size in a text frame.
        
        Args:
            text_frame: PowerPoint text frame object
            font_adjustment: FontAdjustment with size information
        """
        try:
            for paragraph in text_frame.paragraphs:
                for run in paragraph.runs:
                    # Only adjust if font size matches the original size
                    if run.font.size:
                        current_size = run.font.size.pt
                        # Allow some tolerance for floating point differences
                        if abs(current_size - font_adjustment.original_size) <= 1:
                            run.font.size = Pt(font_adjustment.recommended_size)
                            logger.debug(f"Adjusted font size: {current_size}pt → {font_adjustment.recommended_size}pt")
                    else:
                        # If no explicit size set, assume it's the default and adjust if it matches
                        run.font.size = Pt(font_adjustment.recommended_size)
        
        except Exception as e:
            logger.debug(f"Error adjusting text frame font: {e}")

    def _hide_empty_placeholder(self, placeholder, placeholder_name: str) -> None:
        """
        Hide an empty placeholder to avoid style validation warnings.
        
        Args:
            placeholder: Placeholder object to hide
            placeholder_name: Name for logging purposes
        """
        try:
            # Method 1: Try to make the placeholder invisible
            if hasattr(placeholder, 'visible'):
                placeholder.visible = False
                logger.debug(f"Set {placeholder_name} placeholder invisible")
                return
            
            # Method 2: Try to set shape visibility
            if hasattr(placeholder, 'element') and hasattr(placeholder.element, 'set'):
                # Try to add hidden attribute to the shape
                try:
                    placeholder.element.set('hidden', '1')
                    logger.debug(f"Set {placeholder_name} placeholder hidden via XML attribute")
                    return
                except:
                    pass
            
            # Method 3: Move placeholder off-slide (position outside visible area)
            if hasattr(placeholder, 'left') and hasattr(placeholder, 'top'):
                placeholder.left = Inches(-10)  # Move far left, outside slide
                placeholder.top = Inches(-10)   # Move far up, outside slide
                logger.debug(f"Moved {placeholder_name} placeholder off-slide to hide it")
                return
            
            # Method 4: Set placeholder size to zero
            if hasattr(placeholder, 'width') and hasattr(placeholder, 'height'):
                placeholder.width = Inches(0.01)  # Minimal size
                placeholder.height = Inches(0.01)
                logger.debug(f"Set {placeholder_name} placeholder to minimal size")
                return
            
            # Method 5: Clear any default text that might be in the placeholder
            if hasattr(placeholder, 'text_frame') and placeholder.text_frame:
                placeholder.text_frame.clear()
                if hasattr(placeholder.text_frame, 'auto_size'):
                    try:
                        placeholder.text_frame.auto_size = MSO_AUTO_SIZE.NONE
                    except:
                        pass
                logger.debug(f"Cleared {placeholder_name} placeholder text frame")
                return
            
            logger.warning(f"Could not hide {placeholder_name} placeholder - no suitable method found")
            
        except Exception as e:
            logger.warning(f"Failed to hide {placeholder_name} placeholder: {e}")

    def _remove_empty_placeholders_from_slide(self, slide) -> int:
        """
        Remove or hide all empty placeholders from a slide to avoid style warnings.
        
        Args:
            slide: Slide object to process
            
        Returns:
            Number of placeholders hidden/removed
        """
        hidden_count = 0
        
        try:
            for placeholder in slide.placeholders:
                # Check if placeholder is empty
                is_empty = True
                
                if hasattr(placeholder, 'text_frame') and placeholder.text_frame:
                    if placeholder.text_frame.text.strip():
                        is_empty = False
                elif hasattr(placeholder, 'text') and placeholder.text:
                    if placeholder.text.strip():
                        is_empty = False
                
                # Hide empty placeholders (except title which should always be visible)
                if is_empty and placeholder.placeholder_format.type != 1:  # Not TITLE
                    placeholder_type_name = self._get_placeholder_type_name(placeholder.placeholder_format.type)
                    self._hide_empty_placeholder(placeholder, placeholder_type_name)
                    hidden_count += 1
        
        except Exception as e:
            logger.warning(f"Error processing empty placeholders: {e}")
        
        return hidden_count

    def _get_placeholder_type_name(self, placeholder_type: int) -> str:
        """
        Get human-readable name for placeholder type.
        
        Args:
            placeholder_type: PowerPoint placeholder type constant
            
        Returns:
            Human-readable placeholder type name
        """
        type_names = {
            1: "title",
            2: "body", 
            3: "subtitle",
            4: "center_title",
            5: "center_subtitle",
            6: "date_time",
            7: "footer",
            8: "header",
            9: "slide_number",
            10: "picture",
            11: "chart",
            12: "table",
            13: "clip_art",
            14: "organization_chart",
            15: "media_clip",
            16: "vertical_object",
            17: "vertical_body",
            18: "picture"
        }
        
        return type_names.get(placeholder_type, f"unknown_type_{placeholder_type}")