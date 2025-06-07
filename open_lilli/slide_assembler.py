"""Slide assembler for building PowerPoint presentations."""

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Union

from pptx import Presentation
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE_TYPE

from .models import Outline, SlidePlan, StyleValidationConfig, FontInfo, BulletInfo, FontAdjustment, NativeChartData
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
        elif slide_plan.bullets:
            self._add_bullet_content(existing_slide, slide_plan.bullets, language) # Pass language
        
        # Add visuals
        if "native_chart" in slide_visuals and slide_plan.chart_data:
            self._add_native_chart(existing_slide, slide_plan.chart_data)
        elif "chart" in slide_visuals:
            self._add_chart_image(existing_slide, slide_visuals["chart"])
        
        if "process_flow" in slide_visuals:
            self._add_process_flow(existing_slide, slide_visuals["process_flow"])
        
        if "image" in slide_visuals:
            self._add_image(existing_slide, slide_visuals["image"])
        
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
        elif slide_plan.bullets:
            self._add_bullet_content(slide, slide_plan.bullets, language) # Pass language
        
        # Add visuals
        if "native_chart" in slide_visuals and slide_plan.chart_data:
            self._add_native_chart(slide, slide_plan.chart_data)
        elif "chart" in slide_visuals:
            self._add_chart_image(slide, slide_visuals["chart"])
        
        if "process_flow" in slide_visuals:
            self._add_process_flow(slide, slide_visuals["process_flow"])
        
        if "image" in slide_visuals:
            self._add_image(slide, slide_visuals["image"])
        
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
            for placeholder in slide.placeholders:
                if placeholder.placeholder_format.type == 3:  # SUBTITLE
                    subtitle_text = ""
                    if slide_plan.bullets: # Assuming bullets might be used as subtitle parts
                        subtitle_text = " • ".join(slide_plan.bullets[:2])
                    elif hasattr(slide_plan, 'subtitle') and slide_plan.subtitle:
                        subtitle_text = slide_plan.subtitle
                    
                    if subtitle_text:
                        placeholder.text = subtitle_text
                        # Apply RTL alignment for subtitle if needed
                        if language.lower() in RTL_LANGUAGES:
                            if placeholder.text_frame and placeholder.text_frame.paragraphs:
                                for para in placeholder.text_frame.paragraphs:
                                    para.alignment = PP_ALIGN.RIGHT
                        logger.debug(f"Added subtitle: {subtitle_text}")
                    break
                    
        except Exception as e:
            logger.error(f"Failed to add title slide content: {e}")

    def _add_bullet_content(self, slide, bullets: List[str], language: str) -> None: # Add language
        """Add bullet points to slide."""
        try:
            # Find content placeholder
            content_placeholder = None
            
            for placeholder in slide.placeholders:
                ph_type = placeholder.placeholder_format.type
                if ph_type in (2, 7):  # BODY or OBJECT
                    content_placeholder = placeholder
                    break
            
            if not content_placeholder:
                logger.warning("No content placeholder found, looking for any text placeholder")
                # Fallback: look for any placeholder we can use
                for placeholder in slide.placeholders:
                    if hasattr(placeholder, 'text_frame'):
                        content_placeholder = placeholder
                        break
            
            if content_placeholder and hasattr(content_placeholder, 'text_frame'):
                text_frame = content_placeholder.text_frame
                text_frame.clear()  # Clear existing content
                
                # Add bullets
                for i, bullet_text in enumerate(bullets):
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
                    p.level = 0  # Top-level bullet
                    
                    # Apply formatting (font size will be adjusted later if needed)
                    if p.runs:
                        run = p.runs[0]
                        run.font.size = Pt(18)  # Default size

                        # Proactive font selection for bullets
                        specific_font_name = self.template_parser.template_style.language_specific_fonts.get(language.lower())
                        if specific_font_name:
                            run.font.name = specific_font_name
                            logger.debug(f"Applied language-specific font '{specific_font_name}' for bullets in language '{language.lower()}'")

                # RTL alignment for bullets
                if language.lower() in RTL_LANGUAGES:
                    for para in text_frame.paragraphs:
                        if para.text and para.text.strip(): # Only align if paragraph has text
                            para.alignment = PP_ALIGN.RIGHT
                
                logger.debug(f"Added {len(bullets)} bullet points")
            else:
                logger.warning("No suitable placeholder found for bullet content")
                
        except Exception as e:
            logger.error(f"Failed to add bullet content: {e}")

    def _add_chart_image(self, slide, chart_path: str) -> None:
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
                picture_placeholder.insert_picture(str(chart_path))
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
                
                slide.shapes.add_picture(str(chart_path), left, top, img_width, img_height)
                logger.debug(f"Added chart as floating image: {chart_path}")
                
        except Exception as e:
            logger.error(f"Failed to add chart image '{chart_path}': {e}")

    def _add_image(self, slide, image_path: str) -> None:
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
                picture_placeholder.insert_picture(str(image_path))
                logger.debug(f"Inserted image into picture placeholder: {image_path}")
            else:
                # Add as free-floating image
                # Position in upper right area if no chart is present
                slide_width = slide.part.presentation.slide_width
                
                img_width = Inches(3)
                img_height = Inches(2)
                left = slide_width - img_width - Inches(0.5)
                top = Inches(1)
                
                slide.shapes.add_picture(str(image_path), left, top, img_width, img_height)
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

    def _add_process_flow(self, slide, flow_path: str) -> None:
        """Add process flow SVG to slide."""
        try:
            flow_path = Path(flow_path)
            if not flow_path.exists():
                logger.error(f"Process flow file not found: {flow_path}")
                return
            
            # Convert SVG to image for insertion
            # For now, just add as image - could be enhanced to embed SVG directly
            self._add_image(slide, flow_path)
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
                shape_violations = self._validate_shape_style(shape, slide_index, language) # Pass language
                violations.extend(shape_violations)
            except Exception as e:
                logger.debug(f"Failed to validate shape in slide {slide_index}: {e}")
                continue
        
        return violations

    def _validate_shape_style(self, shape, slide_index: int, language: str) -> List[Dict[str, any]]: # Add language
        """
        Validate style for a single shape.
        
        Args:
            shape: Shape object to validate
            slide_index: Index of the slide containing this shape
            language: Language code for the presentation
            
        Returns:
            List of violation dictionaries
        """
        violations = []
        
        # Only validate text-containing shapes
        if not hasattr(shape, 'text_frame') or not shape.text_frame:
            return violations
        
        text_frame = shape.text_frame
        
        # Get placeholder information if available
        placeholder_type = None
        if hasattr(shape, 'placeholder_format') and shape.placeholder_format:
            placeholder_type = shape.placeholder_format.type
        
        # Validate each paragraph
        for para_index, paragraph in enumerate(text_frame.paragraphs):
            para_violations = self._validate_paragraph_style(
                paragraph, placeholder_type, slide_index, para_index, language # Pass language
            )
            violations.extend(para_violations)
        
        # Check for empty placeholders
        if not self.validation_config.allow_empty_placeholders:
            if not text_frame.text.strip() and placeholder_type:
                violations.append({
                    'type': 'empty_placeholder',
                    'description': f"Empty placeholder of type {placeholder_type}",
                    'slide_index': slide_index,
                    'placeholder_type': placeholder_type
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