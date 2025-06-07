"""Slide assembler for building PowerPoint presentations."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

from pptx import Presentation
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor

from .models import Outline, SlidePlan
from .template_parser import TemplateParser

logger = logging.getLogger(__name__)


class SlideAssembler:
    """Assembles PowerPoint presentations from templates and generated content."""

    def __init__(self, template_parser: TemplateParser):
        """
        Initialize the slide assembler.
        
        Args:
            template_parser: TemplateParser instance for template access
        """
        self.template_parser = template_parser
        self.max_title_length = 60
        self.max_bullet_length = 120
        
        logger.info("SlideAssembler initialized")

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
        
        logger.info(f"Assembling presentation with {len(slides)} slides")
        logger.info(f"Output path: {output_path}")
        
        # Create new presentation from template
        prs = Presentation(str(self.template_parser.template_path))
        
        # Clear any existing slides
        while len(prs.slides) > 0:
            slide_to_remove = prs.slides[0]
            slide_index = prs.slides._sldIdLst.remove(slide_to_remove.slide_id)
        
        # Add slides
        for slide_plan in slides:
            try:
                self._add_slide(prs, slide_plan, visuals.get(slide_plan.index, {}))
                logger.debug(f"Added slide {slide_plan.index}: {slide_plan.title}")
            except Exception as e:
                logger.error(f"Failed to add slide {slide_plan.index}: {e}")
                # Add a basic slide as fallback
                self._add_fallback_slide(prs, slide_plan)
        
        # Apply presentation metadata
        self._apply_metadata(prs, outline)
        
        # Save presentation
        prs.save(str(output_path))
        
        logger.info(f"Successfully assembled presentation: {output_path}")
        return output_path

    def _add_slide(
        self,
        prs: Presentation,
        slide_plan: SlidePlan,
        slide_visuals: Dict[str, str]
    ) -> None:
        """Add a single slide to the presentation."""
        
        # Get the appropriate layout
        layout_index = slide_plan.layout_id or 0
        if layout_index >= len(prs.slide_layouts):
            logger.warning(f"Layout index {layout_index} out of range, using 0")
            layout_index = 0
        
        layout = prs.slide_layouts[layout_index]
        slide = prs.slides.add_slide(layout)
        
        # Add title
        self._add_title(slide, slide_plan.title)
        
        # Add content based on slide type
        if slide_plan.slide_type == "title":
            self._add_title_slide_content(slide, slide_plan)
        elif slide_plan.bullets:
            self._add_bullet_content(slide, slide_plan.bullets)
        
        # Add visuals
        if "chart" in slide_visuals:
            self._add_chart_image(slide, slide_visuals["chart"])
        
        if "image" in slide_visuals:
            self._add_image(slide, slide_visuals["image"])
        
        # Add speaker notes
        if slide_plan.speaker_notes:
            self._add_speaker_notes(slide, slide_plan.speaker_notes)

    def _add_title(self, slide, title: str) -> None:
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
                        
                logger.debug(f"Added title: {title}")
            else:
                logger.warning("No title placeholder found in slide")
                
        except Exception as e:
            logger.error(f"Failed to add title '{title}': {e}")

    def _add_title_slide_content(self, slide, slide_plan: SlidePlan) -> None:
        """Add content specific to title slides."""
        try:
            # Try to find subtitle placeholder
            for placeholder in slide.placeholders:
                if placeholder.placeholder_format.type == 3:  # SUBTITLE
                    subtitle_text = ""
                    if slide_plan.bullets:
                        subtitle_text = " • ".join(slide_plan.bullets[:2])
                    elif hasattr(slide_plan, 'subtitle') and slide_plan.subtitle:
                        subtitle_text = slide_plan.subtitle
                    
                    if subtitle_text:
                        placeholder.text = subtitle_text
                        logger.debug(f"Added subtitle: {subtitle_text}")
                    break
                    
        except Exception as e:
            logger.error(f"Failed to add title slide content: {e}")

    def _add_bullet_content(self, slide, bullets: List[str]) -> None:
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
                    
                    # Apply formatting
                    if p.runs:
                        run = p.runs[0]
                        run.font.size = Pt(18)
                
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
        content: Optional[List[str]] = None
    ) -> None:
        """
        Create a slide using a specific layout by name.
        
        Args:
            prs: Presentation object
            layout_name: Name of the layout to use
            title: Slide title
            content: Optional list of content items
        """
        try:
            layout_index = self.template_parser.get_layout_index(layout_name)
            layout = prs.slide_layouts[layout_index]
            slide = prs.slides.add_slide(layout)
            
            # Add title
            if hasattr(slide.shapes, 'title'):
                slide.shapes.title.text = title
            
            # Add content if provided
            if content:
                self._add_bullet_content(slide, content)
            
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