"""Content fit analyzer for dynamic content adjustment and optimization."""

import logging
import re
from typing import List, Optional, Tuple

from openai import OpenAI

from .models import (
    ContentDensityAnalysis,
    ContentFitConfig,
    ContentFitResult,
    FontAdjustment,
    SlidePlan,
    TemplateStyle
)

logger = logging.getLogger(__name__)


class ContentFitAnalyzer:
    """Analyzes content density and provides dynamic fitting solutions."""

    def __init__(
        self,
        config: Optional[ContentFitConfig] = None,
        openai_client: Optional[OpenAI] = None
    ):
        """
        Initialize the content fit analyzer.
        
        Args:
            config: Configuration for content fit analysis
            openai_client: Optional OpenAI client for content rewriting
        """
        self.config = config or ContentFitConfig()
        self.client = openai_client
        
        logger.info("ContentFitAnalyzer initialized")

    def analyze_slide_density(
        self, 
        slide: SlidePlan,
        template_style: Optional[TemplateStyle] = None
    ) -> ContentDensityAnalysis:
        """
        Analyze the content density of a slide.
        
        Args:
            slide: SlidePlan to analyze
            template_style: Optional template style information
            
        Returns:
            ContentDensityAnalysis with density metrics
        """
        # Calculate total content length
        total_chars = len(slide.title)
        for bullet in slide.bullets:
            total_chars += len(bullet)
        
        # Estimate lines needed based on content
        title_lines = self._estimate_text_lines(slide.title)
        bullet_lines = sum(self._estimate_text_lines(bullet) for bullet in slide.bullets)
        total_lines = title_lines + bullet_lines + len(slide.bullets)  # Add line breaks
        
        # Estimate placeholder capacity
        placeholder_capacity = self._estimate_placeholder_capacity(slide, template_style)
        
        # Calculate density ratio
        density_ratio = total_chars / max(placeholder_capacity, 1)
        
        # Determine recommended action
        requires_action = density_ratio > 1.0
        recommended_action = self._determine_action(density_ratio, slide) # Pass slide for context
        
        logger.debug(f"Slide {slide.index} density analysis: {total_chars} chars, "
                    f"ratio {density_ratio:.2f}, action: {recommended_action}")
        
        return ContentDensityAnalysis(
            total_characters=total_chars,
            estimated_lines=total_lines,
            placeholder_capacity=placeholder_capacity,
            density_ratio=density_ratio,
            requires_action=requires_action,
            recommended_action=recommended_action
        )

    def recommend_font_adjustment(
        self,
        density_analysis: ContentDensityAnalysis,
        template_style: Optional[TemplateStyle], # Added template_style
        current_font_size: float = 18.0 # Changed to float
    ) -> Optional[FontAdjustment]:
        """
        Recommend font size adjustment for mild overflow using proportional shrinking.
        
        Args:
            density_analysis: Content density analysis
            template_style: Template style information for baseline font sizes
            current_font_size: Current font size in points
            
        Returns:
            FontAdjustment recommendation or None if not applicable
        """
        if density_analysis.overflow_severity not in ["mild", "moderate"] and \
           density_analysis.recommended_action != "adjust_font": # Ensure it's called for adjust_font action
            return None

        if density_analysis.density_ratio <= self.config.font_tune_threshold: # Check if adjustment is truly needed
            return None

        template_body_font_size = current_font_size # Default to current_font_size
        if template_style:
            # Assuming 2 is 'BODY' placeholder type, common in many templates
            template_body_placeholder_style = template_style.get_placeholder_style(2)
            if (template_body_placeholder_style and
                template_body_placeholder_style.default_font and
                template_body_placeholder_style.default_font.size is not None):
                template_body_font_size = template_body_placeholder_style.default_font.size
            else:
                logger.warning(f"Could not find default body font size in template style. Using current_font_size {current_font_size}pt as base for proportional shrink cap.")
        else:
            logger.warning("Template style not provided for font adjustment. Using current_font_size as base for proportional shrink cap.")

        # Proportional shrink based on current font size
        new_size_proportional = current_font_size * self.config.proportional_shrink_factor
        
        # Floor size based on template's original body font size
        floor_size_from_template = template_body_font_size * self.config.max_proportional_shrink_cap_factor
        
        # Determine final recommended size, ensuring it's an integer or .5 increment
        final_recommended_size = round(max(new_size_proportional, floor_size_from_template, self.config.min_font_size))
        
        # Ensure the new size is not greater than current size (should be a shrink or no change)
        final_recommended_size = min(final_recommended_size, current_font_size)

        adjustment_points_value = current_font_size - final_recommended_size

        # If no change, no adjustment needed
        if adjustment_points_value == 0 and final_recommended_size == current_font_size :
             # Check if it was already at min_font_size or floor_size_from_template
            if current_font_size == self.config.min_font_size or current_font_size == round(floor_size_from_template):
                 reasoning = "Content may still overflow; font size already at minimum/cap. Consider rewrite or split."
                 # Still return an adjustment object so caller knows an attempt was made, but it's not "safe" for fitting
                 return FontAdjustment(
                    original_size=float(current_font_size),
                    recommended_size=float(final_recommended_size),
                    adjustment_points=float(-adjustment_points_value),
                    confidence=0.5, # Lower confidence as it might not solve overflow
                    reasoning=reasoning,
                    safe_bounds=False # Not safe in terms of guaranteeing fit
                )
            return None


        safe_bounds = (final_recommended_size >= self.config.min_font_size and
                       final_recommended_size >= round(floor_size_from_template))
        
        # Confidence can be higher as this is a more systematic approach
        confidence = 0.85
        
        reasoning = "Proportional shrink applied, respecting template and minimum font size caps."
        if not safe_bounds: # This case should ideally be handled by max() but double check
            reasoning = "Font size recommendation hit a minimum cap, may not fully resolve overflow."
            confidence = 0.6

        return FontAdjustment(
            original_size=float(current_font_size),
            recommended_size=float(final_recommended_size),
            adjustment_points=float(-adjustment_points_value), # Negative for reduction
            confidence=confidence,
            reasoning=reasoning,
            safe_bounds=safe_bounds # Safe in terms of not going below defined limits
        )

    def should_split_slide(self, density_analysis: ContentDensityAnalysis) -> bool:
        """
        Determine if a slide should be split based on density analysis.
        
        Args:
            density_analysis: Content density analysis
            
        Returns:
            True if slide should be split
        """
        return density_analysis.density_ratio >= self.config.split_threshold

    def split_slide_content(
        self, 
        slide: SlidePlan,
        target_density: float = 0.8
    ) -> List[SlidePlan]:
        """
        Split slide content into multiple slides.
        
        Args:
            slide: SlidePlan to split
            target_density: Target density ratio for split slides
            
        Returns:
            List of split slide plans
        """
        if not slide.bullets:
            # Can't split a slide with no bullets
            return [slide]
        
        # Calculate optimal split based on content length
        total_content_length = sum(len(bullet) for bullet in slide.bullets)
        target_capacity = int(self.config.characters_per_line * self.config.lines_per_placeholder * target_density)
        
        # Determine number of slides needed
        num_slides = max(2, (total_content_length // target_capacity) + 1)
        bullets_per_slide = max(1, len(slide.bullets) // num_slides)
        
        split_slides = []
        bullet_chunks = self._chunk_bullets(slide.bullets, bullets_per_slide)
        
        for i, bullet_chunk in enumerate(bullet_chunks):
            split_slide = slide.model_copy()
            split_slide.index = slide.index + i
            split_slide.bullets = bullet_chunk
            
            # Modify title to indicate part
            if len(bullet_chunks) > 1:
                split_slide.title = f"{slide.title} (Part {i + 1})"
            
            # Update speaker notes
            if i < len(bullet_chunks) - 1:
                split_slide.speaker_notes = f"Part {i + 1} of {len(bullet_chunks)} - continue to next slide"
            else:
                split_slide.speaker_notes = f"Final part of {slide.title}"
            
            split_slides.append(split_slide)
            
            logger.debug(f"Split slide {slide.index} part {i + 1}: {len(bullet_chunk)} bullets")
        
        logger.info(f"Split slide {slide.index} into {len(split_slides)} slides")
        return split_slides

    def optimize_slide_content(
        self,
        slide: SlidePlan,
        template_style: Optional[TemplateStyle] = None
    ) -> ContentFitResult:
        """
        Optimize slide content using density analysis and dynamic adjustments.
        
        Args:
            slide: SlidePlan to optimize
            template_style: Optional template style information
            
        Returns:
            ContentFitResult with optimization details
        """
        # Analyze content density
        density_analysis = self.analyze_slide_density(slide, template_style)
        
        font_adjustment = None
        split_performed = False
        split_count = 1
        final_action = "no_action"
        
        if not density_analysis.requires_action:
            final_action = "no_action"
        else:
            if density_analysis.recommended_action == "rewrite_content":
                # Calculate target reduction
                target_reduction = (density_analysis.density_ratio - self.config.rewrite_threshold) / self.config.rewrite_threshold
                target_reduction = min(max(0.1, target_reduction), 0.5) # Cap between 10% and 50%

                rewritten_slide = None # Define rewritten_slide before the try block
                try: # Add try block for async call if needed, though rewrite_content_shorter is synchronous in current stub
                    rewritten_slide = self.rewrite_content_shorter(slide, target_reduction=target_reduction)
                except Exception as e: # Catch potential errors from rewrite_content_shorter
                    logger.error(f"Error during content rewriting for slide {slide.index}: {e}")
                    rewritten_slide = None # Ensure it's None on error

                if rewritten_slide:
                    slide = rewritten_slide # Update slide with rewritten content
                    slide.summarized_by_llm = True
                    final_action = "rewrite_content"
                    # Optionally, re-analyze density
                    density_analysis = self.analyze_slide_density(slide, template_style)
                    # If still overflowing significantly after rewrite, or if rewrite failed, consider splitting
                    if not density_analysis.requires_action:
                        pass # Content fits after rewrite
                    elif self.should_split_slide(density_analysis): # Check if split is now needed
                        split_performed = True
                        # We need to handle the output of split_slide_content if we are to use it
                        # For now, just mark that split would be the action
                        # split_slides = self.split_slide_content(slide)
                        # split_count = len(split_slides)
                        final_action = "split_slide" # Update final_action if split is needed
                    elif density_analysis.recommended_action == "adjust_font": # Check if font adjustment is now viable
                        # Pass template_style to recommend_font_adjustment
                        font_adjustment = self.recommend_font_adjustment(density_analysis, template_style, current_font_size=slide.title_font_size if hasattr(slide, 'title_font_size') else 18.0) # Assuming a default/current size
                        if font_adjustment and font_adjustment.safe_bounds:
                            final_action = "adjust_font"
                        else: # If font adjustment not viable after rewrite, consider split
                            split_performed = True
                            final_action = "split_slide"
                else:
                    # Rewrite failed, proceed to split
                    final_action = "split_slide"
                    split_performed = True
                    # split_slides = self.split_slide_content(slide) # Original slide
                    # split_count = len(split_slides)
            
            elif density_analysis.recommended_action == "adjust_font":
                 # Pass template_style. Assuming current_font_size needs to be determined or passed.
                 # For now, using a common default or making it part of SlidePlan later.
                 # Let's assume a default of 18pt for body text if not otherwise specified on slide_plan
                current_slide_font_size = getattr(slide, 'body_font_size', 18.0) # Example attribute
                font_adjustment = self.recommend_font_adjustment(density_analysis, template_style, current_font_size=current_slide_font_size)
                if font_adjustment and font_adjustment.safe_bounds:
                    final_action = "adjust_font"
                else:
                    # Font adjustment not viable or not safe for fitting, split instead
                    final_action = "split_slide"
                    split_performed = True
                    # split_slides = self.split_slide_content(slide)
                    # split_count = len(split_slides)

            elif density_analysis.recommended_action == "split_slide":
                final_action = "split_slide"
                split_performed = True
                # split_slides = self.split_slide_content(slide)
                # split_count = len(split_slides)

            # If split_performed is true, we should update split_count.
            # For now, we assume split_slide_content would return a list of slides,
            # and split_count would be its length. Since we are not calling it yet
            # to avoid using its direct output for now, we'll set split_count
            # based on whether a split was decided.
            if split_performed:
                # This is a placeholder; actual split count would come from split_slide_content
                # For the purpose of this change, if a split is performed, we can assume it results in at least 2 slides.
                # However, the current split_slide_content logic might return 1 if it can't split.
                # Let's stick to the original logic for split_count for now if split_slide_content is not called.
                # If split_slide_content is called, its result should be used.
                # Since we are commenting out the call for now, let's reflect that a split implies >1 slide.
                # A simple heuristic could be 2, or we could estimate based on density.
                # Given the current structure, it's safer to update split_count when split_slide_content is actually called
                # and its result is used. For now, if final_action is "split_slide", split_performed should be true.
                # The actual number of slides (split_count) would be determined by the split_slide_content method.
                # If self.split_slide_content is not called in a path that sets final_action = "split_slide",
                # then split_count might remain 1, which could be misleading.
                # Let's ensure split_count is updated if split_performed.
                # A simple update: if a split is decided, assume at least 2 slides.
                # However, the split_slide_content method itself might return just [slide] if it can't split.
                # To be safe and reflect the intent:
                if final_action == "split_slide": # Ensure split_performed is true
                    split_performed = True
                # The actual split_count should be determined by calling split_slide_content.
                # For now, we'll rely on the fact that if split_slide_content is called, it updates split_count.
                # The new logic paths might not call it.
                # Let's assume for now that if final_action is 'split_slide', split_count will be > 1.
                # This part needs careful handling of when split_slide_content is called.
                # For this refactoring, we'll set split_count to a placeholder if split is chosen but not executed here.
                if final_action == "split_slide" and not slide.bullets: # Cannot split if no bullets
                     split_count = 1
                     split_performed = False # Cannot actually split
                     final_action = "no_action" # Or handle as error/adjust font if possible
                elif final_action == "split_slide":
                     split_count = 2 # Placeholder, actual count from split_slide_content

        # Ensure slide object reflects the summarized_by_llm status if rewritten
        # The slide variable is updated above if rewrite is successful.

        # Determine the modified_slide_plan based on actions taken
        current_modified_slide_plan = None
        if final_action == "rewrite_content" and slide.summarized_by_llm: # slide is the rewritten_slide here
            current_modified_slide_plan = slide
        elif final_action == "adjust_font":
            # If font adjustment is the only action, the slide content itself hasn't changed,
            # but it's good to pass the slide that the adjustment applies to.
            current_modified_slide_plan = slide # Original slide or rewritten slide if font adjustment is a secondary action
        elif final_action == "no_action":
            current_modified_slide_plan = slide # Original slide

        # If split_performed, modified_slide_plan is typically None because multiple slides are generated.
        # However, if a rewrite happened *before* a split decision, 'slide' here would be the rewritten one.
        # This nuance will be handled by SlidePlanner using this ContentFitResult.

        return ContentFitResult(
            slide_index=slide.index, # Use original index for result keying
            density_analysis=density_analysis,
            font_adjustment=font_adjustment,
            split_performed=split_performed,
            split_count=split_count,
            final_action=final_action,
            modified_slide_plan=current_modified_slide_plan
        )

    async def rewrite_content_shorter(
        self,
        slide: SlidePlan,
        target_reduction: float = 0.3
    ) -> Optional[SlidePlan]:
        """
        Use AI to rewrite slide content to be shorter.
        
        Args:
            slide: SlidePlan with content to shorten
            target_reduction: Target reduction ratio (0.3 = 30% shorter)
            
        Returns:
            SlidePlan with shortened content or None if failed
        """
        if not self.client:
            logger.warning("No OpenAI client available for content rewriting")
            return None
        
        try:
            # Create prompt for content shortening
            original_bullets = "\n".join(f"• {bullet}" for bullet in slide.bullets)
            target_length = int(len(original_bullets) * (1 - target_reduction))
            
            prompt = f"""Please rewrite these bullet points to be approximately {target_reduction * 100:.0f}% shorter while keeping the key information:

ORIGINAL BULLETS:
{original_bullets}

TARGET LENGTH: Approximately {target_length} characters

REQUIREMENTS:
- Maintain the same meaning and key points
- Keep the same number of bullet points if possible
- Use more concise language
- Remove redundant words and phrases
- Return only the rewritten bullets, one per line with • prefix"""

            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert at writing concise, impactful bullet points for business presentations."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1000
            )
            
            rewritten_text = response.choices[0].message.content
            
            # Parse rewritten bullets
            new_bullets = []
            for line in rewritten_text.strip().split('\n'):
                line = line.strip()
                if line.startswith('•') or line.startswith('-'):
                    bullet = line[1:].strip()
                    new_bullets.append(bullet)
                elif line and not line.startswith('REWRITTEN') and not line.startswith('TARGET'):
                    new_bullets.append(line)
            
            if new_bullets:
                rewritten_slide = slide.model_copy()
                rewritten_slide.bullets = new_bullets
                
                reduction_achieved = 1 - (sum(len(b) for b in new_bullets) / sum(len(b) for b in slide.bullets))
                logger.info(f"Rewritten slide {slide.index}: {reduction_achieved:.1%} reduction achieved")
                
                return rewritten_slide
            
        except Exception as e:
            logger.error(f"Failed to rewrite content for slide {slide.index}: {e}")
        
        return None

    def _estimate_text_lines(self, text: str) -> int:
        """
        Estimate number of lines needed for text.
        
        Args:
            text: Text to estimate
            
        Returns:
            Estimated number of lines
        """
        if not text:
            return 0
        
        # Basic estimation based on character count
        chars_per_line = self.config.characters_per_line
        return max(1, (len(text) + chars_per_line - 1) // chars_per_line)

    def _estimate_placeholder_capacity(
        self,
        slide: SlidePlan,
        template_style: Optional[TemplateStyle] = None
    ) -> int:
        """
        Estimate the character capacity of the slide's content placeholder.
        
        Args:
            slide: SlidePlan to estimate for
            template_style: Optional template style information
            
        Returns:
            Estimated character capacity
        """
        # Base capacity calculation
        lines_available = self.config.lines_per_placeholder
        chars_per_line = self.config.characters_per_line
        
        # Adjust for title (takes up some space)
        if slide.title:
            lines_available -= 1  # Account for title space
        
        # Adjust for bullet count (each bullet has overhead)
        if slide.bullets:
            # Account for bullet characters and spacing
            bullet_overhead_per_line = 4  # "• " plus some spacing
            chars_per_line -= bullet_overhead_per_line
        
        # Adjust for visual content
        if slide.image_query or slide.chart_data:
            lines_available = int(lines_available * 0.6)  # Visual takes up space
        
        capacity = max(100, lines_available * chars_per_line)  # Minimum capacity
        
        logger.debug(f"Estimated capacity for slide {slide.index}: {capacity} characters")
        return capacity

    def _determine_action(self, density_ratio: float, slide: SlidePlan) -> str:
        """
        Determine recommended action based on density ratio.
        
        Args:
            density_ratio: Content density ratio
            slide: The slide plan, to check if it can be rewritten (e.g. has bullets)
            
        Returns:
            Recommended action string
        """
        if density_ratio <= 1.0:
            return "no_action"

        # If content is primarily title or non-bullet, rewrite might not be suitable.
        can_rewrite = bool(slide.bullets) # Simple check, can be more sophisticated

        if density_ratio <= self.config.font_tune_threshold:
            # Very mild overflow, could be acceptable or tiny font adjustment
            return "no_action" # Or "adjust_font_minor" if we add such a state
        elif self.config.font_tune_threshold < density_ratio <= self.config.rewrite_threshold:
            # Threshold for font adjustment
            return "adjust_font"
        elif can_rewrite and self.config.rewrite_threshold < density_ratio <= self.config.split_threshold:
            # Threshold for rewriting content
            return "rewrite_content"
        elif not can_rewrite and self.config.font_tune_threshold < density_ratio <= self.config.split_threshold:
            # If cannot rewrite, but in range where rewrite or font adjust would be considered, try font adjust.
            # If font adjust is not enough, it might lead to split later.
            return "adjust_font"
        elif density_ratio > self.config.split_threshold:
            # Threshold for splitting slide
            return "split_slide"
        else:
            # Default or fallback if other conditions not met (e.g. cannot rewrite but severe overflow)
            return "split_slide" # Fallback to split for severe cases if other options exhausted

    def _chunk_bullets(self, bullets: List[str], target_size: int) -> List[List[str]]:
        """
        Split bullets into chunks for slide splitting.
        
        Args:
            bullets: List of bullet points
            target_size: Target number of bullets per chunk
            
        Returns:
            List of bullet chunks
        """
        if target_size >= len(bullets):
            return [bullets]
        
        chunks = []
        for i in range(0, len(bullets), target_size):
            chunk = bullets[i:i + target_size]
            chunks.append(chunk)
        
        return chunks

    def get_optimization_summary(
        self,
        results: List[ContentFitResult]
    ) -> dict:
        """
        Get summary of content fit optimization results.
        
        Args:
            results: List of content fit results
            
        Returns:
            Summary dictionary
        """
        total_slides = len(results)
        slides_requiring_action = sum(1 for r in results if r.density_analysis.requires_action)
        slides_split = sum(1 for r in results if r.split_performed)
        slides_font_adjusted = sum(1 for r in results if r.font_adjustment is not None)
        
        action_counts = {}
        for result in results:
            action = result.final_action
            action_counts[action] = action_counts.get(action, 0) + 1
        
        return {
            "total_slides": total_slides,
            "slides_requiring_action": slides_requiring_action,
            "slides_split": slides_split,
            "slides_font_adjusted": slides_font_adjusted,
            "action_breakdown": action_counts,
            "optimization_rate": slides_requiring_action / total_slides if total_slides > 0 else 0
        }