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
        recommended_action = self._determine_action(density_ratio)
        
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
        current_font_size: int = 18
    ) -> Optional[FontAdjustment]:
        """
        Recommend font size adjustment for mild overflow.
        
        Args:
            density_analysis: Content density analysis
            current_font_size: Current font size in points
            
        Returns:
            FontAdjustment recommendation or None if not applicable
        """
        # Only adjust font for mild to moderate overflow
        if density_analysis.overflow_severity not in ["mild", "moderate"]:
            return None
        
        # Calculate required reduction based on overflow
        overflow_factor = density_analysis.density_ratio
        
        if overflow_factor <= self.config.font_tune_threshold:
            return None  # No adjustment needed
        
        # Calculate font size reduction needed
        # For mild overflow (1.1-1.2), reduce by 1-2 points
        # For moderate overflow (1.2-1.5), reduce by 2-3 points
        if overflow_factor <= 1.2:
            reduction = 1
        elif overflow_factor <= 1.3:
            reduction = 2
        else:
            reduction = min(3, self.config.font_adjustment_limit)
        
        new_size = current_font_size - reduction
        
        # Check bounds
        if new_size < self.config.min_font_size:
            new_size = self.config.min_font_size
            reduction = current_font_size - new_size
        
        safe_bounds = (
            new_size >= self.config.min_font_size and 
            reduction <= self.config.font_adjustment_limit
        )
        
        # Calculate confidence based on overflow severity
        if overflow_factor <= 1.15:
            confidence = 0.9
        elif overflow_factor <= 1.25:
            confidence = 0.8
        else:
            confidence = 0.7
        
        reasoning = f"{density_analysis.overflow_severity.title()} overflow detected, reducing font size within safe bounds"
        
        return FontAdjustment(
            original_size=current_font_size,
            recommended_size=new_size,
            adjustment_points=-reduction,
            confidence=confidence,
            reasoning=reasoning,
            safe_bounds=safe_bounds
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
        
        elif density_analysis.overflow_severity == "severe" or self.should_split_slide(density_analysis):
            # Severe overflow - split slide
            split_performed = True
            split_slides = self.split_slide_content(slide)
            split_count = len(split_slides)
            final_action = "split_slide"
            
        elif density_analysis.overflow_severity in ["mild", "moderate"]:
            # Mild to moderate overflow - try font adjustment
            font_adjustment = self.recommend_font_adjustment(density_analysis)
            if font_adjustment and font_adjustment.safe_bounds:
                final_action = "adjust_font"
            else:
                # Font adjustment not viable - split instead
                split_performed = True
                split_slides = self.split_slide_content(slide)
                split_count = len(split_slides)
                final_action = "split_slide"
        
        return ContentFitResult(
            slide_index=slide.index,
            density_analysis=density_analysis,
            font_adjustment=font_adjustment,
            split_performed=split_performed,
            split_count=split_count,
            final_action=final_action
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

    def _determine_action(self, density_ratio: float) -> str:
        """
        Determine recommended action based on density ratio.
        
        Args:
            density_ratio: Content density ratio
            
        Returns:
            Recommended action string
        """
        if density_ratio <= 1.0:
            return "no_action"
        elif density_ratio <= self.config.font_tune_threshold:
            return "no_action"
        elif density_ratio <= 1.2:
            return "adjust_font"
        elif density_ratio < self.config.split_threshold:
            return "adjust_font_or_rewrite"
        else:
            return "split_slide"

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