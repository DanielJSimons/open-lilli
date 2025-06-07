"""Slide planner for converting outlines into detailed slide plans."""

import logging
from typing import Dict, List, Optional

from openai import OpenAI

from .content_fit_analyzer import ContentFitAnalyzer
from .layout_recommender import LayoutRecommender
from .models import (
    ContentFitConfig, 
    GenerationConfig, 
    Outline, 
    SlidePlan, 
    VectorStoreConfig
)
from .template_parser import TemplateParser

logger = logging.getLogger(__name__)


class SlidePlanner:
    """Converts outlines into detailed slide plans with ML-assisted layout selection."""

    def __init__(
        self, 
        template_parser: TemplateParser,
        openai_client: Optional[OpenAI] = None,
        vector_config: Optional[VectorStoreConfig] = None,
        enable_ml_layouts: bool = True,
        content_fit_config: Optional[ContentFitConfig] = None
    ):
        """
        Initialize the slide planner.
        
        Args:
            template_parser: TemplateParser instance for layout information
            openai_client: Optional OpenAI client for ML layout recommendations
            vector_config: Vector store configuration for ML system
            enable_ml_layouts: Whether to use ML-assisted layout selection
            content_fit_config: Configuration for content fit analysis
        """
        self.template_parser = template_parser
        self.layout_priorities = self._define_layout_priorities()
        self.enable_ml_layouts = enable_ml_layouts and openai_client is not None
        
        # Initialize ML layout recommender if enabled
        if self.enable_ml_layouts:
            self.layout_recommender = LayoutRecommender(openai_client, vector_config)
            logger.info("SlidePlanner initialized with ML layout recommendations enabled")
        else:
            self.layout_recommender = None
            logger.info("SlidePlanner initialized with rule-based layout selection only")
        
        # Initialize content fit analyzer
        self.content_fit_analyzer = ContentFitAnalyzer(content_fit_config, openai_client)
        
        logger.info(f"SlidePlanner initialized with {len(self.template_parser.layout_map)} available layouts")

    def _define_layout_priorities(self) -> Dict[str, List[str]]:
        """
        Define layout priorities for different slide types.
        
        Returns:
            Dictionary mapping slide types to ordered lists of preferred layouts
        """
        return {
            "title": ["title", "section", "content"],
            "content": ["content", "title", "blank"],
            "image": ["image", "image_content", "content", "blank"],
            "chart": ["content", "image_content", "blank"],
            "two_column": ["two_column", "content", "blank"],
            "section": ["section", "title", "content"],
            "blank": ["blank", "content"]
        }

    def plan_slides(
        self, 
        outline: Outline, 
        config: Optional[GenerationConfig] = None
    ) -> List[SlidePlan]:
        """
        Convert an outline into detailed slide plans.
        
        Args:
            outline: Outline to convert
            config: Generation configuration
            
        Returns:
            List of detailed slide plans with layouts assigned
        """
        config = config or GenerationConfig()
        
        logger.info(f"Planning slides for outline with {outline.slide_count} slides")
        
        planned_slides = []
        
        for slide in outline.slides:
            # Apply planning logic
            planned_slide = self._plan_individual_slide(slide, config)
            planned_slides.append(planned_slide)
        
        # Apply global optimizations
        planned_slides = self._optimize_slide_sequence(planned_slides, config)
        
        # Apply content fit optimization
        planned_slides = self._optimize_content_fit(planned_slides, config)
        
        # Validate the plan
        self._validate_slide_plan(planned_slides, config)
        
        logger.info(f"Successfully planned {len(planned_slides)} slides")
        return planned_slides

    def _plan_individual_slide(
        self, 
        slide: SlidePlan, 
        config: GenerationConfig
    ) -> SlidePlan:
        """
        Plan an individual slide with enhanced details.
        
        Args:
            slide: Original slide plan
            config: Generation configuration
            
        Returns:
            Enhanced slide plan
        """
        # Start with the original slide data
        planned_slide = slide.model_copy()
        
        # Assign layout using ML or rule-based selection
        layout_recommendation = self._select_layout_with_ml(planned_slide)
        planned_slide.layout_id = layout_recommendation.layout_id
        
        # Store ML recommendation metadata if available
        if hasattr(planned_slide, 'ml_recommendation'):
            planned_slide.ml_recommendation = layout_recommendation
        else:
            # Add as a custom attribute for debugging
            planned_slide.__dict__['ml_recommendation'] = layout_recommendation
        
        # Apply bullet point limits
        if len(planned_slide.bullets) > config.max_bullets_per_slide:
            # Option 1: Truncate bullets
            if config.max_bullets_per_slide > 0:
                planned_slide.bullets = planned_slide.bullets[:config.max_bullets_per_slide]
                logger.warning(f"Slide {slide.index}: Truncated bullets from {len(slide.bullets)} to {len(planned_slide.bullets)}")
        
        # Enhance image queries based on content
        if config.include_images and not planned_slide.image_query:
            planned_slide.image_query = self._generate_image_query(planned_slide)
        elif not config.include_images:
            planned_slide.image_query = None
        
        # Handle chart data
        if not config.include_charts:
            planned_slide.chart_data = None
        
        # Add speaker notes if missing
        if not planned_slide.speaker_notes:
            planned_slide.speaker_notes = self._generate_speaker_notes(planned_slide)
        
        return planned_slide

    def _select_layout_with_ml(self, slide: SlidePlan) -> 'LayoutRecommendation':
        """
        Select layout using ML recommendation with rule-based fallback.
        
        Args:
            slide: SlidePlan to get layout recommendation for
            
        Returns:
            LayoutRecommendation object with layout choice and metadata
        """
        # Try ML recommendation first if enabled
        if self.enable_ml_layouts and self.layout_recommender:
            try:
                # Get available layouts for this template
                available_layouts = self.template_parser.layout_map
                
                # Get ML recommendation
                recommendation = self.layout_recommender.recommend_layout(
                    slide, available_layouts
                )
                
                # Log the recommendation for debugging
                logger.debug(f"ML recommendation for slide {slide.index}: "
                           f"{recommendation.slide_type} (confidence: {recommendation.confidence:.2f})")
                
                return recommendation
                
            except Exception as e:
                logger.warning(f"ML layout recommendation failed for slide {slide.index}: {e}")
                # Fall through to rule-based selection
        
        # Use rule-based fallback
        layout_id = self._select_layout(slide.slide_type)
        
        # Create a fallback recommendation object
        from .models import LayoutRecommendation
        return LayoutRecommendation(
            slide_type=slide.slide_type,
            layout_id=layout_id,
            confidence=0.5,
            reasoning=f"Rule-based selection for {slide.slide_type}",
            similar_slides=[],
            fallback_used=True
        )

    def _select_layout(self, slide_type: str) -> int:
        """
        Select the best available layout for a slide type.
        
        Args:
            slide_type: Type of slide
            
        Returns:
            Layout index
        """
        # Get prioritized layout options for this slide type
        preferred_layouts = self.layout_priorities.get(slide_type, ["content", "blank"])
        available_layouts = self.template_parser.list_available_layouts()
        
        # Find the first available preferred layout
        for preferred in preferred_layouts:
            if preferred in available_layouts:
                layout_index = self.template_parser.get_layout_index(preferred)
                logger.debug(f"Selected layout '{preferred}' (index {layout_index}) for slide type '{slide_type}'")
                return layout_index
        
        # Fallback to first available layout
        if available_layouts:
            fallback_layout = available_layouts[0]
            layout_index = self.template_parser.get_layout_index(fallback_layout)
            logger.warning(f"Using fallback layout '{fallback_layout}' for slide type '{slide_type}'")
            return layout_index
        
        # Last resort - use index 0
        logger.error(f"No suitable layout found for slide type '{slide_type}', using index 0")
        return 0

    def _generate_image_query(self, slide: SlidePlan) -> Optional[str]:
        """
        Generate an image search query based on slide content.
        
        Args:
            slide: Slide to generate query for
            
        Returns:
            Image search query or None
        """
        # Skip image generation for certain slide types
        skip_types = {"title", "section"}
        if slide.slide_type in skip_types:
            return None
        
        # Use existing query if present
        if slide.image_query:
            return slide.image_query
        
        # Generate based on title and content
        query_parts = []
        
        # Extract key terms from title
        title_words = slide.title.lower().split()
        business_keywords = {
            "revenue", "growth", "market", "strategy", "performance", 
            "analysis", "results", "team", "product", "customer",
            "innovation", "technology", "partnership", "expansion"
        }
        
        # Find business-relevant words in title
        for word in title_words:
            if word in business_keywords or len(word) > 6:  # Longer words are often key terms
                query_parts.append(word)
        
        # Add some context based on slide type
        if slide.slide_type == "chart":
            query_parts.append("business analytics")
        elif slide.slide_type == "content":
            query_parts.append("business meeting")
        
        # Create query
        if query_parts:
            query = " ".join(query_parts[:3])  # Limit to 3 terms
            logger.debug(f"Generated image query for slide {slide.index}: '{query}'")
            return query
        
        return None

    def _generate_speaker_notes(self, slide: SlidePlan) -> str:
        """
        Generate basic speaker notes for a slide.
        
        Args:
            slide: Slide to generate notes for
            
        Returns:
            Speaker notes text
        """
        if slide.slide_type == "title":
            return "Welcome and introduce the presentation topic"
        elif slide.slide_type == "section":
            return f"Transition to new section: {slide.title}"
        elif slide.bullets:
            bullet_count = len(slide.bullets)
            return f"Cover {bullet_count} key points, allowing time for questions"
        else:
            return f"Present the content of: {slide.title}"

    def _optimize_slide_sequence(
        self, 
        slides: List[SlidePlan], 
        config: GenerationConfig
    ) -> List[SlidePlan]:
        """
        Apply global optimizations to the slide sequence.
        
        Args:
            slides: List of planned slides
            config: Generation configuration
            
        Returns:
            Optimized slide sequence
        """
        optimized_slides = slides.copy()
        
        # Check if we need to split slides with too many bullets
        slides_to_add = []
        slides_to_remove = []
        
        for i, slide in enumerate(optimized_slides):
            if (len(slide.bullets) > config.max_bullets_per_slide and 
                config.max_bullets_per_slide > 0):
                
                # Split the slide
                split_slides = self._split_slide(slide, config.max_bullets_per_slide)
                if len(split_slides) > 1:
                    slides_to_remove.append(i)
                    slides_to_add.extend([(i, split_slide) for split_slide in split_slides])
                    logger.info(f"Split slide {slide.index} into {len(split_slides)} slides")
        
        # Apply the splits (in reverse order to maintain indices)
        for i in reversed(slides_to_remove):
            optimized_slides.pop(i)
        
        for insert_pos, slide in reversed(slides_to_add):
            optimized_slides.insert(insert_pos, slide)
        
        # Re-index slides
        for i, slide in enumerate(optimized_slides):
            slide.index = i
        
        # Apply slide limit
        if len(optimized_slides) > config.max_slides:
            original_count = len(optimized_slides)
            optimized_slides = optimized_slides[:config.max_slides]
            logger.warning(f"Truncated slides from {original_count} to {config.max_slides}")
        
        return optimized_slides

    def _split_slide(self, slide: SlidePlan, max_bullets: int) -> List[SlidePlan]:
        """
        Split a slide with too many bullets into multiple slides.
        
        Args:
            slide: Slide to split
            max_bullets: Maximum bullets per slide
            
        Returns:
            List of split slides
        """
        if len(slide.bullets) <= max_bullets:
            return [slide]
        
        split_slides = []
        bullets = slide.bullets.copy()
        part_num = 1
        
        while bullets:
            # Take the next batch of bullets
            current_bullets = bullets[:max_bullets]
            bullets = bullets[max_bullets:]
            
            # Create the split slide
            split_slide = slide.model_copy()
            split_slide.bullets = current_bullets
            
            # Modify title to indicate part
            if part_num == 1:
                split_slide.title = f"{slide.title} (Part {part_num})"
            else:
                split_slide.title = f"{slide.title} (Part {part_num})"
            
            # Update speaker notes
            if bullets:  # More slides to come
                split_slide.speaker_notes = f"Part {part_num} of {slide.title} - continue to next slide"
            else:  # Last slide
                split_slide.speaker_notes = f"Final part of {slide.title}"
            
            split_slides.append(split_slide)
            part_num += 1
        
        return split_slides

    def _optimize_content_fit(
        self,
        slides: List[SlidePlan],
        config: GenerationConfig
    ) -> List[SlidePlan]:
        """
        Optimize slides for content fit using density analysis and dynamic adjustments.
        
        Args:
            slides: List of planned slides
            config: Generation configuration
            
        Returns:
            Optimized slide sequence with content fit adjustments
        """
        logger.info("Optimizing slides for content fit")
        
        optimized_slides = []
        content_fit_results = []
        
        template_style = getattr(self.template_parser, 'template_style', None)
        
        for slide in slides:
            # Analyze content fit for this slide
            fit_result = self.content_fit_analyzer.optimize_slide_content(
                slide, template_style
            )
            content_fit_results.append(fit_result)
            
            if fit_result.final_action == "split_slide":
                # Split the slide
                split_slides = self.content_fit_analyzer.split_slide_content(slide)
                optimized_slides.extend(split_slides)
                logger.info(f"Split slide {slide.index} into {len(split_slides)} slides due to content overflow")
                
            elif fit_result.final_action == "adjust_font":
                # Keep slide but mark for font adjustment
                optimized_slide = slide.model_copy()
                if hasattr(optimized_slide, '__dict__'):
                    optimized_slide.__dict__['font_adjustment'] = fit_result.font_adjustment
                optimized_slides.append(optimized_slide)
                logger.debug(f"Slide {slide.index} marked for font adjustment: {fit_result.font_adjustment.recommended_size}pt")
                
            else:
                # No action needed
                optimized_slides.append(slide)
        
        # Re-index slides after potential splitting
        for i, slide in enumerate(optimized_slides):
            slide.index = i
        
        # Apply slide limit after content fit optimization
        if len(optimized_slides) > config.max_slides:
            original_count = len(optimized_slides)
            optimized_slides = optimized_slides[:config.max_slides]
            logger.warning(f"Truncated slides from {original_count} to {config.max_slides} after content fit optimization")
        
        # Store content fit results for reporting
        self._last_content_fit_results = content_fit_results
        
        # Log optimization summary
        optimization_summary = self.content_fit_analyzer.get_optimization_summary(content_fit_results)
        logger.info(f"Content fit optimization complete: "
                   f"{optimization_summary['slides_requiring_action']}/{optimization_summary['total_slides']} slides optimized, "
                   f"{optimization_summary['slides_split']} splits, "
                   f"{optimization_summary['slides_font_adjusted']} font adjustments")
        
        return optimized_slides

    def _validate_slide_plan(self, slides: List[SlidePlan], config: GenerationConfig) -> None:
        """
        Validate the final slide plan.
        
        Args:
            slides: Planned slides to validate
            config: Generation configuration
            
        Raises:
            ValueError: If validation fails
        """
        if not slides:
            raise ValueError("No slides in plan")
        
        if len(slides) > config.max_slides:
            raise ValueError(f"Too many slides: {len(slides)} > {config.max_slides}")
        
        # Check for required first slide
        if slides[0].slide_type != "title":
            logger.warning("First slide is not a title slide")
        
        # Validate indices are sequential
        for i, slide in enumerate(slides):
            if slide.index != i:
                logger.warning(f"Slide index mismatch: expected {i}, got {slide.index}")
                slide.index = i  # Fix it
        
        # Check layout assignments
        available_layouts = len(self.template_parser.prs.slide_layouts)
        for slide in slides:
            if slide.layout_id is None:
                raise ValueError(f"Slide {slide.index} has no layout assigned")
            if slide.layout_id >= available_layouts:
                raise ValueError(f"Slide {slide.index} has invalid layout ID: {slide.layout_id}")
        
        logger.info(f"Slide plan validation passed: {len(slides)} slides")

    def analyze_layout_usage(self, slides: List[SlidePlan]) -> Dict[str, int]:
        """
        Analyze layout usage across slides.
        
        Args:
            slides: Planned slides
            
        Returns:
            Dictionary of layout types to usage counts
        """
        layout_usage = {}
        available_layouts = self.template_parser.layout_map
        
        # Create reverse mapping from index to type
        index_to_type = {v: k for k, v in available_layouts.items()}
        
        for slide in slides:
            layout_type = index_to_type.get(slide.layout_id, f"layout_{slide.layout_id}")
            layout_usage[layout_type] = layout_usage.get(layout_type, 0) + 1
        
        return layout_usage

    def get_planning_summary(self, slides: List[SlidePlan]) -> Dict[str, any]:
        """
        Get a summary of the planning results.
        
        Args:
            slides: Planned slides
            
        Returns:
            Planning summary
        """
        layout_usage = self.analyze_layout_usage(slides)
        
        slide_types = {}
        images_count = 0
        charts_count = 0
        total_bullets = 0
        
        for slide in slides:
            # Count slide types
            slide_types[slide.slide_type] = slide_types.get(slide.slide_type, 0) + 1
            
            # Count visuals
            if slide.image_query:
                images_count += 1
            if slide.chart_data:
                charts_count += 1
            
            # Count bullets
            total_bullets += len(slide.bullets)
        
        return {
            "total_slides": len(slides),
            "slide_types": slide_types,
            "layout_usage": layout_usage,
            "visuals": {
                "images": images_count,
                "charts": charts_count
            },
            "content": {
                "total_bullets": total_bullets,
                "avg_bullets_per_slide": total_bullets / len(slides) if slides else 0
            }
        }