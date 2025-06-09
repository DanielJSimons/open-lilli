"""Slide planner for converting outlines into detailed slide plans."""

import logging
from typing import Dict, List, Optional

from openai import OpenAI

from .content_fit_analyzer import ContentFitAnalyzer, SmartContentFitter
from .layout_recommender import LayoutRecommender
from .models import (
    ContentFitConfig, 
    GenerationConfig, 
    Outline, 
    SlidePlan, 
    VectorStoreConfig,
    DesignPattern
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
        content_fit_config: Optional[ContentFitConfig] = None,
        design_pattern: Optional[DesignPattern] = None
    ):
        """
        Initialize the slide planner.
        
        Args:
            template_parser: TemplateParser instance for layout information
            openai_client: Optional OpenAI client for ML layout recommendations
            vector_config: Vector store configuration for ML system
            enable_ml_layouts: Whether to use ML-assisted layout selection
            content_fit_config: Configuration for content fit analysis
            design_pattern: Optional DesignPattern object to influence planning decisions
        """
        self.template_parser = template_parser
        self.design_pattern = design_pattern
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
        
        # Initialize smart content fitter for rebalancing
        self.smart_fitter = SmartContentFitter(self.content_fit_analyzer)
        
        # Define layout up-shift hierarchy (T-99: Extended Layout Upshift Map)
        self.layout_upshift_map = {
            "content": "two_column",           # Standard content → two column for better spacing
            "image": "image_content",          # Image-only → image+content hybrid
            "image_content": "two_column",     # Image+content → two column for more space  
            "two_column": "content_dense",     # Two column → dense content layout if available
            "chart": "image_content",          # Chart → image+content for additional context
            "title": "section",                # Title → section header for more emphasis
            "section": "content",              # Section → content for additional details
            # Custom dense layouts (if template provides them)
            "content_dense": "blank",          # Dense content → blank for maximum flexibility
            "three_column": "content_dense",   # Three column → dense content
            "comparison": "two_column",        # Comparison → two column as fallback
        }

        logger.info(f"SlidePlanner initialized with {len(self.template_parser.layout_map)} available layouts. Upshift map: {self.layout_upshift_map}")

    def _define_layout_priorities(self) -> Dict[str, List[str]]:
        """
        Define layout priorities for different slide types with better variety.
        
        Returns:
            Dictionary mapping slide types to ordered lists of preferred layouts
        """
        return {
            "title": ["title", "section", "content"],
            "content": ["content", "two_column", "content_dense", "blank"],  # Prioritize variety
            "image": ["image", "image_content", "content", "blank"],
            "chart": ["image_content", "content", "content_dense", "two_column", "blank"],  # Better for charts
            "two_column": ["two_column", "comparison", "content_dense", "content", "blank"],
            "section": ["section", "title", "content"],
            "blank": ["blank", "content"],
            "comparison": ["comparison", "two_column", "three_column", "content_dense", "blank"],
            "content_dense": ["content_dense", "three_column", "two_column", "content", "blank"],
            "three_column": ["three_column", "content_dense", "two_column", "comparison", "blank"],
            # Enhanced content-based patterns with semantic support
            "team": ["image_content", "three_column", "two_column", "content", "blank"],  # Team pages need images
            "process": ["image_content", "two_column", "content", "blank"],
            "data": ["content_dense", "two_column", "comparison", "content", "blank"],
            "overview": ["three_column", "content_dense", "two_column", "content", "blank"],
            "portfolio": ["image", "image_content", "three_column", "content", "blank"],  # Portfolio needs images
            "services": ["content_dense", "three_column", "two_column", "content", "blank"]
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
        current_config = (config or GenerationConfig()).model_copy()

        if self.design_pattern and \
           (self.design_pattern.name == "minimalist" or self.design_pattern.primary_intent == "readability"):
            original_max_bullets = current_config.max_bullets_per_slide
            # Reduce by a fixed number or percentage, ensuring it's at least 1
            # Example: reduce by 2, or by 30%
            reduction = 2
            # reduction = int(current_config.max_bullets_per_slide * 0.3)

            current_config.max_bullets_per_slide = max(1, original_max_bullets - reduction)
            if current_config.max_bullets_per_slide != original_max_bullets:
                logger.info(
                    f"Design pattern '{self.design_pattern.name}' (intent: '{self.design_pattern.primary_intent}') "
                    f"adjusted max_bullets_per_slide from {original_max_bullets} "
                    f"to {current_config.max_bullets_per_slide}."
                )
        
        logger.info(f"Planning slides for outline with {outline.slide_count} slides using effective max_bullets: {current_config.max_bullets_per_slide}")
        
        planned_slides = []
        
        for slide in outline.slides:
            # Apply planning logic
            planned_slide = self._plan_individual_slide(slide, current_config)
            planned_slides.append(planned_slide)
        
        # Apply global optimizations
        planned_slides = self._optimize_slide_sequence(planned_slides, current_config)
        
        # Apply content fit optimization
        planned_slides = self._optimize_content_fit(planned_slides, current_config)
        
        # Validate the plan
        self._validate_slide_plan(planned_slides, current_config)
        
        logger.info(f"Successfully planned {len(planned_slides)} slides")
        return planned_slides

    def _plan_individual_slide(
        self, 
        slide: SlidePlan, 
        current_config: GenerationConfig # Use the potentially modified config
    ) -> SlidePlan:
        """
        Plan an individual slide with enhanced details.
        
        Args:
            slide: Original slide plan
            current_config: Generation configuration (possibly adjusted by design pattern)
            
        Returns:
            Enhanced slide plan
        """
        # Start with the original slide data
        planned_slide = slide.model_copy()
        
        # Enhance slide type based on content analysis before layout selection
        enhanced_slide_type = self._analyze_content_patterns(planned_slide)
        if enhanced_slide_type != planned_slide.slide_type:
            logger.debug(f"Enhanced slide {planned_slide.index} type from '{planned_slide.slide_type}' to '{enhanced_slide_type}'")
            planned_slide.slide_type = enhanced_slide_type
        
        # Assign layout using ML or rule-based selection
        layout_recommendation = self._select_layout_with_ml(planned_slide)
        planned_slide.layout_id = layout_recommendation.layout_id
        
        # Store ML recommendation metadata if available
        if hasattr(planned_slide, 'ml_recommendation'):
            planned_slide.ml_recommendation = layout_recommendation
        else:
            # Add as a custom attribute for debugging
            planned_slide.__dict__['ml_recommendation'] = layout_recommendation
        
        # Mark slides that need splitting - removal of hard truncation (T-100: Support hierarchical bullets)
        bullet_texts = planned_slide.get_bullet_texts()
        # Use current_config here
        if len(bullet_texts) > current_config.max_bullets_per_slide and current_config.max_bullets_per_slide > 0:
            # Mark slide for splitting in _optimize_slide_sequence rather than truncating
            planned_slide.needs_splitting = True
            logger.info(f"Slide {slide.index}: Marked for splitting due to {len(bullet_texts)} bullets > {current_config.max_bullets_per_slide} limit")
        
        # Enhance image queries based on content
        if current_config.include_images and not planned_slide.image_query: # Use current_config
            planned_slide.image_query = self._generate_image_query(planned_slide)
        elif not current_config.include_images: # Use current_config
            planned_slide.image_query = None
        
        # Handle chart data
        if not current_config.include_charts: # Use current_config
            planned_slide.chart_data = None
        
        # Add speaker notes if missing
        if not planned_slide.speaker_notes:
            planned_slide.speaker_notes = self._generate_speaker_notes(planned_slide)
        
        return planned_slide
    
    def _analyze_content_patterns(self, slide: SlidePlan) -> str:
        """
        Analyze slide content to determine the most appropriate slide type.
        
        Args:
            slide: SlidePlan to analyze
            
        Returns:
            Enhanced slide type based on content patterns
        """
        title = slide.title.lower()
        bullets = [bullet.lower() for bullet in slide.get_bullet_texts()]
        all_content = f"{title} {' '.join(bullets)}"
        
        # Enhanced pattern detection keywords
        team_words = ['team', 'staff', 'people', 'member', 'employee', 'founder', 'leadership', 'meet', 'our team', 'about us', 'who we are', 'bio', 'profile']
        comparison_words = ['vs', 'versus', 'compared to', 'difference', 'contrast', 'before/after', 'pros and cons']
        process_words = ['step', 'process', 'workflow', 'procedure', 'method', 'approach', 'strategy', 'timeline', 'roadmap']
        data_words = ['data', 'metrics', 'statistics', 'numbers', 'results', 'performance', 'analysis']
        overview_words = ['overview', 'summary', 'key points', 'highlights', 'main', 'primary', 'executive']
        portfolio_words = ['portfolio', 'gallery', 'showcase', 'examples', 'work', 'projects', 'case studies']
        service_words = ['services', 'products', 'features', 'benefits', 'offerings', 'solutions']
        
        # Count bullet points to assess density
        bullet_count = len(bullets)
        
        # Enhanced pattern matching with semantic content detection
        if any(word in all_content for word in team_words):
            return 'team'
        elif any(word in all_content for word in comparison_words) or bullet_count >= 2 and 'comparison' in title:
            return 'comparison'
        elif any(word in all_content for word in process_words) or 'flow' in title:
            return 'process'
        elif any(word in all_content for word in data_words) or slide.chart_data:
            return 'data'
        elif any(word in all_content for word in portfolio_words):
            return 'portfolio'
        elif any(word in all_content for word in service_words):
            return 'services'
        elif any(word in all_content for word in overview_words) or bullet_count >= 6:
            return 'overview'
        elif bullet_count >= 8:  # Very dense content
            return 'content_dense'
        elif bullet_count >= 4 and slide.image_query:  # Image with multiple points
            return 'image_content'
        elif bullet_count == 0 and slide.image_query:  # Image-focused
            return 'image'
        elif 'section' in slide.slide_type or 'intro' in title:
            return 'section'
        
        # Return original slide type if no patterns detected
        return slide.slide_type

    def _select_layout_with_ml(self, slide: SlidePlan) -> 'LayoutRecommendation':
        """
        Select layout using LLM/ML recommendation with rule-based fallback.
        
        Args:
            slide: SlidePlan to get layout recommendation for
            
        Returns:
            LayoutRecommendation object with layout choice and metadata
        """
        # Try visual analysis-based recommendation first if enabled
        if self.enable_ml_layouts and self.layout_recommender:
            try:
                # Get available layouts for this template
                available_layouts = self.template_parser.layout_map
                
                # First try comprehensive visual analysis approach
                try:
                    visual_recommendation = self.layout_recommender.recommend_layout_with_visual_analysis(
                        slide, self.template_parser, available_layouts
                    )
                    
                    # Log the visual analysis recommendation
                    logger.debug(f"Visual analysis recommendation for slide {slide.index}: "
                               f"{visual_recommendation.slide_type} (confidence: {visual_recommendation.confidence:.2f})")
                    
                    # Use visual analysis recommendation if confidence is reasonable
                    if visual_recommendation.confidence >= 0.3:  # Lower threshold for visual analysis
                        return visual_recommendation
                    else:
                        logger.debug(f"Visual analysis confidence too low ({visual_recommendation.confidence:.2f}), trying LLM fallback")
                
                except Exception as e:
                    logger.debug(f"Visual analysis layout recommendation failed for slide {slide.index}: {e}")
                
                # Fallback to LLM-based semantic analysis if visual analysis fails
                try:
                    llm_recommendation = self.layout_recommender.recommend_layout_with_llm(
                        slide, available_layouts
                    )
                    
                    # Log the LLM recommendation for debugging
                    logger.debug(f"LLM fallback recommendation for slide {slide.index}: "
                               f"{llm_recommendation.slide_type} (confidence: {llm_recommendation.confidence:.2f})")
                    
                    # Use LLM recommendation if confidence is reasonable
                    if llm_recommendation.confidence >= 0.4:
                        return llm_recommendation
                    else:
                        logger.debug(f"LLM confidence too low ({llm_recommendation.confidence:.2f}), trying traditional ML")
                
                except Exception as e:
                    logger.debug(f"LLM layout recommendation failed for slide {slide.index}: {e}")
                
                # Final fallback to traditional ML recommendation
                ml_recommendation = self.layout_recommender.recommend_layout(
                    slide, available_layouts
                )
                
                # Log the ML recommendation for debugging
                logger.debug(f"Traditional ML recommendation for slide {slide.index}: "
                           f"{ml_recommendation.slide_type} (confidence: {ml_recommendation.confidence:.2f})")
                
                return ml_recommendation
                
            except Exception as e:
                logger.warning(f"All ML layout recommendations failed for slide {slide.index}: {e}")
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
        Select the best available layout for a slide type using semantic matching.
        
        Args:
            slide_type: Type of slide
            
        Returns:
            Layout index
        """
        # First, try semantic matching for specialized content types
        semantic_matches = self.template_parser.find_layouts_for_content_type([slide_type])
        
        if semantic_matches:
            # Use the first semantically matched layout
            layout_index = semantic_matches[0]
            layout_name = self.template_parser.reverse_layout_map.get(layout_index, f"layout_{layout_index}")
            logger.info(f"Found semantic match: layout '{layout_name}' (index {layout_index}) for slide type '{slide_type}'")
            return layout_index
        
        # If no semantic match, fall back to prioritized layout options
        preferred_layouts = self.layout_priorities.get(slide_type, ["content", "blank"])
        available_layouts = self.template_parser.list_available_layouts()
        
        # Find the first available preferred layout
        for preferred in preferred_layouts:
            if preferred in available_layouts:
                layout_index = self.template_parser.get_layout_index(preferred)
                logger.debug(f"Selected priority layout '{preferred}' (index {layout_index}) for slide type '{slide_type}'")
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
        elif slide.get_bullet_texts():
            bullet_count = len(slide.get_bullet_texts())
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
        
        # Check if we need to split slides marked for splitting
        slides_to_add = []
        slides_to_remove = []
        
        for i, slide in enumerate(optimized_slides):
            # Check for the needs_splitting attribute set in _plan_individual_slide
            needs_splitting = getattr(slide, 'needs_splitting', False)
            if needs_splitting:
                # Split the slide
                split_slides = self._split_slide(slide, config.max_bullets_per_slide)
                if len(split_slides) > 1:
                    slides_to_remove.append(i)
                    slides_to_add.extend([(i, split_slide) for split_slide in split_slides])
                    logger.info(f"Split slide {slide.index} into {len(split_slides)} slides - zero content loss")
                else:
                    # Clear the needs_splitting flag if we couldn't actually split
                    slide.needs_splitting = False
        
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
        Split a slide with too many bullets into multiple slides (T-100: Support hierarchical bullets).
        
        Args:
            slide: Slide to split
            max_bullets: Maximum bullets per slide
            
        Returns:
            List of split slides
        """
        bullet_texts = slide.get_bullet_texts()
        if len(bullet_texts) <= max_bullets:
            return [slide]
        
        split_slides = []
        part_num = 1
        
        # Handle hierarchical vs legacy bullets
        if slide.bullet_hierarchy is not None:
            # For hierarchical bullets, use content fit analyzer's chunking
            from .content_fit_analyzer import ContentFitAnalyzer
            analyzer = ContentFitAnalyzer()
            bullet_chunks = analyzer._chunk_hierarchical_bullets(slide.bullet_hierarchy, max_bullets)
            
            for chunk in bullet_chunks:
                split_slide = slide.model_copy()
                split_slide.bullet_hierarchy = chunk
                split_slide.bullets = []  # Clear legacy bullets
                
                # Modify title to indicate part
                split_slide.title = f"{slide.title} (Part {part_num})"
                
                # Update speaker notes
                if part_num < len(bullet_chunks):
                    split_slide.speaker_notes = f"Part {part_num} of {slide.title} - continue to next slide"
                else:
                    split_slide.speaker_notes = f"Final part of {slide.title}"
                
                split_slides.append(split_slide)
                part_num += 1
        else:
            # Legacy bullet handling
            bullets = slide.bullets.copy()
            
            while bullets:
                # Take the next batch of bullets
                current_bullets = bullets[:max_bullets]
                bullets = bullets[max_bullets:]
                
                # Create the split slide
                split_slide = slide.model_copy()
                split_slide.bullets = current_bullets
                split_slide.bullet_hierarchy = None
                
                # Modify title to indicate part
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
        
        template_style = getattr(self.template_parser, 'template_style', None)
        
        # Step 1: Apply SmartContentFitter rebalancing first
        logger.info("Applying smart content rebalancing across slides")
        rebalanced_slides = self.smart_fitter.rebalance(slides, template_style)
        
        optimized_slides = []
        content_fit_results = []
        
        for i, original_slide_from_input_list in enumerate(slides):
            # Use rebalanced slide if available, otherwise original
            rebalanced_slide = rebalanced_slides[i] if i < len(rebalanced_slides) else original_slide_from_input_list
            
            # Initial optimization attempt (summarization T-74, font tuning T-75)
            fit_result = self.content_fit_analyzer.optimize_slide_content(
                rebalanced_slide, template_style
            )
            content_fit_results.append(fit_result)

            current_slide_plan = fit_result.modified_slide_plan if fit_result.modified_slide_plan else rebalanced_slide
            needs_further_action = fit_result.density_analysis.requires_action or fit_result.final_action == "split_slide"

            if needs_further_action:
                # Step 2a: Try dynamic layout upgrading first (T-92)
                logger.debug(f"Slide {original_slide_from_input_list.index}: Attempting dynamic layout upgrade")
                upgraded_slide = self.content_fit_analyzer.dynamic_layout_upgrading(
                    current_slide_plan, self.template_parser, template_style
                )
                
                if upgraded_slide:
                    # Verify the upgrade actually resolves the issue
                    upgrade_density_analysis = self.content_fit_analyzer.analyze_slide_density(upgraded_slide, template_style)
                    if not upgrade_density_analysis.requires_action:
                        optimized_slides.append(upgraded_slide)
                        logger.info(f"Slide {original_slide_from_input_list.index}: Dynamic layout upgrade successful, content now fits")
                        continue  # Skip to next slide
                    else:
                        current_slide_plan = upgraded_slide  # Use upgraded layout for further processing
                        logger.info(f"Slide {original_slide_from_input_list.index}: Dynamic layout upgrade provided improvement, continuing with further optimization")
                
                # Step 2b: Try legacy layout up-shift if dynamic upgrade didn't fully resolve
                current_layout_id = current_slide_plan.layout_id
                current_layout_type = self.template_parser.get_layout_type_by_id(current_layout_id)
                upshift_attempted_and_succeeded = False

                if current_layout_type and current_layout_type in self.layout_upshift_map:
                    target_upshift_layout_type = self.layout_upshift_map[current_layout_type]
                    try:
                        new_layout_id = self.template_parser.get_layout_index(target_upshift_layout_type)
                        if new_layout_id is not None and new_layout_id != current_layout_id:
                            upshifted_slide_plan = current_slide_plan.model_copy()
                            upshifted_slide_plan.layout_id = new_layout_id
                            # Clear previous font adjustment as new layout might have different defaults
                            if hasattr(upshifted_slide_plan, '__dict__') and 'font_adjustment' in upshifted_slide_plan.__dict__:
                                del upshifted_slide_plan.__dict__['font_adjustment']

                            logger.info(f"Slide {original_slide_from_input_list.index}: Attempting layout up-shift from '{current_layout_type}' (ID: {current_layout_id}) to '{target_upshift_layout_type}' (ID: {new_layout_id}).")

                            new_density_analysis = self.content_fit_analyzer.analyze_slide_density(upshifted_slide_plan, template_style)

                            if not new_density_analysis.requires_action:
                                optimized_slides.append(upshifted_slide_plan)
                                logger.info(f"Slide {original_slide_from_input_list.index}: Layout up-shift successful. Content now fits in '{target_upshift_layout_type}'.")
                                upshift_attempted_and_succeeded = True
                            else:
                                # If up-shift didn't help enough, try font adjustment on the new layout if applicable
                                if new_density_analysis.recommended_action == "adjust_font":
                                    logger.info(f"Slide {original_slide_from_input_list.index}: Up-shifted layout '{target_upshift_layout_type}' still requires action. Attempting font adjustment.")
                                    # Assuming current_font_size needs to be determined for the new layout, or use a default
                                    current_font_size_for_upshifted = getattr(upshifted_slide_plan, 'body_font_size', 18.0)
                                    font_adj_for_upshifted = self.content_fit_analyzer.recommend_font_adjustment(new_density_analysis, template_style, current_font_size=current_font_size_for_upshifted)
                                    if font_adj_for_upshifted and font_adj_for_upshifted.safe_bounds:
                                        if hasattr(upshifted_slide_plan, '__dict__'):
                                             upshifted_slide_plan.__dict__['font_adjustment'] = font_adj_for_upshifted
                                        optimized_slides.append(upshifted_slide_plan)
                                        logger.info(f"Slide {original_slide_from_input_list.index}: Layout up-shifted and font adjusted. Final layout: '{target_upshift_layout_type}'.")
                                        upshift_attempted_and_succeeded = True
                                    else:
                                        logger.info(f"Slide {original_slide_from_input_list.index}: Font adjustment on up-shifted layout not viable. Proceeding to split original/modified slide.")
                                else:
                                     logger.info(f"Slide {original_slide_from_input_list.index}: Up-shifted layout '{target_upshift_layout_type}' still requires action ({new_density_analysis.recommended_action}), but not font adjustment. Proceeding to split.")
                        else:
                            logger.debug(f"Slide {original_slide_from_input_list.index}: Target upshift layout '{target_upshift_layout_type}' not found or same as current.")
                    except ValueError as e:
                        logger.warning(f"Slide {original_slide_from_input_list.index}: Error getting layout index for upshift target '{target_upshift_layout_type}': {e}")
                
                if not upshift_attempted_and_succeeded:
                    logger.info(f"Slide {original_slide_from_input_list.index}: Layout up-shift not attempted or not successful. Proceeding to split.")
                    # Use current_slide_plan which might have been summarized or had font tuned before upshift attempt
                    split_slides = self.content_fit_analyzer.split_slide_content(current_slide_plan)
                    optimized_slides.extend(split_slides)
                    logger.info(f"Split slide {original_slide_from_input_list.index} into {len(split_slides)} slides.")

            else: # No further action needed after initial T-74/T-75 optimization
                # current_slide_plan already incorporates T-74/T-75 changes
                # Font adjustment from fit_result should be applied if present
                slide_to_add = current_slide_plan.model_copy() # current_slide_plan could be original_slide_from_input_list or fit_result.modified_slide_plan
                if fit_result.font_adjustment and not hasattr(slide_to_add.__dict__, 'font_adjustment'): # Apply if not already part of modified_slide_plan's creation logic
                     if hasattr(slide_to_add, '__dict__'):
                        slide_to_add.__dict__['font_adjustment'] = fit_result.font_adjustment
                
                optimized_slides.append(slide_to_add)
                logger.debug(f"Slide {original_slide_from_input_list.index}: Content fits after initial optimization (action: {fit_result.final_action}). Summarized: {slide_to_add.summarized_by_llm}")

        # Re-index slides after potential splitting/rewriting
        for i, slide_obj in enumerate(optimized_slides): # Renamed slide to slide_obj to avoid conflict
            slide_obj.index = i # Corrected variable name here
        
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
            
            # Count bullets (T-100: Support hierarchical bullets)
            total_bullets += len(slide.get_bullet_texts())
        
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