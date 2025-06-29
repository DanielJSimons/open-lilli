"""Slide planner for converting outlines into detailed slide plans."""

import logging
import re
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

    # Removed hardcoded layout priorities - now using pure LLM-based selection

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
        
        # Use LLM to analyze content and determine appropriate slide type
        enhanced_slide_type = self._analyze_content_with_llm(planned_slide)
        if enhanced_slide_type != planned_slide.slide_type:
            logger.info(f"LLM enhanced slide {planned_slide.index} type from '{planned_slide.slide_type}' to '{enhanced_slide_type}'")
            planned_slide.slide_type = enhanced_slide_type
        
        # Use template-driven approach: select layout first, then adapt content
        layout_recommendation = self._select_layout_with_ml(planned_slide)
        planned_slide.layout_id = layout_recommendation.layout_id
        
        # Store ML recommendation metadata if available
        if hasattr(planned_slide, 'ml_recommendation'):
            planned_slide.ml_recommendation = layout_recommendation
        else:
            # Add as a custom attribute for debugging
            planned_slide.__dict__['ml_recommendation'] = layout_recommendation
        
        # CRITICAL: Now adapt content to fit the selected template's structure
        adapted_slide = self._adapt_content_to_template(planned_slide)
        planned_slide = adapted_slide
        
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
    
    def _analyze_content_with_llm(self, slide: SlidePlan) -> str:
        """
        Use LLM to analyze slide content and determine the most appropriate slide type
        based on content intent and available template layouts.
        
        Args:
            slide: SlidePlan to analyze
            
        Returns:
            LLM-determined slide type or original if LLM analysis fails
        """
        try:
            if not self.layout_recommender or not self.enable_ml_layouts:
                logger.debug("LLM analysis not available, using original slide type")
                return slide.slide_type
            
            # Get available template layouts with their real names
            available_layouts = self.template_parser.layout_map
            real_layout_names = self.template_parser._extract_real_layout_names()
            
            # Create comprehensive content summary
            content_summary = self._create_comprehensive_content_summary(slide)
            
            # Create layout options summary
            layout_options = self._create_layout_options_summary(available_layouts, real_layout_names)
            
            # Build LLM prompt for content analysis
            prompt = self._build_content_analysis_prompt(content_summary, layout_options)
            
            # Get LLM analysis
            response = self.layout_recommender.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an expert presentation designer who understands content intent and can match content to appropriate template layouts. Analyze the slide content and determine what type of slide this should be based on its purpose and the available template options."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=150
            )
            
            # Parse LLM response
            analyzed_type = self._parse_content_analysis_response(response.choices[0].message.content, slide.slide_type)
            
            logger.debug(f"LLM content analysis: '{slide.slide_type}' → '{analyzed_type}'")
            return analyzed_type
            
        except Exception as e:
            logger.error(f"LLM content analysis failed for slide {slide.index}: {e}")
            return slide.slide_type
    
    def _create_comprehensive_content_summary(self, slide: SlidePlan) -> str:
        """
        Create a detailed summary of slide content for LLM analysis.
        
        Args:
            slide: SlidePlan to summarize
            
        Returns:
            Comprehensive content description
        """
        summary_parts = []
        
        # Title analysis
        summary_parts.append(f"SLIDE TITLE: '{slide.title}'")
        
        # Content analysis
        bullets = slide.get_bullet_texts()
        if bullets:
            summary_parts.append(f"CONTENT: {len(bullets)} bullet points")
            if len(bullets) <= 3:
                summary_parts.append(f"BULLET POINTS: {bullets}")
            else:
                summary_parts.append(f"SAMPLE BULLETS: {bullets[:2]} ... and {len(bullets)-2} more")
        else:
            summary_parts.append("CONTENT: No bullet points")
        
        # Media and special content
        if slide.image_query:
            summary_parts.append(f"IMAGES: Needs images ({slide.image_query})")
        if slide.chart_data:
            summary_parts.append("CHARTS: Contains data visualization")
        if slide.speaker_notes:
            summary_parts.append(f"NOTES: {slide.speaker_notes[:100]}...")
        
        # Slide context
        summary_parts.append(f"CURRENT_TYPE: {slide.slide_type}")
        summary_parts.append(f"SLIDE_INDEX: {slide.index}")
        
        return " | ".join(summary_parts)
    
    def _adapt_content_to_template(self, slide_plan: SlidePlan) -> SlidePlan:
        """
        Adapt slide content to fit the selected template's structure and expectations.
        This is the core of template-driven content planning.
        
        Args:
            slide_plan: Original slide plan with selected layout_id
            
        Returns:
            Adapted slide plan with content structured for the template
        """
        try:
            if not self.layout_recommender or not self.enable_ml_layouts:
                logger.debug("LLM not available for content adaptation, using original content")
                return slide_plan
            
            # Get the selected template layout
            layout_index = slide_plan.layout_id or 0
            if layout_index >= len(self.template_parser.prs.slide_layouts):
                logger.warning(f"Invalid layout index {layout_index}, using original content")
                return slide_plan
            
            selected_layout = self.template_parser.prs.slide_layouts[layout_index]
            
            # Analyze the template's intended content structure
            template_intent = self._analyze_template_intent(selected_layout, layout_index)
            
            # Adapt content to match template expectations
            adapted_content = self._generate_template_adapted_content(slide_plan, template_intent)
            
            # Create new slide plan with adapted content
            adapted_slide = slide_plan.model_copy()
            
            # Apply adapted content
            if adapted_content.get('title'):
                adapted_slide.title = adapted_content['title']
            if adapted_content.get('bullets'):
                adapted_slide.bullets = adapted_content['bullets']
                adapted_slide.bullet_hierarchy = None  # Clear hierarchy if using bullets
            if adapted_content.get('bullet_hierarchy'):
                adapted_slide.bullet_hierarchy = adapted_content['bullet_hierarchy']
                adapted_slide.bullets = []  # Clear legacy bullets
            
            # Add template-specific content
            if adapted_content.get('subtitle'):
                adapted_slide.__dict__['template_subtitle'] = adapted_content['subtitle']
            if adapted_content.get('footer'):
                adapted_slide.__dict__['template_footer'] = adapted_content['footer']
            if adapted_content.get('additional_content'):
                adapted_slide.__dict__['template_additional'] = adapted_content['additional_content']
            
            logger.info(f"Adapted content for slide {slide_plan.index} to template structure")
            return adapted_slide
            
        except Exception as e:
            logger.error(f"Content adaptation failed for slide {slide_plan.index}: {e}")
            return slide_plan
    
    def _analyze_template_intent(self, layout: 'SlideLayout', layout_index: int) -> Dict[str, any]:
        """
        Analyze the template layout to understand its intended content structure and purpose.
        This is the foundation of template-driven content planning.
        
        Args:
            layout: The selected SlideLayout object
            layout_index: Index of the layout
            
        Returns:
            Dictionary containing template intent analysis
        """
        try:
            if not self.layout_recommender or not self.enable_ml_layouts:
                logger.debug("LLM not available for template intent analysis, using basic analysis")
                return self._basic_template_intent_analysis(layout, layout_index)
            
            # Get comprehensive visual analysis from template parser
            visual_data = self.template_parser.extract_complete_layout_visual_data(layout, layout_index)
            
            # Create detailed layout description for LLM
            layout_description = self._create_layout_description_for_intent_analysis(visual_data, layout_index)
            
            # Build LLM prompt for template intent analysis
            prompt = self._build_template_intent_analysis_prompt(layout_description)
            
            # Get LLM analysis
            response = self.layout_recommender.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an expert presentation template analyst who understands layout design intent. Analyze template layouts to determine their intended content structure and purpose. Focus on what content each placeholder expects and how they relate to each other."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=500
            )
            
            # Parse LLM response
            template_intent = self._parse_template_intent_response(response.choices[0].message.content)
            
            # Enhance with visual data
            template_intent['visual_data'] = visual_data
            template_intent['layout_index'] = layout_index
            
            logger.debug(f"Analyzed template intent for layout {layout_index}: {template_intent.get('primary_purpose', 'unknown')}")
            return template_intent
            
        except Exception as e:
            logger.error(f"Template intent analysis failed for layout {layout_index}: {e}")
            return self._basic_template_intent_analysis(layout, layout_index)
    
    def _basic_template_intent_analysis(self, layout: 'SlideLayout', layout_index: int) -> Dict[str, any]:
        """
        Fallback template intent analysis without LLM.
        
        Args:
            layout: The SlideLayout object
            layout_index: Index of the layout
            
        Returns:
            Basic template intent dictionary
        """
        placeholders = list(layout.placeholders)
        
        # Count placeholder types
        title_count = len([p for p in placeholders if p.placeholder_format.type in (1, 13)])
        content_count = len([p for p in placeholders if p.placeholder_format.type in (2, 7)])
        subtitle_count = len([p for p in placeholders if p.placeholder_format.type == 3])
        image_count = len([p for p in placeholders if p.placeholder_format.type == 18])
        
        # Determine basic purpose
        if content_count >= 3:
            primary_purpose = "multi_column_content"
        elif content_count == 2:
            primary_purpose = "two_column_comparison"
        elif image_count > 0 and content_count > 0:
            primary_purpose = "image_with_content"
        elif image_count > 0:
            primary_purpose = "image_showcase"
        elif title_count > 0 and content_count == 0:
            primary_purpose = "title_section"
        else:
            primary_purpose = "standard_content"
        
        return {
            "primary_purpose": primary_purpose,
            "placeholder_expectations": {
                "title": "Main slide heading" if title_count > 0 else None,
                "subtitle": "Supporting tagline or summary" if subtitle_count > 0 else None,
                "content": f"Main content in {content_count} areas" if content_count > 0 else None,
                "image": f"Visual content in {image_count} areas" if image_count > 0 else None
            },
            "content_relationships": "sequential" if content_count > 1 else "single",
            "visual_hierarchy": "title_dominant" if title_count > 0 else "content_focused",
            "layout_index": layout_index,
            "llm_analysis": False
        }
    
    def _create_layout_description_for_intent_analysis(self, visual_data: Dict, layout_index: int) -> str:
        """
        Create a focused description for template intent analysis.
        
        Args:
            visual_data: Visual analysis data from template parser
            layout_index: Layout index
            
        Returns:
            Description string for LLM analysis
        """
        try:
            description_parts = []
            
            # Basic layout info
            description_parts.append(f"LAYOUT INDEX: {layout_index}")
            
            if visual_data.get('visual_summary'):
                description_parts.append(f"STRUCTURE: {visual_data['visual_summary']}")
            
            # Placeholder inventory
            shapes = visual_data.get('shapes', {})
            placeholders = shapes.get('placeholders', [])
            
            if placeholders:
                placeholder_summary = []
                for ph in placeholders:
                    ph_type = ph.get('placeholder_type', 'unknown')
                    content_type = ph.get('content_type', 'unknown')
                    placeholder_summary.append(f"{content_type}({ph_type})")
                
                description_parts.append(f"PLACEHOLDERS: {', '.join(placeholder_summary)}")
            
            # Spatial relationships
            spatial = visual_data.get('spatial_analysis', {})
            if spatial:
                grid = spatial.get('layout_grid', {})
                if grid.get('type') != 'single':
                    description_parts.append(f"ARRANGEMENT: {grid.get('type')} ({grid.get('columns')}x{grid.get('rows')})")
            
            # Design patterns
            patterns = visual_data.get('design_patterns', {})
            if patterns:
                purpose = patterns.get('primary_purpose', 'general')
                style = patterns.get('layout_style', 'standard')
                description_parts.append(f"DESIGN: {style} style for {purpose} content")
            
            # Content zones
            zones = visual_data.get('content_zones', {})
            zone_info = []
            for zone_name, zone_content in zones.items():
                if zone_content:
                    zone_info.append(f"{zone_name}({len(zone_content)})")
            if zone_info:
                description_parts.append(f"ZONES: {', '.join(zone_info)}")
            
            return " | ".join(description_parts)
            
        except Exception as e:
            logger.debug(f"Failed to create layout description: {e}")
            return f"Layout {layout_index}: Analysis failed"
    
    def _build_template_intent_analysis_prompt(self, layout_description: str) -> str:
        """
        Build LLM prompt for analyzing template intent.
        
        Args:
            layout_description: Description of the layout structure
            
        Returns:
            Complete prompt for template intent analysis
        """
        return f"""Analyze this PowerPoint template layout to understand its design intent and content expectations.

TEMPLATE LAYOUT:
{layout_description}

ANALYSIS REQUIRED:
1. PRIMARY PURPOSE: What is this layout designed to communicate? (e.g., comparison, process, showcase, overview)
2. PLACEHOLDER EXPECTATIONS: What specific content should go in each placeholder type?
3. CONTENT RELATIONSHIPS: How should content areas relate to each other?
4. VISUAL HIERARCHY: What's the intended information flow and emphasis?
5. CONTENT STRUCTURE: How should content be organized to match this template's design?

IMPORTANT: Focus on the template's INTENT, not just what it CAN contain. What was this layout specifically designed for?

Respond in this format:
PRIMARY_PURPOSE: [main communication goal]
TITLE_EXPECTATION: [what the title should communicate]
SUBTILE_EXPECTATION: [what subtitle should contain, if present]
CONTENT_EXPECTATION: [how main content should be structured]
IMAGE_EXPECTATION: [what images should support, if present]
CONTENT_RELATIONSHIPS: [how content areas should relate]
VISUAL_HIERARCHY: [intended information flow]
CONTENT_STRUCTURE: [recommended organization approach]"""
    
    def _parse_template_intent_response(self, response: str) -> Dict[str, any]:
        """
        Parse LLM response for template intent analysis.
        
        Args:
            response: LLM response text
            
        Returns:
            Parsed template intent dictionary
        """
        try:
            intent = {
                "primary_purpose": "general",
                "placeholder_expectations": {},
                "content_relationships": "independent",
                "visual_hierarchy": "balanced",
                "content_structure": "standard",
                "llm_analysis": True
            }
            
            # Extract structured information
            response_lines = response.strip().split('\n')
            
            for line in response_lines:
                line = line.strip()
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip().lower()
                    value = value.strip()
                    
                    if 'primary_purpose' in key:
                        intent['primary_purpose'] = value
                    elif 'title_expectation' in key:
                        intent['placeholder_expectations']['title'] = value
                    elif 'subtitle_expectation' in key:
                        intent['placeholder_expectations']['subtitle'] = value
                    elif 'content_expectation' in key:
                        intent['placeholder_expectations']['content'] = value
                    elif 'image_expectation' in key:
                        intent['placeholder_expectations']['image'] = value
                    elif 'content_relationships' in key:
                        intent['content_relationships'] = value
                    elif 'visual_hierarchy' in key:
                        intent['visual_hierarchy'] = value
                    elif 'content_structure' in key:
                        intent['content_structure'] = value
            
            return intent
            
        except Exception as e:
            logger.error(f"Failed to parse template intent response: {e}")
            return {
                "primary_purpose": "general",
                "placeholder_expectations": {},
                "content_relationships": "independent",
                "visual_hierarchy": "balanced",
                "content_structure": "standard",
                "llm_analysis": False
            }
    
    def _generate_template_adapted_content(self, slide_plan: 'SlidePlan', template_intent: Dict[str, any]) -> Dict[str, any]:
        """
        Generate content that's specifically adapted to fit the template's design intent.
        This is where content gets reshaped to match template expectations.
        
        Args:
            slide_plan: Original slide plan with content
            template_intent: Template intent analysis results
            
        Returns:
            Dictionary with adapted content for different placeholder types
        """
        try:
            if not self.layout_recommender or not self.enable_ml_layouts:
                logger.debug("LLM not available for content adaptation, using original content")
                return self._basic_content_adaptation(slide_plan, template_intent)
            
            # Create comprehensive content summary for adaptation
            original_content_summary = self._create_content_summary_for_adaptation(slide_plan)
            
            # Build adaptation prompt
            adaptation_prompt = self._build_content_adaptation_prompt(
                original_content_summary, template_intent
            )
            
            # Get LLM content adaptation
            response = self.layout_recommender.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an expert content strategist who adapts slide content to fit specific template designs. Transform content to match template intent while preserving the core message. Focus on what each placeholder type expects based on the template's design purpose."},
                    {"role": "user", "content": adaptation_prompt}
                ],
                temperature=0.2,
                max_tokens=800
            )
            
            # Parse adapted content
            adapted_content = self._parse_content_adaptation_response(response.choices[0].message.content)
            
            logger.debug(f"Adapted content for slide {slide_plan.index} to template intent: {template_intent.get('primary_purpose')}")
            return adapted_content
            
        except Exception as e:
            logger.error(f"Content adaptation failed for slide {slide_plan.index}: {e}")
            return self._basic_content_adaptation(slide_plan, template_intent)
    
    def _basic_content_adaptation(self, slide_plan: 'SlidePlan', template_intent: Dict[str, any]) -> Dict[str, any]:
        """
        Fallback content adaptation without LLM.
        
        Args:
            slide_plan: Original slide plan
            template_intent: Template intent analysis
            
        Returns:
            Basic adapted content dictionary
        """
        adapted_content = {
            "title": slide_plan.title,
            "bullets": slide_plan.bullets.copy() if slide_plan.bullets else [],
            "bullet_hierarchy": slide_plan.bullet_hierarchy.copy() if slide_plan.bullet_hierarchy else None
        }
        
        # Generate basic subtitle if template expects it
        placeholder_expectations = template_intent.get('placeholder_expectations', {})
        if placeholder_expectations.get('subtitle'):
            bullets = slide_plan.get_bullet_texts()
            if bullets and len(bullets) > 0:
                # Create subtitle from first bullet or summary
                first_bullet = bullets[0]
                if len(first_bullet) <= 60:
                    adapted_content['subtitle'] = first_bullet
                else:
                    adapted_content['subtitle'] = first_bullet[:57] + "..."
        
        return adapted_content
    
    def _create_content_summary_for_adaptation(self, slide_plan: 'SlidePlan') -> str:
        """
        Create a comprehensive content summary for adaptation.
        
        Args:
            slide_plan: Slide plan to summarize
            
        Returns:
            Content summary string
        """
        summary_parts = []
        
        # Title and type
        summary_parts.append(f"SLIDE TITLE: '{slide_plan.title}'")
        summary_parts.append(f"SLIDE TYPE: {slide_plan.slide_type}")
        
        # Content analysis
        bullets = slide_plan.get_bullet_texts()
        if bullets:
            summary_parts.append(f"MAIN CONTENT: {len(bullets)} bullet points")
            
            # Include actual bullet content
            for i, bullet in enumerate(bullets[:5]):  # Limit to first 5
                summary_parts.append(f"  • {bullet}")
            
            if len(bullets) > 5:
                summary_parts.append(f"  ... and {len(bullets) - 5} more points")
        else:
            summary_parts.append("MAIN CONTENT: No bullet points provided")
        
        # Additional content context
        if slide_plan.image_query:
            summary_parts.append(f"VISUAL CONTEXT: Expects images ({slide_plan.image_query})")
        
        if slide_plan.speaker_notes:
            summary_parts.append(f"SPEAKER NOTES: {slide_plan.speaker_notes[:100]}...")
        
        return "\n".join(summary_parts)
    
    def _build_content_adaptation_prompt(self, content_summary: str, template_intent: Dict[str, any]) -> str:
        """
        Build LLM prompt for content adaptation.
        
        Args:
            content_summary: Summary of original content
            template_intent: Template intent analysis
            
        Returns:
            Complete adaptation prompt
        """
        # Extract key template expectations
        primary_purpose = template_intent.get('primary_purpose', 'general')
        placeholder_expectations = template_intent.get('placeholder_expectations', {})
        content_structure = template_intent.get('content_structure', 'standard')
        content_relationships = template_intent.get('content_relationships', 'independent')
        
        prompt_parts = [
            "Adapt the following slide content to fit this template's specific design intent and placeholder expectations.",
            "",
            "ORIGINAL CONTENT:",
            content_summary,
            "",
            "TEMPLATE INTENT:",
            f"Primary Purpose: {primary_purpose}",
            f"Content Structure: {content_structure}",
            f"Content Relationships: {content_relationships}",
            "",
            "TEMPLATE PLACEHOLDER EXPECTATIONS:"
        ]
        
        # Add specific placeholder expectations
        for placeholder_type, expectation in placeholder_expectations.items():
            if expectation:
                prompt_parts.append(f"• {placeholder_type.upper()}: {expectation}")
        
        prompt_parts.extend([
            "",
            "ADAPTATION TASK:",
            "Reshape the content to match what this template expects. Generate appropriate content for each placeholder type based on the template's design intent.",
            "",
            "IMPORTANT GUIDELINES:",
            "1. Preserve the core message and key information",
            "2. Adapt content structure to match template expectations",
            "3. Generate content specifically for subtitle/footer placeholders if template expects them",
            "4. Ensure content coherence across all placeholders",
            "5. Match the template's intended communication style",
            "",
            "Respond in this format:",
            "ADAPTED_TITLE: [title adapted for this template]",
            "ADAPTED_SUBTITLE: [subtitle if template expects one]",
            "ADAPTED_BULLETS: [bullet points restructured for template]",
            "ADAPTED_FOOTER: [footer content if template expects one]",
            "ADAPTATION_REASONING: [brief explanation of changes made]"
        ])
        
        return "\n".join(prompt_parts)
    
    def _parse_content_adaptation_response(self, response: str) -> Dict[str, any]:
        """
        Parse LLM response for content adaptation.
        
        Args:
            response: LLM response text
            
        Returns:
            Parsed adapted content dictionary
        """
        try:
            adapted_content = {}
            
            # Split response into lines and parse
            response_lines = response.strip().split('\n')
            current_section = None
            current_content = []
            
            for line in response_lines:
                line = line.strip()
                
                if line.startswith('ADAPTED_TITLE:'):
                    current_section = 'title'
                    adapted_content['title'] = line.split(':', 1)[1].strip()
                elif line.startswith('ADAPTED_SUBTITLE:'):
                    current_section = 'subtitle'
                    subtitle = line.split(':', 1)[1].strip()
                    if subtitle and subtitle.lower() not in ['none', 'n/a', 'not needed']:
                        adapted_content['subtitle'] = subtitle
                elif line.startswith('ADAPTED_BULLETS:'):
                    current_section = 'bullets'
                    bullets_text = line.split(':', 1)[1].strip()
                    if bullets_text and bullets_text.lower() not in ['none', 'n/a']:
                        current_content = [bullets_text] if bullets_text else []
                elif line.startswith('ADAPTED_FOOTER:'):
                    current_section = 'footer'
                    footer = line.split(':', 1)[1].strip()
                    if footer and footer.lower() not in ['none', 'n/a', 'not needed']:
                        adapted_content['footer'] = footer
                elif line.startswith('ADAPTATION_REASONING:'):
                    current_section = 'reasoning'
                    adapted_content['reasoning'] = line.split(':', 1)[1].strip()
                elif line.startswith('•') or line.startswith('-'):
                    # Additional bullet points
                    if current_section == 'bullets':
                        bullet_text = line.lstrip('•-').strip()
                        if bullet_text:
                            current_content.append(bullet_text)
            
            # Process bullets
            if current_content and current_section == 'bullets':
                adapted_content['bullets'] = current_content
            
            return adapted_content
            
        except Exception as e:
            logger.error(f"Failed to parse content adaptation response: {e}")
            return {}
    
    def _create_layout_options_summary(self, available_layouts: Dict[str, int], real_names: Dict[int, str]) -> str:
        """
        Create a summary of available template layouts for LLM analysis.
        
        Args:
            available_layouts: Dictionary of layout names to indices
            real_names: Dictionary of layout indices to real names
            
        Returns:
            Summary of layout options
        """
        layout_descriptions = []
        
        for layout_name, layout_index in available_layouts.items():
            real_name = real_names.get(layout_index, layout_name)
            
            # Get visual analysis if available
            try:
                visual_data = self.template_parser.extract_complete_layout_visual_data(
                    self.template_parser.prs.slide_layouts[layout_index], layout_index
                )
                visual_summary = visual_data.get('visual_summary', 'No description available')
                layout_descriptions.append(f"'{real_name}': {visual_summary}")
            except:
                layout_descriptions.append(f"'{real_name}': Layout {layout_index}")
        
        return "\n".join(layout_descriptions)
    
    def _build_content_analysis_prompt(self, content_summary: str, layout_options: str) -> str:
        """
        Build LLM prompt for content analysis and slide type determination.
        
        Args:
            content_summary: Summary of slide content
            layout_options: Summary of available layouts
            
        Returns:
            Complete prompt for LLM
        """
        return f"""Analyze this slide content and determine what type of slide this should be based on its intent and purpose.

SLIDE CONTENT:
{content_summary}

AVAILABLE TEMPLATE LAYOUTS:
{layout_options}

TASK:
Based on the slide content's intent and purpose, determine what type of slide this should be. Consider:
1. What is the main purpose of this slide?
2. What kind of layout would best serve this content?
3. Are there any template layouts that seem specifically designed for this type of content?

Respond with just the most appropriate slide type (one word if possible, like: title, team, comparison, process, data, portfolio, services, content, image, etc.)

SLIDE_TYPE:"""
    
    def _parse_content_analysis_response(self, response: str, original_type: str) -> str:
        """
        Parse LLM response for content analysis.
        
        Args:
            response: LLM response text
            original_type: Original slide type as fallback
            
        Returns:
            Parsed slide type
        """
        try:
            # Extract slide type from response
            response_clean = response.strip().lower()
            
            # Look for slide type after "SLIDE_TYPE:" marker
            if "slide_type:" in response_clean:
                type_part = response_clean.split("slide_type:")[-1].strip()
                slide_type = type_part.split()[0] if type_part.split() else original_type
            else:
                # Take the last word/line as the slide type
                words = response_clean.split()
                slide_type = words[-1] if words else original_type
            
            # Clean up the slide type
            slide_type = slide_type.replace('.', '').replace(',', '').replace('!', '')
            
            # Validate that it's a reasonable slide type
            valid_types = {'title', 'team', 'comparison', 'process', 'data', 'portfolio', 'services', 
                         'content', 'image', 'chart', 'overview', 'section', 'blank', 'timeline', 
                         'gallery', 'about', 'contact', 'features', 'benefits'}
            
            if slide_type in valid_types:
                return slide_type
            else:
                logger.debug(f"LLM returned unknown slide type '{slide_type}', using original '{original_type}'")
                return original_type
                
        except Exception as e:
            logger.error(f"Failed to parse content analysis response: {e}")
            return original_type

    def _select_layout_with_ml(self, slide: SlidePlan) -> 'LayoutRecommendation':
        """
        Select layout using pure LLM analysis - no hardcoded rules or fallbacks.
        
        Args:
            slide: SlidePlan to get layout recommendation for
            
        Returns:
            LayoutRecommendation object with layout choice and metadata
        """
        try:
            # Use pure LLM-based layout selection
            layout_id = self._select_layout_with_llm_only(slide)
            
            # Get layout name for response
            real_names = self.template_parser._extract_real_layout_names()
            layout_name = real_names.get(layout_id, f"layout_{layout_id}")
            
            # Create recommendation object
            from .models import LayoutRecommendation
            return LayoutRecommendation(
                slide_type=layout_name,
                layout_id=layout_id,
                confidence=0.9,  # High confidence in LLM decision
                reasoning=f"LLM selected '{layout_name}' based on content analysis and template matching",
                similar_slides=[],
                fallback_used=False
            )
            
        except Exception as e:
            logger.error(f"LLM layout selection failed for slide {slide.index}: {e}")
            
            # Emergency fallback only if LLM completely fails
            from .models import LayoutRecommendation
            return LayoutRecommendation(
                slide_type="content",
                layout_id=0,
                confidence=0.1,
                reasoning=f"Emergency fallback due to LLM failure: {e}",
                similar_slides=[],
                fallback_used=True
            )

    def _select_layout_with_llm_only(self, slide: SlidePlan) -> int:
        """
        Use pure LLM analysis to select the best layout for a slide.
        No hardcoded rules or priorities - let the LLM decide based on content and available templates.
        
        Args:
            slide: SlidePlan with content to match
            
        Returns:
            Layout index selected by LLM
        """
        try:
            if not self.layout_recommender or not self.enable_ml_layouts:
                logger.warning("LLM not available for layout selection, using first available layout")
                return 0
            
            # Get available layouts with their real names and descriptions
            available_layouts = self.template_parser.layout_map
            real_names = self.template_parser._extract_real_layout_names()
            
            # Create detailed content analysis
            content_summary = self._create_comprehensive_content_summary(slide)
            
            # Create detailed layout descriptions
            layout_descriptions = []
            for layout_name, layout_index in available_layouts.items():
                real_name = real_names.get(layout_index, layout_name)
                
                try:
                    visual_data = self.template_parser.extract_complete_layout_visual_data(
                        self.template_parser.prs.slide_layouts[layout_index], layout_index
                    )
                    visual_summary = visual_data.get('visual_summary', 'No description')
                    recommended_content = visual_data.get('recommended_content_types', [])
                    content_hint = f" (best for: {', '.join(recommended_content[:3])})" if recommended_content else ""
                    
                    layout_descriptions.append(f"INDEX {layout_index}: '{real_name}' - {visual_summary}{content_hint}")
                except:
                    layout_descriptions.append(f"INDEX {layout_index}: '{real_name}' - Basic layout")
            
            # Build LLM prompt for template-driven layout selection
            prompt = f"""You are selecting a template layout that will define HOW this content should be structured and presented. The template you choose will determine what placeholders are available and how content should be adapted.

SLIDE CONTENT TO STRUCTURE:
{content_summary}

AVAILABLE TEMPLATE LAYOUTS (each with specific placeholder design):
{chr(10).join(layout_descriptions)}

CRITICAL SELECTION CRITERIA:
1. PLACEHOLDER COMPATIBILITY: Which template's placeholders best match this content's natural structure?
2. SEMANTIC ALIGNMENT: Which template was designed for this type of content/message?
3. CONTENT ADAPTATION: How well can this content be adapted to fit the template's intended structure?
4. TEMPLATE PURPOSE: What was this template designed to communicate?

IMPORTANT: You are not just placing content into a template - you are choosing the template that will RESHAPE how this content is presented. The selected template will determine:
- What subtitle/footer content should be generated
- How main content should be structured
- What additional elements should be included
- The overall presentation approach

Respond with only the INDEX number of the template that best defines how this content should be structured.

SELECTED_LAYOUT_INDEX:"""
            
            # Get LLM recommendation
            response = self.layout_recommender.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an expert presentation designer who understands that template selection determines content structure. Choose the template that best defines how the content should be organized and presented, not just where to place existing content. Respond only with the layout index number."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=50
            )
            
            # Parse response
            selected_index = self._parse_layout_selection_response(response.choices[0].message.content, available_layouts)
            
            real_name = real_names.get(selected_index, f"layout_{selected_index}")
            logger.info(f"LLM selected layout '{real_name}' (index {selected_index}) for slide: '{slide.title}'")
            
            return selected_index
            
        except Exception as e:
            logger.error(f"LLM layout selection failed for slide {slide.index}: {e}")
            # Fallback to first available layout
            return 0
    
    def _parse_layout_selection_response(self, response: str, available_layouts: Dict[str, int]) -> int:
        """
        Parse LLM response for layout selection.
        
        Args:
            response: LLM response text
            available_layouts: Available layout mappings
            
        Returns:
            Selected layout index
        """
        try:
            # Look for index number in response
            import re
            
            # Try to find "SELECTED_LAYOUT_INDEX:" followed by number
            index_match = re.search(r'SELECTED_LAYOUT_INDEX:\s*(\d+)', response, re.IGNORECASE)
            if index_match:
                selected_index = int(index_match.group(1))
            else:
                # Look for any number in the response
                numbers = re.findall(r'\b(\d+)\b', response)
                if numbers:
                    selected_index = int(numbers[0])  # Take first number found
                else:
                    logger.warning(f"No index found in LLM response: '{response}', using 0")
                    return 0
            
            # Validate the index is within available layouts
            max_index = max(available_layouts.values()) if available_layouts else 0
            if 0 <= selected_index <= max_index:
                return selected_index
            else:
                logger.warning(f"LLM selected invalid index {selected_index}, using 0")
                return 0
                
        except Exception as e:
            logger.error(f"Failed to parse layout selection response: {e}")
            return 0

    def _generate_image_query(self, slide: SlidePlan) -> Optional[str]:
        """
        Generate an image search query using LLM-based content analysis.
        
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
        
        # Generate using LLM if available
        if self.layout_recommender and self.enable_ml_layouts:
            try:
                # Create slide content summary
                content_summary = f"Title: {slide.title}"
                bullets = slide.get_bullet_texts()
                if bullets:
                    content_summary += f" | Content: {'; '.join(bullets[:3])}"
                    if len(bullets) > 3:
                        content_summary += f" ... and {len(bullets)-3} more points"
                
                # Ask LLM to generate appropriate image query
                response = self.layout_recommender.client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "system", 
                            "content": "You are an expert at selecting appropriate business images for presentation slides. Generate concise, professional image search queries."
                        },
                        {
                            "role": "user", 
                            "content": f"Generate a 2-3 word professional image search query for this slide content:\n\n{content_summary}\n\nProvide only the search query, no explanation."
                        }
                    ],
                    temperature=0.1,
                    max_tokens=20
                )
                
                query = response.choices[0].message.content.strip()
                # Clean up the query
                query = query.strip('"').strip("'").strip()
                
                if query and len(query.split()) <= 4:  # Reasonable query length
                    logger.debug(f"LLM generated image query for slide {slide.index}: '{query}'")
                    return query
                    
            except Exception as e:
                logger.debug(f"LLM image query generation failed for slide {slide.index}: {e}")
        
        # Simple fallback based on slide title only (no hardcoded keywords)
        title_words = [word for word in slide.title.split() if len(word) > 4]
        if title_words:
            query = " ".join(title_words[:2])  # Use first 2 significant words
            logger.debug(f"Generated fallback image query for slide {slide.index}: '{query}'")
            return query
        
        return None

    def _generate_speaker_notes(self, slide: SlidePlan) -> str:
        """
        Generate speaker notes using LLM-based content analysis.
        
        Args:
            slide: Slide to generate notes for
            
        Returns:
            Speaker notes text
        """
        # Generate using LLM if available
        if self.layout_recommender and self.enable_ml_layouts:
            try:
                # Create slide content summary
                content_summary = f"Slide Title: {slide.title}"
                bullets = slide.get_bullet_texts()
                if bullets:
                    content_summary += f"\nBullet Points: {'; '.join(bullets)}"
                content_summary += f"\nSlide Type: {slide.slide_type}"
                
                # Ask LLM to generate speaker notes
                response = self.layout_recommender.client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "system", 
                            "content": "You are an expert presentation coach. Generate concise, helpful speaker notes (1-2 sentences) that guide the presenter on how to deliver this slide effectively."
                        },
                        {
                            "role": "user", 
                            "content": f"Generate speaker notes for this slide:\n\n{content_summary}\n\nProvide practical guidance for the presenter in 1-2 sentences."
                        }
                    ],
                    temperature=0.3,
                    max_tokens=100
                )
                
                notes = response.choices[0].message.content.strip()
                if notes:
                    logger.debug(f"LLM generated speaker notes for slide {slide.index}")
                    return notes
                    
            except Exception as e:
                logger.debug(f"LLM speaker notes generation failed for slide {slide.index}: {e}")
        
        # Simple fallback based on content only (no hardcoded rules)
        if slide.get_bullet_texts():
            bullet_count = len(slide.get_bullet_texts())
            return f"Present the {bullet_count} key points, allowing time for audience engagement"
        else:
            return f"Present the content about: {slide.title}"

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