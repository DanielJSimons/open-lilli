"""Content generator for creating polished slide content using OpenAI."""

import json
import logging
import time
import asyncio
from typing import Dict, List, Optional
import yaml # Add this
from pathlib import Path # Add this

import openai
from openai import OpenAI, AsyncOpenAI

from .models import GenerationConfig, SlidePlan
from .template_parser import TemplateParser

# Setup module-level logger, as it's used in the helper function and class methods
logger = logging.getLogger(__name__)

TONE_PROFILES_PATH = Path(__file__).parent / "config" / "tone_profiles.yaml"

def _load_tone_profiles_static(path: Path) -> Dict[str, str]:
    if not path.exists():
        logger.warning(f"Tone profiles file not found at {path}. Returning empty profiles.")
        return {}
    try:
        with open(path, 'r', encoding='utf-8') as f:
            profiles = yaml.safe_load(f)
            if not isinstance(profiles, dict):
                logger.error(f"Tone profiles file at {path} is not a valid dictionary. Returning empty profiles.")
                return {}
            logger.info(f"Successfully loaded tone profiles from {path}")
            return profiles
    except Exception as e:
        logger.error(f"Error loading tone profiles from {path}: {e}. Returning empty profiles.")
        return {}

class ContentGenerator:
    """Generates polished slide content using OpenAI models."""

    def __init__(
        self, 
        client: OpenAI | AsyncOpenAI,
        model: str = "gpt-4", 
        temperature: float = 0.3,
        template_parser: Optional[TemplateParser] = None
    ):
        """
        Initialize the content generator.
        
        Args:
            client: OpenAI client instance
            model: Model name to use
            temperature: Temperature for generation (0.0 to 1.0)
            template_parser: Optional template parser for style context
        """
        self.client = client
        self.model = model
        self.temperature = temperature
        self.template_parser = template_parser
        self.max_retries = 3
        self.retry_delay = 1.0
        # Load tone profiles
        self.tone_profiles = _load_tone_profiles_static(TONE_PROFILES_PATH) # Uses the module-level function

    def generate_content(
        self,
        slides: List[SlidePlan],
        config: Optional[GenerationConfig] = None,
        style_guidance: Optional[str] = None,
        language: str = "en"
    ) -> List[SlidePlan]:
        """
        Generate polished content for all slides.
        
        Args:
            slides: List of slide plans to generate content for
            config: Generation configuration
            style_guidance: Optional style guidance
            language: Language code for content generation
            
        Returns:
            List of slides with generated content
        """
        config = config or GenerationConfig()
        
        logger.info(f"Generating content for {len(slides)} slides in language: {language}")
        
        enhanced_slides = []
        
        for slide in slides:
            try:
                enhanced_slide = self._generate_slide_content(
                    slide, config, style_guidance, language
                )
                enhanced_slides.append(enhanced_slide)
                
                # Small delay to avoid rate limiting
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Failed to generate content for slide {slide.index}: {e}")
                # Use original slide as fallback
                enhanced_slides.append(slide)
        
        logger.info(f"Content generation completed for {len(enhanced_slides)} slides")
        return enhanced_slides

    async def generate_content_async(
        self,
        slides: List[SlidePlan],
        config: Optional[GenerationConfig] = None,
        style_guidance: Optional[str] = None,
        language: str = "en",
    ) -> List[SlidePlan]:
        """Asynchronous version of ``generate_content`` using concurrency."""
        config = config or GenerationConfig()
        logger.info(f"Generating content for {len(slides)} slides in language: {language} (async)")

        tasks = [
            self._generate_slide_content_async(slide, config, style_guidance, language)
            for slide in slides
        ]
        results = await asyncio.gather(*tasks)
        logger.info(f"Content generation completed for {len(results)} slides")
        return list(results)

    def _generate_slide_content(
        self,
        slide: SlidePlan,
        config: GenerationConfig,
        style_guidance: Optional[str],
        language: str
    ) -> SlidePlan:
        """
        Generate content for a single slide.
        
        Args:
            slide: Slide plan to enhance
            config: Generation configuration
            style_guidance: Style guidance
            language: Language code
            
        Returns:
            Enhanced slide with generated content
        """
        # Skip content generation for certain slide types
        if slide.slide_type == "title" and slide.title and not slide.bullets:
            # Title slides usually don't need content generation
            logger.debug(f"Skipping content generation for title slide {slide.index}")
            return slide
        
        prompt = self._build_content_prompt(slide, config, style_guidance, language)
        
        try:
            response_data = self._call_openai_with_retries(prompt)
            enhanced_slide = self._apply_generated_content(slide, response_data)
            
            logger.debug(f"Generated content for slide {slide.index}: {enhanced_slide.title}")
            return enhanced_slide
            
        except Exception as e:
            logger.error(f"Content generation failed for slide {slide.index}: {e}")
            return slide

    async def _generate_slide_content_async(
        self,
        slide: SlidePlan,
        config: GenerationConfig,
        style_guidance: Optional[str],
        language: str,
    ) -> SlidePlan:
        """Asynchronous version of ``_generate_slide_content``."""
        if slide.slide_type == "title" and slide.title and not slide.bullets:
            logger.debug(f"Skipping content generation for title slide {slide.index}")
            return slide

        prompt = self._build_content_prompt(slide, config, style_guidance, language)

        try:
            response_data = await self._call_openai_with_retries_async(prompt)
            enhanced_slide = self._apply_generated_content(slide, response_data)
            logger.debug(f"Generated content for slide {slide.index}: {enhanced_slide.title}")
            return enhanced_slide
        except Exception as e:
            logger.error(f"Content generation failed for slide {slide.index}: {e}")
            return slide

    def _build_content_prompt(
        self,
        slide: SlidePlan,
        config: GenerationConfig,
        style_guidance: Optional[str],
        language: str
    ) -> str:
        """Build the prompt for content generation."""
        
        # Language-specific instructions
        lang_instructions = {
            "en": "Generate content in English",
            "es": "Generate content in Spanish", 
            "fr": "Generate content in French",
            "de": "Generate content in German",
            "zh": "Generate content in Chinese",
            "ja": "Generate content in Japanese"
        }
        
        lang_instruction = lang_instructions.get(
            language, f"Generate content in language code: {language}"
        )
        
        # Build style context
        style_context = ""
        if style_guidance:
            style_context = f"Style guidance: {style_guidance}\n"
        
        # Determine effective tone
        effective_tone = config.tone  # Start with default tone from config
        # self.tone_profiles is from __init__
        if language in self.tone_profiles:
            profile_tone = self.tone_profiles[language]
            # Ensure profile tone is not empty and is a string before overriding
            if profile_tone and isinstance(profile_tone, str):
                effective_tone = profile_tone
                logger.info(f"Using language-specific tone for '{language}': {effective_tone}")
            else:
                logger.warning(f"Empty or invalid tone profile for language '{language}' (value: {profile_tone}), using default from config: {config.tone}")

        style_context += f"Tone: {effective_tone}\n"
        style_context += f"Complexity level: {config.complexity_level}\n"
        
        # Specific instructions for German
        if language == "de":
            german_instructions = "For German, use the formal 'Sie' form. Keep sentences short and direct for good readability (German Flesch-Kincaid reading ease of 14 or less is ideal)."
            style_context += f"German Language Specifics: {german_instructions}\n"

        # Add template style context if available
        if self.template_parser:
            template_style_context = self._build_template_style_context()
            if template_style_context:
                style_context += f"\nTemplate style requirements:\n{template_style_context}"
        
        # Build slide context
        slide_context = f"""Current slide information:
- Index: {slide.index}
- Type: {slide.slide_type}
- Current title: {slide.title}
- Current bullets: {slide.bullets}
- Speaker notes: {slide.speaker_notes or 'None'}"""

        prompt = f"""You are an expert presentation writer. Your task is to enhance and polish slide content for a professional presentation.

{style_context}

{slide_context}

REQUIREMENTS:
- {lang_instruction}
- Maximum {config.max_bullets_per_slide} bullet points
- Keep content concise and impactful
- Ensure bullets are parallel in structure
- Make titles clear and engaging
- Generate helpful speaker notes

SLIDE TYPE GUIDELINES:
- title: Focus on compelling main title and subtitle
- content: Clear title with 3-5 concise bullet points
- section: Section header title, minimal bullets
- image: Title that complements visual, brief bullets
- chart: Data-focused title, bullets highlighting key insights

OUTPUT FORMAT:
Return a JSON object with this structure:

{{
    "title": "Enhanced slide title",
    "bullets": ["Bullet point 1", "Bullet point 2", "Bullet point 3"],
    "speaker_notes": "Enhanced speaker notes with context and talking points"
}}

GUIDELINES:
1. Improve clarity and impact of existing content
2. Ensure professional business language
3. Make bullet points actionable where appropriate
4. Keep titles under 60 characters when possible
5. Speaker notes should provide context and transitions
6. Maintain consistency with the slide type purpose

Generate enhanced content now:"""

        return prompt

    def _call_openai_with_retries(self, prompt: str) -> dict:
        """Call OpenAI API with retry logic."""
        
        for attempt in range(self.max_retries):
            try:
                logger.debug(f"Content generation API call attempt {attempt + 1}")
                
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert presentation content writer. Always respond with valid JSON only."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=self.temperature,
                    max_tokens=1000,
                    response_format={"type": "json_object"}
                )
                
                content = response.choices[0].message.content
                if not content:
                    raise ValueError("Empty response from OpenAI")
                
                try:
                    return json.loads(content)
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON response: {e}")
                    if attempt == self.max_retries - 1:
                        raise ValueError(f"Invalid JSON response: {e}")
                    continue
                
            except openai.RateLimitError:
                if attempt == self.max_retries - 1:
                    raise
                wait_time = self.retry_delay * (2 ** attempt)
                logger.warning(f"Rate limited, waiting {wait_time}s")
                time.sleep(wait_time)
                
            except openai.APIError as e:
                if attempt == self.max_retries - 1:
                    raise
                wait_time = self.retry_delay * (2 ** attempt)
                logger.warning(f"API error: {e}, waiting {wait_time}s")
                time.sleep(wait_time)
        
        raise ValueError("Failed to generate content after all retries")

    async def _call_openai_with_retries_async(self, prompt: str) -> dict:
        """Asynchronous version of ``_call_openai_with_retries``."""
        for attempt in range(self.max_retries):
            try:
                logger.debug(f"Content generation API call attempt {attempt + 1} (async)")
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert presentation content writer. Always respond with valid JSON only.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=self.temperature,
                    max_tokens=1000,
                    response_format={"type": "json_object"},
                )

                content = response.choices[0].message.content
                if not content:
                    raise ValueError("Empty response from OpenAI")

                try:
                    return json.loads(content)
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON response: {e}")
                    if attempt == self.max_retries - 1:
                        raise ValueError(f"Invalid JSON response: {e}")
                    continue

            except openai.RateLimitError:
                if attempt == self.max_retries - 1:
                    raise
                wait_time = self.retry_delay * (2 ** attempt)
                logger.warning(f"Rate limited, waiting {wait_time}s")
                await asyncio.sleep(wait_time)

            except openai.APIError as e:
                if attempt == self.max_retries - 1:
                    raise
                wait_time = self.retry_delay * (2 ** attempt)
                logger.warning(f"API error: {e}, waiting {wait_time}s")
                await asyncio.sleep(wait_time)

        raise ValueError("Failed to generate content after all retries")

    def _apply_generated_content(self, slide: SlidePlan, response_data: dict) -> SlidePlan:
        """Apply generated content to slide."""
        
        enhanced_slide = slide.model_copy()
        
        # Update title if provided and improved
        if "title" in response_data and response_data["title"]:
            new_title = response_data["title"].strip()
            if new_title and len(new_title) > 0:
                enhanced_slide.title = new_title
        
        # Update bullets if provided
        if "bullets" in response_data and isinstance(response_data["bullets"], list):
            new_bullets = [
                bullet.strip() for bullet in response_data["bullets"] 
                if bullet and bullet.strip()
            ]
            if new_bullets:
                enhanced_slide.bullets = new_bullets
        
        # Update speaker notes if provided
        if "speaker_notes" in response_data and response_data["speaker_notes"]:
            new_notes = response_data["speaker_notes"].strip()
            if new_notes:
                enhanced_slide.speaker_notes = new_notes
        
        return enhanced_slide

    def generate_speaker_notes(
        self,
        slide: SlidePlan,
        context: Optional[str] = None,
        language: str = "en"
    ) -> str:
        """
        Generate detailed speaker notes for a slide.
        
        Args:
            slide: Slide to generate notes for
            context: Additional context about the presentation
            language: Language code
            
        Returns:
            Generated speaker notes
        """
        lang_instruction = f"Generate speaker notes in language: {language}"
        
        context_section = ""
        if context:
            context_section = f"Presentation context: {context}\n"
        
        prompt = f"""Generate detailed speaker notes for this slide in a professional presentation.

{context_section}
Slide information:
- Title: {slide.title}
- Type: {slide.slide_type}
- Bullet points: {slide.bullets}

{lang_instruction}

Requirements:
- 2-3 sentences providing context and talking points
- Include smooth transitions to next topics
- Mention timing suggestions if relevant
- Professional, conversational tone
- Help the speaker feel confident and prepared

Return only the speaker notes text, no JSON or formatting."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system", 
                        "content": "You are an expert presentation coach. Generate clear, helpful speaker notes."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.4,
                max_tokens=300
            )
            
            notes = response.choices[0].message.content
            return notes.strip() if notes else ""
            
        except Exception as e:
            logger.error(f"Failed to generate speaker notes: {e}")
            return f"Present the key points of {slide.title}"

    def refine_content(
        self,
        slide: SlidePlan,
        feedback: str,
        language: str = "en"
    ) -> SlidePlan:
        """
        Refine slide content based on feedback.
        
        Args:
            slide: Slide to refine
            feedback: Specific feedback to address
            language: Language code
            
        Returns:
            Refined slide
        """
        prompt = f"""Refine this slide content based on the provided feedback.

Current slide:
- Title: {slide.title}
- Bullets: {slide.bullets}
- Speaker notes: {slide.speaker_notes}

Feedback to address:
{feedback}

Requirements:
- Generate content in language: {language}
- Address the specific feedback
- Maintain professional quality
- Keep the same slide structure

Return JSON with refined title, bullets, and speaker_notes."""

        try:
            response_data = self._call_openai_with_retries(prompt)
            refined_slide = self._apply_generated_content(slide, response_data)
            
            logger.info(f"Refined slide {slide.index} based on feedback")
            return refined_slide
            
        except Exception as e:
            logger.error(f"Failed to refine slide content: {e}")
            return slide

    def batch_generate_content(
        self,
        slides: List[SlidePlan],
        config: Optional[GenerationConfig] = None,
        style_guidance: Optional[str] = None,
        language: str = "en",
        batch_size: int = 5
    ) -> List[SlidePlan]:
        """
        Generate content for multiple slides in batches.
        
        Args:
            slides: Slides to process
            config: Generation configuration
            style_guidance: Style guidance
            language: Language code
            batch_size: Number of slides to process in each batch
            
        Returns:
            List of slides with generated content
        """
        config = config or GenerationConfig()
        enhanced_slides = []
        
        logger.info(f"Batch generating content for {len(slides)} slides (batch size: {batch_size})")
        
        for i in range(0, len(slides), batch_size):
            batch = slides[i:i + batch_size]
            
            logger.info(f"Processing batch {i//batch_size + 1}: slides {i} to {i + len(batch) - 1}")
            
            batch_results = self.generate_content(
                batch, config, style_guidance, language
            )
            
            enhanced_slides.extend(batch_results)
            
            # Longer delay between batches
            if i + batch_size < len(slides):
                time.sleep(2.0)
        
        return enhanced_slides

    def regenerate_specific_slides(
        self,
        slides: List[SlidePlan],
        config: Optional[GenerationConfig] = None,
        style_guidance: Optional[str] = None,
        language: str = "en",
        feedback: Optional[str] = None
    ) -> List[SlidePlan]:
        """
        Regenerate content for specific slides with optional feedback.
        
        Args:
            slides: List of slide plans to regenerate
            config: Generation configuration
            style_guidance: Optional style guidance
            language: Language code for content generation
            feedback: Optional specific feedback to incorporate
            
        Returns:
            List of slides with regenerated content
        """
        config = config or GenerationConfig()
        
        logger.info(f"Regenerating content for {len(slides)} specific slides")
        if feedback:
            logger.info(f"Applying feedback: {feedback[:100]}...")
        
        regenerated_slides = []
        
        for slide in slides:
            try:
                if feedback:
                    # Use feedback-based refinement
                    enhanced_slide = self.refine_content(slide, feedback, language)
                else:
                    # Standard content generation
                    enhanced_slide = self._generate_slide_content(
                        slide, config, style_guidance, language
                    )
                
                regenerated_slides.append(enhanced_slide)
                
                # Small delay to avoid rate limiting
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Failed to regenerate content for slide {slide.index}: {e}")
                # Use original slide as fallback
                regenerated_slides.append(slide)
        
        logger.info(f"Regeneration completed for {len(regenerated_slides)} slides")
        return regenerated_slides

    def get_content_statistics(self, slides: List[SlidePlan]) -> Dict[str, any]:
        """
        Get statistics about slide content.
        
        Args:
            slides: Slides to analyze
            
        Returns:
            Content statistics
        """
        total_bullets = sum(len(slide.bullets) for slide in slides)
        total_words = sum(
            len(slide.title.split()) + 
            sum(len(bullet.split()) for bullet in slide.bullets)
            for slide in slides
        )
        
        slides_with_notes = sum(1 for slide in slides if slide.speaker_notes)
        avg_bullets = total_bullets / len(slides) if slides else 0
        
        slide_types = {}
        for slide in slides:
            slide_types[slide.slide_type] = slide_types.get(slide.slide_type, 0) + 1
        
        return {
            "total_slides": len(slides),
            "total_bullets": total_bullets,
            "total_words": total_words,
            "avg_bullets_per_slide": round(avg_bullets, 2),
            "slides_with_speaker_notes": slides_with_notes,
            "slide_types": slide_types,
            "avg_words_per_slide": round(total_words / len(slides), 2) if slides else 0
        }
    
    def _build_template_style_context(self) -> str:
        """
        Build template style context for content generation prompts.
        
        Returns:
            Style context string with template font and color information
        """
        if not self.template_parser:
            return ""
        
        context_parts = []
        
        # Add theme colors
        if hasattr(self.template_parser, 'palette') and self.template_parser.palette:
            primary_colors = []
            accent_colors = []
            
            for color_name, color_value in self.template_parser.palette.items():
                if color_name in ['dk1', 'lt1']:
                    primary_colors.append(f"{color_name}: {color_value}")
                elif color_name.startswith('acc'):
                    accent_colors.append(f"{color_name}: {color_value}")
            
            if primary_colors:
                context_parts.append(f"- Primary colors: {', '.join(primary_colors)}")
            if accent_colors:
                context_parts.append(f"- Accent colors: {', '.join(accent_colors[:2])}")  # Use first 2 accent colors
        
        # Add font information if available
        if hasattr(self.template_parser, 'template_style') and self.template_parser.template_style:
            template_style = self.template_parser.template_style
            
            # Master font
            if template_style.master_font:
                font_info = f"{template_style.master_font.name}"
                if template_style.master_font.size:
                    font_info += f" ({template_style.master_font.size}pt)"
                context_parts.append(f"- Primary font: {font_info}")
            
            # Theme fonts
            if template_style.theme_fonts:
                if 'major' in template_style.theme_fonts:
                    context_parts.append(f"- Heading font: {template_style.theme_fonts['major']}")
                if 'minor' in template_style.theme_fonts:
                    context_parts.append(f"- Body font: {template_style.theme_fonts['minor']}")
        
        # Add brand voice guidance
        if context_parts:
            brand_voice = "Follow the corporate brand voice using the template's visual identity. "
            brand_voice += "Ensure content aligns with the professional styling indicated by the template design."
            context_parts.insert(0, f"- Brand voice: {brand_voice}")
        
        return "\n".join(context_parts)