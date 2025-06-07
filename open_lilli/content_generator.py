"""Content generator for creating polished slide content using OpenAI."""

import json
import logging
import time
from typing import Dict, List, Optional

import openai
from openai import OpenAI

from .models import GenerationConfig, SlidePlan

logger = logging.getLogger(__name__)


class ContentGenerator:
    """Generates polished slide content using OpenAI models."""

    def __init__(self, client: OpenAI, model: str = "gpt-4", temperature: float = 0.3):
        """
        Initialize the content generator.
        
        Args:
            client: OpenAI client instance
            model: Model name to use
            temperature: Temperature for generation (0.0 to 1.0)
        """
        self.client = client
        self.model = model
        self.temperature = temperature
        self.max_retries = 3
        self.retry_delay = 1.0

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
        
        style_context += f"Tone: {config.tone}\n"
        style_context += f"Complexity level: {config.complexity_level}\n"
        
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