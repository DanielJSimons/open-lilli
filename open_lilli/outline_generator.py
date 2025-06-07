"""Outline generator using OpenAI GPT models."""

import json
import logging
import time
from typing import Optional

import openai
from openai import OpenAI
from pydantic import ValidationError

from .models import GenerationConfig, Outline, SlidePlan

logger = logging.getLogger(__name__)


class OutlineGenerator:
    """Generates structured presentation outlines using OpenAI models."""

    def __init__(self, client: OpenAI, model: str = "gpt-4", temperature: float = 0.3):
        """
        Initialize the outline generator.
        
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

    def generate_outline(
        self, 
        text: str, 
        config: Optional[GenerationConfig] = None,
        title: Optional[str] = None,
        language: str = "en"
    ) -> Outline:
        """
        Generate a structured outline from input text.
        
        Args:
            text: Input content to structure
            config: Generation configuration
            title: Optional title override
            language: Language code for the presentation
            
        Returns:
            Structured outline with planned slides
            
        Raises:
            ValueError: If outline generation fails
            openai.OpenAIError: If API calls fail
        """
        if not text or not text.strip():
            raise ValueError("Input text cannot be empty")
        
        config = config or GenerationConfig()
        
        logger.info(f"Generating outline for {len(text)} characters of text")
        logger.info(f"Config: max_slides={config.max_slides}, language={language}")
        
        # Create the prompt
        prompt = self._build_outline_prompt(text, config, title, language)
        
        # Generate outline with retries
        outline_data = self._call_openai_with_retries(prompt)
        
        # Parse and validate the response
        try:
            outline = Outline(**outline_data)
            logger.info(f"Successfully generated outline with {outline.slide_count} slides")
            return outline
            
        except ValidationError as e:
            logger.error(f"Failed to validate outline structure: {e}")
            raise ValueError(f"Generated outline is invalid: {e}")

    def _build_outline_prompt(
        self, 
        text: str, 
        config: GenerationConfig,
        title: Optional[str],
        language: str
    ) -> str:
        """Build the prompt for outline generation."""
        
        # Language-specific instructions
        lang_instructions = {
            "en": "Generate the presentation in English.",
            "es": "Generate the presentation in Spanish.",
            "fr": "Generate the presentation in French.",
            "de": "Generate the presentation in German.",
            "zh": "Generate the presentation in Chinese.",
            "ja": "Generate the presentation in Japanese.",
        }
        
        lang_instruction = lang_instructions.get(
            language, f"Generate the presentation in language code: {language}"
        )
        
        prompt = f"""You are an expert presentation designer. Create a structured outline for a professional PowerPoint presentation based on the provided content.

CONTENT TO ANALYZE:
{text}

REQUIREMENTS:
- Maximum {config.max_slides} slides
- Maximum {config.max_bullets_per_slide} bullet points per slide
- Tone: {config.tone}
- Complexity level: {config.complexity_level}
- {lang_instruction}
- Include appropriate slide types (title, content, image, chart, etc.)
- Ensure logical flow and clear narrative structure

SLIDE TYPES AVAILABLE:
- "title": Title slide with main title and subtitle
- "content": Standard content slide with title and bullet points
- "image": Image-focused slide with minimal text
- "chart": Data visualization slide
- "two_column": Two-column comparison layout
- "section": Section header/divider slide

OUTPUT FORMAT:
Return a JSON object with this exact structure:

{{
    "language": "{language}",
    "title": "Presentation Title",
    "subtitle": "Optional subtitle",
    "slides": [
        {{
            "index": 0,
            "slide_type": "title",
            "title": "Main Title",
            "bullets": [],
            "image_query": null,
            "chart_data": null,
            "speaker_notes": "Welcome and introduction"
        }},
        {{
            "index": 1,
            "slide_type": "content",
            "title": "Slide Title",
            "bullets": ["Key point 1", "Key point 2", "Key point 3"],
            "image_query": "relevant search term for images",
            "chart_data": null,
            "speaker_notes": "Additional context for speaker"
        }}
    ],
    "style_guidance": "Professional and data-driven presentation style",
    "target_audience": "Executive leadership team"
}}

GUIDELINES:
1. First slide should always be type "title"
2. Use "chart" type when the content mentions specific data or numbers
3. Use "image" type for conceptual or inspirational slides
4. Keep bullet points concise and actionable
5. Include relevant image_query for slides that would benefit from visuals
6. Add speaker_notes with helpful context
7. Ensure slides flow logically from introduction to conclusion
8. For chart slides, include chart_data with the relevant numbers if available

Generate the outline now:"""

        return prompt

    def _call_openai_with_retries(self, prompt: str) -> dict:
        """Call OpenAI API with exponential backoff retry logic."""
        
        for attempt in range(self.max_retries):
            try:
                logger.debug(f"OpenAI API call attempt {attempt + 1}")
                
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert presentation designer. Always respond with valid JSON only."
                        },
                        {
                            "role": "user", 
                            "content": prompt
                        }
                    ],
                    temperature=self.temperature,
                    max_tokens=4000,
                    response_format={"type": "json_object"}
                )
                
                content = response.choices[0].message.content
                if not content:
                    raise ValueError("Empty response from OpenAI")
                
                # Parse JSON response
                try:
                    outline_data = json.loads(content)
                    self._validate_outline_structure(outline_data)
                    return outline_data
                    
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON response: {e}")
                    if attempt == self.max_retries - 1:
                        raise ValueError(f"Invalid JSON response from OpenAI: {e}")
                    continue
                
            except openai.RateLimitError:
                if attempt == self.max_retries - 1:
                    raise
                wait_time = self.retry_delay * (2 ** attempt)
                logger.warning(f"Rate limited, waiting {wait_time}s before retry")
                time.sleep(wait_time)
                
            except openai.APIError as e:
                if attempt == self.max_retries - 1:
                    raise
                wait_time = self.retry_delay * (2 ** attempt)
                logger.warning(f"API error: {e}, waiting {wait_time}s before retry")
                time.sleep(wait_time)
        
        raise ValueError("Failed to generate outline after all retries")

    def _validate_outline_structure(self, data: dict) -> None:
        """Validate the basic structure of the outline data."""
        required_fields = ["language", "title", "slides"]
        for field in required_fields:
            if field not in data:
                raise ValueError(f"Missing required field: {field}")
        
        if not isinstance(data["slides"], list):
            raise ValueError("Slides must be a list")
        
        if len(data["slides"]) == 0:
            raise ValueError("At least one slide is required")
        
        # Validate each slide has required fields
        required_slide_fields = ["index", "slide_type", "title"]
        for i, slide in enumerate(data["slides"]):
            for field in required_slide_fields:
                if field not in slide:
                    raise ValueError(f"Slide {i} missing required field: {field}")

    def refine_outline(
        self, 
        outline: Outline, 
        feedback: str,
        config: Optional[GenerationConfig] = None
    ) -> Outline:
        """
        Refine an existing outline based on feedback.
        
        Args:
            outline: Current outline to refine
            feedback: User feedback or requirements
            config: Generation configuration
            
        Returns:
            Refined outline
        """
        config = config or GenerationConfig()
        
        prompt = f"""You are an expert presentation designer. Refine the following presentation outline based on the user feedback.

CURRENT OUTLINE:
{outline.model_dump_json(indent=2)}

USER FEEDBACK:
{feedback}

REQUIREMENTS:
- Maximum {config.max_slides} slides
- Maintain the same JSON structure
- Address the feedback while preserving good presentation flow
- Keep the same language: {outline.language}

Return the refined outline as a JSON object with the same structure."""

        outline_data = self._call_openai_with_retries(prompt)
        
        try:
            refined_outline = Outline(**outline_data)
            logger.info(f"Successfully refined outline based on feedback")
            return refined_outline
            
        except ValidationError as e:
            logger.error(f"Failed to validate refined outline: {e}")
            raise ValueError(f"Refined outline is invalid: {e}")