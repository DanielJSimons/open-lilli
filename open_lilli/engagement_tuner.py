"""Engagement Prompt Tuner for varied verb choices and rhetorical questions (T-81)."""

import logging
import re
import json
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
from collections import Counter
from enum import Enum

from openai import OpenAI, AsyncOpenAI
from pydantic import BaseModel, Field

from .models import SlidePlan, GenerationConfig

logger = logging.getLogger(__name__)


class EngagementTechnique(str, Enum):
    """Types of engagement techniques."""
    VARIED_VERBS = "varied_verbs"
    RHETORICAL_QUESTIONS = "rhetorical_questions"
    ACTIVE_VOICE = "active_voice"
    COMPELLING_LANGUAGE = "compelling_language"
    AUDIENCE_INTERACTION = "audience_interaction"


@dataclass
class VerbAnalysis:
    """Analysis of verb usage in content."""
    total_verbs: int
    unique_verbs: int
    verb_diversity_ratio: float
    most_common_verbs: List[Tuple[str, int]]
    repeated_verbs: List[str]
    suggested_alternatives: Dict[str, List[str]]


class EngagementMetrics(BaseModel):
    """Metrics for measuring engagement enhancement."""
    
    total_slides: int = Field(..., description="Total number of slides analyzed")
    verb_diversity_ratio: float = Field(..., description="Unique verbs / total verbs ratio")
    baseline_verb_ratio: float = Field(default=0.15, description="Baseline verb diversity ratio")
    rhetorical_questions_added: int = Field(default=0, description="Number of rhetorical questions added")
    target_rhetorical_frequency: float = Field(default=0.2, description="Target: 1 question per 5 slides")
    engagement_score: float = Field(..., ge=0.0, le=10.0, description="Overall engagement score")
    
    @property
    def meets_verb_diversity_target(self) -> bool:
        """Check if verb diversity meets T-81 target of ≥30%."""
        return self.verb_diversity_ratio >= 0.30
    
    @property
    def improvement_over_baseline(self) -> float:
        """Calculate improvement over baseline (15%)."""
        return self.verb_diversity_ratio - self.baseline_verb_ratio
    
    @property
    def rhetorical_question_frequency(self) -> float:
        """Calculate rhetorical question frequency (questions per slide)."""
        if self.total_slides == 0:
            return 0.0
        return self.rhetorical_questions_added / self.total_slides
    
    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "total_slides": 10,
                "verb_diversity_ratio": 0.35,
                "baseline_verb_ratio": 0.15,
                "rhetorical_questions_added": 2,
                "target_rhetorical_frequency": 0.2,
                "engagement_score": 8.5
            }
        }


class EngagementPromptTuner:
    """
    Enhance content generation prompts with engagement techniques (T-81).
    
    Extends ContentGenerator prompts with:
    - Varied verb choices to achieve ≥30% unique verbs (vs baseline 15%)
    - Rhetorical questions every 5 slides
    - Other engagement techniques
    """
    
    def __init__(
        self,
        client: OpenAI | AsyncOpenAI,
        model: str = "gpt-4",
        temperature: float = 0.4  # Slightly higher for creativity
    ):
        """
        Initialize the engagement prompt tuner.
        
        Args:
            client: OpenAI client instance
            model: Model name to use
            temperature: Temperature for generation (moderate for creative variety)
        """
        self.client = client
        self.model = model
        self.temperature = temperature
        
        # Verb alternatives database for variety
        self.verb_alternatives = self._load_verb_alternatives()
        
        logger.info(f"EngagementPromptTuner initialized with model: {model}")
    
    def enhance_content_prompt(
        self,
        base_prompt: str,
        slide_index: int,
        total_slides: int,
        config: GenerationConfig,
        engagement_context: Optional[Dict[str, any]] = None
    ) -> str:
        """
        Enhance a content generation prompt with engagement techniques.
        
        Args:
            base_prompt: Original content generation prompt
            slide_index: Index of current slide (0-based)
            total_slides: Total number of slides in presentation
            config: Generation configuration
            engagement_context: Additional context for engagement
            
        Returns:
            Enhanced prompt with engagement techniques
        """
        # Determine if this slide should have a rhetorical question
        needs_rhetorical_question = self._should_add_rhetorical_question(
            slide_index, total_slides
        )
        
        # Build engagement instructions
        engagement_instructions = self._build_engagement_instructions(
            needs_rhetorical_question, 
            engagement_context
        )
        
        # Insert engagement instructions into the base prompt
        enhanced_prompt = self._inject_engagement_instructions(
            base_prompt, 
            engagement_instructions
        )
        
        return enhanced_prompt
    
    def analyze_verb_diversity(
        self, 
        slides: List[SlidePlan],
        include_speaker_notes: bool = True
    ) -> VerbAnalysis:
        """
        Analyze verb diversity in slide content.
        
        Args:
            slides: List of slides to analyze
            include_speaker_notes: Whether to include speaker notes in analysis
            
        Returns:
            VerbAnalysis with diversity metrics
        """
        # Extract all text content
        all_text = []
        
        for slide in slides:
            # Add title
            if slide.title:
                all_text.append(slide.title)
            
            # Add bullets
            all_text.extend(slide.bullets)
            
            # Add speaker notes if requested
            if include_speaker_notes and slide.speaker_notes:
                all_text.append(slide.speaker_notes)
        
        # Combine all text
        combined_text = " ".join(all_text)
        
        # Extract verbs
        verbs = self._extract_verbs(combined_text)
        
        # Calculate metrics
        total_verbs = len(verbs)
        unique_verbs = len(set(verbs))
        verb_diversity_ratio = unique_verbs / total_verbs if total_verbs > 0 else 0.0
        
        # Find most common verbs
        verb_counts = Counter(verbs)
        most_common = verb_counts.most_common(10)
        
        # Find repeated verbs (used more than twice)
        repeated_verbs = [verb for verb, count in verb_counts.items() if count > 2]
        
        # Generate alternatives for repeated verbs
        suggested_alternatives = {}
        for verb in repeated_verbs[:5]:  # Top 5 repeated verbs
            alternatives = self.verb_alternatives.get(verb, [])
            if alternatives:
                suggested_alternatives[verb] = alternatives[:3]  # Top 3 alternatives
        
        return VerbAnalysis(
            total_verbs=total_verbs,
            unique_verbs=unique_verbs,
            verb_diversity_ratio=verb_diversity_ratio,
            most_common_verbs=most_common,
            repeated_verbs=repeated_verbs,
            suggested_alternatives=suggested_alternatives
        )
    
    def measure_engagement_metrics(
        self, 
        slides: List[SlidePlan],
        baseline_ratio: float = 0.15
    ) -> EngagementMetrics:
        """
        Measure engagement metrics for T-81 validation.
        
        Args:
            slides: List of slides to analyze
            baseline_ratio: Baseline verb diversity ratio (T-81: 15%)
            
        Returns:
            EngagementMetrics with T-81 compliance data
        """
        # Analyze verb diversity
        verb_analysis = self.analyze_verb_diversity(slides)
        
        # Count rhetorical questions
        rhetorical_questions = self._count_rhetorical_questions(slides)
        
        # Calculate engagement score (0-10 scale)
        engagement_score = self._calculate_engagement_score(
            verb_analysis.verb_diversity_ratio,
            rhetorical_questions,
            len(slides)
        )
        
        return EngagementMetrics(
            total_slides=len(slides),
            verb_diversity_ratio=verb_analysis.verb_diversity_ratio,
            baseline_verb_ratio=baseline_ratio,
            rhetorical_questions_added=rhetorical_questions,
            engagement_score=engagement_score
        )
    
    def generate_enhanced_content_batch(
        self,
        slides: List[SlidePlan],
        config: GenerationConfig,
        style_guidance: Optional[str] = None,
        language: str = "en"
    ) -> List[SlidePlan]:
        """
        Generate enhanced content for multiple slides with engagement techniques.
        
        Args:
            slides: List of slides to enhance
            config: Generation configuration
            style_guidance: Optional style guidance
            language: Language code
            
        Returns:
            List of slides with enhanced, engaging content
        """
        enhanced_slides = []
        total_slides = len(slides)
        
        logger.info(f"Generating enhanced content with engagement for {total_slides} slides")
        
        for i, slide in enumerate(slides):
            try:
                # Skip title slides
                if slide.slide_type == "title":
                    enhanced_slides.append(slide)
                    continue
                
                # Generate enhanced content for this slide
                enhanced_slide = self._generate_enhanced_slide_content(
                    slide, i, total_slides, config, style_guidance, language
                )
                enhanced_slides.append(enhanced_slide)
                
            except Exception as e:
                logger.error(f"Failed to enhance slide {i}: {e}")
                enhanced_slides.append(slide)  # Keep original on error
        
        return enhanced_slides
    
    def _should_add_rhetorical_question(self, slide_index: int, total_slides: int) -> bool:
        """Determine if this slide should have a rhetorical question (every 5 slides)."""
        # Add rhetorical questions at positions: 4, 9, 14, etc. (every 5 slides)
        return (slide_index + 1) % 5 == 0 and slide_index > 0
    
    def _build_engagement_instructions(
        self, 
        needs_rhetorical_question: bool,
        engagement_context: Optional[Dict[str, any]] = None
    ) -> str:
        """Build specific engagement instructions for the prompt."""
        
        instructions = []
        
        # Verb variety instructions (always included)
        instructions.append("""
VERB VARIETY REQUIREMENTS:
- Use diverse, compelling verbs throughout the content
- Avoid repetitive verbs like "is", "has", "shows", "provides"  
- Choose strong action verbs: "demonstrates", "reveals", "transforms", "accelerates"
- Vary verb tenses and forms for dynamic language
- Target: Use at least 30% unique verbs (avoid repeating the same verb)""")
        
        # Rhetorical question instructions (conditional)
        if needs_rhetorical_question:
            instructions.append("""
RHETORICAL QUESTION REQUIREMENT:
- Include ONE compelling rhetorical question in this slide's content
- Integrate the question naturally into title or first bullet point
- Make it thought-provoking and relevant to the content
- Examples: "What drives exceptional performance?", "How can we unlock this potential?"
- The question should engage the audience and introduce the key concept""")
        
        # General engagement instructions
        instructions.append("""
ENGAGEMENT TECHNIQUES:
- Use active voice wherever possible
- Choose vivid, specific language over generic terms
- Create compelling, memorable phrasing
- Make content conversational yet professional
- Ensure each bullet point provides clear value to the audience""")
        
        return "\n".join(instructions)
    
    def _inject_engagement_instructions(
        self, 
        base_prompt: str, 
        engagement_instructions: str
    ) -> str:
        """Inject engagement instructions into the base prompt."""
        
        # Find the REQUIREMENTS section and insert engagement instructions
        if "REQUIREMENTS:" in base_prompt:
            # Insert after REQUIREMENTS but before OUTPUT FORMAT
            parts = base_prompt.split("REQUIREMENTS:")
            if len(parts) == 2:
                before_requirements = parts[0]
                after_requirements = parts[1]
                
                # Find OUTPUT FORMAT to insert before it
                if "OUTPUT FORMAT:" in after_requirements:
                    req_parts = after_requirements.split("OUTPUT FORMAT:")
                    requirements_section = req_parts[0]
                    output_section = "OUTPUT FORMAT:" + req_parts[1]
                    
                    enhanced_prompt = (
                        before_requirements +
                        "REQUIREMENTS:" +
                        requirements_section +
                        "\n" + engagement_instructions + "\n\n" +
                        output_section
                    )
                    return enhanced_prompt
        
        # Fallback: append to the end
        return base_prompt + "\n\n" + engagement_instructions
    
    def _generate_enhanced_slide_content(
        self,
        slide: SlidePlan,
        slide_index: int,
        total_slides: int,
        config: GenerationConfig,
        style_guidance: Optional[str],
        language: str
    ) -> SlidePlan:
        """Generate enhanced content for a single slide."""
        
        # Build base prompt (simplified version of ContentGenerator logic)
        base_prompt = self._build_base_content_prompt(
            slide, config, style_guidance, language
        )
        
        # Enhance with engagement techniques
        enhanced_prompt = self.enhance_content_prompt(
            base_prompt, slide_index, total_slides, config
        )
        
        try:
            # Call LLM with enhanced prompt
            response_data = self._call_llm_with_retries(enhanced_prompt)
            
            # Apply generated content to slide
            enhanced_slide = self._apply_enhanced_content(slide, response_data)
            
            logger.debug(f"Enhanced slide {slide_index} with engagement techniques")
            return enhanced_slide
            
        except Exception as e:
            logger.error(f"Failed to generate enhanced content for slide {slide_index}: {e}")
            return slide
    
    def _build_base_content_prompt(
        self,
        slide: SlidePlan,
        config: GenerationConfig,
        style_guidance: Optional[str],
        language: str
    ) -> str:
        """Build a base content prompt (simplified ContentGenerator logic)."""
        
        style_context = ""
        if style_guidance:
            style_context = f"Style guidance: {style_guidance}\n"
        style_context += f"Tone: {config.tone}\n"
        style_context += f"Complexity level: {config.complexity_level}\n"
        
        slide_context = f"""Current slide information:
- Index: {slide.index}
- Type: {slide.slide_type}
- Current title: {slide.title}
- Current bullets: {slide.bullets}"""
        
        prompt = f"""You are an expert presentation writer focused on creating engaging, compelling content.

{style_context}

{slide_context}

REQUIREMENTS:
- Generate content in {language}
- Maximum {config.max_bullets_per_slide} bullet points
- Keep content concise and impactful
- Ensure bullets are parallel in structure
- Make titles clear and engaging

OUTPUT FORMAT:
Return a JSON object with this structure:
{{
    "title": "Enhanced slide title",
    "bullets": ["Bullet point 1", "Bullet point 2", "Bullet point 3"],
    "speaker_notes": "Enhanced speaker notes"
}}

Generate enhanced content now:"""
        
        return prompt
    
    def _apply_enhanced_content(self, slide: SlidePlan, response_data: dict) -> SlidePlan:
        """Apply generated content to slide."""
        try:
            # Create new slide with enhanced content
            enhanced_slide = SlidePlan(
                index=slide.index,
                slide_type=slide.slide_type,
                title=response_data.get("title", slide.title),
                bullets=response_data.get("bullets", slide.bullets),
                image_query=slide.image_query,
                chart_data=slide.chart_data,
                speaker_notes=response_data.get("speaker_notes", slide.speaker_notes),
                layout_id=slide.layout_id
            )
            
            return enhanced_slide
            
        except Exception as e:
            logger.error(f"Failed to apply enhanced content: {e}")
            return slide
    
    def _extract_verbs(self, text: str) -> List[str]:
        """Extract verbs from text using simple pattern matching."""
        
        # Common verb patterns and forms
        verb_patterns = [
            # Common verbs
            r'\b(is|are|was|were|be|been|being)\b',
            r'\b(has|have|had|having)\b', 
            r'\b(do|does|did|done|doing)\b',
            r'\b(can|could|will|would|shall|should|may|might|must)\b',
            # Action verbs ending in common suffixes
            r'\b\w+(?:ing|ed|es|s)\b',
            # Base form verbs (simplified - would need POS tagging for accuracy)
            r'\b(?:show|provide|create|make|take|give|get|see|know|think|come|go|work|look|feel|become|leave|put|mean|keep|let|begin|seem|help|talk|turn|start|might|move|live|believe|hold|bring|happen|write|hear|play|run|move|try|ask|need|feel|become|leave|put)\b'
        ]
        
        verbs = []
        text_lower = text.lower()
        
        for pattern in verb_patterns:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            verbs.extend(matches)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_verbs = []
        for verb in verbs:
            if verb not in seen:
                seen.add(verb)
                unique_verbs.append(verb)
        
        return unique_verbs
    
    def _count_rhetorical_questions(self, slides: List[SlidePlan]) -> int:
        """Count rhetorical questions in slide content."""
        question_count = 0
        
        for slide in slides:
            # Check title
            if slide.title and "?" in slide.title:
                question_count += slide.title.count("?")
            
            # Check bullets
            for bullet in slide.bullets:
                if "?" in bullet:
                    question_count += bullet.count("?")
        
        return question_count
    
    def _calculate_engagement_score(
        self, 
        verb_diversity: float, 
        rhetorical_questions: int, 
        total_slides: int
    ) -> float:
        """Calculate overall engagement score (0-10 scale)."""
        
        # Verb diversity component (50% of score)
        verb_score = min(10, (verb_diversity / 0.30) * 5)  # Max 5 points for 30% diversity
        
        # Rhetorical question component (30% of score)
        target_questions = max(1, total_slides // 5)  # 1 per 5 slides
        question_score = min(3, (rhetorical_questions / target_questions) * 3)
        
        # Bonus for exceeding targets (20% of score)
        bonus_score = 0
        if verb_diversity > 0.35:  # Exceeds 35%
            bonus_score += 1
        if rhetorical_questions > target_questions:
            bonus_score += 1
        
        total_score = verb_score + question_score + bonus_score
        return min(10.0, max(0.0, total_score))
    
    def _load_verb_alternatives(self) -> Dict[str, List[str]]:
        """Load verb alternatives database for variety."""
        return {
            "show": ["demonstrate", "reveal", "illustrate", "display", "exhibit"],
            "provide": ["deliver", "offer", "supply", "furnish", "present"],
            "create": ["generate", "develop", "build", "establish", "construct"],
            "make": ["produce", "manufacture", "craft", "form", "design"],
            "help": ["assist", "support", "enable", "facilitate", "empower"],
            "improve": ["enhance", "optimize", "refine", "elevate", "strengthen"],
            "increase": ["boost", "expand", "amplify", "escalate", "accelerate"],
            "reduce": ["minimize", "decrease", "diminish", "lower", "streamline"],
            "use": ["utilize", "employ", "leverage", "apply", "implement"],
            "get": ["obtain", "acquire", "secure", "achieve", "attain"],
            "give": ["provide", "deliver", "offer", "supply", "contribute"],
            "see": ["observe", "witness", "discover", "identify", "recognize"],
            "find": ["discover", "identify", "locate", "uncover", "detect"],
            "work": ["operate", "function", "perform", "execute", "collaborate"],
            "grow": ["expand", "develop", "flourish", "scale", "evolve"],
            "start": ["initiate", "launch", "commence", "begin", "kickoff"],
            "change": ["transform", "modify", "adapt", "evolve", "revolutionize"],
            "build": ["construct", "develop", "establish", "create", "forge"],
            "drive": ["propel", "fuel", "power", "accelerate", "catalyze"],
            "lead": ["guide", "direct", "spearhead", "champion", "pioneer"]
        }
    
    def _supports_json_mode(self) -> bool:
        """Check if the current model supports JSON mode."""
        json_mode_models = [
            "gpt-4", "gpt-4-0613", "gpt-4-1106-preview", "gpt-4-0125-preview",
            "gpt-4-turbo-preview", "gpt-4-turbo", "gpt-4o", "gpt-4o-mini",
            "gpt-3.5-turbo", "gpt-3.5-turbo-0613", "gpt-3.5-turbo-1106", "gpt-3.5-turbo-0125"
        ]
        return any(model in self.model for model in json_mode_models)

    def _call_llm_with_retries(self, prompt: str) -> dict:
        """Call LLM API with retry logic."""
        import time
        import openai
        
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                # Build request parameters
                request_params = {
                    "model": self.model,
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are an expert presentation writer focused on engaging, varied language. Always respond with valid JSON only."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    "temperature": self.temperature,
                    "max_tokens": 2000
                }
                
                # Only add response_format for models that support it
                if self._supports_json_mode():
                    request_params["response_format"] = {"type": "json_object"}
                
                response = self.client.chat.completions.create(**request_params)
                
                content = response.choices[0].message.content
                if not content:
                    raise ValueError("Empty response from LLM")
                
                return json.loads(content)
                
            except openai.RateLimitError:
                if attempt == max_retries - 1:
                    raise
                wait_time = 2 ** attempt
                time.sleep(wait_time)
                
            except openai.APIError as e:
                if attempt == max_retries - 1:
                    raise
                wait_time = 2 ** attempt
                time.sleep(wait_time)
        
        raise ValueError("Failed to get LLM response after all retries")
    
    def validate_t81_requirements(
        self, 
        metrics: EngagementMetrics
    ) -> Dict[str, bool]:
        """Validate T-81 requirements."""
        
        validation_results = {}
        
        # Check verb diversity requirement: ≥30% unique verbs
        meets_verb_target = metrics.meets_verb_diversity_target
        validation_results["verb_diversity_target"] = meets_verb_target
        
        # Check improvement over baseline: should be significantly above 15%
        significant_improvement = metrics.improvement_over_baseline >= 0.10  # At least 10% improvement
        validation_results["significant_improvement"] = significant_improvement
        
        # Check rhetorical question frequency: roughly 1 per 5 slides
        adequate_questions = metrics.rhetorical_question_frequency >= 0.15  # At least 15% of slides
        validation_results["rhetorical_questions"] = adequate_questions
        
        logger.info(f"T-81 validation - Verb diversity: {meets_verb_target}, "
                   f"Improvement: {significant_improvement}, Questions: {adequate_questions}")
        
        return validation_results