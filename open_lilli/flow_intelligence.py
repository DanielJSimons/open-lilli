"""Flow Intelligence module for narrative flow analysis and transition generation (T-80)."""

import logging
import json
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from openai import OpenAI, AsyncOpenAI
from pydantic import BaseModel, Field

from .models import SlidePlan, ReviewFeedback

logger = logging.getLogger(__name__)


class TransitionType(str, Enum):
    """Types of transitions between slides."""
    SEQUENCE = "sequence"  # "Next, we'll examine..."
    CAUSE_EFFECT = "cause_effect"  # "As a result of this..."
    CONTRAST = "contrast"  # "However, when we look at..."
    AMPLIFICATION = "amplification"  # "Building on this insight..."
    SUMMARY = "summary"  # "To summarize the key points..."
    BRIDGE = "bridge"  # "This brings us to our next topic..."
    EMPHASIS = "emphasis"  # "Most importantly..."
    CONCLUSION = "conclusion"  # "Finally, let's consider..."


@dataclass
class TransitionSuggestion:
    """Represents a suggested transition between slides."""
    from_slide_index: int
    to_slide_index: int
    transition_type: TransitionType
    linking_sentence: str
    context_summary: str
    confidence: float
    insertion_location: str  # "speaker_notes", "slide_end", "slide_start"


class FlowAnalysisResult(BaseModel):
    """Result of narrative flow analysis."""
    
    total_slides: int = Field(..., description="Total number of slides analyzed")
    transitions_generated: List[TransitionSuggestion] = Field(
        default_factory=list, 
        description="Generated transition suggestions"
    )
    flow_score: float = Field(..., ge=0.0, le=5.0, description="Overall flow coherence score (0-5)")
    narrative_gaps: List[str] = Field(
        default_factory=list, 
        description="Identified gaps in narrative flow"
    )
    coherence_issues: List[ReviewFeedback] = Field(
        default_factory=list,
        description="Specific coherence issues found"
    )
    processing_time_seconds: float = Field(..., description="Analysis processing time")
    
    @property
    def transition_coverage(self) -> float:
        """Calculate percentage of slide transitions covered."""
        max_transitions = max(0, self.total_slides - 1)
        if max_transitions == 0:
            return 1.0
        return len(self.transitions_generated) / max_transitions
    
    @property
    def meets_transition_requirement(self) -> bool:
        """Check if deck meets T-80 requirement of >= (N-1) transitions."""
        required_transitions = max(0, self.total_slides - 1)
        return len(self.transitions_generated) >= required_transitions
    
    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "total_slides": 5,
                "transitions_generated": [
                    {
                        "from_slide_index": 0,
                        "to_slide_index": 1,
                        "transition_type": "bridge",
                        "linking_sentence": "With this foundation established, let's examine the key market trends that are shaping our industry.",
                        "context_summary": "Moving from introduction to market analysis",
                        "confidence": 0.9,
                        "insertion_location": "speaker_notes"
                    }
                ],
                "flow_score": 4.2,
                "narrative_gaps": ["Missing connection between slides 3 and 4"],
                "coherence_issues": [],
                "processing_time_seconds": 3.5
            }
        }


class FlowIntelligence:
    """AI-powered narrative flow analysis and transition generation for presentations."""
    
    def __init__(
        self,
        client: OpenAI | AsyncOpenAI,
        model: str = "gpt-4",
        temperature: float = 0.3,
        max_retries: int = 3
    ):
        """
        Initialize the flow intelligence system.
        
        Args:
            client: OpenAI client instance
            model: Model name to use for flow analysis
            temperature: Temperature for generation (moderate for creative transitions)
            max_retries: Maximum number of retries for API calls
        """
        self.client = client
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        
        logger.info(f"FlowIntelligence initialized with model: {model}")
    
    def analyze_and_enhance_flow(
        self,
        slides: List[SlidePlan],
        target_coherence: float = 4.0,
        insert_transitions: bool = True
    ) -> FlowAnalysisResult:
        """
        Analyze presentation flow and generate transition suggestions (T-80).
        
        Args:
            slides: List of slides to analyze
            target_coherence: Target coherence score (T-80 requirement: > 4.0/5)
            insert_transitions: Whether to insert transitions into slide notes
            
        Returns:
            FlowAnalysisResult with transitions and analysis
        """
        import time
        start_time = time.time()
        
        logger.info(f"Analyzing narrative flow for {len(slides)} slides")
        
        # Generate transition suggestions
        transitions = self._generate_transition_suggestions(slides)
        
        # Analyze overall flow coherence
        flow_score, narrative_gaps = self._analyze_flow_coherence(slides, transitions)
        
        # Identify specific coherence issues
        coherence_issues = self._identify_coherence_issues(slides, transitions)
        
        # Insert transitions into slide notes if requested
        if insert_transitions:
            self._insert_transitions_to_slides(slides, transitions)
        
        processing_time = time.time() - start_time
        
        result = FlowAnalysisResult(
            total_slides=len(slides),
            transitions_generated=transitions,
            flow_score=flow_score,
            narrative_gaps=narrative_gaps,
            coherence_issues=coherence_issues,
            processing_time_seconds=processing_time
        )
        
        logger.info(
            f"Flow analysis completed: {len(transitions)} transitions, "
            f"score {flow_score:.1f}/5, coverage {result.transition_coverage:.1%}"
        )
        
        return result
    
    def _generate_transition_suggestions(self, slides: List[SlidePlan]) -> List[TransitionSuggestion]:
        """Generate transition suggestions for the entire outline."""
        
        if len(slides) < 2:
            return []
        
        # Create presentation context for better transitions
        presentation_context = self._create_presentation_context(slides)
        
        # Generate transitions for each slide pair
        transitions = []
        
        for i in range(len(slides) - 1):
            current_slide = slides[i]
            next_slide = slides[i + 1]
            
            transition = self._generate_single_transition(
                current_slide, 
                next_slide, 
                presentation_context,
                slide_position=i + 1  # 1-based position
            )
            
            if transition:
                transitions.append(transition)
        
        return transitions
    
    def _generate_single_transition(
        self,
        current_slide: SlidePlan,
        next_slide: SlidePlan,
        presentation_context: str,
        slide_position: int
    ) -> Optional[TransitionSuggestion]:
        """Generate a single transition between two slides."""
        
        prompt = f"""You are an expert presentation consultant specializing in narrative flow. Generate a smooth transition between these slides.

PRESENTATION CONTEXT:
{presentation_context}

CURRENT SLIDE (#{current_slide.index + 1}):
Title: {current_slide.title}
Type: {current_slide.slide_type}
Content: {' • '.join(current_slide.bullets[:3]) if current_slide.bullets else 'No bullets'}

NEXT SLIDE (#{next_slide.index + 1}):
Title: {next_slide.title}
Type: {next_slide.slide_type}
Content: {' • '.join(next_slide.bullets[:3]) if next_slide.bullets else 'No bullets'}

POSITION: Slide {slide_position} of presentation

INSTRUCTIONS:
1. Create a smooth, natural transition sentence that bridges these slides
2. Choose the most appropriate transition type from: sequence, cause_effect, contrast, amplification, summary, bridge, emphasis, conclusion
3. Make the transition feel conversational and engaging
4. Consider the narrative arc and logical flow
5. Keep the transition concise but meaningful (1-2 sentences max)

Return a JSON object with this structure:
{{
    "linking_sentence": "Your transition sentence here",
    "transition_type": "bridge",
    "context_summary": "Brief explanation of the connection",
    "confidence": 0.9,
    "reasoning": "Why this transition works"
}}

Focus on creating seamless narrative flow that guides the audience naturally from one concept to the next."""
        
        try:
            response_data = self._call_llm_with_retries(prompt)
            
            # Parse response
            linking_sentence = response_data.get("linking_sentence", "")
            transition_type = response_data.get("transition_type", "bridge")
            context_summary = response_data.get("context_summary", "")
            confidence = response_data.get("confidence", 0.5)
            
            if not linking_sentence:
                logger.warning(f"No transition generated for slides {current_slide.index} -> {next_slide.index}")
                return None
            
            return TransitionSuggestion(
                from_slide_index=current_slide.index,
                to_slide_index=next_slide.index,
                transition_type=TransitionType(transition_type.lower()),
                linking_sentence=linking_sentence,
                context_summary=context_summary,
                confidence=confidence,
                insertion_location="speaker_notes"
            )
            
        except Exception as e:
            logger.error(f"Failed to generate transition for slides {current_slide.index} -> {next_slide.index}: {e}")
            return None
    
    def _create_presentation_context(self, slides: List[SlidePlan]) -> str:
        """Create context summary for better transition generation."""
        
        # Extract key themes and structure
        titles = [slide.title for slide in slides]
        slide_types = [slide.slide_type for slide in slides]
        
        # Identify sections
        sections = []
        current_section = None
        
        for slide in slides:
            if slide.slide_type in ["title", "section"]:
                current_section = slide.title
                sections.append(current_section)
        
        context = f"""PRESENTATION OVERVIEW:
Total Slides: {len(slides)}
Sections: {' → '.join(sections) if sections else 'Single section'}

SLIDE SEQUENCE:
"""
        
        for i, slide in enumerate(slides, 1):
            context += f"{i}. {slide.title} ({slide.slide_type})\n"
        
        return context
    
    def _analyze_flow_coherence(
        self, 
        slides: List[SlidePlan], 
        transitions: List[TransitionSuggestion]
    ) -> Tuple[float, List[str]]:
        """Analyze overall flow coherence and identify gaps."""
        
        # Create comprehensive flow summary
        flow_summary = self._create_detailed_flow_summary(slides, transitions)
        
        prompt = f"""You are an expert presentation consultant. Analyze the narrative flow and coherence of this presentation.

{flow_summary}

EVALUATION CRITERIA:
1. Logical progression of ideas (25%)
2. Smooth transitions between concepts (25%) 
3. Clear narrative arc and story structure (25%)
4. Appropriate pacing and information flow (25%)

INSTRUCTIONS:
1. Evaluate the overall coherence on a scale of 0.0 to 5.0 (where 5.0 is perfect)
2. Identify specific narrative gaps or disconnects
3. Consider whether the flow guides the audience naturally
4. Assess if the story has clear beginning, middle, and end

Return a JSON object:
{{
    "coherence_score": 4.2,
    "narrative_gaps": [
        "Missing connection between market analysis and financial projections",
        "Abrupt jump from problems to solutions without clear bridge"
    ],
    "flow_strengths": [
        "Strong opening that establishes context",
        "Clear progression through main arguments"
    ],
    "improvement_areas": [
        "Add transitional slide before recommendations",
        "Strengthen conclusion linkage to introduction"
    ]
}}

Be specific and actionable in identifying gaps and improvements."""
        
        try:
            response_data = self._call_llm_with_retries(prompt)
            
            coherence_score = response_data.get("coherence_score", 3.0)
            narrative_gaps = response_data.get("narrative_gaps", [])
            
            # Ensure score is within bounds
            coherence_score = max(0.0, min(5.0, coherence_score))
            
            return coherence_score, narrative_gaps
            
        except Exception as e:
            logger.error(f"Failed to analyze flow coherence: {e}")
            return 3.0, ["Unable to analyze flow due to processing error"]
    
    def _create_detailed_flow_summary(
        self, 
        slides: List[SlidePlan], 
        transitions: List[TransitionSuggestion]
    ) -> str:
        """Create detailed summary for flow analysis."""
        
        summary = f"PRESENTATION FLOW ANALYSIS:\n"
        summary += f"Total slides: {len(slides)}\n"
        summary += f"Generated transitions: {len(transitions)}\n\n"
        
        # Add slide-by-slide flow
        for i, slide in enumerate(slides):
            summary += f"SLIDE {i + 1}: {slide.title}\n"
            summary += f"  Type: {slide.slide_type}\n"
            
            if slide.bullets:
                summary += f"  Key points: {'; '.join(slide.bullets[:2])}\n"
            
            # Add transition to next slide
            transition = next((t for t in transitions if t.from_slide_index == i), None)
            if transition:
                summary += f"  → Transition: {transition.linking_sentence}\n"
            
            summary += "\n"
        
        return summary
    
    def _identify_coherence_issues(
        self, 
        slides: List[SlidePlan], 
        transitions: List[TransitionSuggestion]
    ) -> List[ReviewFeedback]:
        """Identify specific coherence issues for ReviewFeedback."""
        
        issues = []
        
        # Check for missing transitions
        for i in range(len(slides) - 1):
            has_transition = any(t.from_slide_index == i for t in transitions)
            if not has_transition:
                issues.append(ReviewFeedback(
                    slide_index=i,
                    severity="medium",
                    category="flow",
                    message=f"Missing transition from slide {i + 1} to {i + 2}",
                    suggestion=f"Add linking sentence to connect '{slides[i].title}' to '{slides[i + 1].title}'"
                ))
        
        # Check for abrupt topic changes
        for i in range(len(slides) - 1):
            current_title = slides[i].title.lower()
            next_title = slides[i + 1].title.lower()
            
            # Simple keyword overlap check
            current_words = set(re.findall(r'\w+', current_title))
            next_words = set(re.findall(r'\w+', next_title))
            
            overlap = len(current_words.intersection(next_words))
            if overlap == 0 and len(current_words) > 1 and len(next_words) > 1:
                issues.append(ReviewFeedback(
                    slide_index=i,
                    severity="low", 
                    category="flow",
                    message=f"Potential topic disconnect between slides {i + 1} and {i + 2}",
                    suggestion="Consider adding transitional content to bridge these topics"
                ))
        
        return issues
    
    def _insert_transitions_to_slides(
        self, 
        slides: List[SlidePlan], 
        transitions: List[TransitionSuggestion]
    ) -> None:
        """Insert transition suggestions into slide speaker notes."""
        
        for transition in transitions:
            if transition.from_slide_index < len(slides):
                slide = slides[transition.from_slide_index]
                
                # Format transition for speaker notes
                transition_note = f"\n[TRANSITION] {transition.linking_sentence}"
                
                # Add to existing speaker notes or create new ones
                if slide.speaker_notes:
                    slide.speaker_notes += transition_note
                else:
                    slide.speaker_notes = transition_note.strip()
                
                logger.debug(f"Inserted transition into slide {slide.index + 1} speaker notes")
    
    def _supports_json_mode(self) -> bool:
        """Check if the current model supports JSON mode."""
        json_mode_models = [
            "gpt-4-1106-preview",
            "gpt-4-0125-preview",
            "gpt-4-turbo-preview",
            "gpt-4-turbo",
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-1106",
            "gpt-3.5-turbo-0125",
        ]
        return any(model in self.model for model in json_mode_models)

    def _call_llm_with_retries(self, prompt: str) -> dict:
        """Call LLM API with retry logic."""
        import time
        import openai
        
        for attempt in range(self.max_retries):
            try:
                logger.debug(f"LLM API call attempt {attempt + 1}")
                
                # Build request parameters
                request_params = {
                    "model": self.model,
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are an expert presentation flow consultant. Always respond with valid JSON only."
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
                wait_time = 2 ** attempt
                logger.warning(f"Rate limited, waiting {wait_time}s")
                time.sleep(wait_time)
                
            except openai.APIError as e:
                if attempt == self.max_retries - 1:
                    raise
                wait_time = 2 ** attempt
                logger.warning(f"API error: {e}, waiting {wait_time}s")
                time.sleep(wait_time)
        
        raise ValueError("Failed to get LLM response after all retries")
    
    def validate_transition_requirements(
        self, 
        slides: List[SlidePlan],
        result: FlowAnalysisResult
    ) -> Dict[str, bool]:
        """Validate T-80 requirements."""
        
        validation_results = {}
        
        # Check transition coverage requirement: >= (N-1) transitions
        required_transitions = max(0, len(slides) - 1)
        has_sufficient_transitions = len(result.transitions_generated) >= required_transitions
        validation_results["sufficient_transitions"] = has_sufficient_transitions
        
        # Check coherence score requirement: > 4.0/5
        meets_coherence_target = result.flow_score > 4.0
        validation_results["coherence_target"] = meets_coherence_target
        
        # Check that transitions are actually inserted in notes
        slides_with_transitions = 0
        for slide in slides[:-1]:  # Exclude last slide
            if slide.speaker_notes and "[TRANSITION]" in slide.speaker_notes:
                slides_with_transitions += 1
        
        transitions_inserted = slides_with_transitions >= required_transitions
        validation_results["transitions_inserted"] = transitions_inserted
        
        logger.info(f"T-80 validation - Transitions: {has_sufficient_transitions}, "
                   f"Coherence: {meets_coherence_target}, Inserted: {transitions_inserted}")
        
        return validation_results
    
    def generate_flow_report(
        self, 
        slides: List[SlidePlan], 
        result: FlowAnalysisResult
    ) -> Dict[str, any]:
        """Generate comprehensive flow analysis report."""
        
        validation = self.validate_transition_requirements(slides, result)
        
        report = {
            "summary": {
                "total_slides": result.total_slides,
                "transitions_generated": len(result.transitions_generated),
                "flow_score": result.flow_score,
                "transition_coverage": result.transition_coverage,
                "processing_time_seconds": result.processing_time_seconds
            },
            "t80_compliance": {
                "meets_transition_requirement": result.meets_transition_requirement,
                "meets_coherence_target": result.flow_score > 4.0,
                "transitions_inserted": validation["transitions_inserted"],
                "overall_compliance": all(validation.values())
            },
            "transitions": [
                {
                    "from_slide": t.from_slide_index + 1,
                    "to_slide": t.to_slide_index + 1,
                    "type": t.transition_type.value,
                    "sentence": t.linking_sentence,
                    "confidence": t.confidence
                }
                for t in result.transitions_generated
            ],
            "narrative_gaps": result.narrative_gaps,
            "coherence_issues": [
                {
                    "slide": issue.slide_index + 1,
                    "severity": issue.severity,
                    "message": issue.message,
                    "suggestion": issue.suggestion
                }
                for issue in result.coherence_issues
            ]
        }
        
        return report