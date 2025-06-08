"""LLM-Based Visual Proofreader for detecting design issues in presentations."""

import logging
import json
import re
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum

from openai import OpenAI, AsyncOpenAI
from pydantic import BaseModel, Field
from pptx import Presentation

from .models import SlidePlan, ReviewFeedback

logger = logging.getLogger(__name__)


class DesignIssueType(str, Enum):
    """Types of design issues that can be detected."""
    CAPITALIZATION = "capitalization"
    FORMATTING = "formatting"
    CONSISTENCY = "consistency"
    ALIGNMENT = "alignment"
    SPACING = "spacing"
    TYPOGRAPHY = "typography"
    COLOR = "color"
    HIERARCHY = "hierarchy"


@dataclass
class SlidePreview:
    """Lightweight text representation of a slide for LLM analysis."""
    slide_index: int
    title: str
    bullet_points: List[str]
    image_alt_text: Optional[str] = None
    chart_description: Optional[str] = None
    speaker_notes: Optional[str] = None


class DesignIssue(BaseModel):
    """Represents a design issue detected by the LLM."""
    
    slide_index: int = Field(..., description="Index of the slide with the issue")
    issue_type: DesignIssueType = Field(..., description="Type of design issue")
    severity: str = Field(..., description="Severity: low, medium, high, critical")
    element: str = Field(..., description="Specific element with the issue (title, bullet, etc.)")
    original_text: str = Field(..., description="Original problematic text")
    corrected_text: Optional[str] = Field(None, description="Suggested correction")
    description: str = Field(..., description="Description of the issue")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score (0.0-1.0)")
    
    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "slide_index": 2,
                "issue_type": "capitalization",
                "severity": "medium",
                "element": "title",
                "original_text": "market ANALYSIS and trends",
                "corrected_text": "Market Analysis and Trends",
                "description": "Inconsistent capitalization in slide title",
                "confidence": 0.9
            }
        }


class ProofreadingResult(BaseModel):
    """Result of the visual proofreading process."""
    
    total_slides: int = Field(..., description="Total number of slides processed")
    issues_found: List[DesignIssue] = Field(default_factory=list, description="Design issues detected")
    processing_time_seconds: float = Field(..., description="Time taken for proofreading")
    model_used: str = Field(..., description="LLM model used for analysis")
    
    @property
    def issue_count_by_type(self) -> Dict[str, int]:
        """Count issues by type."""
        counts = {}
        for issue in self.issues_found:
            counts[issue.issue_type] = counts.get(issue.issue_type, 0) + 1
        return counts
    
    @property
    def high_confidence_issues(self) -> List[DesignIssue]:
        """Get issues with confidence >= 0.8."""
        return [issue for issue in self.issues_found if issue.confidence >= 0.8]
    
    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "total_slides": 10,
                "issues_found": [
                    {
                        "slide_index": 2,
                        "issue_type": "capitalization",
                        "severity": "medium",
                        "element": "title",
                        "original_text": "market ANALYSIS and trends",
                        "corrected_text": "Market Analysis and Trends",
                        "description": "Inconsistent capitalization in slide title",
                        "confidence": 0.9
                    }
                ],
                "processing_time_seconds": 5.2,
                "model_used": "gpt-4"
            }
        }


class VisualProofreader:
    """LLM-Based Visual Proofreader for detecting design issues in slide presentations."""
    
    def __init__(
        self,
        client: Union[OpenAI, AsyncOpenAI],
        model: str = "gpt-4",
        temperature: float = 0.1,
        max_retries: int = 3
    ):
        """
        Initialize the visual proofreader.
        
        Args:
            client: OpenAI client instance
            model: Model name to use for proofreading
            temperature: Temperature for generation (lower for more consistent detection)
            max_retries: Maximum number of retries for API calls
        """
        self.client = client
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        
        logger.info(f"VisualProofreader initialized with model: {model}")
    
    def proofread_slides(
        self,
        slides: List[SlidePlan],
        focus_areas: Optional[List[DesignIssueType]] = None,
        enable_corrections: bool = True
    ) -> ProofreadingResult:
        """
        Proofread slides for design issues.
        
        Args:
            slides: List of slides to proofread
            focus_areas: Specific types of issues to focus on (None for all)
            enable_corrections: Whether to generate suggested corrections
            
        Returns:
            ProofreadingResult with detected issues
        """
        import time
        start_time = time.time()
        
        logger.info(f"Starting visual proofreading of {len(slides)} slides")
        
        # Convert slides to lightweight preview format
        slide_previews = [self._create_slide_preview(slide) for slide in slides]
        
        # Set default focus areas if none specified
        if focus_areas is None:
            focus_areas = [
                DesignIssueType.CAPITALIZATION,
                DesignIssueType.FORMATTING,
                DesignIssueType.CONSISTENCY,
                DesignIssueType.TYPOGRAPHY
            ]
        
        # Perform proofreading
        issues = self._detect_design_issues(
            slide_previews, 
            focus_areas, 
            enable_corrections
        )
        
        processing_time = time.time() - start_time
        
        result = ProofreadingResult(
            total_slides=len(slides),
            issues_found=issues,
            processing_time_seconds=processing_time,
            model_used=self.model
        )
        
        logger.info(
            f"Proofreading completed: {len(issues)} issues found in {processing_time:.2f}s"
        )
        
        return result
    
    def proofread_single_slide(
        self,
        slide: SlidePlan,
        focus_areas: Optional[List[DesignIssueType]] = None,
        enable_corrections: bool = True
    ) -> List[DesignIssue]:
        """
        Proofread a single slide for design issues.
        
        Args:
            slide: Slide to proofread
            focus_areas: Specific types of issues to focus on
            enable_corrections: Whether to generate suggested corrections
            
        Returns:
            List of design issues found
        """
        logger.debug(f"Proofreading slide {slide.index}: {slide.title}")
        
        slide_preview = self._create_slide_preview(slide)
        
        if focus_areas is None:
            focus_areas = [
                DesignIssueType.CAPITALIZATION,
                DesignIssueType.FORMATTING,
                DesignIssueType.CONSISTENCY,
                DesignIssueType.TYPOGRAPHY
            ]
        
        return self._detect_design_issues([slide_preview], focus_areas, enable_corrections)
    
    def test_capitalization_detection(
        self,
        test_slides: List[SlidePlan],
        seeded_errors: List[Dict[str, str]]
    ) -> Dict[str, float]:
        """
        Test the capitalization detection accuracy with seeded errors.
        
        Args:
            test_slides: Slides with seeded capitalization errors
            seeded_errors: List of expected errors with slide_index and element keys
            
        Returns:
            Dictionary with detection accuracy metrics
        """
        logger.info(f"Testing capitalization detection on {len(test_slides)} slides with {len(seeded_errors)} seeded errors")
        
        # Run proofreading focused on capitalization
        result = self.proofread_slides(
            test_slides,
            focus_areas=[DesignIssueType.CAPITALIZATION],
            enable_corrections=True
        )
        
        # Filter to capitalization issues only
        cap_issues = [
            issue for issue in result.issues_found 
            if issue.issue_type == DesignIssueType.CAPITALIZATION
        ]
        
        # Calculate detection metrics
        detected_errors = set()
        for issue in cap_issues:
            key = f"{issue.slide_index}_{issue.element}"
            detected_errors.add(key)
        
        expected_errors = set()
        for error in seeded_errors:
            key = f"{error['slide_index']}_{error['element']}"
            expected_errors.add(key)
        
        true_positives = len(detected_errors.intersection(expected_errors))
        false_positives = len(detected_errors - expected_errors)
        false_negatives = len(expected_errors - detected_errors)
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        accuracy_metrics = {
            "detection_rate": recall,  # What T-79 specifically mentions
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "total_seeded_errors": len(seeded_errors),
            "total_detected_issues": len(cap_issues)
        }
        
        logger.info(f"Capitalization detection rate: {recall:.1%} ({true_positives}/{len(seeded_errors)} seeded errors detected)")
        
        return accuracy_metrics
    
    def _create_slide_preview(self, slide: SlidePlan) -> SlidePreview:
        """
        Create a lightweight text preview of a slide.
        
        Args:
            slide: SlidePlan to convert to preview
            
        Returns:
            SlidePreview with text representation
        """
        # Generate alt-text for images if present
        image_alt_text = None
        if slide.image_query:
            image_alt_text = f"Image related to: {slide.image_query}"
        
        # Generate description for charts if present
        chart_description = None
        if slide.chart_data:
            if isinstance(slide.chart_data, dict):
                chart_type = slide.chart_data.get('type', 'unknown')
                chart_title = slide.chart_data.get('title', 'Untitled Chart')
                chart_description = f"{chart_type.title()} chart: {chart_title}"
            else:
                chart_description = "Chart or data visualization"
        
        return SlidePreview(
            slide_index=slide.index,
            title=slide.title,
            bullet_points=slide.bullets,
            image_alt_text=image_alt_text,
            chart_description=chart_description,
            speaker_notes=slide.speaker_notes
        )
    
    def _detect_design_issues(
        self,
        slide_previews: List[SlidePreview],
        focus_areas: List[DesignIssueType],
        enable_corrections: bool
    ) -> List[DesignIssue]:
        """
        Detect design issues using LLM analysis.
        
        Args:
            slide_previews: List of slide previews to analyze
            focus_areas: Types of issues to focus on
            enable_corrections: Whether to generate corrections
            
        Returns:
            List of detected design issues
        """
        # Create analysis prompt
        prompt = self._build_proofreading_prompt(
            slide_previews, 
            focus_areas, 
            enable_corrections
        )
        
        try:
            # Call LLM for analysis
            response_data = self._call_llm_with_retries(prompt)
            
            # Parse and validate issues
            issues = self._parse_issues_response(response_data)
            
            logger.debug(f"LLM detected {len(issues)} design issues")
            
            return issues
            
        except Exception as e:
            logger.error(f"Design issue detection failed: {e}")
            return []
    
    def _build_proofreading_prompt(
        self,
        slide_previews: List[SlidePreview],
        focus_areas: List[DesignIssueType],
        enable_corrections: bool
    ) -> str:
        """
        Build the proofreading prompt for LLM analysis.
        
        Args:
            slide_previews: List of slide previews
            focus_areas: Types of issues to focus on
            enable_corrections: Whether to include correction requests
            
        Returns:
            Formatted prompt string
        """
        # Create slide content summary
        slides_text = ""
        for preview in slide_previews:
            slides_text += f"\n--- SLIDE {preview.slide_index + 1} ---"
            slides_text += f"\nTitle: {preview.title}"
            
            if preview.bullet_points:
                slides_text += "\nBullet Points:"
                for i, bullet in enumerate(preview.bullet_points, 1):
                    slides_text += f"\n  {i}. {bullet}"
            
            if preview.image_alt_text:
                slides_text += f"\nImage: {preview.image_alt_text}"
            
            if preview.chart_description:
                slides_text += f"\nChart: {preview.chart_description}"
            
            if preview.speaker_notes:
                slides_text += f"\nSpeaker Notes: {preview.speaker_notes[:100]}{'...' if len(preview.speaker_notes) > 100 else ''}"
            
            slides_text += "\n"
        
        # Create focus areas description
        focus_descriptions = {
            DesignIssueType.CAPITALIZATION: "Inconsistent capitalization (title case, sentence case, ALL CAPS, etc.)",
            DesignIssueType.FORMATTING: "Text formatting issues (bold, italic, underline inconsistencies)",
            DesignIssueType.CONSISTENCY: "Inconsistent style, terminology, or formatting across slides",
            DesignIssueType.ALIGNMENT: "Text alignment and spacing issues",
            DesignIssueType.SPACING: "Inconsistent spacing, line breaks, or whitespace",
            DesignIssueType.TYPOGRAPHY: "Font, size, or text style inconsistencies",
            DesignIssueType.COLOR: "Color usage and contrast issues",
            DesignIssueType.HIERARCHY: "Visual hierarchy and information organization issues"
        }
        
        focus_text = ""
        for area in focus_areas:
            description = focus_descriptions.get(area, area.value)
            focus_text += f"\n- {area.value.upper()}: {description}"
        
        correction_instruction = ""
        if enable_corrections:
            correction_instruction = '''
- "corrected_text": "Suggested correction for the text"'''
        
        prompt = f"""You are an expert presentation design consultant with a keen eye for visual consistency and professional formatting. Analyze the following presentation slides for design issues.

SLIDES TO ANALYZE:
{slides_text}

FOCUS AREAS - Look specifically for these types of issues:{focus_text}

INSTRUCTIONS:
1. Examine each slide's title, bullet points, and other text elements
2. Identify design inconsistencies and formatting issues
3. Pay special attention to capitalization patterns - flag mixed cases like "market ANALYSIS" or "Sales TARGETS"
4. Look for typography inconsistencies and formatting problems
5. Consider professional presentation standards

Return a JSON array of design issues found. Each issue should have this structure:
[
    {{
        "slide_index": 0,
        "issue_type": "capitalization",
        "severity": "medium",
        "element": "title",
        "original_text": "market ANALYSIS and trends",{correction_instruction}
        "description": "Inconsistent capitalization in slide title - mix of lowercase and uppercase",
        "confidence": 0.9
    }}
]

SEVERITY GUIDELINES:
- "critical": Major issues that significantly impact readability/professionalism
- "high": Important issues that should be addressed
- "medium": Noticeable issues that detract from quality
- "low": Minor issues or style preferences

ELEMENT TYPES: "title", "bullet", "subtitle", "notes", "chart_title", "image_caption"

Be thorough but focus on genuine issues that would be noticed by a professional audience. Aim for high confidence scores (0.8+) for clear violations of presentation standards."""

        return prompt
    
    def _supports_json_mode(self) -> bool:
        """Check if the current model supports JSON mode."""
        json_mode_models = [
            "gpt-4", "gpt-4-0613", "gpt-4-1106-preview", "gpt-4-0125-preview",
            "gpt-4-turbo-preview", "gpt-4-turbo", "gpt-4o", "gpt-4o-mini",
            "gpt-3.5-turbo", "gpt-3.5-turbo-0613", "gpt-3.5-turbo-1106", "gpt-3.5-turbo-0125"
        ]
        return any(model in self.model for model in json_mode_models)

    def _call_llm_with_retries(self, prompt: str) -> dict:
        """
        Call LLM API with retry logic.
        
        Args:
            prompt: Prompt to send to the LLM
            
        Returns:
            Parsed JSON response
        """
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
                            "content": "You are an expert presentation design consultant. Always respond with valid JSON only."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    "temperature": self.temperature,
                    "max_tokens": 3000
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
                    logger.debug(f"Response content: {content}")
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
    
    def _parse_issues_response(self, response_data: dict) -> List[DesignIssue]:
        """
        Parse and validate the LLM response into DesignIssue objects.
        
        Args:
            response_data: JSON response from LLM
            
        Returns:
            List of validated DesignIssue objects
        """
        issues = []
        
        # Handle different response formats
        if isinstance(response_data, list):
            issues_data = response_data
        elif isinstance(response_data, dict) and "issues" in response_data:
            issues_data = response_data["issues"]
        elif isinstance(response_data, dict) and "items" in response_data:
            issues_data = response_data["items"]
        elif isinstance(response_data, dict) and len(response_data) == 1:
            # Single key response, use its value
            key = list(response_data.keys())[0]
            issues_data = response_data[key]
        else:
            logger.error(f"Unexpected response format: {response_data}")
            return []
        
        if not isinstance(issues_data, list):
            logger.error(f"Expected list of issues, got: {type(issues_data)}")
            return []
        
        for item in issues_data:
            try:
                # Validate and create DesignIssue
                issue = DesignIssue(
                    slide_index=item.get("slide_index", 0),
                    issue_type=item.get("issue_type", "formatting"),
                    severity=item.get("severity", "medium"),
                    element=item.get("element", "unknown"),
                    original_text=item.get("original_text", ""),
                    corrected_text=item.get("corrected_text"),
                    description=item.get("description", ""),
                    confidence=item.get("confidence", 0.5)
                )
                issues.append(issue)
                
            except Exception as e:
                logger.warning(f"Invalid issue item: {item}, error: {e}")
                continue
        
        return issues
    
    def convert_to_review_feedback(
        self,
        issues: List[DesignIssue]
    ) -> List[ReviewFeedback]:
        """
        Convert DesignIssue objects to ReviewFeedback format for integration.
        
        Args:
            issues: List of design issues
            
        Returns:
            List of ReviewFeedback objects
        """
        feedback_list = []
        
        for issue in issues:
            # Create suggestion text
            suggestion = f"Fix {issue.issue_type} issue in {issue.element}"
            if issue.corrected_text:
                suggestion += f": change '{issue.original_text}' to '{issue.corrected_text}'"
            
            feedback = ReviewFeedback(
                slide_index=issue.slide_index,
                severity=issue.severity,
                category="design",
                message=issue.description,
                suggestion=suggestion
            )
            feedback_list.append(feedback)
        
        return feedback_list
    
    def generate_test_slides_with_errors(
        self,
        base_slides: List[SlidePlan],
        error_types: List[DesignIssueType],
        error_count: int = 10
    ) -> Tuple[List[SlidePlan], List[Dict[str, str]]]:
        """
        Generate test slides with seeded design errors for testing detection accuracy.
        
        Args:
            base_slides: Base slides to modify
            error_types: Types of errors to seed
            error_count: Number of errors to seed
            
        Returns:
            Tuple of (modified_slides, seeded_errors_list)
        """
        import random
        import copy
        
        modified_slides = copy.deepcopy(base_slides)
        seeded_errors = []
        
        capitalization_patterns = [
            lambda text: text.upper(),  # ALL CAPS
            lambda text: text.lower(),  # all lowercase
            lambda text: ' '.join([word.upper() if i % 2 == 0 else word.lower() 
                                 for i, word in enumerate(text.split())]),  # aLtErNaTiNg
            lambda text: text.replace(' ', ' ').replace('and', 'AND').replace('the', 'THE'),  # Random caps
        ]
        
        errors_created = 0
        max_attempts = error_count * 3  # Prevent infinite loop
        attempts = 0
        
        while errors_created < error_count and attempts < max_attempts:
            attempts += 1
            
            # Randomly select a slide
            slide = random.choice(modified_slides)
            
            # Randomly select error type
            error_type = random.choice(error_types)
            
            if error_type == DesignIssueType.CAPITALIZATION:
                # Randomly choose title or bullet
                if random.choice([True, False]) and slide.title:
                    # Modify title
                    original_title = slide.title
                    pattern = random.choice(capitalization_patterns)
                    slide.title = pattern(original_title)
                    
                    seeded_errors.append({
                        "slide_index": slide.index,
                        "element": "title",
                        "original": original_title,
                        "modified": slide.title,
                        "error_type": "capitalization"
                    })
                    errors_created += 1
                    
                elif slide.bullets:
                    # Modify a random bullet
                    bullet_idx = random.randint(0, len(slide.bullets) - 1)
                    original_bullet = slide.bullets[bullet_idx]
                    pattern = random.choice(capitalization_patterns)
                    slide.bullets[bullet_idx] = pattern(original_bullet)
                    
                    seeded_errors.append({
                        "slide_index": slide.index,
                        "element": "bullet",
                        "original": original_bullet,
                        "modified": slide.bullets[bullet_idx],
                        "error_type": "capitalization"
                    })
                    errors_created += 1
        
        logger.info(f"Generated {errors_created} seeded errors in test slides")
        
        return modified_slides, seeded_errors