"""Reviewer for AI-powered presentation critique and refinement."""

import json
import logging
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import openai
from openai import OpenAI
from pydantic import ValidationError

from .models import QualityGateResult, QualityGates, ReviewFeedback, SlidePlan

logger = logging.getLogger(__name__)


def calculate_readability_score(text: str) -> float:
    """
    Calculate a simplified readability score based on Flesch Reading Ease.
    
    This is a simplified version that estimates reading difficulty.
    Higher scores indicate more complex text.
    
    Args:
        text: Text to analyze
        
    Returns:
        Grade level (0-20+, where 9 = 9th grade level)
    """
    if not text or not text.strip():
        return 0.0
    
    # Clean text and split into sentences
    clean_text = re.sub(r'[^\w\s\.]', ' ', text)
    sentences = [s.strip() for s in re.split(r'[.!?]+', clean_text) if s.strip()]
    
    if not sentences:
        return 0.0
    
    # Count words and syllables
    total_words = 0
    total_syllables = 0
    
    for sentence in sentences:
        words = sentence.split()
        total_words += len(words)
        
        for word in words:
            # Simple syllable counting
            syllables = count_syllables(word.lower())
            total_syllables += syllables
    
    if total_words == 0:
        return 0.0
    
    # Calculate average sentence length and syllables per word
    avg_sentence_length = total_words / len(sentences)
    avg_syllables_per_word = total_syllables / total_words
    
    # Simplified Flesch-Kincaid Grade Level formula
    # Adjusted for presentation context (typically shorter sentences)
    grade_level = (0.39 * avg_sentence_length) + (11.8 * avg_syllables_per_word) - 15.59
    
    # Ensure reasonable bounds
    return max(0.0, min(20.0, grade_level))


def count_syllables(word: str) -> int:
    """
    Count syllables in a word using simple heuristics.
    
    Args:
        word: Word to count syllables for
        
    Returns:
        Number of syllables (minimum 1)
    """
    if not word:
        return 0
    
    word = word.lower()
    
    # Remove common endings that don't add syllables
    word = re.sub(r'[.,:;!?]', '', word)
    
    # Count vowel groups
    vowels = 'aeiouy'
    syllable_count = 0
    prev_was_vowel = False
    
    for char in word:
        is_vowel = char in vowels
        if is_vowel and not prev_was_vowel:
            syllable_count += 1
        prev_was_vowel = is_vowel
    
    # Handle silent 'e'
    if word.endswith('e') and syllable_count > 1:
        syllable_count -= 1
    
    # Ensure at least 1 syllable
    return max(1, syllable_count)


class Reviewer:
    """AI-powered presentation reviewer for quality feedback and iterative refinement."""

    def __init__(self, client: OpenAI, model: str = "gpt-4", temperature: float = 0.2):
        """
        Initialize the reviewer.
        
        Args:
            client: OpenAI client instance
            model: Model name to use for review
            temperature: Temperature for generation (lower for more consistent reviews)
        """
        self.client = client
        self.model = model
        self.temperature = temperature
        self.max_retries = 3
        self.retry_delay = 1.0

    def review_presentation(
        self,
        slides: List[SlidePlan],
        presentation_context: Optional[str] = None,
        review_criteria: Optional[Dict[str, any]] = None,
        include_quality_gates: bool = False,
        quality_gates: Optional[QualityGates] = None
    ) -> Union[List[ReviewFeedback], Tuple[List[ReviewFeedback], QualityGateResult]]:
        """
        Review an entire presentation and provide feedback.
        
        Args:
            slides: List of slides to review
            presentation_context: Context about the presentation (audience, purpose, etc.)
            review_criteria: Specific criteria to focus on during review
            include_quality_gates: If True, also evaluate quality gates and return tuple
            quality_gates: Quality gate configuration (uses defaults if None)
            
        Returns:
            List of feedback items for improvement, or tuple of (feedback, quality_gates_result)
            if include_quality_gates is True
        """
        logger.info(f"Reviewing presentation with {len(slides)} slides")
        
        review_criteria = review_criteria or self._get_default_criteria()
        
        # Create presentation summary for review
        presentation_summary = self._create_presentation_summary(slides, presentation_context)
        
        # Generate review prompt
        prompt = self._build_review_prompt(presentation_summary, review_criteria)
        
        try:
            # Get AI feedback
            feedback_data = self._call_openai_with_retries(prompt)
            
            # Parse and validate feedback
            feedback_list = self._parse_feedback_response(feedback_data)
            
            logger.info(f"Generated {len(feedback_list)} feedback items")
            
            # Evaluate quality gates if requested
            if include_quality_gates:
                quality_result = self.evaluate_quality_gates(slides, feedback_list, quality_gates)
                return feedback_list, quality_result
            
            return feedback_list
            
        except Exception as e:
            logger.error(f"Presentation review failed: {e}")
            if include_quality_gates:
                # Return empty feedback and a failing quality gates result
                empty_feedback = []
                default_gates = quality_gates or QualityGates()
                failing_result = QualityGateResult(
                    status="needs_fix",
                    gate_results={
                        "bullet_count": False,
                        "readability": False,
                        "style_errors": False,
                        "overall_score": False
                    },
                    violations=["Review failed due to an error"],
                    recommendations=["Please retry the review"],
                    metrics={}
                )
                return empty_feedback, failing_result
            return []

    def review_individual_slide(
        self,
        slide: SlidePlan,
        slide_context: Optional[str] = None
    ) -> List[ReviewFeedback]:
        """
        Review an individual slide in detail.
        
        Args:
            slide: Slide to review
            slide_context: Additional context about this slide's role
            
        Returns:
            List of feedback items for this slide
        """
        logger.debug(f"Reviewing individual slide {slide.index}: {slide.title}")
        
        prompt = self._build_slide_review_prompt(slide, slide_context)
        
        try:
            feedback_data = self._call_openai_with_retries(prompt)
            feedback_list = self._parse_feedback_response(feedback_data)
            
            # Ensure all feedback is for this slide
            for feedback in feedback_list:
                feedback.slide_index = slide.index
            
            return feedback_list
            
        except Exception as e:
            logger.error(f"Slide review failed for slide {slide.index}: {e}")
            return []

    def check_presentation_flow(self, slides: List[SlidePlan]) -> List[ReviewFeedback]:
        """
        Check the logical flow and narrative structure of the presentation.
        
        Args:
            slides: List of slides to check flow for
            
        Returns:
            List of flow-related feedback
        """
        logger.info("Checking presentation flow and narrative structure")
        
        # Create flow summary
        flow_summary = self._create_flow_summary(slides)
        
        prompt = f"""You are an expert presentation consultant. Analyze the flow and narrative structure of this presentation.

PRESENTATION FLOW:
{flow_summary}

FOCUS AREAS:
1. Logical progression of ideas
2. Smooth transitions between sections
3. Clear narrative arc (beginning, middle, end)
4. Appropriate pacing and content distribution
5. Missing or redundant content

Return a JSON array of feedback items with this structure:
[
    {{
        "slide_index": 2,
        "severity": "medium",
        "category": "flow",
        "message": "Abrupt transition from introduction to detailed analysis",
        "suggestion": "Add an overview slide to bridge the gap"
    }}
]

Provide 3-5 specific, actionable feedback items focused on improving the presentation flow."""

        try:
            feedback_data = self._call_openai_with_retries(prompt)
            return self._parse_feedback_response(feedback_data)
            
        except Exception as e:
            logger.error(f"Flow check failed: {e}")
            return []

    def evaluate_quality_gates(
        self,
        slides: List[SlidePlan],
        feedback: List[ReviewFeedback],
        gates: Optional[QualityGates] = None
    ) -> QualityGateResult:
        """
        Evaluate presentation against quality gates to determine pass/fail status.
        
        Args:
            slides: List of slides to evaluate
            feedback: List of review feedback items
            gates: Quality gate configuration (uses defaults if None)
            
        Returns:
            QualityGateResult with pass/fail status and detailed metrics
        """
        logger.info("Evaluating presentation against quality gates")
        
        if gates is None:
            gates = QualityGates()
        
        # Initialize results
        gate_results = {}
        violations = []
        recommendations = []
        metrics = {}
        
        # 1. Check bullet count per slide
        max_bullets_found = 0
        bullet_violations = []
        
        for slide in slides:
            bullet_count = len(slide.bullets)
            if bullet_count > max_bullets_found:
                max_bullets_found = bullet_count
            
            if bullet_count > gates.max_bullets_per_slide:
                violation = f"Slide {slide.index + 1} has {bullet_count} bullets (exceeds limit of {gates.max_bullets_per_slide})"
                violations.append(violation)
                bullet_violations.append(slide.index)
        
        gate_results["bullet_count"] = len(bullet_violations) == 0
        metrics["max_bullets_found"] = max_bullets_found
        
        if bullet_violations:
            recommendations.append(f"Reduce bullet points on slides {', '.join(str(i+1) for i in bullet_violations)} to {gates.max_bullets_per_slide} or fewer")
        
        # 2. Check readability grade
        readability_scores = []
        readability_violations = []
        
        for slide in slides:
            # Combine title and bullet text for readability analysis
            slide_text = slide.title
            if slide.bullets:
                slide_text += ". " + ". ".join(slide.bullets)
            
            if slide_text.strip():
                readability_grade = calculate_readability_score(slide_text)
                readability_scores.append(readability_grade)
                
                if readability_grade > gates.max_readability_grade:
                    violation = f"Slide {slide.index + 1} has readability grade {readability_grade:.1f} (exceeds limit of {gates.max_readability_grade})"
                    violations.append(violation)
                    readability_violations.append(slide.index)
        
        gate_results["readability"] = len(readability_violations) == 0
        
        if readability_scores:
            metrics["avg_readability_grade"] = sum(readability_scores) / len(readability_scores)
            metrics["max_readability_grade"] = max(readability_scores)
        else:
            metrics["avg_readability_grade"] = 0.0
            metrics["max_readability_grade"] = 0.0
        
        if readability_violations:
            recommendations.append(f"Simplify language on slides {', '.join(str(i+1) for i in readability_violations)} to improve readability")
        
        # 3. Check style errors
        style_error_count = sum(1 for f in feedback if f.category == "design" or f.category == "consistency")
        gate_results["style_errors"] = style_error_count <= gates.max_style_errors
        metrics["style_error_count"] = style_error_count
        
        if style_error_count > gates.max_style_errors:
            violations.append(f"Found {style_error_count} style errors (exceeds limit of {gates.max_style_errors})")
            recommendations.append("Address style and consistency issues identified in feedback")
        
        # 4. Check overall score
        review_summary = self.get_review_summary(feedback)
        overall_score = review_summary["overall_score"]
        gate_results["overall_score"] = overall_score >= gates.min_overall_score
        metrics["overall_score"] = overall_score
        
        if overall_score < gates.min_overall_score:
            violations.append(f"Overall score {overall_score} is below minimum threshold of {gates.min_overall_score}")
            recommendations.append("Address critical and high severity feedback to improve overall score")
        
        # Determine overall status
        all_gates_passed = all(gate_results.values())
        status = "pass" if all_gates_passed else "needs_fix"
        
        logger.info(f"Quality gates evaluation completed: {status} ({sum(gate_results.values())}/{len(gate_results)} gates passed)")
        
        return QualityGateResult(
            status=status,
            gate_results=gate_results,
            violations=violations,
            recommendations=recommendations,
            metrics=metrics
        )

    def _create_presentation_summary(
        self,
        slides: List[SlidePlan],
        context: Optional[str]
    ) -> str:
        """Create a summary of the presentation for review."""
        
        summary_parts = []
        
        if context:
            summary_parts.append(f"CONTEXT: {context}\n")
        
        summary_parts.append(f"TOTAL SLIDES: {len(slides)}\n")
        
        summary_parts.append("SLIDE BREAKDOWN:")
        for slide in slides:
            slide_summary = f"\nSlide {slide.index + 1} ({slide.slide_type}): {slide.title}"
            
            if slide.bullets:
                slide_summary += f"\n  • {len(slide.bullets)} bullet points"
                if len(slide.bullets) <= 3:
                    for bullet in slide.bullets:
                        slide_summary += f"\n    - {bullet[:60]}{'...' if len(bullet) > 60 else ''}"
            
            if slide.image_query:
                slide_summary += f"\n  • Image: {slide.image_query}"
            
            if slide.chart_data:
                slide_summary += f"\n  • Chart: {slide.chart_data.get('type', 'unknown')} chart"
            
            summary_parts.append(slide_summary)
        
        return "\n".join(summary_parts)

    def _create_flow_summary(self, slides: List[SlidePlan]) -> str:
        """Create a flow-focused summary of the presentation."""
        
        flow_parts = []
        
        flow_parts.append("PRESENTATION STRUCTURE:")
        
        current_section = None
        for i, slide in enumerate(slides):
            # Detect section changes
            if slide.slide_type in ["title", "section"]:
                current_section = slide.title
                flow_parts.append(f"\n[SECTION] {slide.title}")
            else:
                if current_section:
                    flow_parts.append(f"  {i+1}. {slide.title}")
                else:
                    flow_parts.append(f"{i+1}. {slide.title}")
            
            # Add brief content summary
            if slide.bullets:
                main_points = [bullet[:40] + "..." if len(bullet) > 40 else bullet 
                              for bullet in slide.bullets[:2]]
                flow_parts.append(f"     → {' | '.join(main_points)}")
        
        # Add transition analysis
        flow_parts.append("\nTRANSITION ANALYSIS:")
        for i in range(len(slides) - 1):
            current = slides[i]
            next_slide = slides[i + 1]
            
            transition = f"Slide {i+1} → Slide {i+2}: "
            transition += f"'{current.title}' to '{next_slide.title}'"
            flow_parts.append(transition)
        
        return "\n".join(flow_parts)

    def _build_review_prompt(
        self,
        presentation_summary: str,
        criteria: Dict[str, any]
    ) -> str:
        """Build the main review prompt."""
        
        criteria_text = ""
        for category, details in criteria.items():
            criteria_text += f"\n- {category.upper()}: {details['description']}"
            if details.get('weight'):
                criteria_text += f" (Weight: {details['weight']})"
        
        prompt = f"""You are an expert presentation consultant with experience reviewing corporate presentations. Provide constructive feedback to improve this presentation.

{presentation_summary}

REVIEW CRITERIA:{criteria_text}

INSTRUCTIONS:
1. Focus on actionable, specific improvements
2. Consider the overall narrative and flow
3. Identify slides that may be too dense or sparse
4. Check for consistency in tone and style
5. Look for missing transitions or unclear connections
6. Assess visual balance and content distribution

Return a JSON array of feedback items. Each item should have:
- slide_index: Which slide (0-based index, or -1 for presentation-wide issues)
- severity: "low", "medium", "high", or "critical"
- category: "content", "flow", "design", "clarity", "consistency"
- message: Clear description of the issue
- suggestion: Specific recommendation for improvement

Provide 5-10 most important feedback items that would significantly improve the presentation."""

        return prompt

    def _build_slide_review_prompt(
        self,
        slide: SlidePlan,
        context: Optional[str]
    ) -> str:
        """Build prompt for individual slide review."""
        
        context_text = f"Context: {context}\n" if context else ""
        
        prompt = f"""You are an expert presentation consultant. Review this individual slide for improvement opportunities.

{context_text}
SLIDE DETAILS:
- Index: {slide.index}
- Type: {slide.slide_type}
- Title: {slide.title}
- Bullets: {slide.bullets}
- Speaker Notes: {slide.speaker_notes or 'None'}
- Has Image: {'Yes' if slide.image_query else 'No'}
- Has Chart: {'Yes' if slide.chart_data else 'No'}

REVIEW FOCUS:
1. Title clarity and impact
2. Content density and readability
3. Bullet point structure and parallelism
4. Visual elements appropriateness
5. Speaker notes quality
6. Overall slide effectiveness

Return a JSON array of specific feedback items for this slide. Include 2-4 actionable suggestions."""

        return prompt

    def _get_default_criteria(self) -> Dict[str, any]:
        """Get default review criteria."""
        return {
            "clarity": {
                "description": "Content is clear, concise, and easy to understand",
                "weight": "high"
            },
            "flow": {
                "description": "Logical progression and smooth transitions between ideas",
                "weight": "high"
            },
            "content": {
                "description": "Appropriate content density and meaningful information",
                "weight": "medium"
            },
            "consistency": {
                "description": "Consistent tone, style, and formatting throughout",
                "weight": "medium"
            },
            "engagement": {
                "description": "Content engages audience and maintains interest",
                "weight": "medium"
            }
        }

    def _call_openai_with_retries(self, prompt: str) -> dict:
        """Call OpenAI API with retry logic."""
        
        for attempt in range(self.max_retries):
            try:
                logger.debug(f"Review API call attempt {attempt + 1}")
                
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert presentation consultant. Always respond with valid JSON only."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=self.temperature,
                    max_tokens=2000,
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
        
        raise ValueError("Failed to get review after all retries")

    def _parse_feedback_response(self, response_data: dict) -> List[ReviewFeedback]:
        """Parse and validate feedback response from AI."""
        
        feedback_list = []
        
        # Handle different response formats
        if isinstance(response_data, list):
            feedback_data = response_data
        elif isinstance(response_data, dict) and "feedback" in response_data:
            feedback_data = response_data["feedback"]
        elif isinstance(response_data, dict) and "items" in response_data:
            feedback_data = response_data["items"]
        else:
            logger.error(f"Unexpected response format: {response_data}")
            return []
        
        for item in feedback_data:
            try:
                # Ensure required fields
                feedback_item = ReviewFeedback(
                    slide_index=item.get("slide_index", -1),
                    severity=item.get("severity", "medium"),
                    category=item.get("category", "general"),
                    message=item.get("message", ""),
                    suggestion=item.get("suggestion")
                )
                feedback_list.append(feedback_item)
                
            except ValidationError as e:
                logger.warning(f"Invalid feedback item: {item}, error: {e}")
                continue
        
        return feedback_list

    def prioritize_feedback(
        self,
        feedback_list: List[ReviewFeedback]
    ) -> List[ReviewFeedback]:
        """
        Prioritize feedback items by severity and category.
        
        Args:
            feedback_list: List of feedback items to prioritize
            
        Returns:
            Sorted list with highest priority items first
        """
        severity_order = {"critical": 4, "high": 3, "medium": 2, "low": 1}
        category_order = {"flow": 4, "clarity": 3, "content": 2, "consistency": 1, "design": 1}
        
        def priority_score(feedback: ReviewFeedback) -> tuple:
            severity_score = severity_order.get(feedback.severity, 0)
            category_score = category_order.get(feedback.category, 0)
            return (severity_score, category_score)
        
        return sorted(feedback_list, key=priority_score, reverse=True)

    def filter_feedback(
        self,
        feedback_list: List[ReviewFeedback],
        min_severity: str = "low",
        categories: Optional[List[str]] = None,
        slide_indices: Optional[List[int]] = None
    ) -> List[ReviewFeedback]:
        """
        Filter feedback based on criteria.
        
        Args:
            feedback_list: List of feedback to filter
            min_severity: Minimum severity level to include
            categories: List of categories to include (None for all)
            slide_indices: List of slide indices to include (None for all)
            
        Returns:
            Filtered list of feedback
        """
        severity_levels = {"low": 1, "medium": 2, "high": 3, "critical": 4}
        min_level = severity_levels.get(min_severity, 1)
        
        filtered = []
        for feedback in feedback_list:
            # Check severity
            feedback_level = severity_levels.get(feedback.severity, 1)
            if feedback_level < min_level:
                continue
            
            # Check category
            if categories and feedback.category not in categories:
                continue
            
            # Check slide index
            if slide_indices and feedback.slide_index not in slide_indices:
                continue
            
            filtered.append(feedback)
        
        return filtered

    def generate_improvement_plan(
        self,
        feedback_list: List[ReviewFeedback]
    ) -> Dict[str, any]:
        """
        Generate an improvement plan based on feedback.
        
        Args:
            feedback_list: List of feedback items
            
        Returns:
            Structured improvement plan
        """
        plan = {
            "total_issues": len(feedback_list),
            "by_severity": {},
            "by_category": {},
            "by_slide": {},
            "quick_wins": [],
            "major_improvements": [],
            "action_items": []
        }
        
        # Categorize feedback
        for feedback in feedback_list:
            # By severity
            plan["by_severity"][feedback.severity] = plan["by_severity"].get(feedback.severity, 0) + 1
            
            # By category
            plan["by_category"][feedback.category] = plan["by_category"].get(feedback.category, 0) + 1
            
            # By slide
            slide_key = f"slide_{feedback.slide_index}"
            if slide_key not in plan["by_slide"]:
                plan["by_slide"][slide_key] = []
            plan["by_slide"][slide_key].append(feedback.message)
            
            # Categorize by effort level
            if feedback.severity in ["low", "medium"] and len(feedback.message) < 100:
                plan["quick_wins"].append({
                    "slide": feedback.slide_index,
                    "issue": feedback.message,
                    "fix": feedback.suggestion
                })
            elif feedback.severity in ["high", "critical"]:
                plan["major_improvements"].append({
                    "slide": feedback.slide_index,
                    "issue": feedback.message,
                    "fix": feedback.suggestion
                })
        
        # Generate action items
        prioritized = self.prioritize_feedback(feedback_list)
        for i, feedback in enumerate(prioritized[:10]):  # Top 10 items
            plan["action_items"].append({
                "priority": i + 1,
                "slide": feedback.slide_index,
                "category": feedback.category,
                "action": feedback.suggestion or "Review and improve based on feedback",
                "severity": feedback.severity
            })
        
        return plan

    def get_review_summary(
        self,
        feedback_list: List[ReviewFeedback]
    ) -> Dict[str, any]:
        """
        Get a summary of the review results.
        
        Args:
            feedback_list: List of feedback items
            
        Returns:
            Review summary statistics
        """
        if not feedback_list:
            return {
                "total_feedback": 0,
                "overall_score": 10,  # Perfect if no issues
                "summary": "No issues found - presentation looks good!"
            }
        
        severity_counts = {}
        category_counts = {}
        slides_with_issues = set()
        
        for feedback in feedback_list:
            severity_counts[feedback.severity] = severity_counts.get(feedback.severity, 0) + 1
            category_counts[feedback.category] = category_counts.get(feedback.category, 0) + 1
            if feedback.slide_index >= 0:
                slides_with_issues.add(feedback.slide_index)
        
        # Calculate overall score (0-10 scale)
        severity_weights = {"critical": -3, "high": -2, "medium": -1, "low": -0.5}
        total_deduction = sum(severity_weights.get(sev, 0) * count 
                             for sev, count in severity_counts.items())
        overall_score = max(0, min(10, 10 + total_deduction))
        
        # Generate summary text
        critical_count = severity_counts.get("critical", 0)
        high_count = severity_counts.get("high", 0)
        
        if critical_count > 0:
            summary = f"{critical_count} critical issues need immediate attention"
        elif high_count > 0:
            summary = f"{high_count} high-priority improvements recommended"
        elif len(feedback_list) > 5:
            summary = "Several minor improvements would enhance the presentation"
        else:
            summary = "Good presentation with some minor refinements possible"
        
        return {
            "total_feedback": len(feedback_list),
            "severity_breakdown": severity_counts,
            "category_breakdown": category_counts,
            "slides_with_issues": len(slides_with_issues),
            "overall_score": round(overall_score, 1),
            "summary": summary
        }