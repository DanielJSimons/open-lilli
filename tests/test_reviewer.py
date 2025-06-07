"""Tests for reviewer."""

import json
from unittest.mock import Mock

import pytest
from openai import OpenAI

from open_lilli.models import QualityGateResult, QualityGates, ReviewFeedback, SlidePlan
from open_lilli.reviewer import Reviewer, calculate_readability_score, count_syllables


class TestReviewer:
    """Tests for Reviewer class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_client = Mock(spec=OpenAI)
        self.reviewer = Reviewer(self.mock_client)

    def create_test_slides(self) -> list[SlidePlan]:
        """Create test slides for review."""
        return [
            SlidePlan(
                index=0,
                slide_type="title",
                title="Presentation Title",
                bullets=[],
                speaker_notes="Welcome to the presentation"
            ),
            SlidePlan(
                index=1,
                slide_type="content",
                title="Main Content",
                bullets=["Point 1", "Point 2", "Point 3"],
                speaker_notes="Discuss the main points",
                image_query="business meeting"
            ),
            SlidePlan(
                index=2,
                slide_type="chart",
                title="Performance Data",
                bullets=["Key insight"],
                chart_data={"type": "bar", "values": [1, 2, 3]}
            )
        ]

    def test_init(self):
        """Test Reviewer initialization."""
        assert self.reviewer.client == self.mock_client
        assert self.reviewer.model == "gpt-4"
        assert self.reviewer.temperature == 0.2

    def test_review_presentation_success(self):
        """Test successful presentation review."""
        slides = self.create_test_slides()
        
        # Mock API response
        mock_response_data = {
            "feedback": [
                {
                    "slide_index": 1,
                    "severity": "medium",
                    "category": "content",
                    "message": "Too many bullet points on this slide",
                    "suggestion": "Consider splitting into two slides"
                },
                {
                    "slide_index": 2,
                    "severity": "low",
                    "category": "design",
                    "message": "Chart could use better labels",
                    "suggestion": "Add descriptive axis labels"
                }
            ]
        }
        
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = json.dumps(mock_response_data)
        
        self.mock_client.chat.completions.create.return_value = mock_response
        
        feedback = self.reviewer.review_presentation(slides)
        
        assert len(feedback) == 2
        assert isinstance(feedback[0], ReviewFeedback)
        assert feedback[0].slide_index == 1
        assert feedback[0].severity == "medium"
        assert feedback[1].slide_index == 2
        assert feedback[1].severity == "low"

    def test_review_individual_slide(self):
        """Test individual slide review."""
        slide = self.create_test_slides()[1]
        
        mock_response_data = [
            {
                "slide_index": 1,
                "severity": "medium",
                "category": "clarity",
                "message": "Title could be more specific",
                "suggestion": "Use a more descriptive title"
            }
        ]
        
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = json.dumps(mock_response_data)
        
        self.mock_client.chat.completions.create.return_value = mock_response
        
        feedback = self.reviewer.review_individual_slide(slide)
        
        assert len(feedback) == 1
        assert feedback[0].slide_index == 1  # Should be set to the slide's index

    def test_check_presentation_flow(self):
        """Test presentation flow checking."""
        slides = self.create_test_slides()
        
        mock_response_data = [
            {
                "slide_index": 1,
                "severity": "medium",
                "category": "flow",
                "message": "Abrupt transition from title to content",
                "suggestion": "Add an overview slide"
            }
        ]
        
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = json.dumps(mock_response_data)
        
        self.mock_client.chat.completions.create.return_value = mock_response
        
        feedback = self.reviewer.check_presentation_flow(slides)
        
        assert len(feedback) == 1
        assert feedback[0].category == "flow"

    def test_create_presentation_summary(self):
        """Test presentation summary creation."""
        slides = self.create_test_slides()
        context = "Executive presentation for Q4 review"
        
        summary = self.reviewer._create_presentation_summary(slides, context)
        
        assert "CONTEXT: Executive presentation" in summary
        assert "TOTAL SLIDES: 3" in summary
        assert "Presentation Title" in summary
        assert "Main Content" in summary
        assert "Performance Data" in summary
        assert "3 bullet points" in summary

    def test_create_flow_summary(self):
        """Test flow summary creation."""
        slides = self.create_test_slides()
        
        flow_summary = self.reviewer._create_flow_summary(slides)
        
        assert "PRESENTATION STRUCTURE:" in flow_summary
        assert "TRANSITION ANALYSIS:" in flow_summary
        assert "Presentation Title" in flow_summary
        assert "Main Content" in flow_summary
        assert "Slide 1 → Slide 2" in flow_summary

    def test_parse_feedback_response_list_format(self):
        """Test parsing feedback in list format."""
        response_data = [
            {
                "slide_index": 0,
                "severity": "high",
                "category": "content",
                "message": "Test message",
                "suggestion": "Test suggestion"
            }
        ]
        
        feedback = self.reviewer._parse_feedback_response(response_data)
        
        assert len(feedback) == 1
        assert feedback[0].slide_index == 0
        assert feedback[0].severity == "high"

    def test_parse_feedback_response_dict_format(self):
        """Test parsing feedback in dictionary format."""
        response_data = {
            "feedback": [
                {
                    "slide_index": 1,
                    "severity": "medium",
                    "category": "design",
                    "message": "Test message"
                }
            ]
        }
        
        feedback = self.reviewer._parse_feedback_response(response_data)
        
        assert len(feedback) == 1
        assert feedback[0].slide_index == 1

    def test_parse_feedback_response_invalid_item(self):
        """Test handling of invalid feedback items."""
        response_data = [
            {
                "slide_index": 0,
                "severity": "high",
                "category": "content",
                "message": "Valid item"
            },
            {
                # Missing required fields
                "severity": "medium"
            }
        ]
        
        feedback = self.reviewer._parse_feedback_response(response_data)
        
        # Should only include the valid item
        assert len(feedback) == 1
        assert feedback[0].message == "Valid item"

    def test_prioritize_feedback(self):
        """Test feedback prioritization."""
        feedback_list = [
            ReviewFeedback(
                slide_index=0,
                severity="low",
                category="design",
                message="Minor issue"
            ),
            ReviewFeedback(
                slide_index=1,
                severity="critical",
                category="flow",
                message="Critical issue"
            ),
            ReviewFeedback(
                slide_index=2,
                severity="medium",
                category="content",
                message="Medium issue"
            )
        ]
        
        prioritized = self.reviewer.prioritize_feedback(feedback_list)
        
        # Critical should be first, low should be last
        assert prioritized[0].severity == "critical"
        assert prioritized[-1].severity == "low"

    def test_filter_feedback_by_severity(self):
        """Test filtering feedback by severity."""
        feedback_list = [
            ReviewFeedback(
                slide_index=0, severity="low", category="design", message="Low issue"
            ),
            ReviewFeedback(
                slide_index=1, severity="high", category="content", message="High issue"
            ),
            ReviewFeedback(
                slide_index=2, severity="medium", category="flow", message="Medium issue"
            )
        ]
        
        # Filter for medium and above
        filtered = self.reviewer.filter_feedback(feedback_list, min_severity="medium")
        
        assert len(filtered) == 2
        assert all(f.severity in ["medium", "high"] for f in filtered)

    def test_filter_feedback_by_category(self):
        """Test filtering feedback by category."""
        feedback_list = [
            ReviewFeedback(
                slide_index=0, severity="medium", category="design", message="Design issue"
            ),
            ReviewFeedback(
                slide_index=1, severity="medium", category="content", message="Content issue"
            ),
            ReviewFeedback(
                slide_index=2, severity="medium", category="flow", message="Flow issue"
            )
        ]
        
        # Filter for specific categories
        filtered = self.reviewer.filter_feedback(
            feedback_list, categories=["content", "flow"]
        )
        
        assert len(filtered) == 2
        assert all(f.category in ["content", "flow"] for f in filtered)

    def test_filter_feedback_by_slide_indices(self):
        """Test filtering feedback by slide indices."""
        feedback_list = [
            ReviewFeedback(
                slide_index=0, severity="medium", category="design", message="Slide 0"
            ),
            ReviewFeedback(
                slide_index=1, severity="medium", category="content", message="Slide 1"
            ),
            ReviewFeedback(
                slide_index=2, severity="medium", category="flow", message="Slide 2"
            )
        ]
        
        # Filter for specific slides
        filtered = self.reviewer.filter_feedback(
            feedback_list, slide_indices=[0, 2]
        )
        
        assert len(filtered) == 2
        assert all(f.slide_index in [0, 2] for f in filtered)

    def test_generate_improvement_plan(self):
        """Test improvement plan generation."""
        feedback_list = [
            ReviewFeedback(
                slide_index=0, severity="critical", category="flow", 
                message="Critical flow issue", suggestion="Fix flow"
            ),
            ReviewFeedback(
                slide_index=1, severity="low", category="design",
                message="Minor design issue", suggestion="Improve design"
            ),
            ReviewFeedback(
                slide_index=1, severity="medium", category="content",
                message="Content needs work", suggestion="Revise content"
            )
        ]
        
        plan = self.reviewer.generate_improvement_plan(feedback_list)
        
        assert plan["total_issues"] == 3
        assert plan["by_severity"]["critical"] == 1
        assert plan["by_severity"]["low"] == 1
        assert plan["by_severity"]["medium"] == 1
        assert plan["by_category"]["flow"] == 1
        assert plan["by_category"]["design"] == 1
        assert plan["by_category"]["content"] == 1
        assert len(plan["major_improvements"]) == 1  # Critical issue
        assert len(plan["quick_wins"]) == 2  # Low and medium issues
        assert len(plan["action_items"]) == 3

    def test_get_review_summary_no_issues(self):
        """Test review summary with no issues."""
        summary = self.reviewer.get_review_summary([])
        
        assert summary["total_feedback"] == 0
        assert summary["overall_score"] == 10
        assert "No issues found" in summary["summary"]

    def test_get_review_summary_with_issues(self):
        """Test review summary with various issues."""
        feedback_list = [
            ReviewFeedback(
                slide_index=0, severity="critical", category="flow", message="Critical issue"
            ),
            ReviewFeedback(
                slide_index=1, severity="high", category="content", message="High issue"
            ),
            ReviewFeedback(
                slide_index=2, severity="medium", category="design", message="Medium issue"
            ),
            ReviewFeedback(
                slide_index=2, severity="low", category="clarity", message="Low issue"
            )
        ]
        
        summary = self.reviewer.get_review_summary(feedback_list)
        
        assert summary["total_feedback"] == 4
        assert summary["severity_breakdown"]["critical"] == 1
        assert summary["severity_breakdown"]["high"] == 1
        assert summary["severity_breakdown"]["medium"] == 1
        assert summary["severity_breakdown"]["low"] == 1
        assert summary["slides_with_issues"] == 3
        assert summary["overall_score"] < 10  # Should be reduced due to issues
        assert "critical" in summary["summary"].lower()

    def test_get_default_criteria(self):
        """Test default review criteria."""
        criteria = self.reviewer._get_default_criteria()
        
        assert "clarity" in criteria
        assert "flow" in criteria
        assert "content" in criteria
        assert "consistency" in criteria
        assert "engagement" in criteria
        
        # Check structure
        for category, details in criteria.items():
            assert "description" in details
            assert "weight" in details

    def test_api_failure_handling(self):
        """Test handling of API failures."""
        slides = self.create_test_slides()
        
        # Mock API failure
        self.mock_client.chat.completions.create.side_effect = Exception("API Error")
        
        feedback = self.reviewer.review_presentation(slides)
        
        # Should return empty list on failure
        assert feedback == []

    def test_invalid_json_handling(self):
        """Test handling of invalid JSON response."""
        slides = self.create_test_slides()
        
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "invalid json {"
        
        self.mock_client.chat.completions.create.return_value = mock_response
        
        feedback = self.reviewer.review_presentation(slides)
        
        # Should return empty list on JSON parse failure
        assert feedback == []


class TestReadabilityAssessment:
    """Tests for readability assessment functions."""
    
    def test_count_syllables_simple_words(self):
        """Test syllable counting for simple words."""
        assert count_syllables("cat") == 1
        assert count_syllables("dog") == 1
        assert count_syllables("hello") == 2
        assert count_syllables("business") == 3
        assert count_syllables("presentation") == 4
        
    def test_count_syllables_edge_cases(self):
        """Test syllable counting edge cases."""
        assert count_syllables("") == 0
        assert count_syllables("a") == 1
        assert count_syllables("the") == 1
        assert count_syllables("beautiful") == 3  # silent 'e'
        assert count_syllables("create") == 2  # silent 'e'
        
    def test_count_syllables_complex_words(self):
        """Test syllable counting for complex words."""
        assert count_syllables("organization") >= 4
        assert count_syllables("university") >= 4
        assert count_syllables("development") >= 3
        
    def test_calculate_readability_score_empty_text(self):
        """Test readability calculation with empty text."""
        assert calculate_readability_score("") == 0.0
        assert calculate_readability_score("   ") == 0.0
        assert calculate_readability_score(None) == 0.0
        
    def test_calculate_readability_score_simple_text(self):
        """Test readability calculation with simple text."""
        simple_text = "The cat sat on the mat. It was a big cat."
        score = calculate_readability_score(simple_text)
        assert 0 <= score <= 20
        assert score < 8  # Should be easy to read
        
    def test_calculate_readability_score_complex_text(self):
        """Test readability calculation with complex text."""
        complex_text = "The organizational infrastructure necessitates comprehensive implementation methodologies."
        score = calculate_readability_score(complex_text)
        assert score > 10  # Should be more difficult to read
        
    def test_calculate_readability_score_presentation_text(self):
        """Test readability calculation with typical presentation text."""
        presentation_text = "Market growth increased by 15% year-over-year. Customer satisfaction improved significantly."
        score = calculate_readability_score(presentation_text)
        assert 5 <= score <= 12  # Reasonable business presentation level


class TestQualityGates:
    """Tests for quality gate evaluation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_client = Mock(spec=OpenAI)
        self.reviewer = Reviewer(self.mock_client)
        
    def create_test_slides_with_bullets(self, bullet_counts: list) -> list[SlidePlan]:
        """Create test slides with specified bullet counts."""
        slides = []
        for i, count in enumerate(bullet_counts):
            bullets = [f"Bullet point {j+1}" for j in range(count)]
            slides.append(SlidePlan(
                index=i,
                slide_type="content",
                title=f"Slide {i+1} Title",
                bullets=bullets
            ))
        return slides
        
    def create_test_slides_with_readability(self) -> list[SlidePlan]:
        """Create test slides with varying readability levels."""
        return [
            SlidePlan(
                index=0,
                slide_type="content",
                title="Simple Title",
                bullets=["Simple text.", "Easy to read.", "Short words."]
            ),
            SlidePlan(
                index=1,
                slide_type="content", 
                title="Complex Organizational Infrastructure Analysis",
                bullets=[
                    "Comprehensive implementation methodologies necessitate organizational restructuring.",
                    "Sophisticated analytical frameworks require interdisciplinary collaboration mechanisms.",
                    "Multifaceted strategic initiatives demand extensive stakeholder engagement protocols."
                ]
            )
        ]
        
    def create_test_feedback_with_style_errors(self, error_count: int) -> list[ReviewFeedback]:
        """Create test feedback with specified number of style errors."""
        feedback = []
        
        # Add non-style errors first
        feedback.append(ReviewFeedback(
            slide_index=0,
            severity="medium",
            category="content",
            message="Content could be improved"
        ))
        
        # Add style errors
        for i in range(error_count):
            if i % 2 == 0:
                category = "design"
            else:
                category = "consistency"
                
            feedback.append(ReviewFeedback(
                slide_index=i,
                severity="medium",
                category=category,
                message=f"Style error {i+1}"
            ))
            
        return feedback
        
    def test_quality_gates_default_config(self):
        """Test quality gates with default configuration."""
        gates = QualityGates()
        assert gates.max_bullets_per_slide == 7
        assert gates.max_readability_grade == 9.0
        assert gates.max_style_errors == 0
        assert gates.min_overall_score == 7.0
        
    def test_quality_gates_custom_config(self):
        """Test quality gates with custom configuration."""
        gates = QualityGates(
            max_bullets_per_slide=5,
            max_readability_grade=8.0,
            max_style_errors=2,
            min_overall_score=8.0
        )
        assert gates.max_bullets_per_slide == 5
        assert gates.max_readability_grade == 8.0
        assert gates.max_style_errors == 2
        assert gates.min_overall_score == 8.0
        
    def test_evaluate_quality_gates_all_pass(self):
        """Test quality gates evaluation when all gates pass."""
        slides = self.create_test_slides_with_bullets([3, 2, 4])  # All under limit of 7
        feedback = []  # No feedback means high score and no style errors
        
        result = self.reviewer.evaluate_quality_gates(slides, feedback)
        
        assert result.status == "pass"
        assert result.gate_results["bullet_count"] is True
        assert result.gate_results["readability"] is True
        assert result.gate_results["style_errors"] is True
        assert result.gate_results["overall_score"] is True
        assert len(result.violations) == 0
        assert result.passed_gates == 4
        assert result.total_gates == 4
        assert result.pass_rate == 100.0
        
    def test_evaluate_quality_gates_bullet_count_fail(self):
        """Test quality gates when bullet count exceeds limit."""
        slides = self.create_test_slides_with_bullets([3, 8, 4])  # Second slide exceeds limit
        feedback = []
        
        result = self.reviewer.evaluate_quality_gates(slides, feedback)
        
        assert result.status == "needs_fix"
        assert result.gate_results["bullet_count"] is False
        assert any("Slide 2 has 8 bullets" in violation for violation in result.violations)
        assert any("Reduce bullet points" in rec for rec in result.recommendations)
        assert result.metrics["max_bullets_found"] == 8
        
    def test_evaluate_quality_gates_readability_fail(self):
        """Test quality gates when readability exceeds limit."""
        slides = self.create_test_slides_with_readability()
        feedback = []
        
        # Use strict readability limit
        gates = QualityGates(max_readability_grade=6.0)
        result = self.reviewer.evaluate_quality_gates(slides, feedback, gates)
        
        assert result.status == "needs_fix"
        assert result.gate_results["readability"] is False
        assert any("readability grade" in violation for violation in result.violations)
        assert any("Simplify language" in rec for rec in result.recommendations)
        assert "avg_readability_grade" in result.metrics
        assert "max_readability_grade" in result.metrics
        
    def test_evaluate_quality_gates_style_errors_fail(self):
        """Test quality gates when style errors exceed limit."""
        slides = self.create_test_slides_with_bullets([3, 2])
        feedback = self.create_test_feedback_with_style_errors(2)  # 2 style errors, limit is 0
        
        result = self.reviewer.evaluate_quality_gates(slides, feedback)
        
        assert result.status == "needs_fix"
        assert result.gate_results["style_errors"] is False
        assert any("2 style errors" in violation for violation in result.violations)
        assert any("style and consistency" in rec for rec in result.recommendations)
        assert result.metrics["style_error_count"] == 2
        
    def test_evaluate_quality_gates_overall_score_fail(self):
        """Test quality gates when overall score is too low."""
        slides = self.create_test_slides_with_bullets([3, 2])
        
        # Create feedback that will result in low score
        feedback = [
            ReviewFeedback(
                slide_index=0,
                severity="critical",
                category="content",
                message="Critical issue 1"
            ),
            ReviewFeedback(
                slide_index=1, 
                severity="critical",
                category="flow",
                message="Critical issue 2"
            ),
            ReviewFeedback(
                slide_index=0,
                severity="high",
                category="clarity",
                message="High priority issue"
            )
        ]
        
        result = self.reviewer.evaluate_quality_gates(slides, feedback)
        
        assert result.status == "needs_fix"
        assert result.gate_results["overall_score"] is False
        assert any("Overall score" in violation for violation in result.violations)
        assert any("critical and high severity" in rec for rec in result.recommendations)
        assert result.metrics["overall_score"] < 7.0
        
    def test_evaluate_quality_gates_multiple_failures(self):
        """Test quality gates with multiple failing gates."""
        slides = self.create_test_slides_with_bullets([3, 10, 2])  # Bullet count failure
        feedback = self.create_test_feedback_with_style_errors(3)  # Style error failure
        
        result = self.reviewer.evaluate_quality_gates(slides, feedback)
        
        assert result.status == "needs_fix"
        assert result.gate_results["bullet_count"] is False
        assert result.gate_results["style_errors"] is False
        assert len(result.violations) >= 2
        assert len(result.recommendations) >= 2
        assert result.passed_gates < result.total_gates
        assert result.pass_rate < 100.0
        
    def test_evaluate_quality_gates_custom_thresholds(self):
        """Test quality gates with custom thresholds."""
        slides = self.create_test_slides_with_bullets([6, 5, 4])  # Would pass default but fail custom
        feedback = []
        
        # Custom stricter limits
        gates = QualityGates(
            max_bullets_per_slide=5,
            max_readability_grade=7.0,
            max_style_errors=0,
            min_overall_score=8.0
        )
        
        result = self.reviewer.evaluate_quality_gates(slides, feedback, gates)
        
        assert result.status == "needs_fix"
        assert result.gate_results["bullet_count"] is False  # First slide has 6 bullets
        assert any("exceeds limit of 5" in violation for violation in result.violations)
        
    def test_evaluate_quality_gates_empty_slides(self):
        """Test quality gates with empty slides."""
        slides = []
        feedback = []
        
        result = self.reviewer.evaluate_quality_gates(slides, feedback)
        
        assert result.status == "pass"  # Empty presentation technically passes
        assert result.gate_results["bullet_count"] is True
        assert result.gate_results["readability"] is True
        assert result.gate_results["style_errors"] is True
        assert result.gate_results["overall_score"] is True  # Empty feedback gives perfect score
        assert result.metrics["max_bullets_found"] == 0
        
    def test_quality_gate_result_properties(self):
        """Test QualityGateResult properties."""
        result = QualityGateResult(
            status="needs_fix",
            gate_results={
                "gate1": True,
                "gate2": False,
                "gate3": True,
                "gate4": False
            },
            violations=["Violation 1", "Violation 2"],
            recommendations=["Rec 1", "Rec 2"]
        )
        
        assert result.passed_gates == 2
        assert result.total_gates == 4
        assert result.pass_rate == 50.0
        
    def test_review_presentation_with_quality_gates(self):
        """Test review_presentation with quality gates enabled."""
        slides = self.create_test_slides_with_bullets([3, 2, 4])
        
        # Mock successful review
        mock_response_data = [
            {
                "slide_index": 0,
                "severity": "low",
                "category": "content", 
                "message": "Minor improvement needed"
            }
        ]
        
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = json.dumps(mock_response_data)
        self.mock_client.chat.completions.create.return_value = mock_response
        
        # Test with quality gates enabled
        feedback, quality_result = self.reviewer.review_presentation(
            slides, 
            include_quality_gates=True
        )
        
        assert isinstance(feedback, list)
        assert isinstance(quality_result, QualityGateResult)
        assert len(feedback) == 1
        assert quality_result.status in ["pass", "needs_fix"]
        
    def test_review_presentation_backward_compatibility(self):
        """Test that review_presentation maintains backward compatibility."""
        slides = self.create_test_slides_with_bullets([3, 2, 4])
        
        # Mock successful review
        mock_response_data = [
            {
                "slide_index": 0,
                "severity": "low",
                "category": "content",
                "message": "Minor improvement needed"
            }
        ]
        
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = json.dumps(mock_response_data)
        self.mock_client.chat.completions.create.return_value = mock_response
        
        # Test without quality gates (default behavior)
        feedback = self.reviewer.review_presentation(slides)
        
        assert isinstance(feedback, list)
        assert len(feedback) == 1
        
    def test_review_presentation_with_quality_gates_api_failure(self):
        """Test review_presentation with quality gates when API fails."""
        slides = self.create_test_slides_with_bullets([3, 2, 4])
        
        # Mock API failure
        self.mock_client.chat.completions.create.side_effect = Exception("API Error")
        
        feedback, quality_result = self.reviewer.review_presentation(
            slides,
            include_quality_gates=True
        )
        
        assert feedback == []
        assert isinstance(quality_result, QualityGateResult)
        assert quality_result.status == "needs_fix"
        assert "Review failed due to an error" in quality_result.violations