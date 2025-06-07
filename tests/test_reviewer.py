"""Tests for reviewer."""

import json
from unittest.mock import Mock

import pytest
from openai import OpenAI

from open_lilli.models import (
    QualityGateResult,
    QualityGates,
    ReviewFeedback,
    SlidePlan,
    TemplateStyle,
    FontInfo,
    PlaceholderStyleInfo
)
from open_lilli.reviewer import Reviewer, calculate_readability_score, count_syllables, calculate_contrast_ratio


# --- Tests for calculate_contrast_ratio ---

def test_calculate_contrast_ratio():
    """Test the calculate_contrast_ratio function with various color pairs."""
    # Black and White (Max contrast)
    assert calculate_contrast_ratio("#000000", "#FFFFFF") == pytest.approx(21.0)
    assert calculate_contrast_ratio("000000", "FFFFFF") == pytest.approx(21.0)

    # Red and White
    assert calculate_contrast_ratio("#FF0000", "#FFFFFF") == pytest.approx(3.998, abs=0.001)

    # Blue and White
    assert calculate_contrast_ratio("#0000FF", "#FFFFFF") == pytest.approx(8.592, abs=0.001)

    # Green and Black (Note: WCAG formula for Green #00FF00 vs Black #000000 is (0.7208 + 0.05) / (0 + 0.05) = 15.416)
    # The L values are: L_green = 0.2126 * ((0/255)/12.92) + 0.7152 * ((255/255)/1.0) + 0.0722 * ((0/255)/12.92) = 0.7152
    # My previous implementation used a slightly different calculation for R,G,B components of luminance.
    # Let's recalculate based on the implementation:
    # For #00FF00 (Green): r=0, g=1, b=0. R_lum=0, G_lum=1, B_lum=0. Lum_green = 0.7152
    # For #000000 (Black): r=0, g=0, b=0. R_lum=0, G_lum=0, B_lum=0. Lum_black = 0
    # Ratio = (0.7152 + 0.05) / (0 + 0.05) = 0.7652 / 0.05 = 15.304
    assert calculate_contrast_ratio("#00FF00", "#000000") == pytest.approx(15.304, abs=0.001)

    # Grey and Dark Grey (Example: #767676 on #242424) - L1=0.200, L2=0.022. Ratio=(0.200+0.05)/(0.022+0.05) = 3.47
    # For #767676 (Grey): R,G,B = 118. sRGB=0.4627. Lum_channel = ((0.4627+0.055)/1.055)**2.4 = 0.1786
    # Lum_grey = 0.2126*0.1786 + 0.7152*0.1786 + 0.0722*0.1786 = 0.1786
    # For #242424 (Dark Grey): R,G,B = 36. sRGB=0.1412. Lum_channel = ((0.1412+0.055)/1.055)**2.4 = 0.0245
    # Lum_dark_grey = 0.0245
    # Ratio = (0.1786 + 0.05) / (0.0245 + 0.05) = 0.2286 / 0.0745 = 3.068
    assert calculate_contrast_ratio("#767676", "#242424") == pytest.approx(3.068, abs=0.001)

    # Identical colors
    assert calculate_contrast_ratio("#1A2B3C", "#1A2B3C") == pytest.approx(1.0)
    assert calculate_contrast_ratio("1A2B3C", "1a2b3c") == pytest.approx(1.0) # Case insensitivity for hex

    # Test with known failing examples if any handy, or ensure logic is sound for thresholds
    # Light Gray on White (should be low)
    assert calculate_contrast_ratio("#D3D3D3", "#FFFFFF") == pytest.approx(1.63, abs=0.01) # L_D3D3D3 = 0.601, L_FFFFFF=1. Ratio = (1+0.05)/(0.601+0.05) = 1.613

    # Test with known passing examples for AA
    # Dark Slate Gray (#2F4F4F) on Light Gray (#D3D3D3)
    # L_2F4F4F (R=47,G=79,B=79): sR=0.184, sG=0.310, sB=0.310 -> Rlum=0.029, Glum=0.072, Blum=0.072 -> L=0.066
    # L_D3D3D3 (R=211,G=211,B=211): sRGB=0.827 -> L=0.629
    # Ratio = (0.629+0.05)/(0.066+0.05) = 0.679 / 0.116 = 5.85
    assert calculate_contrast_ratio("#2F4F4F", "#D3D3D3") == pytest.approx(5.85, abs=0.01)


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

    @pytest.mark.asyncio
    async def test_review_presentation_async_success(self):
        """Test async review_presentation."""
        from unittest.mock import AsyncMock

        async_client = AsyncMock()
        reviewer = Reviewer(async_client)
        slides = self.create_test_slides()

        mock_response_data = {"feedback": [{"slide_index": 0, "severity": "low", "category": "content", "message": "ok", "suggestion": "none"}]}
        mock_resp = Mock()
        mock_resp.choices = [Mock()]
        mock_resp.choices[0].message.content = json.dumps(mock_response_data)
        async_client.chat.completions.create.return_value = mock_resp

        feedback = await reviewer.review_presentation_async(slides)
        assert len(feedback) == 1
        async_client.chat.completions.create.assert_called_once()

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

    def test_evaluate_quality_gates_contrast_check_pass(self):
        """Test contrast check passes with good contrast."""
        mock_client = Mock(spec=OpenAI)
        template_style = TemplateStyle(
            master_font=FontInfo(name="Arial", size=12, color="#000000"),
            theme_colors={'lt1': '#FFFFFF', 'dk1': '#000000'},
            placeholder_styles={
                2: PlaceholderStyleInfo( # Body placeholder type
                    placeholder_type=2,
                    type_name="BODY",
                    default_font=FontInfo(name="Arial", size=12, color="#000000"), # Black text
                    fill_color="#FFFFFF", # White background
                    bullet_styles=[]
                )
            }
        )
        reviewer = Reviewer(client=mock_client, template_style=template_style)
        slides = [SlidePlan(index=0, slide_type="content", title="Test Slide Pass")]
        gates = QualityGates() # Default min_contrast_ratio = 4.5

        result = reviewer.evaluate_quality_gates(slides, [], gates)

        assert result.gate_results.get("contrast_check") is True
        assert result.metrics.get("min_contrast_ratio_found") == pytest.approx(21.0)
        assert not any("Poor contrast ratio" in v for v in result.violations)

    def test_evaluate_quality_gates_contrast_check_fail(self):
        """Test contrast check fails with poor contrast."""
        mock_client = Mock(spec=OpenAI)
        # Gray #808080 on White #FFFFFF. Luminance for #808080 is approx 0.221. Ratio = (1+0.05)/(0.221+0.05) = 1.05/0.271 = 3.87
        template_style = TemplateStyle(
            master_font=FontInfo(name="Arial", size=12, color="#808080"),
            theme_colors={'lt1': '#FFFFFF', 'dk1': '#000000'},
            placeholder_styles={
                2: PlaceholderStyleInfo(
                    placeholder_type=2,
                    type_name="BODY",
                    default_font=FontInfo(name="Arial", size=12, color="#808080"), # Gray text
                    fill_color="#FFFFFF", # White background
                    bullet_styles=[]
                )
            }
        )
        reviewer = Reviewer(client=mock_client, template_style=template_style)
        slides = [SlidePlan(index=0, slide_type="content", title="Test Slide Fail")]
        gates = QualityGates()

        result = reviewer.evaluate_quality_gates(slides, [], gates)

        assert result.gate_results.get("contrast_check") is False
        assert result.metrics.get("min_contrast_ratio_found") == pytest.approx(3.87, abs=0.01)
        assert any("Slide 1 (Body Placeholder): Poor contrast ratio 3.87" in v for v in result.violations)

    def test_evaluate_quality_gates_contrast_check_fallback_logic(self):
        """Test contrast check with fallback color logic."""
        mock_client = Mock(spec=OpenAI)

        # Scenario A: Text from master_font, BG from theme_colors['lt1']
        # Blue text (#0000FF) on White BG (#FFFFFF) -> Ratio ~8.59
        ts_fallback_master = TemplateStyle(
            master_font=FontInfo(name="Arial", size=12, color="#0000FF"),
            theme_colors={'lt1': '#FFFFFF', 'dk1': '#000000'},
            placeholder_styles={
                2: PlaceholderStyleInfo(placeholder_type=2, type_name="BODY", default_font=None, fill_color=None, bullet_styles=[])
            }
        )
        reviewer_a = Reviewer(client=mock_client, template_style=ts_fallback_master)
        slides_a = [SlidePlan(index=0, slide_type="content", title="Fallback Test A")]
        gates = QualityGates()
        result_a = reviewer_a.evaluate_quality_gates(slides_a, [], gates)

        assert result_a.gate_results.get("contrast_check") is True, "Scenario A failed"
        assert result_a.metrics.get("min_contrast_ratio_found") == pytest.approx(8.59, abs=0.01), "Scenario A metrics failed"

        # Scenario B: Text from default (#000000), BG from placeholder_style.fill_color
        # Black text (#000000) on Green BG (#00FF00) -> Ratio ~15.3
        ts_fallback_fill = TemplateStyle(
            master_font=None, # No master font color
            theme_colors={'lt1': '#CCCCCC', 'dk1': '#333333'}, # Theme colors not used for BG in this case
            placeholder_styles={
                2: PlaceholderStyleInfo(
                    placeholder_type=2,
                    type_name="BODY",
                    default_font=None, # No placeholder font
                    fill_color="#00FF00", # Green BG
                    bullet_styles=[])
            }
        )
        reviewer_b = Reviewer(client=mock_client, template_style=ts_fallback_fill)
        slides_b = [SlidePlan(index=0, slide_type="content", title="Fallback Test B")]
        result_b = reviewer_b.evaluate_quality_gates(slides_b, [], gates)

        assert result_b.gate_results.get("contrast_check") is True, "Scenario B failed"
        assert result_b.metrics.get("min_contrast_ratio_found") == pytest.approx(15.304, abs=0.001), "Scenario B metrics failed"

    def test_evaluate_quality_gates_no_template_style(self):
        """Test contrast check when template_style is None."""
        mock_client = Mock(spec=OpenAI)
        reviewer = Reviewer(client=mock_client, template_style=None) # No template style
        slides = [SlidePlan(index=0, slide_type="content", title="No Template Style Test")]
        gates = QualityGates()

        result = reviewer.evaluate_quality_gates(slides, [], gates)

        assert result.gate_results.get("contrast_check") is False
        assert any("Contrast check could not be performed: Template style information is missing." in v for v in result.violations)
        assert result.metrics.get("min_contrast_ratio_found") == 0.0 # Default if no ratios calculated