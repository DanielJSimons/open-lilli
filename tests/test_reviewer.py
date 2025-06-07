"""Tests for reviewer."""

import json
from unittest.mock import Mock

import pytest
from openai import OpenAI

from open_lilli.models import ReviewFeedback, SlidePlan
from open_lilli.reviewer import Reviewer


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