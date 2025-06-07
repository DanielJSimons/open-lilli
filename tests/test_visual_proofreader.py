"""Tests for the Visual Proofreader module."""

import pytest
import json
from unittest.mock import Mock, patch
from typing import List

from open_lilli.visual_proofreader import (
    VisualProofreader,
    DesignIssue,
    DesignIssueType,
    SlidePreview,
    ProofreadingResult
)
from open_lilli.models import SlidePlan, ReviewFeedback


class TestVisualProofreader:
    """Test cases for VisualProofreader class."""
    
    @pytest.fixture
    def mock_openai_client(self):
        """Create a mock OpenAI client."""
        client = Mock()
        return client
    
    @pytest.fixture
    def proofreader(self, mock_openai_client):
        """Create a VisualProofreader instance with mocked client."""
        return VisualProofreader(
            client=mock_openai_client,
            model="gpt-4",
            temperature=0.1
        )
    
    @pytest.fixture
    def sample_slides(self):
        """Create sample slides for testing."""
        return [
            SlidePlan(
                index=0,
                slide_type="title",
                title="BUSINESS Overview",
                bullets=[],
                image_query=None,
                chart_data=None,
                speaker_notes=None,
                layout_id=0
            ),
            SlidePlan(
                index=1,
                slide_type="content",
                title="market ANALYSIS and trends",
                bullets=[
                    "REVENUE growth of 15%",
                    "customer satisfaction IMPROVED",
                    "Market share INCREASED significantly"
                ],
                image_query="business charts",
                chart_data={"type": "bar", "title": "Revenue GROWTH"},
                speaker_notes="Focus on growth metrics",
                layout_id=1
            ),
            SlidePlan(
                index=2,
                slide_type="content",
                title="Next Steps",
                bullets=[
                    "Implement new strategy",
                    "Track performance metrics",
                    "Review quarterly results"
                ],
                image_query=None,
                chart_data=None,
                speaker_notes=None,
                layout_id=1
            )
        ]
    
    @pytest.fixture
    def mock_llm_response(self):
        """Mock LLM response with design issues."""
        return {
            "issues": [
                {
                    "slide_index": 0,
                    "issue_type": "capitalization",
                    "severity": "medium",
                    "element": "title",
                    "original_text": "BUSINESS Overview",
                    "corrected_text": "Business Overview",
                    "description": "Title uses inconsistent capitalization - should be title case",
                    "confidence": 0.9
                },
                {
                    "slide_index": 1,
                    "issue_type": "capitalization",
                    "severity": "high",
                    "element": "title",
                    "original_text": "market ANALYSIS and trends",
                    "corrected_text": "Market Analysis and Trends",
                    "description": "Mixed case in title - unprofessional appearance",
                    "confidence": 0.95
                },
                {
                    "slide_index": 1,
                    "issue_type": "capitalization",
                    "severity": "medium",
                    "element": "bullet",
                    "original_text": "REVENUE growth of 15%",
                    "corrected_text": "Revenue growth of 15%",
                    "description": "Inconsistent capitalization in bullet point",
                    "confidence": 0.85
                }
            ]
        }
    
    def test_create_slide_preview(self, proofreader, sample_slides):
        """Test slide preview creation."""
        slide = sample_slides[1]  # Content slide with various elements
        
        preview = proofreader._create_slide_preview(slide)
        
        assert isinstance(preview, SlidePreview)
        assert preview.slide_index == 1
        assert preview.title == "market ANALYSIS and trends"
        assert len(preview.bullet_points) == 3
        assert preview.image_alt_text == "Image related to: business charts"
        assert preview.chart_description == "Bar chart: Revenue GROWTH"
        assert preview.speaker_notes == "Focus on growth metrics"
    
    def test_create_slide_preview_minimal(self, proofreader):
        """Test slide preview creation with minimal content."""
        slide = SlidePlan(
            index=0,
            slide_type="title",
            title="Simple Title",
            bullets=[],
            layout_id=0
        )
        
        preview = proofreader._create_slide_preview(slide)
        
        assert preview.slide_index == 0
        assert preview.title == "Simple Title"
        assert preview.bullet_points == []
        assert preview.image_alt_text is None
        assert preview.chart_description is None
        assert preview.speaker_notes is None
    
    def test_build_proofreading_prompt(self, proofreader):
        """Test proofreading prompt generation."""
        slide_previews = [
            SlidePreview(
                slide_index=0,
                title="Test TITLE",
                bullet_points=["First bullet", "SECOND bullet"],
                image_alt_text="Test image",
                chart_description="Test chart"
            )
        ]
        
        focus_areas = [DesignIssueType.CAPITALIZATION, DesignIssueType.FORMATTING]
        
        prompt = proofreader._build_proofreading_prompt(
            slide_previews, 
            focus_areas, 
            enable_corrections=True
        )
        
        assert "Test TITLE" in prompt
        assert "SECOND bullet" in prompt
        assert "CAPITALIZATION" in prompt
        assert "FORMATTING" in prompt
        assert "corrected_text" in prompt
        assert "JSON" in prompt.upper()
    
    def test_parse_issues_response(self, proofreader, mock_llm_response):
        """Test parsing of LLM response into DesignIssue objects."""
        issues = proofreader._parse_issues_response(mock_llm_response)
        
        assert len(issues) == 3
        
        first_issue = issues[0]
        assert isinstance(first_issue, DesignIssue)
        assert first_issue.slide_index == 0
        assert first_issue.issue_type == DesignIssueType.CAPITALIZATION
        assert first_issue.severity == "medium"
        assert first_issue.element == "title"
        assert first_issue.original_text == "BUSINESS Overview"
        assert first_issue.corrected_text == "Business Overview"
        assert first_issue.confidence == 0.9
    
    def test_parse_issues_response_alternative_formats(self, proofreader):
        """Test parsing different response formats."""
        # Test direct list format
        response1 = [
            {
                "slide_index": 0,
                "issue_type": "capitalization",
                "severity": "medium",
                "element": "title",
                "original_text": "Test",
                "description": "Test issue",
                "confidence": 0.8
            }
        ]
        
        issues1 = proofreader._parse_issues_response(response1)
        assert len(issues1) == 1
        
        # Test single key format
        response2 = {
            "detected_issues": [
                {
                    "slide_index": 1,
                    "issue_type": "formatting",
                    "severity": "low",
                    "element": "bullet",
                    "original_text": "Test bullet",
                    "description": "Test formatting issue",
                    "confidence": 0.7
                }
            ]
        }
        
        issues2 = proofreader._parse_issues_response(response2)
        assert len(issues2) == 1
    
    def test_convert_to_review_feedback(self, proofreader):
        """Test conversion of DesignIssue to ReviewFeedback."""
        issues = [
            DesignIssue(
                slide_index=1,
                issue_type=DesignIssueType.CAPITALIZATION,
                severity="medium",
                element="title",
                original_text="test TITLE",
                corrected_text="Test Title",
                description="Inconsistent capitalization",
                confidence=0.9
            )
        ]
        
        feedback_list = proofreader.convert_to_review_feedback(issues)
        
        assert len(feedback_list) == 1
        feedback = feedback_list[0]
        assert isinstance(feedback, ReviewFeedback)
        assert feedback.slide_index == 1
        assert feedback.severity == "medium"
        assert feedback.category == "design"
        assert "Inconsistent capitalization" in feedback.message
        assert "test TITLE" in feedback.suggestion
        assert "Test Title" in feedback.suggestion
    
    @patch('time.time')
    def test_proofread_slides_success(self, mock_time, proofreader, sample_slides, mock_llm_response):
        """Test successful slide proofreading."""
        # Mock time for processing duration
        mock_time.side_effect = [0.0, 2.5]  # start_time, end_time
        
        # Mock LLM call
        proofreader._call_llm_with_retries = Mock(return_value=mock_llm_response)
        
        result = proofreader.proofread_slides(
            sample_slides,
            focus_areas=[DesignIssueType.CAPITALIZATION],
            enable_corrections=True
        )
        
        assert isinstance(result, ProofreadingResult)
        assert result.total_slides == 3
        assert len(result.issues_found) == 3
        assert result.processing_time_seconds == 2.5
        assert result.model_used == "gpt-4"
        
        # Check issue count by type
        issue_counts = result.issue_count_by_type
        assert issue_counts["capitalization"] == 3
        
        # Check high confidence issues
        high_conf_issues = result.high_confidence_issues
        assert len(high_conf_issues) == 3  # All have confidence >= 0.8
    
    def test_proofread_single_slide(self, proofreader, sample_slides, mock_llm_response):
        """Test proofreading of a single slide."""
        slide = sample_slides[1]  # Slide with issues
        
        # Mock LLM call to return issues for this slide
        single_slide_response = {
            "issues": [issue for issue in mock_llm_response["issues"] if issue["slide_index"] == 1]
        }
        proofreader._call_llm_with_retries = Mock(return_value=single_slide_response)
        
        issues = proofreader.proofread_single_slide(
            slide,
            focus_areas=[DesignIssueType.CAPITALIZATION]
        )
        
        assert len(issues) == 2  # Title and bullet issue for slide 1
        assert all(issue.slide_index == 1 for issue in issues)
        assert all(issue.issue_type == DesignIssueType.CAPITALIZATION for issue in issues)
    
    def test_generate_test_slides_with_errors(self, proofreader, sample_slides):
        """Test generation of test slides with seeded errors."""
        base_slides = [sample_slides[2]]  # Use clean slide
        
        modified_slides, seeded_errors = proofreader.generate_test_slides_with_errors(
            base_slides,
            error_types=[DesignIssueType.CAPITALIZATION],
            error_count=2
        )
        
        assert len(modified_slides) == 1
        assert len(seeded_errors) <= 2  # May be fewer if not enough content to modify
        
        # Verify that slides were modified
        for error in seeded_errors:
            assert "slide_index" in error
            assert "element" in error
            assert "original" in error
            assert "modified" in error
            assert error["error_type"] == "capitalization"
    
    def test_test_capitalization_detection(self, proofreader, sample_slides):
        """Test the capitalization detection accuracy testing."""
        # Create seeded errors
        seeded_errors = [
            {"slide_index": 0, "element": "title"},
            {"slide_index": 1, "element": "title"},
            {"slide_index": 1, "element": "bullet"}
        ]
        
        # Mock proofreading result
        mock_result = ProofreadingResult(
            total_slides=3,
            issues_found=[
                DesignIssue(
                    slide_index=0,
                    issue_type=DesignIssueType.CAPITALIZATION,
                    severity="medium",
                    element="title",
                    original_text="Test",
                    description="Test",
                    confidence=0.9
                ),
                DesignIssue(
                    slide_index=1,
                    issue_type=DesignIssueType.CAPITALIZATION,
                    severity="high",
                    element="title",
                    original_text="Test",
                    description="Test",
                    confidence=0.95
                ),
                # Missing bullet detection (false negative)
            ],
            processing_time_seconds=1.0,
            model_used="gpt-4"
        )
        
        proofreader.proofread_slides = Mock(return_value=mock_result)
        
        metrics = proofreader.test_capitalization_detection(sample_slides, seeded_errors)
        
        assert "detection_rate" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics
        assert "true_positives" in metrics
        assert "false_positives" in metrics
        assert "false_negatives" in metrics
        
        # Should detect 2 out of 3 seeded errors
        assert metrics["true_positives"] == 2
        assert metrics["false_negatives"] == 1
        assert metrics["detection_rate"] == 2/3  # 66.7%
    
    def test_call_llm_with_retries_success(self, proofreader, mock_openai_client):
        """Test successful LLM API call."""
        # Mock successful response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = '{"issues": []}'
        
        mock_openai_client.chat.completions.create.return_value = mock_response
        
        result = proofreader._call_llm_with_retries("test prompt")
        
        assert result == {"issues": []}
        mock_openai_client.chat.completions.create.assert_called_once()
    
    def test_call_llm_with_retries_json_error(self, proofreader, mock_openai_client):
        """Test LLM API call with JSON parsing error."""
        # Mock response with invalid JSON
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = 'invalid json'
        
        mock_openai_client.chat.completions.create.return_value = mock_response
        
        with pytest.raises(ValueError, match="Invalid JSON response"):
            proofreader._call_llm_with_retries("test prompt")
    
    @patch('time.sleep')
    def test_call_llm_with_retries_rate_limit(self, mock_sleep, proofreader, mock_openai_client):
        """Test LLM API call with rate limiting."""
        from openai import RateLimitError
        
        # Mock rate limit error then success
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = '{"issues": []}'
        
        mock_openai_client.chat.completions.create.side_effect = [
            RateLimitError("Rate limited", response=Mock(), body=None),
            mock_response
        ]
        
        result = proofreader._call_llm_with_retries("test prompt")
        
        assert result == {"issues": []}
        assert mock_openai_client.chat.completions.create.call_count == 2
        mock_sleep.assert_called_once_with(1)  # 2^0 = 1 second wait
    
    def test_error_handling_in_proofread_slides(self, proofreader, sample_slides):
        """Test error handling in main proofreading method."""
        # Mock LLM call to raise exception
        proofreader._call_llm_with_retries = Mock(side_effect=Exception("API Error"))
        
        result = proofreader.proofread_slides(sample_slides)
        
        # Should return empty result on error
        assert isinstance(result, ProofreadingResult)
        assert result.total_slides == 3
        assert len(result.issues_found) == 0
    
    def test_design_issue_validation(self):
        """Test DesignIssue model validation."""
        # Valid issue
        issue = DesignIssue(
            slide_index=1,
            issue_type=DesignIssueType.CAPITALIZATION,
            severity="medium",
            element="title",
            original_text="Test TEXT",
            corrected_text="Test Text",
            description="Inconsistent capitalization",
            confidence=0.9
        )
        
        assert issue.slide_index == 1
        assert issue.issue_type == DesignIssueType.CAPITALIZATION
        assert issue.confidence == 0.9
        
        # Test confidence bounds
        with pytest.raises(ValueError):
            DesignIssue(
                slide_index=1,
                issue_type=DesignIssueType.CAPITALIZATION,
                severity="medium",
                element="title",
                original_text="Test",
                description="Test",
                confidence=1.5  # Invalid: > 1.0
            )
    
    def test_proofreading_result_properties(self):
        """Test ProofreadingResult computed properties."""
        issues = [
            DesignIssue(
                slide_index=0,
                issue_type=DesignIssueType.CAPITALIZATION,
                severity="medium",
                element="title",
                original_text="Test",
                description="Test",
                confidence=0.9
            ),
            DesignIssue(
                slide_index=1,
                issue_type=DesignIssueType.FORMATTING,
                severity="low",
                element="bullet",
                original_text="Test",
                description="Test",
                confidence=0.7
            ),
            DesignIssue(
                slide_index=1,
                issue_type=DesignIssueType.CAPITALIZATION,
                severity="high",
                element="title",
                original_text="Test",
                description="Test",
                confidence=0.85
            )
        ]
        
        result = ProofreadingResult(
            total_slides=2,
            issues_found=issues,
            processing_time_seconds=3.0,
            model_used="gpt-4"
        )
        
        # Test issue count by type
        counts = result.issue_count_by_type
        assert counts["capitalization"] == 2
        assert counts["formatting"] == 1
        
        # Test high confidence issues (>= 0.8)
        high_conf = result.high_confidence_issues
        assert len(high_conf) == 2  # 0.9 and 0.85
        assert all(issue.confidence >= 0.8 for issue in high_conf)


class TestCapitalizationDetection:
    """Specific tests for capitalization error detection."""
    
    @pytest.fixture
    def slides_with_cap_errors(self):
        """Create slides with various capitalization errors."""
        return [
            SlidePlan(
                index=0,
                slide_type="title",
                title="QUARTERLY RESULTS",  # All caps
                bullets=[],
                layout_id=0
            ),
            SlidePlan(
                index=1,
                slide_type="content",
                title="market ANALYSIS and trends",  # Mixed case
                bullets=[
                    "REVENUE increased by 15%",  # All caps start
                    "customer satisfaction improved",  # All lowercase
                    "Market Share GREW significantly",  # Random caps
                ],
                layout_id=1
            ),
            SlidePlan(
                index=2,
                slide_type="content", 
                title="Next Steps",  # Correct
                bullets=[
                    "Implement new strategy",  # Correct
                    "Track performance metrics",  # Correct
                ],
                layout_id=1
            )
        ]
    
    def test_capitalization_error_seeding(self):
        """Test the seeding of capitalization errors."""
        base_slides = [
            SlidePlan(
                index=0,
                slide_type="content",
                title="Clean Title",
                bullets=["Clean bullet one", "Clean bullet two"],
                layout_id=1
            )
        ]
        
        proofreader = VisualProofreader(Mock(), "gpt-4")
        
        modified_slides, seeded_errors = proofreader.generate_test_slides_with_errors(
            base_slides,
            error_types=[DesignIssueType.CAPITALIZATION],
            error_count=3
        )
        
        # Should have created some errors
        assert len(seeded_errors) > 0
        
        # All errors should be capitalization type
        for error in seeded_errors:
            assert error["error_type"] == "capitalization"
            assert error["original"] != error["modified"]
    
    def test_90_percent_detection_rate_simulation(self):
        """Simulate the 90% detection rate mentioned in T-79."""
        # Create a proofreader with mocked LLM
        client = Mock()
        proofreader = VisualProofreader(client, "gpt-4")
        
        # Create 10 seeded errors
        seeded_errors = [
            {"slide_index": i // 2, "element": "title" if i % 2 == 0 else "bullet"}
            for i in range(10)
        ]
        
        # Mock detection of 9 out of 10 errors (90%)
        detected_issues = []
        for i, error in enumerate(seeded_errors):
            if i < 9:  # Detect first 9, miss the last one
                detected_issues.append(
                    DesignIssue(
                        slide_index=error["slide_index"],
                        issue_type=DesignIssueType.CAPITALIZATION,
                        severity="medium",
                        element=error["element"],
                        original_text=f"Error {i}",
                        description=f"Capitalization issue {i}",
                        confidence=0.85
                    )
                )
        
        mock_result = ProofreadingResult(
            total_slides=5,
            issues_found=detected_issues,
            processing_time_seconds=2.0,
            model_used="gpt-4"
        )
        
        proofreader.proofread_slides = Mock(return_value=mock_result)
        
        # Test the detection
        slides = [SlidePlan(index=i, slide_type="content", title=f"Slide {i}", bullets=[], layout_id=1) for i in range(5)]
        metrics = proofreader.test_capitalization_detection(slides, seeded_errors)
        
        # Should achieve 90% detection rate
        assert metrics["detection_rate"] == 0.9
        assert metrics["true_positives"] == 9
        assert metrics["false_negatives"] == 1
        assert metrics["total_seeded_errors"] == 10


if __name__ == "__main__":
    pytest.main([__file__])