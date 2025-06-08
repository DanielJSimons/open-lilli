"""Tests for Flow Intelligence module (T-80)."""

import pytest
from unittest.mock import Mock, patch
from typing import List

from open_lilli.flow_intelligence import (
    FlowIntelligence,
    TransitionSuggestion,
    TransitionType,
    FlowAnalysisResult
)
from open_lilli.models import SlidePlan, ReviewFeedback


class TestFlowIntelligence:
    """Test cases for FlowIntelligence class."""
    
    @pytest.fixture
    def mock_openai_client(self):
        """Create a mock OpenAI client."""
        client = Mock()
        return client
    
    @pytest.fixture
    def flow_intelligence(self, mock_openai_client):
        """Create a FlowIntelligence instance with mocked client."""
        return FlowIntelligence(
            client=mock_openai_client,
            model="gpt-4.1",
            temperature=0.3
        )
    
    @pytest.fixture
    def sample_slides(self):
        """Create sample slides for testing."""
        return [
            SlidePlan(
                index=0,
                slide_type="title",
                title="Business Strategy Overview",
                bullets=[],
                speaker_notes="Welcome to our strategy presentation",
                layout_id=0
            ),
            SlidePlan(
                index=1,
                slide_type="content",
                title="Market Analysis",
                bullets=[
                    "Current market size is $10B",
                    "Growth rate of 15% annually",
                    "Key competitors identified"
                ],
                speaker_notes="Discuss market conditions",
                layout_id=1
            ),
            SlidePlan(
                index=2,
                slide_type="content",
                title="Financial Projections",
                bullets=[
                    "Revenue targets for next 3 years",
                    "Cost optimization opportunities",
                    "ROI expectations"
                ],
                speaker_notes="Present financial outlook",
                layout_id=1
            ),
            SlidePlan(
                index=3,
                slide_type="content",
                title="Implementation Plan",
                bullets=[
                    "Phase 1: Foundation building",
                    "Phase 2: Market expansion",
                    "Phase 3: Optimization"
                ],
                speaker_notes="Outline execution strategy",
                layout_id=1
            )
        ]
    
    @pytest.fixture
    def mock_transition_response(self):
        """Mock LLM response for transition generation."""
        return {
            "linking_sentence": "With this market foundation established, let's examine how these insights translate into our financial projections.",
            "transition_type": "bridge",
            "context_summary": "Moving from market analysis to financial implications",
            "confidence": 0.9,
            "reasoning": "Natural progression from market data to financial planning"
        }
    
    @pytest.fixture
    def mock_coherence_response(self):
        """Mock LLM response for coherence analysis."""
        return {
            "coherence_score": 4.2,
            "narrative_gaps": [
                "Missing connection between market analysis and implementation timeline"
            ],
            "flow_strengths": [
                "Clear progression from strategy to execution",
                "Well-structured three-part narrative"
            ],
            "improvement_areas": [
                "Add transitional slide between projections and implementation",
                "Strengthen conclusion linkage to opening strategy"
            ]
        }
    
    def test_create_presentation_context(self, flow_intelligence, sample_slides):
        """Test presentation context creation."""
        context = flow_intelligence._create_presentation_context(sample_slides)
        
        assert "Total Slides: 4" in context
        assert "SLIDE SEQUENCE:" in context
        assert "Business Strategy Overview" in context
        assert "Market Analysis" in context
        assert "Financial Projections" in context
        assert "Implementation Plan" in context
    
    def test_generate_single_transition(self, flow_intelligence, sample_slides, mock_transition_response):
        """Test single transition generation."""
        flow_intelligence._call_llm_with_retries = Mock(return_value=mock_transition_response)
        
        current_slide = sample_slides[1]  # Market Analysis
        next_slide = sample_slides[2]     # Financial Projections
        context = "Test presentation context"
        
        transition = flow_intelligence._generate_single_transition(
            current_slide, next_slide, context, slide_position=2
        )
        
        assert transition is not None
        assert transition.from_slide_index == 1
        assert transition.to_slide_index == 2
        assert transition.transition_type == TransitionType.BRIDGE
        assert "market foundation established" in transition.linking_sentence
        assert transition.confidence == 0.9
        assert transition.insertion_location == "speaker_notes"
    
    def test_generate_transition_suggestions(self, flow_intelligence, sample_slides, mock_transition_response):
        """Test transition generation for all slide pairs."""
        flow_intelligence._call_llm_with_retries = Mock(return_value=mock_transition_response)
        
        transitions = flow_intelligence._generate_transition_suggestions(sample_slides)
        
        # Should generate N-1 transitions for N slides
        assert len(transitions) == 3  # 4 slides = 3 transitions
        
        # Check all transition pairs
        expected_pairs = [(0, 1), (1, 2), (2, 3)]
        actual_pairs = [(t.from_slide_index, t.to_slide_index) for t in transitions]
        assert actual_pairs == expected_pairs
    
    def test_analyze_flow_coherence(self, flow_intelligence, sample_slides, mock_coherence_response):
        """Test flow coherence analysis."""
        flow_intelligence._call_llm_with_retries = Mock(return_value=mock_coherence_response)
        
        transitions = []  # Empty transitions for test
        coherence_score, narrative_gaps = flow_intelligence._analyze_flow_coherence(
            sample_slides, transitions
        )
        
        assert coherence_score == 4.2
        assert len(narrative_gaps) == 1
        assert "Missing connection between market analysis" in narrative_gaps[0]
    
    def test_identify_coherence_issues(self, flow_intelligence, sample_slides):
        """Test identification of specific coherence issues."""
        transitions = []  # No transitions provided
        
        issues = flow_intelligence._identify_coherence_issues(sample_slides, transitions)
        
        # Should identify missing transitions
        assert len(issues) == 3  # 3 missing transitions
        
        for issue in issues:
            assert isinstance(issue, ReviewFeedback)
            assert issue.category == "flow"
            assert "Missing transition" in issue.message
    
    def test_insert_transitions_to_slides(self, flow_intelligence, sample_slides):
        """Test insertion of transitions into slide speaker notes."""
        transitions = [
            TransitionSuggestion(
                from_slide_index=1,
                to_slide_index=2,
                transition_type=TransitionType.BRIDGE,
                linking_sentence="This brings us to our financial analysis.",
                context_summary="Market to finance transition",
                confidence=0.9,
                insertion_location="speaker_notes"
            )
        ]
        
        # Make a copy to avoid modifying the fixture
        test_slides = [
            SlidePlan(
                index=slide.index,
                slide_type=slide.slide_type,
                title=slide.title,
                bullets=slide.bullets,
                speaker_notes=slide.speaker_notes,
                layout_id=slide.layout_id
            )
            for slide in sample_slides
        ]
        
        flow_intelligence._insert_transitions_to_slides(test_slides, transitions)
        
        # Check that transition was added to slide 1
        slide_1 = test_slides[1]
        assert "[TRANSITION]" in slide_1.speaker_notes
        assert "This brings us to our financial analysis" in slide_1.speaker_notes
    
    @patch('time.time')
    def test_analyze_and_enhance_flow(self, mock_time, flow_intelligence, sample_slides, 
                                     mock_transition_response, mock_coherence_response):
        """Test complete flow analysis and enhancement."""
        # Mock time for processing duration
        mock_time.side_effect = [0.0, 3.5]  # start_time, end_time
        
        # Mock LLM calls
        flow_intelligence._call_llm_with_retries = Mock(
            side_effect=[
                mock_transition_response,  # Transition 1
                mock_transition_response,  # Transition 2
                mock_transition_response,  # Transition 3
                mock_coherence_response    # Coherence analysis
            ]
        )
        
        result = flow_intelligence.analyze_and_enhance_flow(
            sample_slides,
            target_coherence=4.0,
            insert_transitions=True
        )
        
        assert isinstance(result, FlowAnalysisResult)
        assert result.total_slides == 4
        assert len(result.transitions_generated) == 3
        assert result.flow_score == 4.2
        assert result.processing_time_seconds == 3.5
        assert result.meets_transition_requirement  # Should have N-1 transitions
        assert result.transition_coverage == 1.0   # 100% coverage
    
    def test_validate_transition_requirements(self, flow_intelligence, sample_slides):
        """Test validation of T-80 requirements."""
        
        # Create result with sufficient transitions and good coherence
        result = FlowAnalysisResult(
            total_slides=4,
            transitions_generated=[
                TransitionSuggestion(0, 1, TransitionType.BRIDGE, "Test 1", "Context", 0.9, "speaker_notes"),
                TransitionSuggestion(1, 2, TransitionType.SEQUENCE, "Test 2", "Context", 0.8, "speaker_notes"),
                TransitionSuggestion(2, 3, TransitionType.SUMMARY, "Test 3", "Context", 0.85, "speaker_notes")
            ],
            flow_score=4.3,
            narrative_gaps=[],
            coherence_issues=[],
            processing_time_seconds=2.0
        )
        
        # Mock slides with transitions in speaker notes
        test_slides = sample_slides.copy()
        for slide in test_slides[:-1]:  # All but last slide
            slide.speaker_notes += "\n[TRANSITION] Test transition"
        
        validation = flow_intelligence.validate_transition_requirements(test_slides, result)
        
        assert validation["sufficient_transitions"] is True
        assert validation["coherence_target"] is True  # 4.3 > 4.0
        assert validation["transitions_inserted"] is True
    
    def test_generate_flow_report(self, flow_intelligence, sample_slides):
        """Test comprehensive flow report generation."""
        
        # Create result with good metrics
        result = FlowAnalysisResult(
            total_slides=4,
            transitions_generated=[
                TransitionSuggestion(0, 1, TransitionType.BRIDGE, "Test transition", "Context", 0.9, "speaker_notes")
            ],
            flow_score=4.2,
            narrative_gaps=["Test gap"],
            coherence_issues=[
                ReviewFeedback(slide_index=1, severity="low", category="flow", 
                              message="Test issue", suggestion="Test suggestion")
            ],
            processing_time_seconds=2.5
        )
        
        report = flow_intelligence.generate_flow_report(sample_slides, result)
        
        assert "summary" in report
        assert "t80_compliance" in report
        assert "transitions" in report
        assert "narrative_gaps" in report
        assert "coherence_issues" in report
        
        # Check summary
        assert report["summary"]["total_slides"] == 4
        assert report["summary"]["flow_score"] == 4.2
        
        # Check transitions
        assert len(report["transitions"]) == 1
        assert report["transitions"][0]["from_slide"] == 1  # 1-based indexing
        assert report["transitions"][0]["to_slide"] == 2
    
    def test_flow_analysis_result_properties(self):
        """Test FlowAnalysisResult computed properties."""
        
        result = FlowAnalysisResult(
            total_slides=5,
            transitions_generated=[
                TransitionSuggestion(0, 1, TransitionType.BRIDGE, "Test 1", "Context", 0.9, "speaker_notes"),
                TransitionSuggestion(1, 2, TransitionType.SEQUENCE, "Test 2", "Context", 0.8, "speaker_notes"),
                TransitionSuggestion(2, 3, TransitionType.SUMMARY, "Test 3", "Context", 0.85, "speaker_notes")
            ],
            flow_score=4.1,
            narrative_gaps=[],
            coherence_issues=[],
            processing_time_seconds=2.0
        )
        
        # Test transition coverage: 3 transitions for 5 slides = 3/4 = 75%
        assert result.transition_coverage == 0.75
        
        # Test meets requirement: needs 4 transitions (N-1), has 3 = False
        assert result.meets_transition_requirement is False
        
        # Add one more transition to meet requirement
        result.transitions_generated.append(
            TransitionSuggestion(3, 4, TransitionType.CONCLUSION, "Test 4", "Context", 0.9, "speaker_notes")
        )
        
        assert result.transition_coverage == 1.0  # 4/4 = 100%
        assert result.meets_transition_requirement is True  # 4 >= 4
    
    def test_transition_suggestion_dataclass(self):
        """Test TransitionSuggestion dataclass functionality."""
        
        transition = TransitionSuggestion(
            from_slide_index=1,
            to_slide_index=2,
            transition_type=TransitionType.CAUSE_EFFECT,
            linking_sentence="As a result of these findings...",
            context_summary="Cause-effect relationship",
            confidence=0.85,
            insertion_location="speaker_notes"
        )
        
        assert transition.from_slide_index == 1
        assert transition.to_slide_index == 2
        assert transition.transition_type == TransitionType.CAUSE_EFFECT
        assert "As a result" in transition.linking_sentence
        assert transition.confidence == 0.85
    
    def test_error_handling_in_transition_generation(self, flow_intelligence, sample_slides):
        """Test error handling in transition generation."""
        
        # Mock LLM call to raise exception
        flow_intelligence._call_llm_with_retries = Mock(side_effect=Exception("API Error"))
        
        transitions = flow_intelligence._generate_transition_suggestions(sample_slides)
        
        # Should return empty list on error
        assert len(transitions) == 0
    
    def test_empty_slides_handling(self, flow_intelligence):
        """Test handling of empty slide lists."""
        
        empty_slides = []
        transitions = flow_intelligence._generate_transition_suggestions(empty_slides)
        assert len(transitions) == 0
        
        single_slide = [SlidePlan(index=0, slide_type="title", title="Only Slide", bullets=[], layout_id=0)]
        transitions = flow_intelligence._generate_transition_suggestions(single_slide)
        assert len(transitions) == 0
    
    def test_detailed_flow_summary_creation(self, flow_intelligence, sample_slides):
        """Test creation of detailed flow summary."""
        
        transitions = [
            TransitionSuggestion(0, 1, TransitionType.BRIDGE, "Moving to analysis...", "Context", 0.9, "speaker_notes"),
            TransitionSuggestion(1, 2, TransitionType.SEQUENCE, "Next, we examine...", "Context", 0.8, "speaker_notes")
        ]
        
        summary = flow_intelligence._create_detailed_flow_summary(sample_slides, transitions)
        
        assert "PRESENTATION FLOW ANALYSIS:" in summary
        assert "Total slides: 4" in summary
        assert "Generated transitions: 2" in summary
        assert "SLIDE 1: Business Strategy Overview" in summary
        assert "→ Transition: Moving to analysis..." in summary
        assert "SLIDE 2: Market Analysis" in summary
        assert "→ Transition: Next, we examine..." in summary


class TestTransitionTypes:
    """Test TransitionType enum functionality."""
    
    def test_transition_type_values(self):
        """Test that all transition types have correct values."""
        
        assert TransitionType.SEQUENCE == "sequence"
        assert TransitionType.CAUSE_EFFECT == "cause_effect"
        assert TransitionType.CONTRAST == "contrast"
        assert TransitionType.AMPLIFICATION == "amplification"
        assert TransitionType.SUMMARY == "summary"
        assert TransitionType.BRIDGE == "bridge"
        assert TransitionType.EMPHASIS == "emphasis"
        assert TransitionType.CONCLUSION == "conclusion"
    
    def test_transition_type_creation(self):
        """Test creating TransitionType from string."""
        
        assert TransitionType("bridge") == TransitionType.BRIDGE
        assert TransitionType("sequence") == TransitionType.SEQUENCE
        assert TransitionType("cause_effect") == TransitionType.CAUSE_EFFECT


if __name__ == "__main__":
    pytest.main([__file__])