"""Tests for Engagement Prompt Tuner module (T-81)."""

import pytest
from unittest.mock import Mock, patch
from typing import List

from open_lilli.engagement_tuner import (
    EngagementPromptTuner,
    EngagementTechnique,
    VerbAnalysis,
    EngagementMetrics
)
from open_lilli.models import SlidePlan, GenerationConfig


class TestEngagementPromptTuner:
    """Test cases for EngagementPromptTuner class."""
    
    @pytest.fixture
    def mock_openai_client(self):
        """Create a mock OpenAI client."""
        client = Mock()
        return client
    
    @pytest.fixture
    def engagement_tuner(self, mock_openai_client):
        """Create an EngagementPromptTuner instance with mocked client."""
        return EngagementPromptTuner(
            client=mock_openai_client,
            model="gpt-4.1",
            temperature=0.4
        )
    
    @pytest.fixture
    def sample_slides_baseline(self):
        """Create sample slides with baseline verb diversity (around 15%)."""
        return [
            SlidePlan(
                index=0,
                slide_type="content",
                title="Market Overview",
                bullets=[
                    "Market is growing rapidly",
                    "Company is expanding operations", 
                    "Revenue is increasing steadily"
                ],
                speaker_notes="Market is showing positive trends",
                layout_id=1
            ),
            SlidePlan(
                index=1,
                slide_type="content",
                title="Financial Results",
                bullets=[
                    "Sales are up 20% this quarter",
                    "Costs are down 10% from last year",
                    "Profits are exceeding expectations"
                ],
                speaker_notes="Numbers are looking very good",
                layout_id=1
            ),
            SlidePlan(
                index=2,
                slide_type="content",
                title="Strategic Initiatives",
                bullets=[
                    "Team is implementing new processes",
                    "Technology is being upgraded",
                    "Training is being provided to staff"
                ],
                speaker_notes="Progress is being made on all fronts",
                layout_id=1
            )
        ]
    
    @pytest.fixture
    def sample_slides_enhanced(self):
        """Create sample slides with enhanced verb diversity (around 35%)."""
        return [
            SlidePlan(
                index=0,
                slide_type="content",
                title="Market Transformation",
                bullets=[
                    "Markets accelerate toward digital solutions",
                    "Companies revolutionize operational frameworks",
                    "Revenue streams diversify across channels"
                ],
                speaker_notes="Trends demonstrate significant evolution",
                layout_id=1
            ),
            SlidePlan(
                index=1,
                slide_type="content",
                title="Financial Performance",
                bullets=[
                    "Sales skyrocketed 20% this quarter",
                    "Expenses plummeted 10% from previous year",
                    "Profits surpassed ambitious projections"
                ],
                speaker_notes="Metrics reflect exceptional achievement",
                layout_id=1
            ),
            SlidePlan(
                index=2,
                slide_type="content",
                title="Strategic Initiatives",
                bullets=[
                    "Teams orchestrate innovative methodologies",
                    "Technology transforms core capabilities",
                    "Training empowers workforce excellence"
                ],
                speaker_notes="Initiatives catalyze organizational growth",
                layout_id=1
            )
        ]
    
    @pytest.fixture
    def sample_slides_with_questions(self):
        """Create sample slides with rhetorical questions."""
        return [
            SlidePlan(
                index=0,
                slide_type="content",
                title="Market Analysis",
                bullets=["Key market trends", "Competitive landscape"],
                layout_id=1
            ),
            SlidePlan(
                index=1,
                slide_type="content", 
                title="Financial Overview",
                bullets=["Revenue performance", "Cost optimization"],
                layout_id=1
            ),
            SlidePlan(
                index=2,
                slide_type="content",
                title="Strategic Direction",
                bullets=["Future roadmap", "Investment priorities"],
                layout_id=1
            ),
            SlidePlan(
                index=3,
                slide_type="content",
                title="Implementation Plan",
                bullets=["Timeline and milestones", "Resource allocation"],
                layout_id=1
            ),
            SlidePlan(
                index=4,
                slide_type="content",
                title="What drives exceptional growth in today's market?",  # Rhetorical question
                bullets=[
                    "Innovation and agility",
                    "Customer-centric approach",
                    "How can we leverage these insights?"  # Another question
                ],
                layout_id=1
            )
        ]
    
    @pytest.fixture
    def mock_enhanced_content_response(self):
        """Mock LLM response for enhanced content generation."""
        return {
            "title": "What propels market transformation?",
            "bullets": [
                "Digital innovations revolutionize customer experiences",
                "Agile methodologies accelerate product development",
                "Data analytics optimize strategic decisions"
            ],
            "speaker_notes": "These factors catalyze unprecedented growth opportunities"
        }
    
    def test_should_add_rhetorical_question(self, engagement_tuner):
        """Test logic for when to add rhetorical questions (every 5 slides)."""
        
        # Test various slide positions
        assert engagement_tuner._should_add_rhetorical_question(0, 10) is False  # Slide 1
        assert engagement_tuner._should_add_rhetorical_question(1, 10) is False  # Slide 2
        assert engagement_tuner._should_add_rhetorical_question(2, 10) is False  # Slide 3
        assert engagement_tuner._should_add_rhetorical_question(3, 10) is False  # Slide 4
        assert engagement_tuner._should_add_rhetorical_question(4, 10) is True   # Slide 5 (every 5th)
        assert engagement_tuner._should_add_rhetorical_question(5, 10) is False  # Slide 6
        assert engagement_tuner._should_add_rhetorical_question(8, 10) is False  # Slide 9
        assert engagement_tuner._should_add_rhetorical_question(9, 10) is True   # Slide 10 (every 5th)
    
    def test_build_engagement_instructions_without_question(self, engagement_tuner):
        """Test building engagement instructions without rhetorical question."""
        
        instructions = engagement_tuner._build_engagement_instructions(
            needs_rhetorical_question=False,
            engagement_context=None
        )
        
        assert "VERB VARIETY REQUIREMENTS:" in instructions
        assert "diverse, compelling verbs" in instructions
        assert "30% unique verbs" in instructions
        assert "RHETORICAL QUESTION REQUIREMENT:" not in instructions
        assert "ENGAGEMENT TECHNIQUES:" in instructions
    
    def test_build_engagement_instructions_with_question(self, engagement_tuner):
        """Test building engagement instructions with rhetorical question."""
        
        instructions = engagement_tuner._build_engagement_instructions(
            needs_rhetorical_question=True,
            engagement_context=None
        )
        
        assert "VERB VARIETY REQUIREMENTS:" in instructions
        assert "RHETORICAL QUESTION REQUIREMENT:" in instructions
        assert "Include ONE compelling rhetorical question" in instructions
        assert "thought-provoking and relevant" in instructions
        assert "ENGAGEMENT TECHNIQUES:" in instructions
    
    def test_inject_engagement_instructions(self, engagement_tuner):
        """Test injection of engagement instructions into base prompt."""
        
        base_prompt = """You are an expert presentation writer.

REQUIREMENTS:
- Generate content in English
- Maximum 5 bullet points
- Keep content concise

OUTPUT FORMAT:
Return a JSON object with this structure:
{"title": "Enhanced title"}

Generate content now:"""
        
        engagement_instructions = """
ENGAGEMENT TECHNIQUES:
- Use varied verbs
- Add rhetorical questions"""
        
        enhanced_prompt = engagement_tuner._inject_engagement_instructions(
            base_prompt, engagement_instructions
        )
        
        # Check that engagement instructions were inserted
        assert "ENGAGEMENT TECHNIQUES:" in enhanced_prompt
        assert "Use varied verbs" in enhanced_prompt
        assert enhanced_prompt.count("REQUIREMENTS:") == 1
        assert enhanced_prompt.count("OUTPUT FORMAT:") == 1
        
        # Check order: REQUIREMENTS -> original requirements -> engagement -> OUTPUT FORMAT
        req_pos = enhanced_prompt.find("REQUIREMENTS:")
        engage_pos = enhanced_prompt.find("ENGAGEMENT TECHNIQUES:")
        output_pos = enhanced_prompt.find("OUTPUT FORMAT:")
        
        assert req_pos < engage_pos < output_pos
    
    def test_extract_verbs_basic(self, engagement_tuner):
        """Test basic verb extraction from text."""
        
        text = "The company is growing rapidly. Sales have increased significantly. We are implementing new strategies."
        
        verbs = engagement_tuner._extract_verbs(text)
        
        # Should extract various verb forms
        assert "is" in verbs
        assert "growing" in verbs
        assert "have" in verbs
        assert "increased" in verbs
        assert "are" in verbs
        assert "implementing" in verbs
        
        # Should not extract non-verbs
        assert "company" not in verbs
        assert "sales" not in verbs
        assert "strategies" not in verbs
    
    def test_extract_verbs_comprehensive(self, engagement_tuner):
        """Test comprehensive verb extraction with various forms."""
        
        text = """Markets accelerate toward digital solutions. Companies revolutionize operational frameworks. 
                  Revenue streams diversify across channels. Teams orchestrate innovative methodologies."""
        
        verbs = engagement_tuner._extract_verbs(text)
        
        # Should extract action verbs
        assert "accelerate" in verbs
        assert "revolutionize" in verbs  
        assert "diversify" in verbs
        assert "orchestrate" in verbs
        
        # Check for uniqueness (no duplicates)
        assert len(verbs) == len(set(verbs))
    
    def test_analyze_verb_diversity_baseline(self, engagement_tuner, sample_slides_baseline):
        """Test verb diversity analysis on baseline slides (should be around 15%)."""
        
        analysis = engagement_tuner.analyze_verb_diversity(sample_slides_baseline)
        
        assert isinstance(analysis, VerbAnalysis)
        assert analysis.total_verbs > 0
        assert analysis.unique_verbs > 0
        assert analysis.verb_diversity_ratio > 0.0
        
        # Should be around baseline (15% or lower for repetitive content)
        assert analysis.verb_diversity_ratio <= 0.25  # Allow some variance
        
        # Should identify repeated verbs
        assert len(analysis.repeated_verbs) > 0
        assert "is" in analysis.repeated_verbs or "are" in analysis.repeated_verbs
        
        # Should have common verbs identified
        assert len(analysis.most_common_verbs) > 0
    
    def test_analyze_verb_diversity_enhanced(self, engagement_tuner, sample_slides_enhanced):
        """Test verb diversity analysis on enhanced slides (should be around 35%)."""
        
        analysis = engagement_tuner.analyze_verb_diversity(sample_slides_enhanced)
        
        assert isinstance(analysis, VerbAnalysis)
        assert analysis.total_verbs > 0
        assert analysis.unique_verbs > 0
        
        # Should achieve higher diversity ratio
        assert analysis.verb_diversity_ratio >= 0.30  # T-81 target
        
        # Should have fewer repeated verbs
        assert len(analysis.repeated_verbs) < 5  # More variety = fewer repeats
    
    def test_count_rhetorical_questions(self, engagement_tuner, sample_slides_with_questions):
        """Test counting rhetorical questions in slides."""
        
        question_count = engagement_tuner._count_rhetorical_questions(sample_slides_with_questions)
        
        # Should count questions in title and bullets
        assert question_count == 2  # One in title, one in bullet
    
    def test_calculate_engagement_score(self, engagement_tuner):
        """Test engagement score calculation."""
        
        # Test high-performing metrics
        high_score = engagement_tuner._calculate_engagement_score(
            verb_diversity=0.35,      # Above 30% target
            rhetorical_questions=2,   # Good number of questions
            total_slides=10          # 2 questions for 10 slides = 1 per 5
        )
        
        assert high_score >= 8.0  # Should be high score
        assert high_score <= 10.0
        
        # Test low-performing metrics
        low_score = engagement_tuner._calculate_engagement_score(
            verb_diversity=0.12,     # Below target
            rhetorical_questions=0,  # No questions
            total_slides=10
        )
        
        assert low_score <= 5.0  # Should be low score
        assert low_score >= 0.0
    
    def test_measure_engagement_metrics_baseline(self, engagement_tuner, sample_slides_baseline):
        """Test engagement metrics measurement on baseline slides."""
        
        metrics = engagement_tuner.measure_engagement_metrics(sample_slides_baseline)
        
        assert isinstance(metrics, EngagementMetrics)
        assert metrics.total_slides == 3
        assert metrics.baseline_verb_ratio == 0.15
        assert metrics.verb_diversity_ratio > 0.0
        
        # Should not meet T-81 target initially
        assert metrics.meets_verb_diversity_target is False  # Below 30%
        
        # Check computed properties
        improvement = metrics.improvement_over_baseline
        assert improvement == metrics.verb_diversity_ratio - 0.15
    
    def test_measure_engagement_metrics_enhanced(self, engagement_tuner, sample_slides_enhanced):
        """Test engagement metrics measurement on enhanced slides."""
        
        metrics = engagement_tuner.measure_engagement_metrics(sample_slides_enhanced)
        
        assert isinstance(metrics, EngagementMetrics)
        assert metrics.total_slides == 3
        
        # Should meet T-81 target
        assert metrics.meets_verb_diversity_target is True  # Above 30%
        assert metrics.verb_diversity_ratio >= 0.30
        
        # Should show significant improvement
        assert metrics.improvement_over_baseline >= 0.10
    
    def test_enhance_content_prompt_no_question(self, engagement_tuner):
        """Test content prompt enhancement without rhetorical question."""
        
        base_prompt = """You are an expert presentation writer.

REQUIREMENTS:
- Generate content in English
- Maximum 5 bullet points

OUTPUT FORMAT:
Return JSON object.

Generate content:"""
        
        enhanced_prompt = engagement_tuner.enhance_content_prompt(
            base_prompt=base_prompt,
            slide_index=1,  # Not a 5th slide
            total_slides=10,
            config=GenerationConfig(),
            engagement_context=None
        )
        
        assert "VERB VARIETY REQUIREMENTS:" in enhanced_prompt
        assert "RHETORICAL QUESTION REQUIREMENT:" not in enhanced_prompt
        assert "ENGAGEMENT TECHNIQUES:" in enhanced_prompt
    
    def test_enhance_content_prompt_with_question(self, engagement_tuner):
        """Test content prompt enhancement with rhetorical question."""
        
        base_prompt = """You are an expert presentation writer.

REQUIREMENTS:
- Generate content in English

OUTPUT FORMAT:
Return JSON object."""
        
        enhanced_prompt = engagement_tuner.enhance_content_prompt(
            base_prompt=base_prompt,
            slide_index=4,  # 5th slide (0-based index)
            total_slides=10,
            config=GenerationConfig(),
            engagement_context=None
        )
        
        assert "VERB VARIETY REQUIREMENTS:" in enhanced_prompt
        assert "RHETORICAL QUESTION REQUIREMENT:" in enhanced_prompt
        assert "Include ONE compelling rhetorical question" in enhanced_prompt
        assert "ENGAGEMENT TECHNIQUES:" in enhanced_prompt
    
    def test_generate_enhanced_slide_content(self, engagement_tuner, mock_enhanced_content_response):
        """Test enhanced content generation for a single slide."""
        
        engagement_tuner._call_llm_with_retries = Mock(return_value=mock_enhanced_content_response)
        
        slide = SlidePlan(
            index=4,  # 5th slide - should get rhetorical question
            slide_type="content",
            title="Market Trends",
            bullets=["Growth patterns", "Consumer behavior"],
            layout_id=1
        )
        
        enhanced_slide = engagement_tuner._generate_enhanced_slide_content(
            slide=slide,
            slide_index=4,
            total_slides=10,
            config=GenerationConfig(),
            style_guidance=None,
            language="en"
        )
        
        # Should have enhanced content
        assert enhanced_slide.title == "What propels market transformation?"
        assert "revolutionize" in enhanced_slide.bullets[0]
        assert "accelerate" in enhanced_slide.bullets[1]
        assert "optimize" in enhanced_slide.bullets[2]
        assert "catalyze" in enhanced_slide.speaker_notes
    
    def test_generate_enhanced_content_batch(self, engagement_tuner, sample_slides_baseline, 
                                           mock_enhanced_content_response):
        """Test batch enhancement of multiple slides."""
        
        engagement_tuner._call_llm_with_retries = Mock(return_value=mock_enhanced_content_response)
        
        enhanced_slides = engagement_tuner.generate_enhanced_content_batch(
            slides=sample_slides_baseline,
            config=GenerationConfig(),
            style_guidance=None,
            language="en"
        )
        
        assert len(enhanced_slides) == 3
        
        # Check that content was enhanced (all should have same mock response)
        for slide in enhanced_slides:
            assert "propels market transformation" in slide.title
            assert any("revolutionize" in bullet for bullet in slide.bullets)
    
    def test_validate_t81_requirements_success(self, engagement_tuner):
        """Test T-81 requirements validation with successful metrics."""
        
        metrics = EngagementMetrics(
            total_slides=10,
            verb_diversity_ratio=0.35,      # Above 30% target
            baseline_verb_ratio=0.15,
            rhetorical_questions_added=3,   # Good coverage
            engagement_score=8.5
        )
        
        validation = engagement_tuner.validate_t81_requirements(metrics)
        
        assert validation["verb_diversity_target"] is True   # ≥30%
        assert validation["significant_improvement"] is True  # ≥10% improvement over baseline
        assert validation["rhetorical_questions"] is True    # ≥15% of slides
    
    def test_validate_t81_requirements_failure(self, engagement_tuner):
        """Test T-81 requirements validation with failing metrics."""
        
        metrics = EngagementMetrics(
            total_slides=10,
            verb_diversity_ratio=0.20,      # Below 30% target
            baseline_verb_ratio=0.15,
            rhetorical_questions_added=0,   # No questions
            engagement_score=4.0
        )
        
        validation = engagement_tuner.validate_t81_requirements(metrics)
        
        assert validation["verb_diversity_target"] is False  # <30%
        assert validation["significant_improvement"] is False # Only 5% improvement
        assert validation["rhetorical_questions"] is False   # 0% of slides
    
    def test_verb_alternatives_database(self, engagement_tuner):
        """Test verb alternatives database."""
        
        alternatives = engagement_tuner.verb_alternatives
        
        # Should have common verbs with alternatives
        assert "show" in alternatives
        assert "demonstrate" in alternatives["show"]
        assert "reveal" in alternatives["show"]
        
        assert "improve" in alternatives
        assert "enhance" in alternatives["improve"]
        assert "optimize" in alternatives["improve"]
        
        assert "increase" in alternatives
        assert "boost" in alternatives["increase"]
        assert "amplify" in alternatives["increase"]
    
    def test_error_handling_in_content_generation(self, engagement_tuner, sample_slides_baseline):
        """Test error handling in enhanced content generation."""
        
        # Mock LLM call to raise exception
        engagement_tuner._call_llm_with_retries = Mock(side_effect=Exception("API Error"))
        
        slide = sample_slides_baseline[0]
        enhanced_slide = engagement_tuner._generate_enhanced_slide_content(
            slide=slide,
            slide_index=0,
            total_slides=3,
            config=GenerationConfig(),
            style_guidance=None,
            language="en"
        )
        
        # Should return original slide on error
        assert enhanced_slide == slide
    
    def test_engagement_metrics_properties(self):
        """Test EngagementMetrics computed properties."""
        
        metrics = EngagementMetrics(
            total_slides=10,
            verb_diversity_ratio=0.32,
            baseline_verb_ratio=0.15,
            rhetorical_questions_added=2,
            engagement_score=7.5
        )
        
        # Test meets_verb_diversity_target
        assert metrics.meets_verb_diversity_target is True  # 0.32 >= 0.30
        
        # Test improvement_over_baseline
        assert metrics.improvement_over_baseline == 0.17  # 0.32 - 0.15
        
        # Test rhetorical_question_frequency
        assert metrics.rhetorical_question_frequency == 0.2  # 2/10 = 0.2


class TestEngagementTechnique:
    """Test EngagementTechnique enum functionality."""
    
    def test_engagement_technique_values(self):
        """Test that all engagement techniques have correct values."""
        
        assert EngagementTechnique.VARIED_VERBS == "varied_verbs"
        assert EngagementTechnique.RHETORICAL_QUESTIONS == "rhetorical_questions"
        assert EngagementTechnique.ACTIVE_VOICE == "active_voice"
        assert EngagementTechnique.COMPELLING_LANGUAGE == "compelling_language"
        assert EngagementTechnique.AUDIENCE_INTERACTION == "audience_interaction"


if __name__ == "__main__":
    pytest.main([__file__])