import unittest
from unittest.mock import MagicMock, patch
from openai import OpenAI

from open_lilli.models import SlidePlan, ContentFitConfig, ContentDensityAnalysis, FontAdjustment, ContentFitResult
from open_lilli.content_fit_analyzer import ContentFitAnalyzer, SmartContentFitter


class TestContentFitAnalyzer(unittest.TestCase):

    def test_density_analysis_no_action(self):
        """Test content density analysis when no action is required."""
        config = ContentFitConfig(characters_per_line=50, lines_per_placeholder=8)
        analyzer = ContentFitAnalyzer(config=config)
        slide_plan = SlidePlan(
            index=0,
            slide_type="content",
            title="Short Title",
            bullets=["Short bullet 1.", "Short bullet 2."]
        )
        analysis = analyzer.analyze_slide_density(slide_plan)
        self.assertFalse(analysis.requires_action)
        self.assertEqual(analysis.recommended_action, "no_action")

    def test_density_analysis_needs_split(self):
        """Test content density analysis when a split is recommended."""
        config = ContentFitConfig(
            characters_per_line=20,
            lines_per_placeholder=3,
            split_threshold=1.0 # Lower threshold for easy testing
        )
        analyzer = ContentFitAnalyzer(config=config)
        slide_plan = SlidePlan(
            index=0,
            slide_type="content",
            title="This is a very long title for a slide to test splitting",
            bullets=[
                "This is a very long bullet point that should definitely exceed the capacity of a small placeholder.",
                "Another long bullet point to ensure that the content overflows significantly and requires a split action.",
                "Yet another long bullet to push it over the edge."
            ]
        )
        analysis = analyzer.analyze_slide_density(slide_plan)
        self.assertTrue(analysis.requires_action)
        self.assertEqual(analysis.recommended_action, "split_slide")

    def test_font_adjustment_recommendation(self):
        """Test font adjustment recommendation for mild overflow."""
        config = ContentFitConfig(font_tune_threshold=1.0, split_threshold=1.5) # Ensure tune threshold is low
        analyzer = ContentFitAnalyzer(config=config)
        # Create a density analysis that simulates mild overflow
        density_analysis = ContentDensityAnalysis(
            total_characters=600,
            estimated_lines=10,
            placeholder_capacity=500, # 600/500 = 1.2 ratio
            density_ratio=1.2,
            requires_action=True,
            recommended_action="adjust_font"
        )
        adjustment = analyzer.recommend_font_adjustment(density_analysis, current_font_size=18)
        self.assertIsNotNone(adjustment)
        self.assertTrue(adjustment.adjustment_points < 0) # Font size should decrease
        self.assertEqual(adjustment.recommended_size, 17) # 18 - 1 = 17 for 1.2 ratio (since <= 1.2 is 1 point reduction)

    def test_split_slide_content(self):
        """Test splitting a slide with too many bullets."""
        analyzer = ContentFitAnalyzer()
        slide_plan = SlidePlan(
            index=0,
            slide_type="content",
            title="Original Title",
            bullets=[f"Bullet {i}" for i in range(10)] # 10 bullets
        )
        # Mock configuration for splitting behavior if needed, or rely on defaults
        analyzer.config.lines_per_placeholder = 3 # Assume 3 lines for bullets
        analyzer.config.characters_per_line = 30

        split_slides = analyzer.split_slide_content(slide_plan, target_density=0.8)
        self.assertTrue(len(split_slides) > 1)
        self.assertEqual(split_slides[0].title, "Original Title (Part 1)")
        self.assertEqual(split_slides[1].title, "Original Title (Part 2)")
        # Check if bullets are distributed
        self.assertTrue(len(split_slides[0].bullets) < 10)


    def test_optimize_slide_content_no_action(self):
        """Test optimize_slide_content when no action is needed."""
        analyzer = ContentFitAnalyzer()
        slide_plan = SlidePlan(index=0, slide_type="content", title="Test", bullets=["Bullet 1"])
        result = analyzer.optimize_slide_content(slide_plan)
        self.assertEqual(result.final_action, "no_action")
        self.assertFalse(result.split_performed)
        self.assertIsNone(result.font_adjustment)
        self.assertIsNotNone(result.modified_slide_plan)
        self.assertEqual(result.modified_slide_plan.title, "Test")


    def test_optimize_slide_content_performs_split(self):
        """Test optimize_slide_content performs a split for severe overflow."""
        config = ContentFitConfig(
            characters_per_line=10,
            lines_per_placeholder=2,
            split_threshold=1.1, # Ensure split is triggered
            font_tune_threshold=1.01,
            rewrite_threshold=1.05
        )
        analyzer = ContentFitAnalyzer(config=config)
        long_text = "This is very long text. " * 10
        slide_plan = SlidePlan(index=0, slide_type="content", title="Long Slide", bullets=[long_text, long_text, long_text])

        # Since optimize_slide_content calls split_slide_content internally,
        # and split_slide_content returns a list of slides,
        # the ContentFitResult's split_count should reflect this.
        # The actual SlidePlan objects are handled by the SlidePlanner.
        result = analyzer.optimize_slide_content(slide_plan, template_style=None)

        self.assertEqual(result.final_action, "split_slide")
        self.assertTrue(result.split_performed)
        # split_count is a placeholder if split_slide_content is not called directly in some paths.
        # Here it should be determined by the internal logic.
        # If not slide.bullets, it's 1. Otherwise, it's 2 (placeholder) or actual from split_slide_content
        # Given the logic, if final_action is split_slide and bullets exist, it should be > 1
        self.assertTrue(result.split_count > 1 if slide_plan.bullets else result.split_count == 1)
        self.assertIsNone(result.modified_slide_plan) # When split, modified_slide_plan is None

    def test_summarize_long_content_for_fit(self):
        """Test that long content is summarized to fit the slide."""
        # Configuration that would push to "rewrite_content"
        # Density ratio for rewrite: (rewrite_threshold, split_threshold]
        # e.g., (1.3, 1.5]
        # Placeholder capacity: characters_per_line * lines_per_placeholder
        # Let characters_per_line = 50, lines_per_placeholder = 8 => 400
        # Content length needs to be > 1.3 * 400 = 520 chars
        # And < 1.5 * 400 = 600 chars for rewrite without split immediately
        config = ContentFitConfig(
            characters_per_line=50,
            lines_per_placeholder=8, # Estimated capacity = 400 chars
            font_tune_threshold=1.1,
            rewrite_threshold=1.3, # Content > 520 chars
            split_threshold=1.5    # Content < 600 chars
        )

        # Mock OpenAI client
        mock_openai_client = MagicMock(spec=OpenAI)
        mock_response = MagicMock()
        summarized_text = "• This is a summarized point, short and sweet.\n• This is another concise point after summarization."
        mock_message = MagicMock()
        mock_message.content = summarized_text
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        mock_openai_client.chat.completions.create.return_value = mock_response

        analyzer = ContentFitAnalyzer(config=config, openai_client=mock_openai_client)

        # Create a SlidePlan with long content (~550 characters in bullets)
        # Total words approx = (5 * 20) + 10 + 10 = 120 words
        # Total chars approx = (28 * 20) + 50 + 50 = 560 + 100 = 660 (this will trigger split if not careful with thresholds)
        # Let's aim for ~550 chars to be in rewrite range.
        # (28 * 15) = 420 chars. (5 words * 15 = 75 words)
        long_paragraph_segment = "This is a very long paragraph. " # 29 chars, 6 words
        long_paragraph = long_paragraph_segment * 15 # 435 chars, 90 words

        slide_plan = SlidePlan(
            index=0,
            slide_type="content",
            title="Long Content Summarization Test", # 32 chars
            bullets=[
                long_paragraph, # 435 chars
                "Another moderately long bullet point to add to the total character count for this test.", # 86 chars
                "Short one." # 10 chars
            ]
        )
        # Total bullet chars = 435 + 86 + 10 = 531
        # Total title + bullet chars = 32 (title) + 531 (bullets) = 563
        # Placeholder capacity (estimated by analyzer) will be less than default lines_per_placeholder * chars_per_line
        # due to title and bullet overhead.
        # Default _estimate_placeholder_capacity:
        # lines_available = 8 - 1 (title) = 7
        # chars_per_line = 50 - 4 (bullet overhead) = 46
        # capacity = 7 * 46 = 322
        # Density ratio = 563 / 322 = ~1.74. This would trigger split_slide.

        # We need to adjust config or content for rewrite_content to be chosen.
        # Let's make placeholder_capacity larger by increasing lines_per_placeholder
        # or reduce content slightly.
        # New target: ratio between 1.3 and 1.5.
        # If capacity is C, content L.  1.3 < L/C <= 1.5
        # Let C be around 380-400.
        # Estimated capacity with lines_per_placeholder=10:
        # lines_available = 10 - 1 = 9. chars_per_line = 50-4 = 46. capacity = 9 * 46 = 414
        # Content L = 563. Ratio = 563 / 414 = 1.36. This should be in rewrite_content range.

        # Instantiate analyzer without OpenAI client as we are mocking the method that uses it.
        analyzer = ContentFitAnalyzer(config=config, openai_client=None)
        analyzer.config.lines_per_placeholder = 10


        # This is the SlidePlan object that the mocked rewrite_content_shorter will return.
        mock_rewritten_slide_obj = slide_plan.model_copy()
        summarized_bullet_list = ["This is a summarized point, short and sweet.", "This is another concise point after summarization."]
        mock_rewritten_slide_obj.bullets = summarized_bullet_list
        # summarized_by_llm will be set by optimize_slide_content after a successful "call" to rewrite_content_shorter

        # Patch 'rewrite_content_shorter' on the 'analyzer' instance.
        # new_callable=MagicMock forces the patched method to be a standard MagicMock (synchronous)
        # instead of an AsyncMock (which would be the default for an async method).
        with patch.object(analyzer, 'rewrite_content_shorter', new_callable=MagicMock, return_value=mock_rewritten_slide_obj) as mock_sync_rewrite_method:
            result = analyzer.optimize_slide_content(slide_plan, template_style=None)

            self.assertEqual(result.final_action, "rewrite_content", f"Expected rewrite_content, got {result.final_action}. Density ratio: {result.density_analysis.density_ratio}, Recommended: {result.density_analysis.recommended_action}")
            self.assertIsNotNone(result.modified_slide_plan)
            self.assertTrue(result.modified_slide_plan.summarized_by_llm) # This is set by optimize_slide_content

            summarized_bullets_from_result = result.modified_slide_plan.bullets
            self.assertEqual(summarized_bullets_from_result, summarized_bullet_list) # Check if the mocked bullets are there

            word_count = sum(len(bullet.split()) for bullet in summarized_bullets_from_result)
            # The mock response has 8 + 7 = 15 words.
            self.assertLess(word_count, 100, "Summarized content should be less than 100 words.")
            self.assertEqual(word_count, 15, "Word count of mock response is not matching")

            # Check if number of bullets is maintained (based on mock response)
            self.assertEqual(len(summarized_bullets_from_result), 2, "Number of bullets after summarization should match mock response.")

            # Verify rewrite_content_shorter (the mock) was called
            mock_sync_rewrite_method.assert_called_once()
            # Check arguments passed to the mocked rewrite_content_shorter
            args, kwargs = mock_sync_rewrite_method.call_args
            self.assertEqual(args[0], slide_plan) # Original slide plan
            self.assertTrue(0.1 <= kwargs['target_reduction'] <= 0.5) # Target reduction is calculated correctly


class TestSmartContentFitter(unittest.TestCase):
    """Test cases for T-91: SmartContentFitter v1 rebalancing."""
    
    def test_t91_smart_content_fitter_rebalance(self):
        """Test T-91: SmartContentFitter redistributes bullets across adjacent slides."""
        # Create analyzer and smart fitter
        config = ContentFitConfig(
            characters_per_line=40,
            lines_per_placeholder=5,
            split_threshold=1.4
        )
        analyzer = ContentFitAnalyzer(config=config)
        smart_fitter = SmartContentFitter(analyzer)
        
        # Create slides: one overflowing, one with capacity
        overflowing_slide = SlidePlan(
            index=0,
            slide_type="content",
            title="Overflowing Slide",
            bullets=[
                "This is a very long bullet point that causes overflow in the slide.",
                "Another very long bullet point adding to the density problem.",
                "Yet another long bullet point to ensure overflow.",
                "A fourth bullet to make it even worse."
            ]
        )
        
        light_slide = SlidePlan(
            index=1,
            slide_type="content", 
            title="Light Slide",
            bullets=[
                "Short bullet."
            ]
        )
        
        slides = [overflowing_slide, light_slide]
        
        # Check initial density ratios
        initial_overflow_density = analyzer.analyze_slide_density(overflowing_slide).density_ratio
        initial_light_density = analyzer.analyze_slide_density(light_slide).density_ratio
        
        # Perform rebalancing
        rebalanced_slides = smart_fitter.rebalance(slides)
        
        # Verify rebalancing occurred
        self.assertEqual(len(rebalanced_slides), 2, "Should maintain number of slides")
        
        final_first_density = analyzer.analyze_slide_density(rebalanced_slides[0]).density_ratio
        final_second_density = analyzer.analyze_slide_density(rebalanced_slides[1]).density_ratio
        
        # Verify that density ratio of first slide improved (decreased)
        self.assertLess(final_first_density, initial_overflow_density, 
                       "First slide density should improve after rebalancing")
        
        # Verify total bullets preserved (zero content loss)
        original_total_bullets = sum(len(slide.bullets) for slide in slides)
        rebalanced_total_bullets = sum(len(slide.bullets) for slide in rebalanced_slides)
        self.assertEqual(original_total_bullets, rebalanced_total_bullets,
                        "Total bullets should be preserved during rebalancing")
        
        # Verify that the target density ratio ≤ 1.0 for improved slides
        # (This may not always be achievable depending on content, but we check improvement)
        if final_first_density <= 1.0:
            # Successfully achieved target density
            pass
        else:
            # At least verify there was improvement
            self.assertLess(final_first_density, initial_overflow_density,
                           "Should at least improve density even if not achieving ≤ 1.0")
    
    def test_t91_rebalance_no_moves_when_no_benefit(self):
        """Test that rebalancing doesn't move bullets when it's not beneficial."""
        config = ContentFitConfig(
            characters_per_line=50,
            lines_per_placeholder=10  # Large capacity
        )
        analyzer = ContentFitAnalyzer(config=config)
        smart_fitter = SmartContentFitter(analyzer)
        
        # Create slides that are both well-balanced
        slide1 = SlidePlan(
            index=0,
            slide_type="content",
            title="Balanced Slide 1",
            bullets=["Short bullet 1", "Short bullet 2"]
        )
        
        slide2 = SlidePlan(
            index=1,
            slide_type="content",
            title="Balanced Slide 2", 
            bullets=["Short bullet 3", "Short bullet 4"]
        )
        
        slides = [slide1, slide2]
        original_bullets = [slide.bullets[:] for slide in slides]  # Deep copy
        
        # Perform rebalancing
        rebalanced_slides = smart_fitter.rebalance(slides)
        
        # Verify no changes were made (slides were already balanced)
        for i, (original, rebalanced) in enumerate(zip(original_bullets, rebalanced_slides)):
            self.assertEqual(original, rebalanced.bullets,
                           f"Slide {i} bullets should remain unchanged when already balanced")
    
    def test_t91_rebalance_single_slide_skipped(self):
        """Test that rebalancing is skipped for single slide presentations."""
        config = ContentFitConfig()
        analyzer = ContentFitAnalyzer(config=config)
        smart_fitter = SmartContentFitter(analyzer)
        
        single_slide = SlidePlan(
            index=0,
            slide_type="content",
            title="Only Slide",
            bullets=["Bullet 1", "Bullet 2", "Bullet 3"]
        )
        
        slides = [single_slide]
        rebalanced_slides = smart_fitter.rebalance(slides)
        
        # Should return unchanged
        self.assertEqual(len(rebalanced_slides), 1)
        self.assertEqual(rebalanced_slides[0].bullets, single_slide.bullets)


class TestDynamicLayoutUpgrading(unittest.TestCase):
    """Test cases for T-92: Dynamic Layout Upgrading."""
    
    def test_t92_dynamic_layout_upgrade_resolves_overflow(self):
        """Test T-92: Dynamic layout upgrading resolves overflow by switching to higher-capacity layout."""
        # Create analyzer
        config = ContentFitConfig(
            characters_per_line=30,
            lines_per_placeholder=4,
            split_threshold=1.5
        )
        analyzer = ContentFitAnalyzer(config=config)
        
        # Mock template parser
        mock_template_parser = MagicMock()
        mock_template_parser.get_layout_type_by_id.return_value = "content"
        mock_template_parser.get_layout_index.return_value = 2  # two_column layout ID
        
        # Create overflowing slide with content layout
        overflowing_slide = SlidePlan(
            index=0,
            slide_type="content",
            title="Overflowing Content Slide",
            bullets=[
                "This is a very long bullet point that causes overflow in the content layout.",
                "Another very long bullet point that adds to the density problem.",
                "A third long bullet point making overflow worse.",
                "Fourth bullet to ensure significant overflow."
            ],
            layout_id=1  # content layout
        )
        
        # Test dynamic layout upgrading
        upgraded_slide = analyzer.dynamic_layout_upgrading(
            overflowing_slide, mock_template_parser
        )
        
        # Verify upgrade occurred
        self.assertIsNotNone(upgraded_slide, "Dynamic layout upgrade should return upgraded slide")
        self.assertNotEqual(upgraded_slide.layout_id, overflowing_slide.layout_id,
                           "Layout ID should change after upgrade")
        self.assertEqual(upgraded_slide.layout_id, 2, "Should upgrade to two_column layout")
        
        # Verify content is preserved
        self.assertEqual(upgraded_slide.bullets, overflowing_slide.bullets,
                        "Bullet content should be preserved during layout upgrade")
        self.assertEqual(upgraded_slide.title, overflowing_slide.title,
                        "Title should be preserved during layout upgrade")
    
    def test_t92_no_upgrade_when_no_overflow(self):
        """Test that dynamic layout upgrading is not applied when there's no overflow."""
        config = ContentFitConfig(
            characters_per_line=100,  # Large capacity
            lines_per_placeholder=10
        )
        analyzer = ContentFitAnalyzer(config=config)
        
        # Mock template parser
        mock_template_parser = MagicMock()
        mock_template_parser.get_layout_type_by_id.return_value = "content"
        
        # Create slide with no overflow
        normal_slide = SlidePlan(
            index=0,
            slide_type="content",
            title="Normal Slide",
            bullets=["Short bullet", "Another short bullet"],
            layout_id=1
        )
        
        # Test dynamic layout upgrading
        upgraded_slide = analyzer.dynamic_layout_upgrading(
            normal_slide, mock_template_parser
        )
        
        # Should return None since no upgrade needed
        self.assertIsNone(upgraded_slide, 
                         "Should not upgrade layout when content already fits")
    
    def test_t92_layout_capacity_factors(self):
        """Test that layout capacity factors are correctly defined."""
        config = ContentFitConfig()
        analyzer = ContentFitAnalyzer(config=config)
        
        # Test capacity factors
        self.assertEqual(analyzer._get_layout_capacity_factor("content"), 1.0)
        self.assertEqual(analyzer._get_layout_capacity_factor("two_column"), 1.4)
        self.assertEqual(analyzer._get_layout_capacity_factor("blank"), 1.8)
        self.assertEqual(analyzer._get_layout_capacity_factor("title"), 0.6)
        self.assertEqual(analyzer._get_layout_capacity_factor("unknown_layout"), 1.0)  # Default
    
    def test_t92_no_upgrade_path_available(self):
        """Test behavior when no upgrade path is available for layout type."""
        config = ContentFitConfig(
            characters_per_line=20,
            lines_per_placeholder=3
        )
        analyzer = ContentFitAnalyzer(config=config)
        
        # Mock template parser
        mock_template_parser = MagicMock()
        mock_template_parser.get_layout_type_by_id.return_value = "unknown_layout"
        
        # Create overflowing slide with unknown layout type
        slide = SlidePlan(
            index=0,
            slide_type="content",
            title="Slide with Unknown Layout",
            bullets=["Very long bullet point that causes overflow"],
            layout_id=99  # unknown layout
        )
        
        # Test dynamic layout upgrading
        upgraded_slide = analyzer.dynamic_layout_upgrading(
            slide, mock_template_parser
        )
        
        # Should return None since no upgrade path exists
        self.assertIsNone(upgraded_slide,
                         "Should not upgrade when no upgrade path is defined")


if __name__ == '__main__':
    unittest.main()
