#!/usr/bin/env python3
"""
Demo script for testing the dynamic content fit system.

This script demonstrates Epic C2's dynamic content fit features:
- T-49: Density heuristic for content overflow detection and slide splitting
- T-50: Auto font size tuner for handling mild overflow scenarios

Usage:
    python examples/content_fit_demo.py
"""

import sys
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("⚠️  OpenAI not installed. Demonstrating configuration only.")

from open_lilli.models import (
    SlidePlan,
    ContentFitConfig,
    ContentDensityAnalysis,
    FontAdjustment
)

def create_stress_test_slides() -> list[SlidePlan]:
    """Create slides with various content density scenarios for testing."""
    
    # Normal content slide
    normal_slide = SlidePlan(
        index=0,
        slide_type="content",
        title="Normal Content Slide",
        bullets=[
            "This is a reasonable bullet point",
            "Another normal bullet with standard length",
            "Final bullet point for comparison"
        ],
        layout_id=1
    )
    
    # Mild overflow slide
    mild_overflow_slide = SlidePlan(
        index=1,
        slide_type="content", 
        title="Mildly Dense Content That Might Need Font Adjustment",
        bullets=[
            "This bullet point contains significantly more text than typical presentations would normally include in a single line item",
            "Another verbose bullet point that continues to add substantial detail and context beyond what would fit comfortably",
            "A third detailed bullet explaining complex concepts with multiple clauses and comprehensive information that extends length",
            "Additional bullet point with supplementary details and explanatory content that adds to the overall density"
        ],
        layout_id=1
    )
    
    # Severe overflow slide (should be split)
    severe_overflow_slide = SlidePlan(
        index=2,
        slide_type="content",
        title="Severely Overloaded Content Slide Requiring Immediate Splitting Action",
        bullets=[
            "This is an extremely verbose bullet point that contains far too much information for a single slide presentation format and would definitely cause significant overflow issues requiring immediate attention and corrective action through slide splitting mechanisms",
            "Another exceptionally detailed bullet point with comprehensive explanations, multiple subclauses, extensive detail, supporting context, background information, implementation specifics, technical requirements, business justification, risk assessment, timeline considerations, resource allocation details, and success metrics that clearly exceed reasonable content density limits",
            "A third massively detailed bullet containing exhaustive information about complex business processes, technological implementations, strategic considerations, operational procedures, compliance requirements, quality assurance protocols, testing methodologies, deployment strategies, monitoring approaches, maintenance procedures, troubleshooting guides, and performance optimization techniques",
            "Additional comprehensive bullet point covering project management methodologies, stakeholder engagement strategies, communication protocols, risk mitigation approaches, change management procedures, training requirements, documentation standards, reporting mechanisms, escalation procedures, and continuous improvement processes that absolutely require slide division",
            "Yet another extensive bullet point discussing market analysis, competitive landscape, customer segmentation, value proposition development, pricing strategies, distribution channels, partnership opportunities, technology platforms, scalability considerations, and long-term growth planning initiatives",
            "Final overwhelming bullet containing detailed financial projections, budget allocations, cost-benefit analysis, return on investment calculations, revenue forecasting, expense management, cash flow planning, and comprehensive fiscal responsibility measures that make this slide impossible to display effectively"
        ],
        layout_id=1
    )
    
    # Edge case: single very long bullet
    single_long_bullet_slide = SlidePlan(
        index=3,
        slide_type="content",
        title="Single Extremely Long Bullet Point Scenario",
        bullets=[
            "This single bullet point is designed to test the system's ability to handle one extremely long piece of content that contains multiple sentences with extensive detail, comprehensive explanations, detailed analysis, supporting evidence, contextual information, background details, implementation guidance, best practices, lessons learned, case studies, performance metrics, success factors, risk considerations, mitigation strategies, and comprehensive recommendations that would normally be split across multiple bullets but instead appears as one continuous stream of information that challenges our content density algorithms and font adjustment mechanisms to determine the most appropriate response strategy."
        ],
        layout_id=1
    )
    
    return [normal_slide, mild_overflow_slide, severe_overflow_slide, single_long_bullet_slide]

def demo_density_analysis():
    """Demonstrate content density analysis."""
    print("📊 Testing Content Density Analysis...")
    
    # Test slides
    test_slides = create_stress_test_slides()
    config = ContentFitConfig()
    
    print(f"   ⚙️  Configuration:")
    print(f"      Characters per line: {config.characters_per_line}")
    print(f"      Lines per placeholder: {config.lines_per_placeholder}")
    print(f"      Split threshold: {config.split_threshold}")
    print(f"      Font tune threshold: {config.font_tune_threshold}")
    print()
    
    from open_lilli.content_fit_analyzer import ContentFitAnalyzer
    analyzer = ContentFitAnalyzer(config)
    
    for slide in test_slides:
        density_analysis = analyzer.analyze_slide_density(slide)
        
        print(f"   📝 Slide {slide.index}: {slide.title[:40]}...")
        print(f"      Total characters: {density_analysis.total_characters}")
        print(f"      Estimated lines: {density_analysis.estimated_lines}")
        print(f"      Placeholder capacity: {density_analysis.placeholder_capacity}")
        print(f"      Density ratio: {density_analysis.density_ratio:.2f}")
        print(f"      Overflow severity: {density_analysis.overflow_severity}")
        print(f"      Recommended action: {density_analysis.recommended_action}")
        print()

def demo_font_adjustment():
    """Demonstrate auto font size tuner."""
    print("🔤 Testing Auto Font Size Tuner...")
    
    # Test scenarios with different overflow levels
    test_scenarios = [
        ("No overflow", 0.8, 18),
        ("Mild overflow", 1.15, 18),
        ("Moderate overflow", 1.25, 18),
        ("Severe overflow", 1.6, 18),
        ("Already small font", 1.2, 14),
        ("Large font with overflow", 1.3, 22)
    ]
    
    from open_lilli.content_fit_analyzer import ContentFitAnalyzer
    analyzer = ContentFitAnalyzer()
    
    for scenario_name, density_ratio, current_size in test_scenarios:
        # Create mock density analysis
        density_analysis = ContentDensityAnalysis(
            total_characters=int(400 * density_ratio),
            estimated_lines=int(8 * density_ratio),
            placeholder_capacity=400,
            density_ratio=density_ratio,
            requires_action=density_ratio > 1.0,
            recommended_action="adjust_font" if 1.0 < density_ratio < 1.3 else "split_slide"
        )
        
        font_adjustment = analyzer.recommend_font_adjustment(density_analysis, current_size)
        
        print(f"   📐 {scenario_name}:")
        print(f"      Density ratio: {density_ratio:.2f}")
        print(f"      Current font: {current_size}pt")
        
        if font_adjustment:
            print(f"      Recommended: {font_adjustment.recommended_size}pt")
            print(f"      Adjustment: {font_adjustment.adjustment_points:+d}pt")
            print(f"      Confidence: {font_adjustment.confidence:.2f}")
            print(f"      Safe bounds: {font_adjustment.safe_bounds}")
            print(f"      Reasoning: {font_adjustment.reasoning}")
        else:
            print(f"      No font adjustment recommended")
        print()

def demo_slide_splitting():
    """Demonstrate slide splitting for severe overflow."""
    print("✂️  Testing Slide Splitting...")
    
    # Get the severe overflow slide
    test_slides = create_stress_test_slides()
    severe_slide = test_slides[2]  # The one with 6 long bullets
    
    from open_lilli.content_fit_analyzer import ContentFitAnalyzer
    analyzer = ContentFitAnalyzer()
    
    print(f"   📝 Original slide: {severe_slide.title[:50]}...")
    print(f"      Bullets: {len(severe_slide.bullets)}")
    print(f"      Total characters: {sum(len(b) for b in severe_slide.bullets)}")
    
    # Perform split
    split_slides = analyzer.split_slide_content(severe_slide)
    
    print(f"   ✂️  Split result:")
    print(f"      Generated slides: {len(split_slides)}")
    
    for i, split_slide in enumerate(split_slides):
        char_count = sum(len(b) for b in split_slide.bullets)
        print(f"      Slide {i+1}: '{split_slide.title}' - {len(split_slide.bullets)} bullets, {char_count} chars")

def demo_integration():
    """Demonstrate integration with SlidePlanner."""
    print("🔧 Testing SlidePlanner Integration...")
    
    if not OPENAI_AVAILABLE:
        print("   ⚠️  OpenAI not available. Showing integration concept only.")
        print("   🏗️  Integration features:")
        print("      • Content fit analysis during slide planning")
        print("      • Automatic slide splitting for severe overflow")
        print("      • Font size adjustments for mild overflow")
        print("      • Integration with template style information")
        print("      • Reporting of optimization actions taken")
        return
    
    print("   🔄 SlidePlanner integration would:")
    print("      1. Analyze each slide for content density")
    print("      2. Apply splitting for severe overflow (ratio ≥1.3)")
    print("      3. Apply font adjustment for mild overflow (1.1-1.3)")
    print("      4. Pass font adjustments to SlideAssembler")
    print("      5. Report optimization summary")

def demo_stress_testing():
    """Demonstrate stress testing scenarios."""
    print("🚨 Stress Testing Scenarios...")
    
    print("   📋 Test Case T-49: 200-word bullet list")
    
    # Create a 200-word bullet
    long_bullet = " ".join([
        "comprehensive", "business", "analysis", "methodology", "framework", "implementation",
        "strategy", "development", "process", "optimization", "performance", "measurement",
        "quality", "assurance", "risk", "management", "stakeholder", "engagement", "project",
        "deliverables", "timeline", "milestones", "budget", "allocation", "resource", "planning",
        "technical", "requirements", "system", "architecture", "integration", "testing", "deployment",
        "monitoring", "maintenance", "documentation", "training", "support", "continuous",
        "improvement", "best", "practices", "lessons", "learned", "success", "metrics", "key",
        "performance", "indicators", "return", "investment", "cost", "benefit", "analysis",
        "market", "research", "competitive", "landscape", "customer", "segmentation", "value",
        "proposition", "pricing", "strategy", "distribution", "channels", "partnership", "opportunities",
        "technology", "platforms", "scalability", "considerations", "future", "growth", "planning",
        "financial", "projections", "revenue", "forecasting", "expense", "management", "cash",
        "flow", "planning", "fiscal", "responsibility", "compliance", "requirements", "regulatory",
        "standards", "audit", "procedures", "internal", "controls", "governance", "oversight",
        "executive", "reporting", "board", "presentations", "investor", "relations", "public",
        "communications", "media", "strategy", "brand", "management", "marketing", "campaigns",
        "sales", "enablement", "customer", "service", "satisfaction", "retention", "loyalty",
        "programs", "market", "expansion", "international", "operations", "cultural", "adaptation",
        "localization", "global", "supply", "chain", "logistics", "operations", "efficiency",
        "productivity", "automation", "digital", "transformation", "innovation", "research",
        "development", "intellectual", "property", "patent", "protection", "competitive", "advantage",
        "sustainable", "practices", "environmental", "responsibility", "social", "impact", "corporate",
        "citizenship", "ethical", "business", "conduct", "transparency", "accountability", "long",
        "term", "value", "creation", "shareholder", "returns", "dividend", "policy", "capital",
        "structure", "debt", "management", "credit", "rating", "financial", "stability", "business",
        "continuity", "disaster", "recovery", "contingency", "planning", "crisis", "management",
        "reputation", "protection", "stakeholder", "communication", "change", "management", "organizational",
        "development", "talent", "acquisition", "employee", "engagement", "performance", "management",
        "compensation", "benefits", "workplace", "culture", "diversity", "inclusion", "professional",
        "development", "succession", "planning", "knowledge", "management", "information", "systems",
        "data", "analytics", "business", "intelligence", "decision", "support", "predictive", "modeling"
    ])
    
    word_count = len(long_bullet.split())
    print(f"      Generated bullet: {word_count} words, {len(long_bullet)} characters")
    print(f"      Preview: {long_bullet[:100]}...")
    
    stress_slide = SlidePlan(
        index=0,
        slide_type="content",
        title="Stress Test: 200-Word Bullet Point",
        bullets=[long_bullet],
        layout_id=1
    )
    
    from open_lilli.content_fit_analyzer import ContentFitAnalyzer
    analyzer = ContentFitAnalyzer()
    
    density_analysis = analyzer.analyze_slide_density(stress_slide)
    print(f"      Density ratio: {density_analysis.density_ratio:.2f}")
    print(f"      Action: {density_analysis.recommended_action}")
    
    if analyzer.should_split_slide(density_analysis):
        split_slides = analyzer.split_slide_content(stress_slide)
        print(f"      ✅ Split into {len(split_slides)} slides")
    else:
        print(f"      ⚠️  No split triggered (threshold: {analyzer.config.split_threshold})")
    
    print()
    print("   📋 Test Case T-50: Font size tuning")
    print("      18pt → 16pt for mild overflow")
    
    font_adjustment = analyzer.recommend_font_adjustment(density_analysis, 18)
    if font_adjustment:
        print(f"      ✅ Font adjustment: {font_adjustment.original_size}pt → {font_adjustment.recommended_size}pt")
        print(f"      Passes StyleValidation: {font_adjustment.safe_bounds}")
    else:
        print("      ⚠️  No font adjustment recommended")

def main():
    """Run the content fit demo."""
    print("📐 Dynamic Content Fit Demo - Epic C2 Implementation")
    print("=" * 60)
    
    # Demo each component
    demo_density_analysis()
    demo_font_adjustment()
    demo_slide_splitting()
    demo_integration()
    demo_stress_testing()
    
    print("\n✨ Implementation Status:")
    print("   ✅ T-49: Density heuristic with slide splitting")
    print("   ✅ T-50: Auto font size tuner (±2pt safe bounds)")
    
    print("\n🔧 Integration Points:")
    print("   • SlidePlanner._optimize_content_fit() applies density analysis")
    print("   • SlideAssembler._apply_font_adjustments() handles font tuning")
    print("   • ContentFitAnalyzer.split_slide_content() splits overflow slides")
    print("   • ConfigFitConfig provides tunable thresholds")
    
    print("\n📋 Expected Results:")
    print("   • 200-word bullet → slide splits into multiple slides")
    print("   • Mild overflow → font size 18pt → 16pt")
    print("   • Font adjustments pass StyleValidation (within safe bounds)")
    print("   • Severe overflow (ratio ≥1.3) → automatic slide splitting")
    
    print("\n🚀 Usage in CLI:")
    print("   # Content fit is automatically applied during generation:")
    print("   ai-ppt generate --template template.pptx --input content.txt")
    print("   # (Content fit optimization runs during slide planning)")

if __name__ == "__main__":
    main()