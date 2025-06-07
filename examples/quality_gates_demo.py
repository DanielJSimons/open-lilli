#!/usr/bin/env python3
"""
Quality Gates Demo

This example demonstrates how to use the new quality gates functionality
in the Open Lilli presentation reviewer.

The quality gates provide objective, quantitative assessment of presentations
to determine if they meet specific quality thresholds.
"""

import json
from typing import List

# Note: This is a demonstration of usage - requires actual imports in practice
# from open_lilli.models import QualityGates, QualityGateResult, ReviewFeedback, SlidePlan
# from open_lilli.reviewer import Reviewer
# from openai import OpenAI

# For demo purposes, we'll use mock classes
class SlidePlan:
    def __init__(self, index, slide_type, title, bullets):
        self.index = index
        self.slide_type = slide_type
        self.title = title
        self.bullets = bullets

class ReviewFeedback:
    def __init__(self, slide_index, severity, category, message, suggestion=None):
        self.slide_index = slide_index
        self.severity = severity
        self.category = category
        self.message = message
        self.suggestion = suggestion

class QualityGates:
    def __init__(self, max_bullets_per_slide=7, max_readability_grade=9.0, 
                 max_style_errors=0, min_overall_score=7.0,
                 min_apca_lc_for_body_text: float = 45.0): # Added APCA
        self.max_bullets_per_slide = max_bullets_per_slide
        self.max_readability_grade = max_readability_grade
        self.max_style_errors = max_style_errors
        self.min_overall_score = min_overall_score
        self.min_apca_lc_for_body_text = min_apca_lc_for_body_text # Store APCA threshold

class QualityGateResult:
    def __init__(self, status, gate_results, violations, recommendations, metrics):
        self.status = status
        self.gate_results = gate_results
        self.violations = violations
        self.recommendations = recommendations
        self.metrics = metrics
    
    @property
    def passed_gates(self):
        return sum(1 for passed in self.gate_results.values() if passed)
    
    @property
    def total_gates(self):
        return len(self.gate_results)
    
    @property
    def pass_rate(self):
        if self.total_gates == 0:
            return 0.0
        return (self.passed_gates / self.total_gates) * 100.0


def create_sample_slides() -> List[SlidePlan]:
    """Create sample slides for demonstration."""
    return [
        SlidePlan(
            index=0,
            slide_type="title",
            title="Q4 Business Review",
            bullets=[]
        ),
        SlidePlan(
            index=1,
            slide_type="content",
            title="Market Performance",
            bullets=[
                "Revenue grew 15% year-over-year",
                "Customer satisfaction increased to 95%",
                "Market share expanded in key segments",
                "Digital transformation initiatives delivered results"
            ]
        ),
        SlidePlan(
            index=2,
            slide_type="content",
            title="Complex Organizational Infrastructure Analysis",
            bullets=[
                "Comprehensive implementation methodologies necessitate organizational restructuring protocols",
                "Sophisticated analytical frameworks require interdisciplinary collaboration mechanisms",
                "Multifaceted strategic initiatives demand extensive stakeholder engagement paradigms",
                "Innovative technological solutions facilitate transformative business optimization processes",
                "Synergistic organizational capabilities enable sustainable competitive advantage development",
                "Advanced operational frameworks support comprehensive performance enhancement initiatives",
                "Strategic partnership developments require sophisticated relationship management protocols",
                "Complex regulatory compliance mechanisms necessitate comprehensive oversight procedures"
            ]
        ),
        SlidePlan(
            index=3,
            slide_type="content",
            title="Next Steps",
            bullets=[
                "Continue growth initiatives",
                "Monitor key metrics",
                "Adjust strategy as needed"
            ]
        )
    ]


def create_sample_feedback() -> List[ReviewFeedback]:
    """Create sample review feedback."""
    return [
        ReviewFeedback(
            slide_index=1,
            severity="low",
            category="content",
            message="Good use of specific metrics",
            suggestion="Consider adding visual charts"
        ),
        ReviewFeedback(
            slide_index=2,
            severity="medium",
            category="design",
            message="Slide has too many complex bullet points",
            suggestion="Break into multiple slides or simplify language"
        ),
        ReviewFeedback(
            slide_index=2,
            severity="high",
            category="consistency",
            message="Overly complex language may confuse audience",
            suggestion="Use simpler, more direct language"
        )
    ]


def demonstrate_quality_gates():
    """Demonstrate quality gates functionality."""
    print("=== Quality Gates Demonstration ===\n")
    
    # Create sample data
    slides = create_sample_slides()
    feedback = create_sample_feedback()
    
    print("Sample Presentation:")
    for slide in slides:
        print(f"  Slide {slide.index + 1}: {slide.title}")
        print(f"    Bullets: {len(slide.bullets)}")
        if slide.bullets:
            for bullet in slide.bullets[:2]:  # Show first 2 bullets
                print(f"      • {bullet[:50]}{'...' if len(bullet) > 50 else ''}")
            if len(slide.bullets) > 2:
                print(f"      ... and {len(slide.bullets) - 2} more")
        print()
    
    print(f"Sample Feedback: {len(feedback)} items")
    for fb in feedback:
        print(f"  • Slide {fb.slide_index + 1} ({fb.severity}): {fb.message}")
    print()
    
    # Demonstrate default quality gates
    print("1. Default Quality Gates Configuration:")
    default_gates = QualityGates()
    print(f"   • Max bullets per slide: {default_gates.max_bullets_per_slide}")
    print(f"   • Max readability grade: {default_gates.max_readability_grade}")
    print(f"   • Max style errors: {default_gates.max_style_errors}")
    print(f"   • Min overall score: {default_gates.min_overall_score}")
    print(f"   • Min APCA Lc for Body Text: {default_gates.min_apca_lc_for_body_text}") # Added APCA line
    print()
    
    # This would be the actual usage with a real reviewer:
    print("2. Quality Gates Evaluation (Simulated):")
    print("   With a real OpenAI client, you would do:")
    print("   ```python")
    print("   client = OpenAI(api_key='your-key')")
    print("   reviewer = Reviewer(client)")
    print("   feedback, quality_result = reviewer.review_presentation(")
    print("       slides, include_quality_gates=True)")
    print("   ```")
    print()
    
    # Simulate what the quality gates evaluation would find
    print("3. Expected Quality Gate Results:")
    print("   Based on the sample slides above:")
    print()
    
    # Bullet count analysis
    print("   Bullet Count Gate:")
    for slide in slides:
        bullet_count = len(slide.bullets)
        status = "PASS" if bullet_count <= 7 else "FAIL"
        print(f"     Slide {slide.index + 1}: {bullet_count} bullets - {status}")
    print()
    
    # Readability analysis (manual estimation)
    print("   Readability Gate:")
    print("     Slide 1 (title): Simple title - PASS")
    print("     Slide 2: Business language, moderate complexity - PASS") 
    print("     Slide 3: Very complex language with long words - FAIL")
    print("     Slide 4: Simple language - PASS")
    print()
    
    # Style errors
    style_errors = sum(1 for fb in feedback if fb.category in ["design", "consistency"])
    print(f"   Style Errors Gate: {style_errors} errors found - {'FAIL' if style_errors > 0 else 'PASS'}")
    print()

    # APCA Contrast Gate
    print("   APCA Contrast Gate:")
    print("     (Assumes specific text/background colors for slides if Reviewer had TemplateStyle)")
    print("     Slide 2 (e.g. Black text #000000 on White #FFFFFF): High Lc value (e.g., ~106) - PASS (abs(Lc) >= 45)")
    print("     Slide with (e.g. Grey text #777777 on White #FFFFFF): Lc value (e.g., ~44) - FAIL (abs(Lc) < 45)")
    print()
    
    # Overall assessment
    print("   Expected Overall Result: NEEDS_FIX")
    print("   Reasons:")
    print("     • Slide 3 has too many bullets (8 > 7)")
    print("     • Slide 3 has overly complex language")
    print("     • 2 style-related feedback items found")
    print()
    
    # Demonstrate custom configuration
    print("4. Custom Quality Gates Configuration:")
    strict_gates = QualityGates(
        max_bullets_per_slide=5,      # Stricter bullet limit
        max_readability_grade=7.0,    # Simpler language required
        max_style_errors=0,           # No style errors allowed
        min_overall_score=8.0,        # Higher quality threshold
        min_apca_lc_for_body_text=60.0 # Custom APCA threshold
    )
    
    print("   Strict Configuration:")
    print(f"     • Max bullets per slide: {strict_gates.max_bullets_per_slide}")
    print(f"     • Max readability grade: {strict_gates.max_readability_grade}")
    print(f"     • Max style errors: {strict_gates.max_style_errors}")
    print(f"     • Min overall score: {strict_gates.min_overall_score}")
    print(f"     • Min APCA Lc for Body Text: {strict_gates.min_apca_lc_for_body_text}") # Added APCA line
    print()
    
    print("   With strict gates, even more slides would fail:")
    print("     • Slide 2 would fail (4 bullets, but complex business language)")
    print("     • Slide 3 would fail (8 bullets, very complex language)")
    print("     • Style errors still present")
    print()
    
    # Show example QualityGateResult
    print("5. Example QualityGateResult Structure:")
    example_result = QualityGateResult(
        status="needs_fix",
        gate_results={
            "bullet_count": False,
            "readability": False,
            "style_errors": False,
            "overall_score": True,
            "contrast_check": False # Added APCA contrast check
        },
        violations=[
            "Slide 3 has 8 bullets (exceeds limit of 7)",
            "Slide 3 has readability grade 12.4 (exceeds limit of 9.0)",
            "Found 2 style errors (exceeds limit of 0)",
            "Slide 2 (Body Placeholder): APCA Lc is 35.50 (Text: #AAAAAA, Background: #FFFFFF). Minimum absolute Lc is 45.0." # Example APCA violation
        ],
        recommendations=[
            "Reduce bullet points on slide 3 to 7 or fewer",
            "Simplify language on slide 3 to improve readability",
            "Address style and consistency issues identified in feedback",
            "Improve text contrast on Slide 2. Ensure an absolute APCA Lc of at least 45.0 for body text." # Example APCA recommendation
        ],
        metrics={
            "max_bullets_found": 8,
            "avg_readability_grade": 8.2,
            "max_readability_grade": 12.4,
            "style_error_count": 2,
            "overall_score": 7.5,
            "min_abs_apca_lc_found": 35.5, # Example APCA metric
            "avg_abs_apca_lc_found": 55.0, # Example APCA metric
            "max_apca_lc_found": 106.0    # Example APCA metric (can be > 100)
        }
    )
    
    print(f"   Status: {example_result.status}")
    print(f"   Gates passed: {example_result.passed_gates}/{example_result.total_gates} ({example_result.pass_rate:.1f}%)")
    print("   Violations:")
    for violation in example_result.violations:
        print(f"     • {violation}")
    print("   Recommendations:")
    for rec in example_result.recommendations:
        print(f"     • {rec}")
    print()
    
    print("6. Integration with Existing Workflow:")
    print("   Quality gates work alongside existing review functionality:")
    print("   • Backward compatible - existing code continues to work")
    print("   • Optional feature - enable with include_quality_gates=True")
    print("   • Quantitative assessment complements qualitative feedback")
    print("   • Configurable thresholds for different use cases")
    print("   • Automated pass/fail determination for CI/CD integration")
    print()
    
    print("=== Quality Gates Demo Complete ===")


if __name__ == "__main__":
    demonstrate_quality_gates()