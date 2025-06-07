#!/usr/bin/env python3
"""
Demo script for testing the ML-assisted layout recommendation system.

This script demonstrates Phase 3's ML layout intelligence:
- T-46: Slide embedding & k-NN layout recommendation
- T-47: SlidePlanner ML integration with confidence threshold
- T-48: Template learning pipeline for building training corpus

Usage:
    python examples/ml_layout_demo.py
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
    VectorStoreConfig, 
    SlideEmbedding,
    LayoutRecommendation
)

def create_sample_training_data() -> list[SlideEmbedding]:
    """Create sample training data for the ML system."""
    return [
        SlideEmbedding(
            slide_id="training_1",
            title="Market Analysis vs Competition",
            content_text="Market Analysis vs Competition: Revenue comparison, Market share analysis, Competitive advantages",
            slide_type="two_column",
            layout_id=3,
            embedding=[0.1, 0.2, 0.3] * 512,  # Mock embedding
            bullet_count=3,
            has_image=False,
            has_chart=True,
            source_file="business_deck.pptx",
            created_at="2024-01-15T10:30:00Z"
        ),
        SlideEmbedding(
            slide_id="training_2", 
            title="Product Comparison Analysis",
            content_text="Product Comparison Analysis: Feature comparison, Price analysis, Customer feedback vs competitor",
            slide_type="two_column",
            layout_id=3,
            embedding=[0.15, 0.25, 0.35] * 512,  # Similar to above
            bullet_count=3,
            has_image=False,
            has_chart=False,
            source_file="product_review.pptx",
            created_at="2024-01-16T14:20:00Z"
        ),
        SlideEmbedding(
            slide_id="training_3",
            title="Financial Overview",
            content_text="Financial Overview: Revenue trends, Expense breakdown, Profit margins, Growth metrics",
            slide_type="content",
            layout_id=1,
            embedding=[0.8, 0.1, 0.4] * 512,  # Different pattern
            bullet_count=4,
            has_image=False,
            has_chart=True,
            source_file="financial_report.pptx", 
            created_at="2024-01-17T09:15:00Z"
        )
    ]

def demo_embedding_system():
    """Demonstrate slide embedding functionality."""
    print("🧬 Testing Slide Embedding System...")
    
    # Create sample slide
    test_slide = SlidePlan(
        index=0,
        slide_type="content",
        title="Q3 Performance vs Q2 Results", 
        bullets=[
            "Revenue increased by 15% quarter-over-quarter",
            "Market share expanded in key demographics",
            "Cost optimization reduced expenses by 8%"
        ],
        layout_id=1
    )
    
    if not OPENAI_AVAILABLE:
        print("   ⚠️  OpenAI not available. Showing mock embedding process.")
        print(f"   📝 Slide: {test_slide.title}")
        print(f"   📊 Content: {len(test_slide.bullets)} bullets")
        print(f"   🎯 Would create embedding for: '{test_slide.title}: {', '.join(test_slide.bullets[:2])}...'")
        return
    
    # In real implementation, would create actual embedding here
    print(f"   ✅ Slide embedding would be created for: {test_slide.title}")

def demo_knn_recommendation():
    """Demonstrate k-NN layout recommendation."""
    print("\n🎯 Testing k-NN Layout Recommendation...")
    
    # Create sample query slide
    query_slide = SlidePlan(
        index=0,
        slide_type="content", 
        title="Sales Performance vs Target",
        bullets=[
            "Actual sales exceeded target by 12%",
            "Regional performance varied significantly",
            "Key customer segments outperformed expectations"
        ],
        layout_id=None  # To be determined by ML
    )
    
    # Mock training data
    training_data = create_sample_training_data()
    
    print(f"   📝 Query slide: {query_slide.title}")
    print(f"   📚 Training corpus: {len(training_data)} examples")
    
    # Simulate similarity matching
    print("   🔍 Finding similar slides...")
    for i, example in enumerate(training_data[:2]):
        similarity = 0.85 if "vs" in example.title.lower() else 0.65
        print(f"      {i+1}. '{example.title}' - similarity: {similarity:.2f}")
    
    # Mock recommendation
    print("   🤖 ML Recommendation:")
    print("      Layout: two_column (confidence: 0.87)")
    print("      Reasoning: Based on comparison keywords and historical 'vs' patterns")
    print("      Similar slides: training_1, training_2")

def demo_confidence_threshold():
    """Demonstrate confidence threshold integration.""" 
    print("\n⚖️  Testing Confidence Threshold Integration...")
    
    config = VectorStoreConfig()
    print(f"   📊 Confidence threshold: {config.confidence_threshold}")
    print(f"   🎯 Similarity threshold: {config.similarity_threshold}")
    
    scenarios = [
        ("High confidence", 0.87, "Use ML recommendation"),
        ("Medium confidence", 0.55, "Use rule-based fallback"),
        ("Low confidence", 0.32, "Use rule-based fallback")
    ]
    
    for scenario, confidence, action in scenarios:
        status = "✅ ML" if confidence >= config.confidence_threshold else "🔄 Fallback"
        print(f"   {status} {scenario}: {confidence:.2f} → {action}")

def demo_template_ingestion():
    """Demonstrate template ingestion pipeline."""
    print("\n📚 Testing Template Ingestion Pipeline...")
    
    print("   🏗️  Ingestion pipeline capabilities:")
    print("      • Extract slides from .pptx files")
    print("      • Create embeddings for slide content")  
    print("      • Store layout-content pairs in vector database")
    print("      • Support batch processing of presentation directories")
    
    print("   📁 CLI Usage Examples:")
    print("      # Ingest single presentation:")
    print("      ai-ppt ingest --pptx business_deck.pptx")
    print()
    print("      # Ingest entire directory:")
    print("      ai-ppt ingest --pptx corpus/ --template template.pptx")
    print()
    print("      # Custom vector store:")
    print("      ai-ppt ingest --pptx corpus/ --vector-store custom_layouts.vec")
    
    # Mock ingestion results
    print("   📊 Sample Ingestion Results:")
    print("      Total slides processed: 1000")
    print("      Successful embeddings: 985 (98.5%)")
    print("      Unique layouts found: 8")
    print("      Vector store size: 2543 embeddings")

def demo_ab_testing():
    """Demonstrate A/B testing concept for ML vs rule-based."""
    print("\n🧪 A/B Testing Framework...")
    
    print("   📈 Expected Improvements:")
    print("      • 15% fewer 'layout mismatch' flags from Reviewer")
    print("      • Better layout selection for comparison slides")
    print("      • Improved consistency with historical patterns")
    
    print("   📊 Metrics to Track:")
    print("      • Layout recommendation accuracy")
    print("      • User satisfaction with slide layouts")
    print("      • Reviewer feedback on layout appropriateness")
    print("      • Time saved in manual layout adjustments")

def main():
    """Run the ML layout demo."""
    print("🤖 ML Layout Intelligence Demo - Phase 3 Implementation")
    print("=" * 60)
    
    # Demo each component
    demo_embedding_system()
    demo_knn_recommendation()
    demo_confidence_threshold()
    demo_template_ingestion()
    demo_ab_testing()
    
    print("\n✨ Implementation Status:")
    print("   ✅ T-46: Slide embedding & k-NN system")
    print("   ✅ T-47: SlidePlanner ML integration") 
    print("   ✅ T-48: Template learning pipeline")
    
    print("\n🚀 Integration Complete:")
    print("   • ML recommendations with ≥0.6 confidence override rule-based")
    print("   • k-NN similarity search using OpenAI embeddings")
    print("   • Vector store (layouts.vec) for historical training data")
    print("   • CLI ingestion pipeline for building corpus")
    print("   • Graceful fallback to rule-based when ML fails")
    
    print("\n📋 Usage Examples:")
    print("   # Build ML training corpus:")
    print("   ai-ppt ingest --pptx historical_presentations/")
    print()
    print("   # Generate with ML-assisted layouts:")
    print("   ai-ppt generate --template template.pptx --input content.txt")
    print("   # (ML layout selection is automatically enabled)")
    
    print("\n🎯 Expected Results:")
    print("   • 'Comparison of X vs Y' → two_column layout in top-3 recommendations")
    print("   • After ingesting 1k slides → vector DB shows >950 unique embeddings")
    print("   • 15% reduction in layout mismatch flags from Reviewer")

if __name__ == "__main__":
    main()