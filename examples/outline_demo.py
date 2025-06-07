"""
Demo script for outline generation.

Run with: python examples/outline_demo.py

Make sure to set OPENAI_API_KEY environment variable.
"""

import json
import os
from pathlib import Path

from openai import OpenAI

# Add the parent directory to the path so we can import open_lilli
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from open_lilli.content_processor import ContentProcessor
from open_lilli.models import GenerationConfig
from open_lilli.outline_generator import OutlineGenerator


def main():
    """Run the outline generation demo."""
    
    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: Please set OPENAI_API_KEY environment variable")
        print("You can create a .env file with: OPENAI_API_KEY=your_key_here")
        return
    
    # Initialize components
    client = OpenAI(api_key=api_key)
    content_processor = ContentProcessor()
    outline_generator = OutlineGenerator(client)
    
    # Load sample content
    content_path = Path(__file__).parent / "sample_content.txt"
    if not content_path.exists():
        print(f"Error: Sample content file not found at {content_path}")
        return
    
    print("Loading sample content...")
    text = content_processor.extract_text(content_path)
    print(f"Loaded {len(text)} characters of content")
    
    # Extract sections for better understanding
    sections = content_processor.extract_sections(text)
    print(f"Found {len(sections)} sections: {list(sections.keys())}")
    
    # Configure generation
    config = GenerationConfig(
        max_slides=12,
        max_bullets_per_slide=4,
        tone="professional",
        complexity_level="intermediate"
    )
    
    print("\nGenerating outline with OpenAI...")
    print(f"Config: {config.max_slides} slides max, {config.tone} tone")
    
    try:
        # Generate the outline
        outline = outline_generator.generate_outline(
            text=text,
            config=config,
            title="Q4 Business Review Presentation",
            language="en"
        )
        
        print(f"\n✅ Successfully generated outline!")
        print(f"Title: {outline.title}")
        if outline.subtitle:
            print(f"Subtitle: {outline.subtitle}")
        print(f"Language: {outline.language}")
        print(f"Slides: {outline.slide_count}")
        print(f"Target audience: {outline.target_audience}")
        print(f"Style guidance: {outline.style_guidance}")
        
        # Print slide details
        print("\n📄 Slide Breakdown:")
        for slide in outline.slides:
            print(f"\nSlide {slide.index + 1}: {slide.title}")
            print(f"  Type: {slide.slide_type}")
            if slide.bullets:
                print(f"  Bullets ({len(slide.bullets)}):")
                for bullet in slide.bullets:
                    print(f"    • {bullet}")
            if slide.image_query:
                print(f"  Image query: {slide.image_query}")
            if slide.chart_data:
                print(f"  Chart data: {slide.chart_data}")
            if slide.speaker_notes:
                print(f"  Notes: {slide.speaker_notes}")
        
        # Save the outline
        output_path = Path(__file__).parent / "generated_outline.json"
        with open(output_path, 'w') as f:
            json.dump(outline.model_dump(), f, indent=2)
        
        print(f"\n💾 Outline saved to: {output_path}")
        
        # Optional: Test refinement
        if input("\nWould you like to test outline refinement? (y/n): ").lower() == 'y':
            feedback = input("Enter feedback for refinement: ")
            
            print("Refining outline based on feedback...")
            refined_outline = outline_generator.refine_outline(outline, feedback)
            
            print(f"\n✅ Refined outline generated!")
            print(f"New title: {refined_outline.title}")
            print(f"New slide count: {refined_outline.slide_count}")
            
            # Save refined version
            refined_path = Path(__file__).parent / "refined_outline.json"
            with open(refined_path, 'w') as f:
                json.dump(refined_outline.model_dump(), f, indent=2)
            print(f"💾 Refined outline saved to: {refined_path}")
        
    except Exception as e:
        print(f"\n❌ Error generating outline: {e}")
        if "API" in str(e):
            print("This might be an API key or network issue.")
        return
    
    print("\n🎉 Demo completed successfully!")


if __name__ == "__main__":
    main()