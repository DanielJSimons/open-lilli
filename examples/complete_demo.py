"""
Complete demonstration of the Open Lilli pipeline.

This script shows how to use all components together to generate a presentation.
Run with: python examples/complete_demo.py

Requirements:
- Set OPENAI_API_KEY environment variable
- Have a template file available
"""

import json
import os
import sys
from pathlib import Path

from openai import OpenAI
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from open_lilli.content_generator import ContentGenerator
from open_lilli.content_processor import ContentProcessor
from open_lilli.models import GenerationConfig
from open_lilli.outline_generator import OutlineGenerator
from open_lilli.reviewer import Reviewer
from open_lilli.slide_assembler import SlideAssembler
from open_lilli.slide_planner import SlidePlanner
from open_lilli.template_parser import TemplateParser
from open_lilli.visual_generator import VisualGenerator

console = Console()


def create_sample_template(template_path: Path):
    """Create a basic sample template for demo purposes."""
    from pptx import Presentation
    
    prs = Presentation()
    prs.save(str(template_path))
    console.print(f"✅ Created sample template: {template_path}")


def create_sample_content(content_path: Path):
    """Create sample content for the demo."""
    content = """# Q4 Business Performance Review

## Executive Summary
Our organization delivered exceptional results in Q4, exceeding targets across all key performance indicators. Revenue grew by 28% year-over-year, reaching $15.2M, while maintaining strong profit margins of 22%. This performance was driven by successful product launches, market expansion, and operational excellence initiatives.

## Financial Performance
The financial results demonstrate the strength of our business model and execution capabilities:
- Total Revenue: $15.2M (28% YoY growth)
- Gross Profit Margin: 65% (up from 62% in Q3)
- Operating Profit Margin: 22% (industry-leading performance)
- Customer Acquisition Cost: Decreased by 15% through improved digital marketing
- Customer Lifetime Value: Increased by 31% due to enhanced retention programs

## Market Expansion Success
Our strategic expansion into new markets yielded significant returns:
- Launched in 3 new geographical regions (EMEA, APAC, LatAm)
- Established partnerships with 12 regional distributors
- Achieved 85% brand recognition in target markets within 6 months
- Generated $3.8M in revenue from new markets (25% of total revenue)

## Product Innovation Highlights
Innovation remained at the core of our competitive advantage:
- Released 2 major product updates with 47 new features
- Achieved 94% customer satisfaction rating (up from 89%)
- Reduced product development cycle time by 23%
- Filed 8 new patents, strengthening our IP portfolio
- Launched AI-powered analytics dashboard, increasing user engagement by 45%

## Operational Excellence
Efficiency improvements contributed significantly to our success:
- Implemented lean manufacturing processes, reducing waste by 18%
- Achieved 99.7% uptime across all systems
- Reduced average customer support response time to 2.3 hours
- Completed digital transformation of core processes
- Improved employee satisfaction scores to 4.6/5.0

## Strategic Partnerships
Key partnerships enhanced our market position and capabilities:
- Formed strategic alliance with TechCorp for AI integration
- Established distribution partnership covering 15 additional countries
- Launched co-development program with innovation labs
- Secured preferred vendor status with 3 Fortune 500 companies

## Future Outlook and 2025 Strategy
Looking ahead, we are well-positioned for continued growth:
- Projected revenue growth of 35-40% in 2025
- Plans to expand into 5 additional markets
- Investment of $4.2M in R&D for next-generation products
- Target to achieve carbon neutrality by end of 2025
- Goal to increase market share from 12% to 18% globally

The foundation built in Q4 sets us up for an exceptional 2025, with strong momentum across all business dimensions."""

    content_path.write_text(content)
    console.print(f"✅ Created sample content: {content_path}")


def main():
    """Run the complete demo."""
    console.print("[bold blue]🚀 Open Lilli Complete Demo[/bold blue]")
    console.print("This demo shows the full AI PowerPoint generation pipeline.\n")
    
    # Check API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        console.print("[red]❌ Error: OPENAI_API_KEY environment variable not set[/red]")
        console.print("Please set your OpenAI API key and try again.")
        return
    
    # Set up paths
    demo_dir = Path(__file__).parent
    template_path = demo_dir / "demo_template.pptx"
    content_path = demo_dir / "demo_content.txt"
    output_path = demo_dir / "demo_output.pptx"
    assets_dir = demo_dir / "demo_assets"
    
    # Create directories
    demo_dir.mkdir(exist_ok=True)
    assets_dir.mkdir(exist_ok=True)
    
    # Create sample files if they don't exist
    if not template_path.exists():
        create_sample_template(template_path)
    
    if not content_path.exists():
        create_sample_content(content_path)
    
    console.print(f"📁 Demo directory: {demo_dir}")
    console.print(f"📄 Content file: {content_path}")
    console.print(f"🎨 Template file: {template_path}")
    console.print(f"📊 Output file: {output_path}")
    console.print(f"🖼️  Assets directory: {assets_dir}\n")
    
    # Initialize OpenAI client
    try:
        openai_client = OpenAI(api_key=api_key)
        console.print("✅ OpenAI client initialized")
    except Exception as e:
        console.print(f"[red]❌ Failed to initialize OpenAI: {e}[/red]")
        return
    
    # Configuration
    config = GenerationConfig(
        max_slides=12,
        max_bullets_per_slide=4,
        tone="professional",
        complexity_level="intermediate",
        include_images=True,
        include_charts=True
    )
    
    console.print(f"⚙️  Configuration: {config.max_slides} slides max, {config.tone} tone\n")
    
    # Run the pipeline
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        
        results = {}
        
        try:
            # Step 1: Process Content
            task1 = progress.add_task("📖 Processing content...", total=None)
            processor = ContentProcessor()
            raw_text = processor.extract_text(content_path)
            sections = processor.extract_sections(raw_text)
            word_count = processor.get_word_count(raw_text)
            
            results["content"] = {
                "text_length": len(raw_text),
                "word_count": word_count,
                "sections": len(sections),
                "reading_time": processor.get_reading_time(raw_text)
            }
            
            console.print(f"✅ Processed {word_count} words in {len(sections)} sections")
            progress.remove_task(task1)
            
            # Step 2: Parse Template
            task2 = progress.add_task("🎨 Analyzing template...", total=None)
            template_parser = TemplateParser(str(template_path))
            template_info = template_parser.get_template_info()
            
            results["template"] = {
                "layouts": template_info["total_layouts"],
                "layout_types": template_info["available_layout_types"],
                "dimensions": template_info["slide_dimensions"]
            }
            
            console.print(f"✅ Analyzed template with {template_info['total_layouts']} layouts")
            progress.remove_task(task2)
            
            # Step 3: Generate Outline
            task3 = progress.add_task("🧠 Generating AI outline...", total=None)
            outline_generator = OutlineGenerator(openai_client)
            outline = outline_generator.generate_outline(
                raw_text, config=config, language="en"
            )
            
            results["outline"] = {
                "slide_count": outline.slide_count,
                "title": outline.title,
                "language": outline.language
            }
            
            console.print(f"✅ Generated outline: '{outline.title}' with {outline.slide_count} slides")
            progress.remove_task(task3)
            
            # Step 4: Plan Slides
            task4 = progress.add_task("📋 Planning slide layouts...", total=None)
            slide_planner = SlidePlanner(template_parser)
            planned_slides = slide_planner.plan_slides(outline, config)
            planning_summary = slide_planner.get_planning_summary(planned_slides)
            
            results["planning"] = planning_summary
            
            console.print(f"✅ Planned {len(planned_slides)} slides with layouts")
            progress.remove_task(task4)
            
            # Step 5: Generate Content
            task5 = progress.add_task("✍️  Generating slide content...", total=None)
            content_generator = ContentGenerator(openai_client)
            enhanced_slides = content_generator.generate_content(
                planned_slides, config, outline.style_guidance, "en"
            )
            content_stats = content_generator.get_content_statistics(enhanced_slides)
            
            results["content_generation"] = content_stats
            
            console.print(f"✅ Generated content for {len(enhanced_slides)} slides")
            progress.remove_task(task5)
            
            # Step 6: Generate Visuals
            task6 = progress.add_task("🎨 Creating visuals...", total=None)
            visual_generator = VisualGenerator(str(assets_dir), template_parser.palette)
            visuals = visual_generator.generate_visuals(enhanced_slides)
            visual_summary = visual_generator.get_visual_summary(visuals)
            
            results["visuals"] = visual_summary
            
            console.print(f"✅ Generated {visual_summary['total_charts']} charts and {visual_summary['total_images']} images")
            progress.remove_task(task6)
            
            # Step 7: Review Presentation
            task7 = progress.add_task("🔍 AI quality review...", total=None)
            reviewer = Reviewer(openai_client)
            feedback = reviewer.review_presentation(enhanced_slides)
            review_summary = reviewer.get_review_summary(feedback)
            
            results["review"] = {
                "score": review_summary["overall_score"],
                "feedback_count": len(feedback),
                "summary": review_summary["summary"]
            }
            
            console.print(f"✅ Review complete - Quality score: {review_summary['overall_score']}/10")
            progress.remove_task(task7)
            
            # Step 8: Assemble Presentation
            task8 = progress.add_task("🔧 Assembling PowerPoint...", total=None)
            slide_assembler = SlideAssembler(template_parser)
            
            # Validate slides
            validation_issues = slide_assembler.validate_slides_before_assembly(enhanced_slides)
            if validation_issues:
                console.print(f"[yellow]⚠️  {len(validation_issues)} validation warnings[/yellow]")
            
            # Create presentation
            final_path = slide_assembler.assemble(
                outline, enhanced_slides, visuals, output_path
            )
            assembly_stats = slide_assembler.get_assembly_statistics(enhanced_slides, visuals)
            
            results["assembly"] = assembly_stats
            
            console.print(f"✅ Assembled presentation: {final_path}")
            progress.remove_task(task8)
            
        except Exception as e:
            progress.stop()
            console.print(f"[red]❌ Pipeline failed: {e}[/red]")
            import traceback
            console.print(traceback.format_exc())
            return
    
    # Display comprehensive results
    console.print("\n[bold green]🎉 Demo Complete![/bold green]")
    
    # Results table
    table = Table(title="Pipeline Results Summary", show_header=True)
    table.add_column("Stage", style="bold cyan")
    table.add_column("Key Metrics", style="white")
    table.add_column("Details", style="dim")
    
    table.add_row(
        "Content Processing",
        f"{results['content']['word_count']} words",
        f"{results['content']['sections']} sections, {results['content']['reading_time']} min read"
    )
    
    table.add_row(
        "Template Analysis", 
        f"{results['template']['layouts']} layouts",
        f"Types: {', '.join(results['template']['layout_types'][:3])}..."
    )
    
    table.add_row(
        "AI Outline",
        f"{results['outline']['slide_count']} slides planned",
        f"Title: {results['outline']['title']}"
    )
    
    table.add_row(
        "Content Generation",
        f"{results['content_generation']['total_bullets']} bullet points",
        f"{results['content_generation']['total_words']} words total"
    )
    
    table.add_row(
        "Visual Generation",
        f"{results['visuals']['total_charts']} charts, {results['visuals']['total_images']} images",
        f"Assets in: {assets_dir}"
    )
    
    table.add_row(
        "AI Review",
        f"Score: {results['review']['score']}/10",
        f"{results['review']['feedback_count']} feedback items"
    )
    
    table.add_row(
        "Final Assembly",
        f"{results['assembly']['total_slides']} slides",
        f"Output: {output_path.name}"
    )
    
    console.print(table)
    
    # Show top feedback items
    if feedback:
        console.print(f"\n[bold blue]📝 Top Quality Feedback[/bold blue]")
        prioritized = reviewer.prioritize_feedback(feedback)
        for i, item in enumerate(prioritized[:3]):
            severity_color = {"critical": "red", "high": "orange3", "medium": "yellow", "low": "green"}.get(item.severity, "white")
            console.print(f"{i+1}. [bold {severity_color}]{item.severity.upper()}[/bold {severity_color}]: {item.message}")
            if item.suggestion:
                console.print(f"   💡 Suggestion: {item.suggestion}")
    
    # File outputs
    console.print(f"\n[bold green]📁 Generated Files:[/bold green]")
    console.print(f"• PowerPoint: [link={output_path}]{output_path}[/link]")
    console.print(f"• Assets: [link={assets_dir}]{assets_dir}[/link]")
    
    if assets_dir.exists():
        asset_files = list(assets_dir.glob("*"))
        if asset_files:
            console.print(f"• {len(asset_files)} visual assets created")
    
    # Save results summary
    results_file = demo_dir / "demo_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    console.print(f"• Results summary: {results_file}")
    
    console.print(f"\n[green]🎯 Demo completed successfully![/green]")
    console.print(f"Open {output_path} in PowerPoint to view your AI-generated presentation!")


if __name__ == "__main__":
    main()