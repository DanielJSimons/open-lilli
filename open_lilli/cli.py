"""Command-line interface for Open Lilli presentation generator."""

import os
import sys
from pathlib import Path
from typing import Optional

import click
from dotenv import load_dotenv
from openai import OpenAI
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from . import __version__
from .content_generator import ContentGenerator
from .content_processor import ContentProcessor
from .models import GenerationConfig
from .outline_generator import OutlineGenerator
from .reviewer import Reviewer
from .slide_assembler import SlideAssembler
from .slide_planner import SlidePlanner
from .template_parser import TemplateParser
from .visual_generator import VisualGenerator

# Load environment variables
load_dotenv()

console = Console()


@click.group()
@click.version_option(version=__version__)
def cli():
    """
    Open Lilli - AI-powered PowerPoint generation tool.
    
    Generate professional presentations from text using AI and templates.
    """
    pass


@cli.command()
@click.option(
    "--template", "-t",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to PowerPoint template (.pptx) file"
)
@click.option(
    "--input", "-i", "input_path",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to input content file (.txt, .md)"
)
@click.option(
    "--output", "-o",
    default="presentation.pptx",
    type=click.Path(path_type=Path),
    help="Output presentation file path"
)
@click.option(
    "--lang", "-l",
    default="en",
    help="Language code for the presentation (e.g., en, es, fr, de)"
)
@click.option(
    "--slides",
    default=15,
    type=int,
    help="Maximum number of slides to generate"
)
@click.option(
    "--tone",
    default="professional",
    type=click.Choice(["professional", "casual", "formal", "friendly"]),
    help="Tone for content generation"
)
@click.option(
    "--complexity",
    default="intermediate",
    type=click.Choice(["basic", "intermediate", "advanced"]),
    help="Complexity level of content"
)
@click.option(
    "--no-images",
    is_flag=True,
    help="Disable image generation/sourcing"
)
@click.option(
    "--no-charts",
    is_flag=True,
    help="Disable chart generation"
)
@click.option(
    "--review/--no-review",
    default=True,
    help="Enable/disable AI review and feedback"
)
@click.option(
    "--assets-dir",
    default="assets",
    type=click.Path(path_type=Path),
    help="Directory for generated assets"
)
@click.option(
    "--model",
    default="gpt-4",
    help="OpenAI model to use"
)
@click.option(
    "--verbose", "-v",
    is_flag=True,
    help="Enable verbose output"
)
def generate(
    template: Path,
    input_path: Path,
    output: Path,
    lang: str,
    slides: int,
    tone: str,
    complexity: str,
    no_images: bool,
    no_charts: bool,
    review: bool,
    assets_dir: Path,
    model: str,
    verbose: bool
):
    """Generate a PowerPoint presentation from input content."""
    
    # Set up logging level
    if verbose:
        import logging
        logging.basicConfig(level=logging.DEBUG)
    
    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        console.print("[red]Error: OPENAI_API_KEY environment variable not set[/red]")
        console.print("Please set your OpenAI API key in the .env file or environment")
        sys.exit(1)
    
    # Initialize OpenAI client
    try:
        openai_client = OpenAI(api_key=api_key)
    except Exception as e:
        console.print(f"[red]Error initializing OpenAI client: {e}[/red]")
        sys.exit(1)
    
    # Create assets directory
    assets_dir.mkdir(exist_ok=True)
    
    # Create generation config
    config = GenerationConfig(
        max_slides=slides,
        tone=tone,
        complexity_level=complexity,
        include_images=not no_images,
        include_charts=not no_charts
    )
    
    console.print(f"[bold green]🎯 Generating presentation from {input_path}[/bold green]")
    console.print(f"Template: {template}")
    console.print(f"Output: {output}")
    console.print(f"Language: {lang}")
    console.print(f"Max slides: {slides}")
    console.print(f"Config: {config.tone} tone, {config.complexity_level} complexity")
    console.print()
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        
        try:
            # Step 1: Process content
            task = progress.add_task("📄 Processing input content...", total=None)
            content_processor = ContentProcessor()
            raw_text = content_processor.extract_text(input_path)
            console.print(f"✅ Processed {len(raw_text)} characters of content")
            progress.remove_task(task)
            
            # Step 2: Parse template
            task = progress.add_task("🎨 Analyzing template...", total=None)
            template_parser = TemplateParser(str(template))
            template_info = template_parser.get_template_info()
            console.print(f"✅ Found {template_info['total_layouts']} layouts in template")
            progress.remove_task(task)
            
            # Step 3: Generate outline
            task = progress.add_task("🧠 Generating outline with AI...", total=None)
            outline_generator = OutlineGenerator(openai_client, model=model)
            outline = outline_generator.generate_outline(
                raw_text, config=config, language=lang
            )
            console.print(f"✅ Generated outline with {outline.slide_count} slides")
            progress.remove_task(task)
            
            # Step 4: Plan slides
            task = progress.add_task("📋 Planning slide layouts...", total=None)
            slide_planner = SlidePlanner(template_parser)
            planned_slides = slide_planner.plan_slides(outline, config)
            planning_summary = slide_planner.get_planning_summary(planned_slides)
            console.print(f"✅ Planned {len(planned_slides)} slides")
            progress.remove_task(task)
            
            # Step 5: Generate content
            task = progress.add_task("✍️ Generating slide content...", total=None)
            content_generator = ContentGenerator(openai_client, model=model)
            enhanced_slides = content_generator.generate_content(
                planned_slides, config, outline.style_guidance, lang
            )
            content_stats = content_generator.get_content_statistics(enhanced_slides)
            console.print(f"✅ Generated content for {len(enhanced_slides)} slides")
            progress.remove_task(task)
            
            # Step 6: Generate visuals
            task = progress.add_task("🎨 Creating visuals...", total=None)
            visual_generator = VisualGenerator(
                str(assets_dir), template_parser.palette
            )
            visuals = visual_generator.generate_visuals(enhanced_slides)
            visual_summary = visual_generator.get_visual_summary(visuals)
            console.print(f"✅ Generated {visual_summary['total_charts']} charts and {visual_summary['total_images']} images")
            progress.remove_task(task)
            
            # Step 7: Review (if enabled)
            feedback = []
            if review:
                task = progress.add_task("🔍 Reviewing presentation quality...", total=None)
                reviewer = Reviewer(openai_client, model=model)
                feedback = reviewer.review_presentation(enhanced_slides)
                review_summary = reviewer.get_review_summary(feedback)
                console.print(f"✅ Review complete - Score: {review_summary['overall_score']}/10")
                progress.remove_task(task)
            
            # Step 8: Assemble presentation
            task = progress.add_task("🔧 Assembling PowerPoint...", total=None)
            slide_assembler = SlideAssembler(template_parser)
            
            # Validate before assembly
            validation_issues = slide_assembler.validate_slides_before_assembly(enhanced_slides)
            if validation_issues:
                console.print(f"[yellow]⚠️  Found {len(validation_issues)} validation issues[/yellow]")
                if verbose:
                    for issue in validation_issues:
                        console.print(f"   • {issue}")
            
            # Create the presentation
            output_path = slide_assembler.assemble(
                outline, enhanced_slides, visuals, output
            )
            assembly_stats = slide_assembler.get_assembly_statistics(enhanced_slides, visuals)
            console.print(f"✅ Assembled presentation: {output_path}")
            progress.remove_task(task)
            
        except Exception as e:
            progress.stop()
            console.print(f"[red]❌ Error during generation: {e}[/red]")
            if verbose:
                import traceback
                console.print(traceback.format_exc())
            sys.exit(1)
    
    # Display results summary
    console.print("\n[bold green]🎉 Generation Complete![/bold green]")
    
    # Create summary table
    table = Table(title="Presentation Summary")
    table.add_column("Metric", style="bold")
    table.add_column("Value")
    
    table.add_row("Output File", str(output_path))
    table.add_row("Total Slides", str(assembly_stats["total_slides"]))
    table.add_row("Total Bullets", str(content_stats["total_bullets"]))
    table.add_row("Total Words", str(content_stats["total_words"]))
    table.add_row("Charts Generated", str(visual_summary["total_charts"]))
    table.add_row("Images Sourced", str(visual_summary["total_images"]))
    
    if review and feedback:
        table.add_row("Review Score", f"{review_summary['overall_score']}/10")
        table.add_row("Feedback Items", str(len(feedback)))
    
    console.print(table)
    
    # Show feedback if review was enabled
    if review and feedback:
        console.print(f"\n[bold blue]📝 Review Feedback ({len(feedback)} items)[/bold blue]")
        
        prioritized_feedback = reviewer.prioritize_feedback(feedback)
        for i, item in enumerate(prioritized_feedback[:5]):  # Show top 5
            severity_color = {
                "critical": "red",
                "high": "orange3",
                "medium": "yellow",
                "low": "green"
            }.get(item.severity, "white")
            
            console.print(f"{i+1}. [bold {severity_color}]{item.severity.upper()}[/bold {severity_color}] "
                         f"(Slide {item.slide_index + 1 if item.slide_index >= 0 else 'General'}): {item.message}")
            if item.suggestion:
                console.print(f"   💡 {item.suggestion}")
        
        if len(feedback) > 5:
            console.print(f"   ... and {len(feedback) - 5} more items")
    
    console.print(f"\n[green]🎯 Presentation saved to: {output_path}[/green]")


@cli.command()
@click.argument("template_path", type=click.Path(exists=True, path_type=Path))
def analyze_template(template_path: Path):
    """Analyze a PowerPoint template and show layout information."""
    
    console.print(f"[bold]Analyzing template: {template_path}[/bold]\n")
    
    try:
        template_parser = TemplateParser(str(template_path))
        template_info = template_parser.get_template_info()
        
        # Basic info
        console.print(f"📁 File: {template_info['template_path']}")
        console.print(f"📐 Dimensions: {template_info['slide_dimensions']['width_inches']:.1f}\" × "
                     f"{template_info['slide_dimensions']['height_inches']:.1f}\"")
        console.print(f"🎨 Layouts: {template_info['total_layouts']}")
        console.print()
        
        # Layout details
        console.print("[bold]Available Layouts:[/bold]")
        for i, layout_type in enumerate(template_info['available_layout_types']):
            layout_index = template_info['layout_mapping'][layout_type]
            analysis = template_parser.analyze_layout_placeholders(layout_type)
            
            console.print(f"{i+1}. [cyan]{layout_type}[/cyan] (index {layout_index})")
            console.print(f"   • {analysis['total_placeholders']} placeholders")
            console.print(f"   • Has title: {analysis['has_title']}")
            console.print(f"   • Has content: {analysis['has_content']}")
            console.print(f"   • Has image: {analysis['has_image']}")
            console.print(f"   • Has chart: {analysis['has_chart']}")
        
        console.print()
        
        # Theme colors
        console.print("[bold]Theme Colors:[/bold]")
        for color_name, color_value in template_info['theme_colors'].items():
            console.print(f"• {color_name}: {color_value}")
        
    except Exception as e:
        console.print(f"[red]Error analyzing template: {e}[/red]")
        sys.exit(1)


@cli.command()
@click.argument("presentation_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--model",
    default="gpt-4",
    help="OpenAI model to use for review"
)
def review(presentation_path: Path, model: str):
    """Review an existing presentation and provide feedback."""
    
    # This is a simplified version - in a full implementation,
    # we'd need to extract text from the existing PowerPoint file
    console.print(f"[yellow]Note: Direct .pptx review not yet implemented[/yellow]")
    console.print(f"To review a presentation:")
    console.print(f"1. Use the --review flag with the generate command")
    console.print(f"2. Or extract content to text and regenerate with review")


@cli.command()
def setup():
    """Set up Open Lilli configuration."""
    
    console.print("[bold]🚀 Open Lilli Setup[/bold]\n")
    
    # Check for .env file
    env_file = Path(".env")
    if env_file.exists():
        console.print("✅ Found existing .env file")
    else:
        console.print("📝 Creating .env file...")
        env_content = """# OpenAI API Configuration
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4
OPENAI_TEMPERATURE=0.3

# Image Generation
UNSPLASH_ACCESS_KEY=your_unsplash_key_here
USE_DALLE=false

# Logging
LOG_LEVEL=INFO
"""
        env_file.write_text(env_content)
        console.print("✅ Created .env file")
    
    # Check API key
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "your_openai_api_key_here":
        console.print("[yellow]⚠️  Please set your OpenAI API key in the .env file[/yellow]")
        console.print("Get your API key from: https://platform.openai.com/api-keys")
    else:
        console.print("✅ OpenAI API key found")
        
        # Test API connection
        try:
            client = OpenAI(api_key=api_key)
            # Make a simple test request
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=5
            )
            console.print("✅ OpenAI API connection successful")
        except Exception as e:
            console.print(f"[red]❌ OpenAI API test failed: {e}[/red]")
    
    # Create directories
    for directory in ["assets", "templates", "examples"]:
        Path(directory).mkdir(exist_ok=True)
        console.print(f"✅ Created {directory}/ directory")
    
    console.print(f"\n[green]🎯 Setup complete! You're ready to generate presentations.[/green]")
    console.print(f"Try: [bold]ai-ppt generate --help[/bold]")


def main():
    """Main entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()