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
from .regeneration_manager import RegenerationManager
from .template_ingestion import TemplateIngestionPipeline
from .models import VectorStoreConfig, ContentFitConfig

# Load environment variables
load_dotenv()

console = Console()


def _configure_logging(debug: bool, log_file: Optional[Path]):
    """Configure root logging handlers and level."""
    import logging

    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file, mode="w"))

    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=handlers,
    )


@click.group()
@click.version_option(version=__version__)
@click.option("--debug", is_flag=True, help="Show stack traces on error")
@click.option("--log-file", type=click.Path(path_type=Path), help="Write debug logs to file")
@click.pass_context
def cli(ctx: click.Context, debug: bool, log_file: Optional[Path]):
    """
    Open Lilli - AI-powered PowerPoint generation tool.

    Generate professional presentations from text using AI and templates.
    """
    _configure_logging(debug, log_file)
    ctx.ensure_object(dict)
    ctx.obj["debug"] = debug

    ctx.obj["log_file"] = log_file

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
    "--auto-refine",
    is_flag=True,
    help="Automatically refine presentation until quality gates pass"
)
@click.option(
    "--max-iterations",
    default=3,
    type=int,
    help="Maximum number of refinement iterations for auto-refine"
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
@click.pass_context
def generate(ctx: click.Context, 
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
    auto_refine: bool,
    max_iterations: int,
    assets_dir: Path,
    model: str,
    verbose: bool
):
    """Generate a PowerPoint presentation from input content."""

    debug = ctx.obj.get("debug", False)
    if verbose:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        console.print("[red]Error: OPENAI_API_KEY environment variable not set[/red]")
        console.print("Hint: run 'open-lilli setup' or set OPENAI_API_KEY in your environment")
        sys.exit(1)
    
    # Initialize OpenAI client
    try:
        openai_client = OpenAI(api_key=api_key)
    except Exception as e:
        console.print(f"[red]Error initializing OpenAI client: {e}[/red]")
        console.print("Hint: verify your API key and network connection")
        sys.exit(1)
    
    # Create assets directory
    assets_dir.mkdir(exist_ok=True)
    
    # Create generation config
    config = GenerationConfig(
        max_slides=slides,
        tone=tone,
        complexity_level=complexity,
        include_images=not no_images,
        include_charts=not no_charts,
        max_iterations=max_iterations
    )
    
    console.print(f"[bold green]🎯 Generating presentation from {input_path}[/bold green]")
    console.print(f"Template: {template}")
    console.print(f"Output: {output}")
    console.print(f"Language: {lang}")
    console.print(f"Max slides: {slides}")
    console.print(f"Config: {config.tone} tone, {config.complexity_level} complexity")
    if auto_refine:
        console.print(f"Auto-refine: enabled (max {max_iterations} iterations)")
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
            
            # Step 4: Plan slides with ML-assisted layout selection
            task = progress.add_task("📋 Planning slide layouts...", total=None)
            
            # Create vector store config for ML layout recommendations
            vector_config = VectorStoreConfig()
            
            # Create content fit config for dynamic content optimization
            content_fit_config = ContentFitConfig()
            
            slide_planner = SlidePlanner(
                template_parser, 
                openai_client=openai_client,
                vector_config=vector_config,
                enable_ml_layouts=True,  # Enable ML recommendations
                content_fit_config=content_fit_config  # Enable content fit optimization
            )
            planned_slides = slide_planner.plan_slides(outline, config)
            planning_summary = slide_planner.get_planning_summary(planned_slides)
            
            # Count ML vs rule-based recommendations
            ml_count = sum(1 for slide in planned_slides 
                          if hasattr(slide, '__dict__') and 
                             'ml_recommendation' in slide.__dict__ and
                             not slide.__dict__['ml_recommendation'].fallback_used)
            
            console.print(f"✅ Planned {len(planned_slides)} slides ({ml_count} ML recommendations)")
            progress.remove_task(task)
            
            # Step 5: Generate content
            task = progress.add_task("✍️ Generating slide content...", total=None)
            content_generator = ContentGenerator(openai_client, model=model, template_parser=template_parser)
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
            
            # Step 7: Review and Auto-refine (if enabled)
            feedback = []
            quality_result = None
            iteration_count = 0
            
            if review or auto_refine:
                task = progress.add_task("🔍 Reviewing presentation quality...", total=None)
                reviewer = Reviewer(openai_client, model=model)
                
                if auto_refine:
                    # Auto-refine mode: iterate until quality gates pass or max iterations reached
                    current_slides = enhanced_slides
                    
                    while iteration_count < config.max_iterations:
                        iteration_count += 1
                        
                        # Review with quality gates
                        feedback, quality_result = reviewer.review_presentation(
                            current_slides, 
                            include_quality_gates=True
                        )
                        
                        review_summary = reviewer.get_review_summary(feedback)
                        console.print(f"✅ Iteration {iteration_count} - Score: {review_summary['overall_score']}/10, "
                                    f"Gates: {quality_result.passed_gates}/{quality_result.total_gates}")
                        
                        # Check if quality gates pass
                        if quality_result.status == "pass":
                            console.print(f"🎉 Quality gates passed on iteration {iteration_count}!")
                            enhanced_slides = current_slides
                            break
                        
                        # If not the last iteration, regenerate failing slides
                        if iteration_count < config.max_iterations:
                            # Identify slides that need regeneration based on feedback
                            failing_slide_indices = []
                            for f in feedback:
                                if f.severity in ["critical", "high"] and f.slide_index >= 0:
                                    if f.slide_index not in failing_slide_indices:
                                        failing_slide_indices.append(f.slide_index)
                            
                            if failing_slide_indices:
                                console.print(f"🔄 Regenerating {len(failing_slide_indices)} slides based on feedback...")
                                
                                # Create feedback string for regeneration
                                feedback_text = "Address these issues: " + "; ".join([
                                    f"Slide {f.slide_index + 1}: {f.message}" 
                                    for f in feedback 
                                    if f.severity in ["critical", "high"] and f.slide_index >= 0
                                ])
                                
                                # Initialize regeneration manager
                                regeneration_manager = RegenerationManager(
                                    template_parser, content_generator, slide_assembler
                                )
                                
                                # Create temporary PPTX for regeneration
                                temp_output = output.parent / f"temp_iteration_{iteration_count}.pptx"
                                slide_assembler.assemble(outline, current_slides, visuals, temp_output)
                                
                                # Regenerate failing slides
                                try:
                                    updated_outline, updated_slides = regeneration_manager.coordinate_selective_regeneration(
                                        temp_output, failing_slide_indices, config, feedback_text, lang
                                    )
                                    current_slides = updated_slides
                                    
                                    # Clean up temp file
                                    if temp_output.exists():
                                        temp_output.unlink()
                                        
                                except Exception as e:
                                    console.print(f"[yellow]⚠️  Regeneration failed: {e}[/yellow]")
                                    break
                            else:
                                # No specific slides to regenerate, break the loop
                                break
                        else:
                            # Last iteration, use current slides
                            enhanced_slides = current_slides
                    
                    if iteration_count >= config.max_iterations and quality_result.status != "pass":
                        console.print(f"[yellow]⚠️  Reached maximum iterations ({config.max_iterations}) without passing all quality gates[/yellow]")
                        console.print(f"Final status: {quality_result.passed_gates}/{quality_result.total_gates} gates passed")
                
                else:
                    # Regular review mode
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
            if verbose or debug:
                import traceback
                console.print(traceback.format_exc())
            else:
                console.print("Run again with --debug or --log-file for details")
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
        review_summary = reviewer.get_review_summary(feedback)  # Ensure we have the latest summary
        table.add_row("Review Score", f"{review_summary['overall_score']}/10")
        table.add_row("Feedback Items", str(len(feedback)))
    
    if auto_refine:
        table.add_row("Refinement Iterations", str(iteration_count))
        if quality_result:
            table.add_row("Quality Gates", f"{quality_result.passed_gates}/{quality_result.total_gates} passed")
            table.add_row("Final Status", quality_result.status.title())
    
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
@click.option(
    "--template", "-t",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to PowerPoint template (.pptx) file"
)
@click.option(
    "--input-pptx", "-p",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to existing presentation (.pptx) file to regenerate"
)
@click.option(
    "--slides", "-s",
    required=True,
    help="Comma-separated slide indices to regenerate (0-based, e.g., '1,3,5')"
)
@click.option(
    "--output", "-o",
    default="regenerated.pptx",
    type=click.Path(path_type=Path),
    help="Output presentation file path"
)
@click.option(
    "--feedback", "-f",
    help="Specific feedback to apply during regeneration"
)
@click.option(
    "--lang", "-l",
    default="en",
    help="Language code for the presentation (e.g., en, es, fr, de)"
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
@click.pass_context
def regenerate(ctx: click.Context,
    template: Path,
    input_pptx: Path,
    slides: str,
    output: Path,
    feedback: Optional[str],
    lang: str,
    tone: str,
    complexity: str,
    no_images: bool,
    no_charts: bool,
    assets_dir: Path,
    model: str,
    verbose: bool
):
    """Regenerate specific slides in an existing PowerPoint presentation."""

    debug = ctx.obj.get("debug", False)
    if verbose:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        console.print("[red]Error: OPENAI_API_KEY environment variable not set[/red]")
        console.print("Hint: run 'open-lilli setup' or set OPENAI_API_KEY in your environment")
        sys.exit(1)
    
    # Parse slide indices
    try:
        target_indices = [int(idx.strip()) for idx in slides.split(",")]
    except ValueError:
        console.print("[red]Error: Invalid slide indices format. Use comma-separated numbers (e.g., '1,3,5')[/red]")
        console.print("Hint: provide indices without spaces or brackets")
        sys.exit(1)
    
    # Initialize OpenAI client
    try:
        openai_client = OpenAI(api_key=api_key)
    except Exception as e:
        console.print(f"[red]Error initializing OpenAI client: {e}[/red]")
        console.print("Hint: verify your API key and network connection")
        sys.exit(1)
    
    # Create assets directory
    assets_dir.mkdir(exist_ok=True)
    
    # Create generation config
    config = GenerationConfig(
        max_slides=50,  # Allow larger number for regeneration
        tone=tone,
        complexity_level=complexity,
        include_images=not no_images,
        include_charts=not no_charts
    )
    
    console.print(f"[bold green]🔄 Regenerating slides {target_indices} from {input_pptx}[/bold green]")
    console.print(f"Template: {template}")
    console.print(f"Output: {output}")
    console.print(f"Language: {lang}")
    if feedback:
        console.print(f"Feedback: {feedback[:100]}...")
    console.print()
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        
        try:
            # Step 1: Parse template
            task = progress.add_task("🎨 Analyzing template...", total=None)
            template_parser = TemplateParser(str(template))
            template_info = template_parser.get_template_info()
            console.print(f"✅ Found {template_info['total_layouts']} layouts in template")
            progress.remove_task(task)
            
            # Step 2: Initialize components
            task = progress.add_task("🔧 Initializing components...", total=None)
            content_generator = ContentGenerator(openai_client, model=model, template_parser=template_parser)
            slide_assembler = SlideAssembler(template_parser)
            regeneration_manager = RegenerationManager(template_parser, content_generator, slide_assembler)
            progress.remove_task(task)
            
            # Step 3: Extract existing slides
            task = progress.add_task("📄 Extracting existing slides...", total=None)
            outline, all_slides = regeneration_manager.extract_slides_from_presentation(input_pptx)
            console.print(f"✅ Extracted {len(all_slides)} slides from existing presentation")
            progress.remove_task(task)
            
            # Step 4: Validate slide indices
            task = progress.add_task("🔍 Validating slide indices...", total=None)
            try:
                target_slides = regeneration_manager.select_slides_for_regeneration(all_slides, target_indices)
                console.print(f"✅ Selected {len(target_slides)} slides for regeneration")
            except ValueError as e:
                progress.remove_task(task)
                console.print(f"[red]❌ {e}[/red]")
                console.print("Hint: ensure slide indices exist in the presentation")
                sys.exit(1)
            progress.remove_task(task)
            
            # Step 5: Regenerate specific slides
            task = progress.add_task("✍️ Regenerating slide content...", total=None)
            updated_outline, updated_slides = regeneration_manager.coordinate_selective_regeneration(
                input_pptx, target_indices, config, feedback, lang
            )
            console.print(f"✅ Regenerated {len(target_indices)} slides")
            progress.remove_task(task)
            
            # Step 6: Generate visuals for updated slides (if needed)
            visuals = {}
            if config.include_images or config.include_charts:
                task = progress.add_task("🎨 Generating visuals for updated slides...", total=None)
                visual_generator = VisualGenerator(str(assets_dir), template_parser.palette)
                # Only generate visuals for the updated slides
                updated_slide_plans = [updated_slides[i] for i in target_indices]
                target_visuals = visual_generator.generate_visuals(updated_slide_plans)
                # Map visuals back to their original indices
                for i, slide_plan in enumerate(updated_slide_plans):
                    if slide_plan.index in target_visuals:
                        visuals[slide_plan.index] = target_visuals[slide_plan.index]
                visual_summary = visual_generator.get_visual_summary(target_visuals)
                console.print(f"✅ Generated {visual_summary['total_charts']} charts and {visual_summary['total_images']} images")
                progress.remove_task(task)
            
            # Step 7: Patch existing presentation
            task = progress.add_task("🔧 Patching presentation...", total=None)
            output_path = slide_assembler.patch_existing_presentation(
                input_pptx, updated_slides, target_indices, visuals, output
            )
            console.print(f"✅ Patched presentation: {output_path}")
            progress.remove_task(task)
            
        except Exception as e:
            progress.stop()
            console.print(f"[red]❌ Error during regeneration: {e}[/red]")
            if verbose or debug:
                import traceback
                console.print(traceback.format_exc())
            else:
                console.print("Run again with --debug or --log-file for details")
            sys.exit(1)
    
    # Display results summary
    console.print("\n[bold green]🎉 Regeneration Complete![/bold green]")
    
    # Create summary table
    table = Table(title="Regeneration Summary")
    table.add_column("Metric", style="bold")
    table.add_column("Value")
    
    table.add_row("Output File", str(output_path))
    table.add_row("Slides Regenerated", str(len(target_indices)))
    table.add_row("Target Indices", ", ".join(map(str, target_indices)))
    table.add_row("Total Slides", str(len(updated_slides)))
    if feedback:
        table.add_row("Feedback Applied", "Yes")
    
    console.print(table)
    console.print(f"\n[green]🎯 Updated presentation saved to: {output_path}[/green]")


@cli.command()
@click.pass_context
@click.argument("template_path", type=click.Path(exists=True, path_type=Path))
def analyze_template(ctx: click.Context, template_path: Path):
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
        if ctx.obj.get("debug", False):
            import traceback
            console.print(traceback.format_exc())
        else:
            console.print("Run again with --debug or --log-file for details")
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


@cli.command()
@click.option(
    "--pptx",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to PowerPoint file or directory containing .pptx files to ingest"
)
@click.option(
    "--template", "-t",
    type=click.Path(exists=True, path_type=Path),
    help="Optional template file for layout detection"
)
@click.option(
    "--vector-store",
    default="layouts.vec",
    type=click.Path(path_type=Path),
    help="Path to vector store file"
)
@click.option(
    "--embedding-model",
    default="text-embedding-3-small",
    help="OpenAI embedding model to use"
)
@click.option(
    "--batch-size",
    default=50,
    type=int,
    help="Number of slides to process in each batch"
)
@click.option(
    "--verbose", "-v",
    is_flag=True,
    help="Enable verbose output"
)
@click.pass_context
def ingest(ctx: click.Context,
    pptx: Path,
    template: Optional[Path],
    vector_store: Path,
    embedding_model: str,
    batch_size: int,
    verbose: bool
):
    """Ingest PowerPoint presentations into the ML training corpus."""

    debug = ctx.obj.get("debug", False)
    if verbose:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        console.print("[red]Error: OPENAI_API_KEY environment variable not set[/red]")
        console.print("Hint: run 'open-lilli setup' or set OPENAI_API_KEY in your environment")
        sys.exit(1)
    
    # Initialize OpenAI client
    try:
        openai_client = OpenAI(api_key=api_key)
    except Exception as e:
        console.print(f"[red]Error initializing OpenAI client: {e}[/red]")
        console.print("Hint: verify your API key and network connection")
        sys.exit(1)
    
    # Create vector store configuration
    vector_config = VectorStoreConfig(
        embedding_model=embedding_model,
        vector_store_path=str(vector_store)
    )
    
    console.print(f"[bold green]📚 Ingesting presentations into ML corpus[/bold green]")
    console.print(f"Source: {pptx}")
    console.print(f"Vector store: {vector_store}")
    console.print(f"Embedding model: {embedding_model}")
    if template:
        console.print(f"Template: {template}")
    console.print()
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        
        try:
            # Initialize ingestion pipeline
            task = progress.add_task("🔧 Initializing ingestion pipeline...", total=None)
            ingestion_pipeline = TemplateIngestionPipeline(openai_client, vector_config)
            progress.remove_task(task)
            
            # Determine if input is file or directory
            if pptx.is_file():
                # Single file ingestion
                task = progress.add_task(f"📄 Processing {pptx.name}...", total=None)
                result = ingestion_pipeline.ingest_single_file(pptx, template)
                progress.remove_task(task)
                
                console.print(f"✅ Processed single file: {pptx.name}")
                
            elif pptx.is_dir():
                # Directory ingestion
                task = progress.add_task("📁 Processing directory...", total=None)
                result = ingestion_pipeline.ingest_directory(pptx, template)
                progress.remove_task(task)
                
                console.print(f"✅ Processed directory: {pptx}")
                
            else:
                console.print(f"[red]Error: {pptx} is neither a file nor directory[/red]")
                console.print("Hint: check the path and try again")
                sys.exit(1)
            
        except Exception as e:
            progress.stop()
            console.print(f"[red]❌ Error during ingestion: {e}[/red]")
            if verbose or debug:
                import traceback
                console.print(traceback.format_exc())
            else:
                console.print("Run again with --debug or --log-file for details")
            sys.exit(1)
    
    # Display results summary
    console.print("\n[bold green]📊 Ingestion Complete![/bold green]")
    
    # Create summary table
    table = Table(title="Ingestion Results")
    table.add_column("Metric", style="bold")
    table.add_column("Value")
    
    table.add_row("Total Slides Processed", str(result.total_slides_processed))
    table.add_row("Successful Embeddings", str(result.successful_embeddings))
    table.add_row("Failed Embeddings", str(result.failed_embeddings))
    table.add_row("Success Rate", f"{result.success_rate:.1f}%")
    table.add_row("Unique Layouts Found", str(result.unique_layouts_found))
    table.add_row("Vector Store Size", str(result.vector_store_size))
    table.add_row("Processing Time", f"{result.processing_time_seconds:.1f}s")
    
    console.print(table)
    
    # Show errors if any
    if result.errors:
        console.print(f"\n[yellow]⚠️  Errors Encountered ({len(result.errors)})[/yellow]")
        for i, error in enumerate(result.errors[:5]):  # Show first 5 errors
            console.print(f"{i+1}. {error}")
        if len(result.errors) > 5:
            console.print(f"   ... and {len(result.errors) - 5} more errors")
    
    # Show corpus summary
    corpus_summary = ingestion_pipeline.get_corpus_summary()
    if "total_embeddings" in corpus_summary:
        console.print(f"\n[bold blue]📈 Updated Corpus Summary[/bold blue]")
        console.print(f"Total embeddings: {corpus_summary['total_embeddings']}")
        console.print(f"Unique layouts: {corpus_summary['unique_layouts']}")
        console.print(f"Layout distribution: {corpus_summary['layout_distribution']}")
    
    console.print(f"\n[green]🎯 Vector store updated: {vector_store}[/green]")
    console.print("ML-assisted layout recommendations are now ready for use!")


def main():
    """Main entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()