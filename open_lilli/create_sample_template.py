"""Create a sample PowerPoint template for testing."""

from pathlib import Path
from pptx import Presentation
from pptx.util import Inches
from pptx.enum.text import PP_ALIGN


def create_sample_template(output_path: str) -> None:
    """
    Create a basic PowerPoint template for testing.
    
    Args:
        output_path: Path where to save the template
    """
    prs = Presentation()
    
    # Get the slide layouts (these come with the default template)
    title_layout = prs.slide_layouts[0]  # Title slide
    content_layout = prs.slide_layouts[1]  # Title and content
    
    # We'll use the existing layouts but could customize them further
    # For a real template, you'd design custom layouts in PowerPoint
    
    # Just save the presentation with default layouts
    # This gives us a basic template to work with
    prs.save(output_path)
    print(f"Sample template created at: {output_path}")


if __name__ == "__main__":
    output_dir = Path(__file__).parent.parent / "templates"
    output_dir.mkdir(exist_ok=True)
    
    template_path = output_dir / "sample_template.pptx"
    create_sample_template(str(template_path))