#!/usr/bin/env python3
"""
Demo script showing theme color extraction from PowerPoint templates.

This script demonstrates how to extract theme colors (dk1, lt1, acc1-6) 
from PowerPoint template files using the TemplateParser class.
"""

import sys
from pathlib import Path

# Add the parent directory to the path so we can import open_lilli
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from open_lilli.template_parser import TemplateParser
    from pptx import Presentation
except ImportError as e:
    print(f"Required dependencies not available: {e}")
    print("Please install python-pptx to run this demo.")
    sys.exit(1)


def create_sample_template_with_colors():
    """
    Create a sample PowerPoint template for demonstration.
    
    Returns:
        Path to the created template
    """
    # Create a basic presentation
    prs = Presentation()
    
    # Save to templates directory
    template_dir = Path(__file__).parent.parent / "templates"
    template_dir.mkdir(exist_ok=True)
    
    template_path = template_dir / "demo_template.pptx"
    prs.save(str(template_path))
    
    return template_path


def demonstrate_theme_extraction():
    """Demonstrate theme color extraction functionality."""
    print("Theme Color Extraction Demo")
    print("=" * 40)
    
    # Check if we have a sample template
    template_dir = Path(__file__).parent.parent / "templates"
    sample_templates = list(template_dir.glob("*.pptx")) if template_dir.exists() else []
    
    if not sample_templates:
        print("Creating a sample template...")
        try:
            template_path = create_sample_template_with_colors()
            print(f"Sample template created: {template_path}")
        except Exception as e:
            print(f"Could not create sample template: {e}")
            return
    else:
        template_path = sample_templates[0]
        print(f"Using existing template: {template_path}")
    
    try:
        # Initialize the template parser
        print(f"\nLoading template: {template_path}")
        parser = TemplateParser(str(template_path))
        
        # Show extracted theme colors
        print(f"\nTheme colors extracted:")
        for color_name, hex_value in parser.palette.items():
            print(f"  {color_name:4s}: {hex_value}")
        
        # Demonstrate the get_theme_colors method directly
        print(f"\nDirect theme color extraction:")
        direct_colors = parser.get_theme_colors()
        if direct_colors:
            for color_name, hex_value in direct_colors.items():
                print(f"  {color_name:4s}: {hex_value}")
        else:
            print("  No theme colors found in XML (using defaults)")
        
        # Show how to get individual colors
        print(f"\nIndividual color access:")
        print(f"  Primary dark color (dk1): {parser.get_theme_color('dk1')}")
        print(f"  Primary light color (lt1): {parser.get_theme_color('lt1')}")
        print(f"  First accent color (acc1): {parser.get_theme_color('acc1')}")
        print(f"  Unknown color (fallback): {parser.get_theme_color('unknown')}")
        
        # Show complete template info
        print(f"\nComplete template information:")
        info = parser.get_template_info()
        print(f"  Template path: {info['template_path']}")
        print(f"  Total layouts: {info['total_layouts']}")
        print(f"  Available layouts: {', '.join(info['available_layout_types'])}")
        print(f"  Slide dimensions: {info['slide_dimensions']['width_inches']:.1f}\" x {info['slide_dimensions']['height_inches']:.1f}\"")
        print(f"  Theme colors: {len(info['theme_colors'])} colors extracted")
        
    except Exception as e:
        print(f"Error processing template: {e}")


def main():
    """Main demo function."""
    try:
        demonstrate_theme_extraction()
        
        print("\n" + "=" * 40)
        print("Demo completed successfully!")
        print("\nThe TemplateParser now supports:")
        print("• Extracting theme colors from ppt/theme/theme1.xml")
        print("• Converting sRGB, system, and preset colors to hex")
        print("• Fallback to default colors when extraction fails")
        print("• Integration with get_template_info() method")
        print("• Robust error handling for invalid templates")
        
    except KeyboardInterrupt:
        print("\nDemo interrupted by user.")
    except Exception as e:
        print(f"\nUnexpected error: {e}")


if __name__ == "__main__":
    main()