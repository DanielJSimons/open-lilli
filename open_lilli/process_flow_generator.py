"""Process flow diagram generator using Mermaid to SVG conversion.

This module implements T-52: Render Mermaid → SVG → insert; ensure SVG recolored with brand palette.
"""

import logging
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from xml.etree import ElementTree as ET

from .models import ProcessFlowConfig, ProcessFlowStep, ProcessFlowType
from .template_parser import TemplateParser

logger = logging.getLogger(__name__)


class ProcessFlowGenerator:
    """Generates process flow diagrams using Mermaid and converts to SVG."""

    def __init__(self, template_parser: Optional[TemplateParser] = None):
        """
        Initialize the process flow generator.
        
        Args:
            template_parser: Optional template parser for color palette access
        """
        self.template_parser = template_parser
        self.mermaid_available = self._check_mermaid_availability()
        
        # Default color scheme
        self.default_colors = {
            'primary': '#1F497D',
            'secondary': '#4F81BD',
            'accent1': '#9BBB59',
            'accent2': '#F79646',
            'accent3': '#8064A2',
            'text': '#000000',
            'background': '#FFFFFF',
            'border': '#404040'
        }
        
        logger.info(f"ProcessFlowGenerator initialized, Mermaid available: {self.mermaid_available}")

    def _check_mermaid_availability(self) -> bool:
        """Check if Mermaid CLI is available."""
        try:
            result = subprocess.run(
                ['mmdc', '--version'], 
                capture_output=True, 
                text=True, 
                timeout=5
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            logger.warning("Mermaid CLI (mmdc) not found. Process flow diagrams will use fallback mode.")
            return False

    def generate_process_flow(
        self,
        flow_config: ProcessFlowConfig,
        output_path: Optional[Path] = None
    ) -> Optional[Path]:
        """
        Generate a process flow diagram.
        
        Args:
            flow_config: Process flow configuration
            output_path: Optional output path for SVG file
            
        Returns:
            Path to generated SVG file or None if failed
        """
        try:
            # Generate Mermaid syntax
            mermaid_code = self._generate_mermaid_code(flow_config)
            
            if self.mermaid_available:
                # Use Mermaid CLI to generate SVG
                svg_path = self._generate_with_mermaid_cli(mermaid_code, output_path)
            else:
                # Use fallback method
                svg_path = self._generate_fallback_svg(flow_config, output_path)
            
            if svg_path and flow_config.use_template_colors:
                # Recolor SVG with template palette
                self._recolor_svg_with_template(svg_path)
            
            logger.info(f"Generated process flow diagram: {svg_path}")
            return svg_path
            
        except Exception as e:
            logger.error(f"Failed to generate process flow: {e}")
            return None

    def _generate_mermaid_code(self, flow_config: ProcessFlowConfig) -> str:
        """
        Generate Mermaid flowchart syntax from configuration.
        
        Args:
            flow_config: Process flow configuration
            
        Returns:
            Mermaid flowchart code
        """
        lines = ["flowchart TD"]  # Top-down direction by default
        
        # Adjust direction based on orientation
        if flow_config.orientation == "horizontal":
            lines[0] = "flowchart LR"  # Left-right
        
        # Add step definitions with shapes
        for step in flow_config.steps:
            shape_start, shape_end = self._get_mermaid_shape(step.step_type)
            
            # Format step label
            label = step.label
            if flow_config.show_step_numbers:
                # Add step number if it's a process step
                if step.step_type == "process":
                    step_num = self._get_step_number(step, flow_config.steps)
                    label = f"{step_num}. {label}"
            
            # Clean label for Mermaid
            clean_label = self._clean_mermaid_text(label)
            
            # Add step definition
            lines.append(f"    {step.id}{shape_start}{clean_label}{shape_end}")
        
        # Add connections
        for step in flow_config.steps:
            for connection in step.connections:
                lines.append(f"    {step.id} --> {connection}")
        
        # Add styling if using template colors
        if flow_config.use_template_colors and self.template_parser:
            style_lines = self._generate_mermaid_styling(flow_config.steps)
            lines.extend(style_lines)
        
        return "\n".join(lines)

    def _get_mermaid_shape(self, step_type: str) -> Tuple[str, str]:
        """
        Get Mermaid shape syntax for step type.
        
        Args:
            step_type: Type of step (start, end, process, decision)
            
        Returns:
            Tuple of (shape_start, shape_end) syntax
        """
        shape_map = {
            "start": ("([", "])"),      # Stadium shape
            "end": ("([", "])"),        # Stadium shape
            "process": ("[", "]"),      # Rectangle
            "decision": ("{", "}"),     # Diamond shape
            "data": ("[(", ")]"),       # Cylinder shape
            "document": ("[[", "]]"),   # Subroutine shape
        }
        
        return shape_map.get(step_type, ("[", "]"))

    def _get_step_number(self, step: ProcessFlowStep, all_steps: List[ProcessFlowStep]) -> int:
        """Get sequential number for a process step."""
        process_steps = [s for s in all_steps if s.step_type == "process"]
        try:
            return process_steps.index(step) + 1
        except ValueError:
            return 1

    def _clean_mermaid_text(self, text: str) -> str:
        """Clean text for Mermaid syntax."""
        # Replace problematic characters
        text = re.sub(r'["\[\]{}()]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _generate_mermaid_styling(self, steps: List[ProcessFlowStep]) -> List[str]:
        """Generate Mermaid styling based on template colors."""
        style_lines = []
        
        if not self.template_parser or not hasattr(self.template_parser, 'palette'):
            return style_lines
        
        palette = self.template_parser.palette
        colors = self._get_template_colors(palette)
        
        # Define class styles
        style_lines.append(f"    classDef startEnd fill:{colors['accent1']},stroke:{colors['primary']},stroke-width:2px,color:{colors['text']}")
        style_lines.append(f"    classDef process fill:{colors['primary']},stroke:{colors['border']},stroke-width:1px,color:{colors['background']}")
        style_lines.append(f"    classDef decision fill:{colors['accent2']},stroke:{colors['primary']},stroke-width:2px,color:{colors['text']}")
        
        # Apply classes to steps
        for step in steps:
            if step.step_type in ["start", "end"]:
                style_lines.append(f"    class {step.id} startEnd")
            elif step.step_type == "decision":
                style_lines.append(f"    class {step.id} decision")
            else:
                style_lines.append(f"    class {step.id} process")
        
        return style_lines

    def _get_template_colors(self, palette: Dict[str, str]) -> Dict[str, str]:
        """Extract and map template colors."""
        colors = self.default_colors.copy()
        
        if 'dk1' in palette:
            colors['primary'] = palette['dk1']
        if 'lt1' in palette:
            colors['background'] = palette['lt1']
        if 'acc1' in palette:
            colors['accent1'] = palette['acc1']
        if 'acc2' in palette:
            colors['accent2'] = palette['acc2']
        if 'acc3' in palette:
            colors['accent3'] = palette['acc3']
        
        return colors

    def _generate_with_mermaid_cli(
        self, 
        mermaid_code: str, 
        output_path: Optional[Path] = None
    ) -> Optional[Path]:
        """
        Generate SVG using Mermaid CLI.
        
        Args:
            mermaid_code: Mermaid flowchart code
            output_path: Optional output path
            
        Returns:
            Path to generated SVG file
        """
        try:
            # Create temporary file for Mermaid code
            with tempfile.NamedTemporaryFile(mode='w', suffix='.mmd', delete=False) as f:
                f.write(mermaid_code)
                mmd_file = Path(f.name)
            
            # Set output path
            if output_path is None:
                output_path = mmd_file.parent / f"{mmd_file.stem}.svg"
            
            # Run Mermaid CLI
            cmd = [
                'mmdc',
                '-i', str(mmd_file),
                '-o', str(output_path),
                '-f', 'svg',
                '-t', 'default',
                '--width', '1200',
                '--height', '800'
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            # Clean up temporary file
            mmd_file.unlink()
            
            if result.returncode == 0 and output_path.exists():
                logger.debug(f"Generated SVG with Mermaid CLI: {output_path}")
                return output_path
            else:
                logger.error(f"Mermaid CLI failed: {result.stderr}")
                return None
                
        except Exception as e:
            logger.error(f"Mermaid CLI generation failed: {e}")
            return None

    def _generate_fallback_svg(
        self,
        flow_config: ProcessFlowConfig,
        output_path: Optional[Path] = None
    ) -> Optional[Path]:
        """
        Generate SVG using fallback method when Mermaid CLI is not available.
        
        Args:
            flow_config: Process flow configuration
            output_path: Optional output path
            
        Returns:
            Path to generated SVG file
        """
        try:
            if output_path is None:
                output_path = Path(tempfile.mktemp(suffix='.svg'))
            
            # Create simple SVG flowchart
            svg_content = self._create_simple_svg_flowchart(flow_config)
            
            # Write SVG file
            output_path.write_text(svg_content, encoding='utf-8')
            
            logger.debug(f"Generated fallback SVG: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Fallback SVG generation failed: {e}")
            return None

    def _create_simple_svg_flowchart(self, flow_config: ProcessFlowConfig) -> str:
        """Create a simple SVG flowchart without Mermaid."""
        
        # Calculate layout
        steps = flow_config.steps
        colors = self._get_template_colors(
            self.template_parser.palette if self.template_parser else {}
        )
        
        # Simple horizontal layout
        step_width = 150
        step_height = 60
        step_spacing = 200
        margin = 50
        
        svg_width = len(steps) * step_spacing + margin * 2
        svg_height = step_height + margin * 2
        
        # Start SVG
        svg_parts = [
            f'<svg width="{svg_width}" height="{svg_height}" xmlns="http://www.w3.org/2000/svg">',
            f'<defs>',
            f'  <style>',
            f'    .step-text {{ font-family: Arial, sans-serif; font-size: 12px; text-anchor: middle; dominant-baseline: middle; }}',
            f'    .arrow {{ stroke: {colors["border"]}; stroke-width: 2; fill: none; marker-end: url(#arrowhead); }}',
            f'  </style>',
            f'  <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">',
            f'    <polygon points="0 0, 10 3.5, 0 7" fill="{colors["border"]}" />',
            f'  </marker>',
            f'</defs>'
        ]
        
        # Add background
        svg_parts.append(f'<rect width="100%" height="100%" fill="{colors["background"]}" />')
        
        # Add title
        title_y = margin // 2
        svg_parts.append(
            f'<text x="{svg_width // 2}" y="{title_y}" class="step-text" '
            f'style="font-size: 16px; font-weight: bold; fill: {colors["primary"]};">'
            f'{flow_config.title}</text>'
        )
        
        # Draw steps
        for i, step in enumerate(steps):
            x = margin + i * step_spacing
            y = margin + title_y
            
            # Choose color based on step type
            if step.step_type in ["start", "end"]:
                fill_color = colors["accent1"]
                text_color = colors["text"]
            elif step.step_type == "decision":
                fill_color = colors["accent2"]
                text_color = colors["text"]
            else:
                fill_color = colors["primary"]
                text_color = colors["background"]
            
            # Draw shape
            if step.step_type == "decision":
                # Diamond shape
                points = f"{x},{y-step_height//2} {x+step_width//2},{y} {x},{y+step_height//2} {x-step_width//2},{y}"
                svg_parts.append(
                    f'<polygon points="{points}" fill="{fill_color}" '
                    f'stroke="{colors["border"]}" stroke-width="1" />'
                )
            elif step.step_type in ["start", "end"]:
                # Rounded rectangle
                svg_parts.append(
                    f'<rect x="{x-step_width//2}" y="{y-step_height//2}" '
                    f'width="{step_width}" height="{step_height}" '
                    f'rx="30" ry="30" fill="{fill_color}" '
                    f'stroke="{colors["border"]}" stroke-width="1" />'
                )
            else:
                # Rectangle
                svg_parts.append(
                    f'<rect x="{x-step_width//2}" y="{y-step_height//2}" '
                    f'width="{step_width}" height="{step_height}" '
                    f'fill="{fill_color}" stroke="{colors["border"]}" stroke-width="1" />'
                )
            
            # Add text
            label = step.label
            if flow_config.show_step_numbers and step.step_type == "process":
                step_num = self._get_step_number(step, steps)
                label = f"{step_num}. {label}"
            
            # Word wrap text if too long
            words = label.split()
            if len(label) > 20:
                mid = len(words) // 2
                line1 = " ".join(words[:mid])
                line2 = " ".join(words[mid:])
                svg_parts.append(
                    f'<text x="{x}" y="{y-5}" class="step-text" fill="{text_color}">{line1}</text>'
                )
                svg_parts.append(
                    f'<text x="{x}" y="{y+10}" class="step-text" fill="{text_color}">{line2}</text>'
                )
            else:
                svg_parts.append(
                    f'<text x="{x}" y="{y}" class="step-text" fill="{text_color}">{label}</text>'
                )
            
            # Draw arrow to next step
            if i < len(steps) - 1:
                arrow_start_x = x + step_width // 2
                arrow_end_x = x + step_spacing - step_width // 2
                svg_parts.append(
                    f'<line x1="{arrow_start_x}" y1="{y}" x2="{arrow_end_x}" y2="{y}" class="arrow" />'
                )
        
        svg_parts.append('</svg>')
        
        return '\n'.join(svg_parts)

    def _recolor_svg_with_template(self, svg_path: Path) -> None:
        """
        Recolor SVG elements to match template palette.
        
        Args:
            svg_path: Path to SVG file to recolor
        """
        try:
            if not self.template_parser or not hasattr(self.template_parser, 'palette'):
                return
            
            # Read SVG content
            svg_content = svg_path.read_text(encoding='utf-8')
            
            # Get template colors
            colors = self._get_template_colors(self.template_parser.palette)
            
            # Color mapping for common SVG colors
            color_replacements = {
                '#1f77b4': colors['primary'],      # Default blue
                '#ff7f0e': colors['accent2'],      # Default orange
                '#2ca02c': colors['accent1'],      # Default green
                '#d62728': colors['accent3'],      # Default red
                '#9467bd': colors['accent3'],      # Default purple
                '#000000': colors['text'],         # Black text
                '#ffffff': colors['background'],   # White background
                'black': colors['text'],
                'white': colors['background']
            }
            
            # Apply color replacements
            modified_content = svg_content
            for old_color, new_color in color_replacements.items():
                # Replace in fill attributes
                modified_content = re.sub(
                    f'fill="{old_color}"',
                    f'fill="{new_color}"',
                    modified_content,
                    flags=re.IGNORECASE
                )
                # Replace in stroke attributes
                modified_content = re.sub(
                    f'stroke="{old_color}"',
                    f'stroke="{new_color}"',
                    modified_content,
                    flags=re.IGNORECASE
                )
                # Replace in style attributes
                modified_content = re.sub(
                    f'fill:\\s*{old_color}',
                    f'fill: {new_color}',
                    modified_content,
                    flags=re.IGNORECASE
                )
                modified_content = re.sub(
                    f'stroke:\\s*{old_color}',
                    f'stroke: {new_color}',
                    modified_content,
                    flags=re.IGNORECASE
                )
            
            # Write modified content back
            svg_path.write_text(modified_content, encoding='utf-8')
            
            logger.debug(f"Recolored SVG with template palette: {svg_path}")
            
        except Exception as e:
            logger.error(f"Failed to recolor SVG: {e}")

    def create_from_text_description(self, description: str, title: str = "Process Flow") -> Optional[ProcessFlowConfig]:
        """
        Create process flow configuration from text description.
        
        Args:
            description: Text description of the process
            title: Flow title
            
        Returns:
            ProcessFlowConfig or None if parsing fails
        """
        try:
            # Simple parsing - look for numbered steps or bullet points
            lines = [line.strip() for line in description.split('\n') if line.strip()]
            
            steps = []
            step_id_counter = 1
            
            for i, line in enumerate(lines):
                # Remove bullet points, numbers, etc.
                clean_line = re.sub(r'^[-*•]\s*', '', line)
                clean_line = re.sub(r'^\d+[.)]\s*', '', clean_line)
                
                if not clean_line:
                    continue
                
                # Determine step type
                step_type = "process"
                if i == 0 or any(word in clean_line.lower() for word in ["start", "begin", "initial"]):
                    step_type = "start"
                elif i == len(lines) - 1 or any(word in clean_line.lower() for word in ["end", "finish", "complete"]):
                    step_type = "end"
                elif any(word in clean_line.lower() for word in ["decide", "check", "if", "whether"]):
                    step_type = "decision"
                
                step_id = f"step{step_id_counter}"
                connections = [f"step{step_id_counter + 1}"] if i < len(lines) - 1 else []
                
                step = ProcessFlowStep(
                    id=step_id,
                    label=clean_line[:50],  # Limit label length
                    step_type=step_type,
                    connections=connections
                )
                
                steps.append(step)
                step_id_counter += 1
            
            if not steps:
                return None
            
            return ProcessFlowConfig(
                flow_type=ProcessFlowType.SEQUENTIAL,
                title=title,
                steps=steps,
                orientation="horizontal",
                use_template_colors=True,
                show_step_numbers=True
            )
            
        except Exception as e:
            logger.error(f"Failed to create flow from text description: {e}")
            return None

    def validate_flow_config(self, flow_config: ProcessFlowConfig) -> List[str]:
        """
        Validate process flow configuration.
        
        Args:
            flow_config: Configuration to validate
            
        Returns:
            List of validation issues
        """
        issues = []
        
        if not flow_config.title:
            issues.append("Flow title is required")
        
        if not flow_config.steps:
            issues.append("At least one step is required")
            return issues
        
        # Check step IDs are unique
        step_ids = [step.id for step in flow_config.steps]
        if len(step_ids) != len(set(step_ids)):
            issues.append("Step IDs must be unique")
        
        # Check connections reference valid step IDs
        for step in flow_config.steps:
            for connection in step.connections:
                if connection not in step_ids:
                    issues.append(f"Step '{step.id}' references unknown step '{connection}'")
        
        # Check for start and end steps
        start_steps = [s for s in flow_config.steps if s.step_type == "start"]
        end_steps = [s for s in flow_config.steps if s.step_type == "end"]
        
        if not start_steps:
            issues.append("Flow should have at least one start step")
        if not end_steps:
            issues.append("Flow should have at least one end step")
        
        return issues