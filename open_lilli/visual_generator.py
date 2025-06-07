"""Visual generator for creating charts and sourcing images."""

import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from urllib.parse import quote_plus

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFont

from .models import SlidePlan, NativeChartData, ProcessFlowConfig, VisualExcellenceConfig
from .native_chart_builder import NativeChartBuilder
from .process_flow_generator import ProcessFlowGenerator
from .corporate_asset_library import CorporateAssetLibrary

logger = logging.getLogger(__name__)


class VisualGenerator:
    """Generates charts and sources images for presentations."""

    def __init__(
        self, 
        output_dir: str = "assets", 
        theme_colors: Optional[Dict[str, str]] = None,
        template_parser=None,
        visual_config: Optional[VisualExcellenceConfig] = None
    ):
        """
        Initialize the visual generator.
        
        Args:
            output_dir: Directory to save generated visuals
            theme_colors: Theme colors from template palette (dk1, lt1, acc1-6) or default colors
            template_parser: Optional template parser for enhanced features
            visual_config: Visual excellence configuration
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.template_parser = template_parser
        self.visual_config = visual_config or VisualExcellenceConfig()
        
        # Initialize Phase 4 components
        self.native_chart_builder = None
        self.process_flow_generator = None
        self.corporate_asset_library = None
        
        if self.visual_config.enable_native_charts:
            self.native_chart_builder = NativeChartBuilder(template_parser)
        
        if self.visual_config.enable_process_flows:
            self.process_flow_generator = ProcessFlowGenerator(template_parser)
        
        if self.visual_config.enable_asset_library and self.visual_config.asset_library:
            self.corporate_asset_library = CorporateAssetLibrary(self.visual_config.asset_library)
        
        # Set template palette or use defaults
        if theme_colors:
            self.template_palette = theme_colors
            # Map template palette to chart color scheme
            self.theme_colors = self._map_template_palette_to_chart_colors(theme_colors)
        else:
            # Default corporate color palette
            self.template_palette = None
            self.theme_colors = {
                "primary": "#1F497D",
                "secondary": "#4F81BD", 
                "accent1": "#9BBB59",
                "accent2": "#F79646",
                "accent3": "#8064A2",
                "text_dark": "#000000",
                "text_light": "#FFFFFF",
                "background": "#FFFFFF",
                "neutral": "#808080"
            }
        
        # Configure matplotlib for professional charts
        plt.style.use('default')
        plt.rcParams.update({
            'font.size': 12,
            'font.family': 'sans-serif',
            'axes.labelsize': 12,
            'axes.titlesize': 14,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.titlesize': 16,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'axes.grid': True,
            'grid.alpha': 0.3
        })
        
        logger.info(f"VisualGenerator initialized with Phase 4 features:")
        logger.info(f"  Output directory: {self.output_dir}")
        logger.info(f"  Native charts: {self.visual_config.enable_native_charts}")
        logger.info(f"  Process flows: {self.visual_config.enable_process_flows}")
        logger.info(f"  Corporate assets: {self.visual_config.enable_asset_library}")

    def generate_visuals(self, slides: List[SlidePlan]) -> Dict[int, Dict[str, str]]:
        """
        Generate all visuals for a list of slides.
        
        Args:
            slides: List of slides to generate visuals for
            
        Returns:
            Dictionary mapping slide indices to generated file paths
        """
        visuals = {}
        
        for slide in slides:
            slide_visuals = {}
            
            # Generate chart if chart data is present
            if slide.chart_data:
                try:
                    # Check if this is native chart data
                    if isinstance(slide.chart_data, dict) and slide.chart_data.get("native_chart"):
                        # Flag for native chart - will be handled during slide assembly
                        slide_visuals["native_chart"] = "pending"
                        logger.info(f"Flagged native chart for slide {slide.index}")
                    else:
                        # Generate traditional chart image
                        chart_path = self.generate_chart(slide)
                        if chart_path:
                            slide_visuals["chart"] = str(chart_path)
                            logger.info(f"Generated chart for slide {slide.index}: {chart_path}")
                except Exception as e:
                    logger.error(f"Failed to generate chart for slide {slide.index}: {e}")
            
            # Generate process flow if flow data is present
            if hasattr(slide, 'process_flow') and slide.process_flow:
                try:
                    flow_path = self.generate_process_flow(slide.process_flow, slide.index)
                    if flow_path:
                        slide_visuals["process_flow"] = str(flow_path)
                        logger.info(f"Generated process flow for slide {slide.index}: {flow_path}")
                except Exception as e:
                    logger.error(f"Failed to generate process flow for slide {slide.index}: {e}")
            
            # Source image if image query is present
            if slide.image_query:
                try:
                    # Try corporate asset library first if enabled
                    if self.corporate_asset_library:
                        image_path = self.corporate_asset_library.get_brand_approved_image(
                            slide.image_query, slide.index
                        )
                    else:
                        image_path = self.source_image(slide.image_query, slide.index)
                    
                    if image_path:
                        slide_visuals["image"] = str(image_path)
                        logger.info(f"Sourced image for slide {slide.index}: {image_path}")
                except Exception as e:
                    logger.error(f"Failed to source image for slide {slide.index}: {e}")
            
            if slide_visuals:
                visuals[slide.index] = slide_visuals
        
        logger.info(f"Generated visuals for {len(visuals)} slides")
        return visuals

    def generate_chart(self, slide: SlidePlan) -> Optional[Path]:
        """
        Generate a chart based on slide chart data.
        
        Args:
            slide: Slide containing chart data
            
        Returns:
            Path to generated chart image or None if failed
        """
        if not slide.chart_data:
            return None
        
        try:
            chart_type = slide.chart_data.get("type", "bar")
            
            if chart_type == "bar":
                return self._generate_bar_chart(slide)
            elif chart_type == "line":
                return self._generate_line_chart(slide)
            elif chart_type == "pie":
                return self._generate_pie_chart(slide)
            elif chart_type == "scatter":
                return self._generate_scatter_chart(slide)
            else:
                logger.warning(f"Unknown chart type: {chart_type}, using bar chart")
                return self._generate_bar_chart(slide)
                
        except Exception as e:
            logger.error(f"Chart generation failed: {e}")
            return None

    def _generate_bar_chart(self, slide: SlidePlan) -> Path:
        """Generate a bar chart."""
        data = slide.chart_data
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Extract data
        categories = data.get("categories", data.get("x", ["A", "B", "C"]))
        values = data.get("values", data.get("y", [1, 2, 3]))
        
        # Create bars with theme colors
        colors = [self.theme_colors["primary"], self.theme_colors["secondary"], 
                 self.theme_colors["accent1"], self.theme_colors["accent2"]]
        bar_colors = colors[:len(categories)]
        
        bars = ax.bar(categories, values, color=bar_colors)
        
        # Customize chart
        title = data.get("title", slide.title)
        ax.set_title(title, fontweight='bold', pad=20)
        ax.set_xlabel(data.get("xlabel", ""))
        ax.set_ylabel(data.get("ylabel", ""))
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.01,
                   f'{height:.1f}', ha='center', va='bottom')
        
        # Style improvements
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save chart
        filename = f"chart_slide_{slide.index}_bar.png"
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        return output_path

    def _generate_line_chart(self, slide: SlidePlan) -> Path:
        """Generate a line chart."""
        data = slide.chart_data
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Extract data
        x_data = data.get("x", list(range(len(data.get("y", [1, 2, 3])))))
        y_data = data.get("y", [1, 2, 3])
        
        # Create line plot
        ax.plot(x_data, y_data, color=self.theme_colors["primary"], 
               linewidth=3, marker='o', markersize=8)
        
        # Customize chart
        title = data.get("title", slide.title)
        ax.set_title(title, fontweight='bold', pad=20)
        ax.set_xlabel(data.get("xlabel", ""))
        ax.set_ylabel(data.get("ylabel", ""))
        
        # Style improvements
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save chart
        filename = f"chart_slide_{slide.index}_line.png"
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close()
        
        return output_path

    def _generate_pie_chart(self, slide: SlidePlan) -> Path:
        """Generate a pie chart."""
        data = slide.chart_data
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Extract data
        labels = data.get("labels", data.get("categories", ["A", "B", "C"]))
        values = data.get("values", data.get("y", [1, 2, 3]))
        
        # Create pie chart with theme colors
        colors = [self.theme_colors["primary"], self.theme_colors["secondary"],
                 self.theme_colors["accent1"], self.theme_colors["accent2"],
                 self.theme_colors["accent3"]]
        
        wedges, texts, autotexts = ax.pie(values, labels=labels, autopct='%1.1f%%',
                                         colors=colors[:len(labels)], startangle=90)
        
        # Customize chart
        title = data.get("title", slide.title)
        ax.set_title(title, fontweight='bold', pad=20)
        
        # Style improvements
        plt.setp(autotexts, size=10, weight="bold")
        ax.axis('equal')
        
        plt.tight_layout()
        
        # Save chart
        filename = f"chart_slide_{slide.index}_pie.png"
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close()
        
        return output_path

    def _generate_scatter_chart(self, slide: SlidePlan) -> Path:
        """Generate a scatter plot."""
        data = slide.chart_data
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Extract data
        x_data = data.get("x", [1, 2, 3, 4])
        y_data = data.get("y", [1, 4, 2, 3])
        
        # Create scatter plot
        ax.scatter(x_data, y_data, color=self.theme_colors["primary"],
                  s=100, alpha=0.7)
        
        # Customize chart
        title = data.get("title", slide.title)
        ax.set_title(title, fontweight='bold', pad=20)
        ax.set_xlabel(data.get("xlabel", ""))
        ax.set_ylabel(data.get("ylabel", ""))
        
        # Style improvements
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save chart
        filename = f"chart_slide_{slide.index}_scatter.png"
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close()
        
        return output_path

    def generate_process_flow(self, flow_config: ProcessFlowConfig, slide_index: int) -> Optional[Path]:
        """
        Generate a process flow diagram.
        
        Args:
            flow_config: Process flow configuration
            slide_index: Slide index for filename
            
        Returns:
            Path to generated SVG file or None if failed
        """
        if not self.process_flow_generator:
            logger.warning("Process flow generator not initialized")
            return None
        
        try:
            # Generate SVG output path
            filename = f"process_flow_slide_{slide_index}.svg"
            output_path = self.output_dir / filename
            
            # Generate process flow
            svg_path = self.process_flow_generator.generate_process_flow(flow_config, output_path)
            
            if svg_path:
                logger.info(f"Generated process flow: {svg_path}")
                return svg_path
            else:
                logger.error("Process flow generation returned None")
                return None
                
        except Exception as e:
            logger.error(f"Process flow generation failed: {e}")
            return None

    def create_native_chart_on_slide(
        self,
        slide,
        chart_config: NativeChartData,
        position: Optional[Tuple] = None
    ) -> bool:
        """
        Create a native PowerPoint chart directly on a slide.
        
        Args:
            slide: PowerPoint slide object
            chart_config: Native chart configuration
            position: Optional (left, top, width, height) position
            
        Returns:
            True if chart was created successfully
        """
        if not self.native_chart_builder:
            logger.warning("Native chart builder not initialized")
            return False
        
        try:
            return self.native_chart_builder.create_native_chart(slide, chart_config, position)
        except Exception as e:
            logger.error(f"Native chart creation failed: {e}")
            return False

    def convert_legacy_to_native_chart(self, legacy_chart_data: Dict) -> Optional[NativeChartData]:
        """
        Convert legacy chart data to native chart configuration.
        
        Args:
            legacy_chart_data: Old format chart data
            
        Returns:
            NativeChartData or None if conversion fails
        """
        if not self.native_chart_builder:
            return None
        
        return self.native_chart_builder.convert_legacy_chart_data(legacy_chart_data)

    def source_image(self, query: str, slide_index: int) -> Optional[Path]:
        """
        Source an image based on query.
        
        Args:
            query: Search query for the image
            slide_index: Slide index for filename
            
        Returns:
            Path to downloaded/generated image or None if failed
        """
        try:
            # Check if we're in strict brand mode
            if (self.corporate_asset_library and 
                self.visual_config.asset_library and
                self.visual_config.asset_library.brand_guidelines_strict):
                logger.info(f"Strict brand mode: skipping external sources for '{query}'")
                return self._generate_placeholder_image(query, slide_index)
            
            # Try to source from Unsplash (free stock photos)
            image_path = self._source_from_unsplash(query, slide_index)
            if image_path:
                return image_path
            
            # Fallback to generating a placeholder
            return self._generate_placeholder_image(query, slide_index)
            
        except Exception as e:
            logger.error(f"Image sourcing failed for query '{query}': {e}")
            return self._generate_placeholder_image(query, slide_index)

    def _source_from_unsplash(self, query: str, slide_index: int) -> Optional[Path]:
        """Source image from Unsplash."""
        try:
            # Clean and encode query
            clean_query = re.sub(r'[^\w\s]', '', query).strip()
            if not clean_query:
                clean_query = "business"
            
            # Use Unsplash's random photo API with search
            url = f"https://source.unsplash.com/1600x900/?{quote_plus(clean_query)}"
            
            logger.debug(f"Requesting image from Unsplash: {url}")
            
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            # Save image
            filename = f"image_slide_{slide_index}_{clean_query.replace(' ', '_')}.jpg"
            output_path = self.output_dir / filename
            
            with open(output_path, 'wb') as f:
                f.write(response.content)
            
            logger.info(f"Downloaded image from Unsplash: {output_path}")
            return output_path
            
        except Exception as e:
            logger.warning(f"Unsplash download failed for '{query}': {e}")
            return None

    def _generate_placeholder_image(self, query: str, slide_index: int) -> Path:
        """Generate a placeholder image with text."""
        # Create a professional-looking placeholder
        width, height = 1600, 900
        img = Image.new('RGB', (width, height), color=self.theme_colors["background"])
        draw = ImageDraw.Draw(img)
        
        # Draw border
        border_color = self.theme_colors["primary"]
        border_width = 8
        draw.rectangle([0, 0, width-1, height-1], outline=border_color, width=border_width)
        
        # Add text
        try:
            # Try to use a decent font
            font_size = 72
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
        except:
            try:
                font = ImageFont.load_default()
                font_size = 36
            except:
                font = None
                font_size = 24
        
        # Main text
        main_text = query.title() if query else "Image Placeholder"
        if font:
            bbox = draw.textbbox((0, 0), main_text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
        else:
            text_width = len(main_text) * font_size * 0.6
            text_height = font_size
        
        x = (width - text_width) // 2
        y = (height - text_height) // 2 - 50
        
        draw.text((x, y), main_text, fill=self.theme_colors["primary"], font=font)
        
        # Subtitle
        subtitle = f"Slide {slide_index + 1} Visual"
        if font:
            try:
                subtitle_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 48)
            except:
                subtitle_font = font
        else:
            subtitle_font = font
        
        if subtitle_font:
            bbox = draw.textbbox((0, 0), subtitle, font=subtitle_font)
            sub_width = bbox[2] - bbox[0]
        else:
            sub_width = len(subtitle) * 24 * 0.6
        
        sub_x = (width - sub_width) // 2
        sub_y = y + text_height + 30
        
        draw.text((sub_x, sub_y), subtitle, fill=self.theme_colors["neutral"], font=subtitle_font)
        
        # Save placeholder
        clean_query = re.sub(r'[^\w\s]', '', query).replace(' ', '_') if query else "placeholder"
        filename = f"placeholder_slide_{slide_index}_{clean_query}.png"
        output_path = self.output_dir / filename
        
        img.save(output_path, 'PNG', quality=95)
        
        logger.info(f"Generated placeholder image: {output_path}")
        return output_path

    def create_icon_visual(self, icon_type: str, slide_index: int) -> Path:
        """
        Create a simple icon visual.
        
        Args:
            icon_type: Type of icon (arrow, checkmark, etc.)
            slide_index: Slide index for filename
            
        Returns:
            Path to generated icon image
        """
        size = 400
        img = Image.new('RGBA', (size, size), color=(255, 255, 255, 0))
        draw = ImageDraw.Draw(img)
        
        color = self.theme_colors["primary"]
        center = size // 2
        
        if icon_type.lower() in ["arrow", "growth"]:
            # Draw an upward arrow
            points = [
                (center, size * 0.2),  # Top point
                (center - size * 0.15, center),  # Left point
                (center - size * 0.05, center),  # Left inner
                (center - size * 0.05, size * 0.8),  # Left bottom
                (center + size * 0.05, size * 0.8),  # Right bottom
                (center + size * 0.05, center),  # Right inner
                (center + size * 0.15, center),  # Right point
            ]
            draw.polygon(points, fill=color)
        
        elif icon_type.lower() in ["check", "success"]:
            # Draw a checkmark
            draw.line([
                (center - size * 0.2, center),
                (center - size * 0.05, center + size * 0.15),
                (center + size * 0.2, center - size * 0.1)
            ], fill=color, width=size // 20)
        
        else:
            # Default: circle
            margin = size * 0.1
            draw.ellipse([margin, margin, size - margin, size - margin], 
                        fill=color, outline=color)
        
        # Save icon
        filename = f"icon_slide_{slide_index}_{icon_type}.png"
        output_path = self.output_dir / filename
        img.save(output_path, 'PNG')
        
        logger.info(f"Generated icon: {output_path}")
        return output_path

    def resize_image(self, image_path: Union[str, Path], target_size: Tuple[int, int]) -> Path:
        """
        Resize an image to target size.
        
        Args:
            image_path: Path to image to resize
            target_size: Target (width, height)
            
        Returns:
            Path to resized image
        """
        image_path = Path(image_path)
        
        with Image.open(image_path) as img:
            # Calculate aspect ratio preserving resize
            img_ratio = img.width / img.height
            target_ratio = target_size[0] / target_size[1]
            
            if img_ratio > target_ratio:
                # Image is wider, fit to width
                new_width = target_size[0]
                new_height = int(target_size[0] / img_ratio)
            else:
                # Image is taller, fit to height
                new_height = target_size[1]
                new_width = int(target_size[1] * img_ratio)
            
            resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Create new image with target size and center the resized image
            final_img = Image.new('RGB', target_size, color='white')
            paste_x = (target_size[0] - new_width) // 2
            paste_y = (target_size[1] - new_height) // 2
            final_img.paste(resized_img, (paste_x, paste_y))
            
            # Save resized image
            resized_path = image_path.parent / f"resized_{image_path.name}"
            final_img.save(resized_path, 'PNG', quality=95)
            
            logger.debug(f"Resized image {image_path} to {target_size}: {resized_path}")
            return resized_path

    def get_visual_summary(self, visuals: Dict[int, Dict[str, str]]) -> Dict[str, any]:
        """
        Get summary of generated visuals.
        
        Args:
            visuals: Dictionary of generated visuals
            
        Returns:
            Summary statistics
        """
        total_slides_with_visuals = len(visuals)
        total_charts = sum(1 for v in visuals.values() if "chart" in v or "native_chart" in v)
        total_images = sum(1 for v in visuals.values() if "image" in v)
        total_process_flows = sum(1 for v in visuals.values() if "process_flow" in v)
        total_native_charts = sum(1 for v in visuals.values() if "native_chart" in v)
        
        chart_types = {}
        for slide_visuals in visuals.values():
            if "chart" in slide_visuals:
                chart_path = slide_visuals["chart"]
                if "bar" in chart_path:
                    chart_types["bar"] = chart_types.get("bar", 0) + 1
                elif "line" in chart_path:
                    chart_types["line"] = chart_types.get("line", 0) + 1
                elif "pie" in chart_path:
                    chart_types["pie"] = chart_types.get("pie", 0) + 1
                elif "scatter" in chart_path:
                    chart_types["scatter"] = chart_types.get("scatter", 0) + 1
        
        return {
            "total_slides_with_visuals": total_slides_with_visuals,
            "total_charts": total_charts,
            "total_images": total_images,
            "total_process_flows": total_process_flows,
            "total_native_charts": total_native_charts,
            "chart_types": chart_types,
            "output_directory": str(self.output_dir),
            "phase4_features": {
                "native_charts_enabled": self.visual_config.enable_native_charts,
                "process_flows_enabled": self.visual_config.enable_process_flows,
                "corporate_assets_enabled": self.visual_config.enable_asset_library,
                "strict_brand_mode": (
                    self.visual_config.asset_library.brand_guidelines_strict 
                    if self.visual_config.asset_library else False
                )
            }
        }
    
    def _map_template_palette_to_chart_colors(self, template_palette: Dict[str, str]) -> Dict[str, str]:
        """
        Map template palette colors (dk1, lt1, acc1-6) to chart color scheme.
        
        This implements T-40 requirement: "default to palette.primary + palette.acc*"
        
        Args:
            template_palette: Template colors with keys like dk1, lt1, acc1, acc2, etc.
            
        Returns:
            Mapped color scheme for charts
        """
        mapped_colors = {}
        
        # Primary color: Use dk1 (dark1) as primary, or first accent if not available
        if 'dk1' in template_palette:
            mapped_colors['primary'] = template_palette['dk1']
        elif 'acc1' in template_palette:
            mapped_colors['primary'] = template_palette['acc1']
        else:
            mapped_colors['primary'] = "#1F497D"  # Fallback
        
        # Secondary color: Use first accent color or dk1
        if 'acc1' in template_palette:
            mapped_colors['secondary'] = template_palette['acc1']
        elif 'dk1' in template_palette:
            mapped_colors['secondary'] = template_palette['dk1']
        else:
            mapped_colors['secondary'] = "#4F81BD"  # Fallback
        
        # Accent colors: Map acc1-6 to accent1-3 plus additional slots
        accent_keys = ['acc1', 'acc2', 'acc3', 'acc4', 'acc5', 'acc6']
        chart_accent_keys = ['accent1', 'accent2', 'accent3']
        
        for i, chart_key in enumerate(chart_accent_keys):
            if i < len(accent_keys) and accent_keys[i] in template_palette:
                mapped_colors[chart_key] = template_palette[accent_keys[i]]
            else:
                # Fallback colors for missing accent colors
                fallbacks = ["#9BBB59", "#F79646", "#8064A2"]
                mapped_colors[chart_key] = fallbacks[i] if i < len(fallbacks) else fallbacks[-1]
        
        # Text colors: Use dk1 for dark text, lt1 for light text
        mapped_colors['text_dark'] = template_palette.get('dk1', '#000000')
        mapped_colors['text_light'] = template_palette.get('lt1', '#FFFFFF')
        
        # Background: Use lt1 (light1) as background
        mapped_colors['background'] = template_palette.get('lt1', '#FFFFFF')
        
        # Neutral: Mix of dark and light or use available accent
        if 'dk1' in template_palette and 'lt1' in template_palette:
            # Create a neutral by mixing dark and light (simplified approach)
            mapped_colors['neutral'] = template_palette.get('acc2', '#808080')
        else:
            mapped_colors['neutral'] = '#808080'
        
        logger.debug(f"Mapped template palette to chart colors: {mapped_colors}")
        return mapped_colors
    
    def get_template_color_usage(self) -> Dict[str, str]:
        """
        Get the current template color usage for debugging/validation.
        
        Returns:
            Dictionary showing how template palette maps to chart colors
        """
        if not self.template_palette:
            return {"status": "Using default colors, no template palette provided"}
        
        return {
            "template_palette": self.template_palette,
            "mapped_chart_colors": self.theme_colors,
            "primary_source": "dk1" if 'dk1' in self.template_palette else "acc1 fallback",
            "accent_sources": [f"acc{i+1}" for i in range(3) if f"acc{i+1}" in self.template_palette]
        }