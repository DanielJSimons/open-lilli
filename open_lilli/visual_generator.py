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

from .models import SlidePlan

logger = logging.getLogger(__name__)


class VisualGenerator:
    """Generates charts and sources images for presentations."""

    def __init__(self, output_dir: str = "assets", theme_colors: Optional[Dict[str, str]] = None):
        """
        Initialize the visual generator.
        
        Args:
            output_dir: Directory to save generated visuals
            theme_colors: Theme colors for styling charts
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Default corporate color palette
        self.theme_colors = theme_colors or {
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
        
        logger.info(f"VisualGenerator initialized, output directory: {self.output_dir}")

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
                    chart_path = self.generate_chart(slide)
                    if chart_path:
                        slide_visuals["chart"] = str(chart_path)
                        logger.info(f"Generated chart for slide {slide.index}: {chart_path}")
                except Exception as e:
                    logger.error(f"Failed to generate chart for slide {slide.index}: {e}")
            
            # Source image if image query is present
            if slide.image_query:
                try:
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
        total_charts = sum(1 for v in visuals.values() if "chart" in v)
        total_images = sum(1 for v in visuals.values() if "image" in v)
        
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
            "chart_types": chart_types,
            "output_directory": str(self.output_dir)
        }