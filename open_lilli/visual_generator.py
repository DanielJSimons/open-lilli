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
import openai # Added for generative AI

from .models import SlidePlan, NativeChartData, ProcessFlowConfig, VisualExcellenceConfig, ChartType, AssetLibraryConfig
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
                    chart_type_str = None
                    is_native_data_obj = isinstance(slide.chart_data, NativeChartData)
                    is_dict_data = isinstance(slide.chart_data, dict)

                    explicitly_native_pending = False
                    if is_dict_data:
                        chart_type_str = slide.chart_data.get("type", "bar").lower()
                        if slide.chart_data.get("native_chart") == "pending":
                            explicitly_native_pending = True
                    elif is_native_data_obj:
                        chart_type_str = slide.chart_data.chart_type.value.lower()

                    # Determine if we should aim for a native chart
                    use_native_chart = False
                    if self.visual_config.enable_native_charts and self.native_chart_builder:
                        if is_native_data_obj: # Already a NativeChartData object
                            use_native_chart = True
                        elif explicitly_native_pending: # Dictionary explicitly asks for native
                            use_native_chart = True
                        elif chart_type_str in ["bar", "line", "column", "area", "doughnut"]: # It's a dict and type is bar/line/column/area/doughnut
                            use_native_chart = True

                    if use_native_chart:
                        slide_visuals["native_chart"] = "pending"
                        logger.info(f"Flagged native chart for slide {slide.index} (type: {chart_type_str})")
                    else:
                        # Generate traditional chart image (PNG)
                        chart_path = self.generate_chart(slide) # This method expects slide.chart_data to be a dict for PNGs
                        if chart_path:
                            slide_visuals["chart"] = str(chart_path)
                            logger.info(f"Generated PNG chart for slide {slide.index} (type: {chart_type_str}): {chart_path}")
                except Exception as e:
                    logger.error(f"Failed to process chart data for slide {slide.index}: {e}")
            
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
                    image_path = None
                    # Try corporate asset library first if enabled
                    if self.corporate_asset_library:
                        target_aspect_ratio: Optional[float] = None
                        # Attempt to find an image placeholder and its aspect ratio
                        if self.template_parser and slide.layout_id is not None:
                            try:
                                layout = self.template_parser.get_layout(slide.layout_id)
                                if layout and hasattr(layout, 'placeholders'):
                                    for ph in layout.placeholders:
                                        # Heuristic: find a picture placeholder.
                                        # Common names: "Picture Placeholder", "Image Placeholder", "Slide Image"
                                        # Common types: "PICTURE", "PIC" (depends on parser)
                                        # Assuming ph.type_name or ph.name and ph.width, ph.height
                                        is_picture_placeholder = False
                                        if hasattr(ph, 'type_name') and isinstance(ph.type_name, str) and "PIC" in ph.type_name.upper():
                                            is_picture_placeholder = True
                                        elif hasattr(ph, 'name') and isinstance(ph.name, str) and "Picture" in ph.name:
                                             is_picture_placeholder = True

                                        if is_picture_placeholder and hasattr(ph, 'width') and hasattr(ph, 'height'):
                                            if ph.width > 0 and ph.height > 0:
                                                target_aspect_ratio = ph.width / ph.height
                                                logger.info(f"Found image placeholder in layout {slide.layout_id} for slide {slide.index}. Target aspect ratio: {target_aspect_ratio:.2f}")
                                                break # Use the first suitable one found
                                    if not target_aspect_ratio:
                                        logger.info(f"No suitable picture placeholder found or dimensions invalid in layout {slide.layout_id} for slide {slide.index}.")
                            except Exception as e:
                                logger.error(f"Error accessing layout or placeholders for slide {slide.index}, layout_id {slide.layout_id}: {e}")
                        else:
                            logger.info(f"No template parser or layout_id for slide {slide.index}, cannot determine target aspect ratio for CAL.")

                        image_path = self.corporate_asset_library.get_brand_approved_image(
                            query=slide.image_query,
                            slide_index=slide.index,
                            orientation=None, # Placeholder for future enhancement
                            dominant_color=None, # Placeholder for future enhancement
                            tags=None, # Placeholder for future enhancement
                            target_aspect_ratio=target_aspect_ratio
                        )
                        if image_path:
                             slide_visuals["image"] = str(image_path)
                             logger.info(f"Sourced image for slide {slide.index} via CorporateAssetLibrary: {image_path}")

                    # If CAL is not enabled, or CAL returned None (signaling fallback is allowed)
                    if not image_path:
                        sourced_image_path = self.source_image(slide.image_query, slide.index, slide)
                        if sourced_image_path:
                            slide_visuals["image"] = str(sourced_image_path)
                            logger.info(f"Sourced image for slide {slide.index} via VisualGenerator (GenAI/Unsplash/Placeholder): {sourced_image_path}")
                        # If sourced_image_path is None, it means all fallbacks within source_image failed or it returned a placeholder.
                        # The placeholder generation is handled within source_image itself if all else fails.
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
            elif chart_type == "area":
                return self._generate_area_chart(slide)
            elif chart_type == "doughnut":
                return self._generate_doughnut_chart(slide)
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

    def _generate_area_chart(self, slide: SlidePlan) -> Path:
        """Generate an area chart."""
        data = slide.chart_data

        fig, ax = plt.subplots(figsize=(10, 6))

        # Extract data
        categories = data.get("categories", data.get("x", list(range(len(data.get("series", [{"values": [1,2,3]}])[0]["values"])))))
        series = data.get("series", [{"name": "Series 1", "values": [1, 2, 3]}])

        # Prepare data for stackplot
        y_data = [s["values"] for s in series]
        labels = [s.get("name", f"Series {i+1}") for i, s in enumerate(series)]

        # Define colors, cycling through theme colors
        available_colors = [
            self.theme_colors["primary"],
            self.theme_colors["secondary"],
            self.theme_colors["accent1"],
            self.theme_colors["accent2"],
            self.theme_colors["accent3"]
        ]
        plot_colors = [available_colors[i % len(available_colors)] for i in range(len(series))]

        # Create stackplot
        ax.stackplot(categories, y_data, labels=labels, colors=plot_colors, alpha=0.7)

        # Customize chart
        title = data.get("title", slide.title)
        ax.set_title(title, fontweight='bold', pad=20)
        ax.set_xlabel(data.get("xlabel", ""))
        ax.set_ylabel(data.get("ylabel", ""))

        # Add legend if multiple series
        if len(series) > 1:
            ax.legend(loc='upper left')

        # Style improvements
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        # Save chart
        filename = f"chart_slide_{slide.index}_area.png"
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

    def _generate_doughnut_chart(self, slide: SlidePlan) -> Path:
        """Generate a doughnut chart."""
        data = slide.chart_data

        fig, ax = plt.subplots(figsize=(8, 8))

        # Extract data
        labels = data.get("labels", data.get("categories", ["A", "B", "C"]))
        values = data.get("values", data.get("y", [1, 2, 3]))

        # Create pie chart with theme colors
        colors = [
            self.theme_colors["primary"],
            self.theme_colors["secondary"],
            self.theme_colors["accent1"],
            self.theme_colors["accent2"],
            self.theme_colors["accent3"]
        ]
        plot_colors = colors[:len(labels)]

        wedges, texts, autotexts = ax.pie(
            values,
            labels=labels,
            autopct='%1.1f%%',
            colors=plot_colors,
            startangle=90,
            wedgeprops=dict(width=0.4) # This creates the doughnut hole, alternative to drawing a circle
        )

        # Customize chart
        title = data.get("title", slide.title)
        ax.set_title(title, fontweight='bold', pad=20)

        # Style improvements
        plt.setp(autotexts, size=10, weight="bold", color=self.theme_colors.get("text_light", "#FFFFFF")) # Make autopct text visible
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

        # Add legend if specified or if there are many items
        if data.get("legend", False) or len(labels) > 5:
            ax.legend(wedges, labels, title="Categories", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))

        plt.tight_layout()

        # Save chart
        filename = f"chart_slide_{slide.index}_doughnut.png"
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

    def source_image(self, query: str, slide_index: int, slide_plan: SlidePlan) -> Optional[Path]:
        """
        Source an image based on query, trying Generative AI then Unsplash.
        
        Args:
            query: Search query for the image
            slide_index: Slide index for filename
            slide_plan: The SlidePlan object for context (used by GenAI)
            
        Returns:
            Path to downloaded/generated image or None if failed
        """
        try:
            # Check if we're in strict brand mode (this check might be redundant if CAL is called first,
            # but good for direct calls to source_image or if CAL logic changes)
            if (self.corporate_asset_library and 
                self.visual_config.asset_library and
                self.visual_config.asset_library.brand_guidelines_strict):
                logger.info(f"Strict brand mode active: VisualGenerator will only create a placeholder if other methods (like CAL) haven't already.")
                # This path should ideally be hit only if CAL decided not to provide an image AND strict mode is on.
                # However, the main generate_visuals logic should prevent source_image from being called if CAL provided an image or placeholder in strict mode.
                return self._generate_placeholder_image(query, slide_index)

            # 1. Try Generative AI if enabled
            if self.visual_config.enable_generative_ai and self.visual_config.asset_library:
                logger.info(f"Attempting Generative AI for query: '{query}'")
                gen_ai_image_path = self._source_from_generative_ai(query, slide_index, slide_plan, self.theme_colors)
                if gen_ai_image_path:
                    logger.info(f"Successfully sourced image from Generative AI: {gen_ai_image_path}")
                    return gen_ai_image_path
                else:
                    logger.info(f"Generative AI did not return an image for query: '{query}'. Proceeding to other sources.")
            else:
                logger.info("Generative AI not enabled or asset_library not configured. Skipping.")

            # 2. Try to source from Unsplash (free stock photos)
            logger.info(f"Attempting Unsplash for query: '{query}'")
            unsplash_image_path = self._source_from_unsplash(query, slide_index)
            if unsplash_image_path:
                logger.info(f"Successfully sourced image from Unsplash: {unsplash_image_path}")
                return unsplash_image_path
            else:
                logger.info(f"Unsplash did not return an image for query: '{query}'. Proceeding to placeholder.")
            
            # 3. Fallback to generating a placeholder if all else fails
            logger.info(f"All external image sources failed for query '{query}'. Generating placeholder.")
            return self._generate_placeholder_image(query, slide_index)
            
        except Exception as e:
            logger.error(f"Image sourcing failed for query '{query}': {e}. Generating placeholder as a last resort.")
            return self._generate_placeholder_image(query, slide_index)

    def _source_from_generative_ai(self, query: str, slide_index: int, slide_context: SlidePlan, palette: Dict[str, str]) -> Optional[Path]:
        """Source an image from a generative AI provider."""
        logger.info(f"Attempting to source image from generative AI for query: '{query}' (slide {slide_index})")

        if not self.visual_config or \
           not self.visual_config.asset_library or \
           not self.visual_config.asset_library.generative_ai_provider:
            logger.warning("Generative AI provider not configured in visual_config.asset_library. Skipping AI image generation.")
            return None

        asset_lib_config: AssetLibraryConfig = self.visual_config.asset_library
        provider = asset_lib_config.generative_ai_provider
        api_key = asset_lib_config.generative_ai_api_key
        model_name = asset_lib_config.generative_ai_model

        if not api_key:
            logger.warning(f"API key for {provider} not configured. Skipping AI image generation.")
            return None

        # Construct detailed prompt
        prompt_parts = [query]
        if slide_context.title:
            prompt_parts.append(f"Slide title: '{slide_context.title}'")
        if slide_context.bullets:
            bullets_str = "; ".join(slide_context.bullets)
            prompt_parts.append(f"Slide bullets: {bullets_str}")

        palette_desc = []
        if palette.get("primary"):
            palette_desc.append(f"primary color {palette['primary']}")
        if palette.get("secondary"):
            palette_desc.append(f"secondary color {palette['secondary']}")
        if palette.get("accent1"): # Assuming theme_colors has accent1, acc1 might be from template_palette
            palette_desc.append(f"accent color {palette.get('accent1', palette.get('acc1'))}") # Check both keys

        if palette_desc:
            prompt_parts.append(f"Use a color palette with: {', '.join(palette_desc)}.")

        prompt_parts.append("Style: photorealistic, suitable for a business presentation slide.")
        detailed_prompt = ". ".join(prompt_parts)
        logger.debug(f"Generative AI prompt for {provider}: {detailed_prompt}")

        try:
            if provider == 'dalle3':
                openai.api_key = api_key
                image_params = {
                    "prompt": detailed_prompt,
                    "n": 1,
                    "size": "1792x1024", # Landscape for DALL-E 3
                    "model": model_name or "dall-e-3"
                }
                logger.info(f"Calling OpenAI Image.create with params: {image_params}")
                response = openai.Image.create(**image_params)
                image_url = response.data[0].url

                logger.info(f"Generated DALL·E 3 image URL: {image_url}")

                # Download the image
                image_response = requests.get(image_url, timeout=30)
                image_response.raise_for_status()

                # Save image
                filename_query_part = re.sub(r'[^\w\s-]', '', query.lower()).replace(' ', '_')[:50]
                filename = f"gen_image_slide_{slide_index}_{filename_query_part}.png"
                output_path = self.output_dir / filename

                with open(output_path, 'wb') as f:
                    f.write(image_response.content)
                logger.info(f"Saved DALL·E 3 image to {output_path}")
                return output_path

            elif provider == 'stablediffusion':
                # Placeholder for Stable Diffusion API call
                logger.warning(f"Stable Diffusion provider selected, but implementation is a placeholder. API key: {api_key}, Model: {model_name}")
                # Example:
                # api_endpoint = "https://api.stablediffusion.com/v1/generate"
                # headers = {"Authorization": f"Bearer {api_key}"}
                # payload = {"prompt": detailed_prompt, "model": model_name, "size": "1024x1024"}
                # response = requests.post(api_endpoint, headers=headers, json=payload)
                # response.raise_for_status()
                # image_data = response.content # or parse JSON for URL
                # ... save image ...
                return self._generate_placeholder_image(f"StableDiffusion: {query}", slide_index) # Return placeholder for now

            else:
                logger.error(f"Unknown generative AI provider: {provider}")
                return None

        except openai.APIError as e: # Specific OpenAI error
            logger.error(f"OpenAI API error for query '{query}': {e}")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Error downloading AI generated image for query '{query}': {e}")
            return None
        except Exception as e:
            logger.error(f"Generative AI image sourcing failed for provider {provider}, query '{query}': {e}")
            return None

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