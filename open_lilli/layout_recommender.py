"""ML-assisted layout recommender for intelligent slide layout selection."""

import json
import logging
import pickle
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from openai import OpenAI

from .models import (
    LayoutRecommendation, 
    SlideEmbedding, 
    SlidePlan, 
    VectorStoreConfig
)

logger = logging.getLogger(__name__)


class LayoutRecommender:
    """ML-powered layout recommender using OpenAI embeddings and k-NN."""

    def __init__(
        self, 
        openai_client: OpenAI,
        config: Optional[VectorStoreConfig] = None
    ):
        """
        Initialize the layout recommender.
        
        Args:
            openai_client: OpenAI client for embeddings
            config: Vector store configuration
        """
        self.client = openai_client
        self.config = config or VectorStoreConfig()
        self.embeddings_cache: List[SlideEmbedding] = []
        self.load_vector_store()
        
        logger.info(f"LayoutRecommender initialized with {len(self.embeddings_cache)} embeddings")
    
    def recommend_layout_with_visual_analysis(self, slide: SlidePlan, template_parser, available_layouts: Dict[str, int]) -> LayoutRecommendation:
        """
        Use LLM to select the best layout based on comprehensive visual analysis of templates.
        
        Args:
            slide: SlidePlan to get layout recommendation for
            template_parser: TemplateParser with visual analysis capabilities
            available_layouts: Dictionary of available layout names to indices
            
        Returns:
            LayoutRecommendation with visual analysis-based selection
        """
        try:
            # Get comprehensive layout descriptions
            layout_descriptions = template_parser.create_layout_descriptions_for_llm()
            
            # Create slide content summary
            slide_summary = self._create_slide_content_summary(slide)
            
            # Build LLM prompt
            prompt = self._build_visual_layout_selection_prompt(slide_summary, layout_descriptions)
            
            # Get LLM recommendation
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an expert presentation designer with deep knowledge of visual layout principles and user experience. Analyze the provided slide content and template layouts to recommend the most appropriate layout."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=300
            )
            
            # Parse LLM response
            recommendation = self._parse_visual_layout_response(response.choices[0].message.content, available_layouts)
            
            logger.debug(f"Visual analysis recommendation for slide {slide.index}: {recommendation.slide_type} (confidence: {recommendation.confidence:.2f})")
            return recommendation
            
        except Exception as e:
            logger.error(f"Visual layout analysis failed for slide {slide.index}: {e}")
            # Fallback to basic recommendation
            return self._create_fallback_recommendation(slide, available_layouts)
    
    def _extract_content_keywords(self, slide: SlidePlan) -> List[str]:
        """
        Extract keywords from slide content for semantic matching.
        
        Args:
            slide: SlidePlan to analyze
            
        Returns:
            List of content keywords
        """
        keywords = []
        
        try:
            # Extract from title
            title_words = slide.title.lower().split()
            keywords.extend(title_words)
            
            # Extract from bullets
            bullets = slide.get_bullet_texts()
            for bullet in bullets:
                bullet_words = bullet.lower().split()
                keywords.extend(bullet_words)
            
            # Add slide type
            keywords.append(slide.slide_type)
            
            # Clean up keywords (remove common stop words)
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were'}
            keywords = [k for k in keywords if k not in stop_words and len(k) > 2]
            
            logger.debug(f"Extracted keywords for semantic matching: {keywords[:10]}...")  # Log first 10
            
        except Exception as e:
            logger.error(f"Failed to extract content keywords: {e}")
            
        return keywords
    
    def _create_slide_content_summary(self, slide: SlidePlan) -> str:
        """
        Create a comprehensive summary of slide content for layout selection.
        
        Args:
            slide: SlidePlan to summarize
            
        Returns:
            Detailed content summary string
        """
        summary_parts = [
            f"SLIDE TITLE: {slide.title}",
            f"SLIDE TYPE: {slide.slide_type}"
        ]
        
        # Add bullet points analysis
        bullets = slide.get_bullet_texts()
        if bullets:
            bullet_count = len(bullets)
            avg_length = sum(len(bullet) for bullet in bullets) / bullet_count
            
            summary_parts.append(f"CONTENT: {bullet_count} bullet points (avg {avg_length:.0f} chars each)")
            
            # Sample bullets for context
            if bullet_count <= 3:
                summary_parts.append(f"BULLETS: {'; '.join(bullets)}")
            else:
                summary_parts.append(f"BULLETS: {'; '.join(bullets[:2])}... and {bullet_count-2} more")
        else:
            summary_parts.append("CONTENT: No bullet points")
        
        # Add media information
        media_elements = []
        if slide.image_query:
            media_elements.append("image")
        if slide.chart_data:
            media_elements.append("chart/data visualization")
        
        if media_elements:
            summary_parts.append(f"MEDIA: {', '.join(media_elements)}")
        else:
            summary_parts.append("MEDIA: Text-only")
        
        # Add speaker notes if available
        if slide.speaker_notes:
            summary_parts.append(f"NOTES: {slide.speaker_notes[:100]}...")
        
        return " | ".join(summary_parts)
    
    def _build_visual_layout_selection_prompt(self, slide_summary: str, layout_descriptions: Dict[str, str]) -> str:
        """
        Build a comprehensive prompt for LLM layout selection.
        
        Args:
            slide_summary: Summary of slide content
            layout_descriptions: Detailed descriptions of available layouts
            
        Returns:
            Complete prompt string
        """
        prompt_parts = [
            "Analyze the slide content and recommend the most appropriate template layout.",
            "",
            "SLIDE TO DESIGN:",
            slide_summary,
            "",
            "AVAILABLE TEMPLATE LAYOUTS:"
        ]
        
        # Add layout descriptions
        for layout_name, description in layout_descriptions.items():
            prompt_parts.append(f"• {layout_name}: {description}")
        
        prompt_parts.extend([
            "",
            "SELECTION CRITERIA:",
            "1. Content fit: Does the layout accommodate the amount and type of content?",
            "2. Visual hierarchy: Does the layout support clear information flow?",
            "3. Media integration: Does the layout properly integrate images/charts if needed?",
            "4. Purpose alignment: Does the layout match the slide's communication goal?",
            "5. Professional appearance: Does the layout create a polished, readable result?",
            "",
            "Respond with this exact format:",
            "RECOMMENDED_LAYOUT: [layout_name]",
            "CONFIDENCE: [0.0-1.0]",
            "REASONING: [2-3 sentences explaining why this layout is best]"
        ])
        
        return "\n".join(prompt_parts)
    
    def _parse_visual_layout_response(self, response: str, available_layouts: Dict[str, int]) -> LayoutRecommendation:
        """
        Parse LLM response and create LayoutRecommendation.
        
        Args:
            response: LLM response text
            available_layouts: Available layout mappings
            
        Returns:
            LayoutRecommendation object
        """
        try:
            # Extract recommended layout
            layout_match = re.search(r'RECOMMENDED_LAYOUT:\s*([^\n]+)', response, re.IGNORECASE)
            recommended_layout = layout_match.group(1).strip() if layout_match else None
            
            # Extract confidence
            confidence_match = re.search(r'CONFIDENCE:\s*([0-9.]+)', response, re.IGNORECASE)
            confidence = float(confidence_match.group(1)) if confidence_match else 0.5
            
            # Extract reasoning
            reasoning_match = re.search(r'REASONING:\s*([^\n]+(?:\n[^\n]+)*)', response, re.IGNORECASE)
            reasoning = reasoning_match.group(1).strip() if reasoning_match else "Visual analysis recommendation"
            
            # Validate layout exists
            if recommended_layout and recommended_layout in available_layouts:
                layout_id = available_layouts[recommended_layout]
                
                return LayoutRecommendation(
                    slide_type=recommended_layout,
                    layout_id=layout_id,
                    confidence=min(max(confidence, 0.0), 1.0),  # Clamp to 0-1
                    reasoning=f"Visual analysis: {reasoning}",
                    similar_slides=[],
                    fallback_used=False
                )
            else:
                # Layout not found, use fallback
                logger.warning(f"LLM recommended unknown layout '{recommended_layout}', using fallback")
                fallback_layout = list(available_layouts.keys())[0] if available_layouts else "content"
                
                return LayoutRecommendation(
                    slide_type=fallback_layout,
                    layout_id=available_layouts.get(fallback_layout, 0),
                    confidence=0.3,
                    reasoning=f"Fallback: LLM recommended unknown layout '{recommended_layout}'",
                    similar_slides=[],
                    fallback_used=True
                )
                
        except Exception as e:
            logger.error(f"Failed to parse visual layout response: {e}")
            return self._create_fallback_recommendation(None, available_layouts)
    
    def _create_fallback_recommendation(self, slide: Optional[SlidePlan], available_layouts: Dict[str, int]) -> LayoutRecommendation:
        """
        Create a fallback recommendation when visual analysis fails.
        
        Args:
            slide: Optional slide plan
            available_layouts: Available layout mappings
            
        Returns:
            Fallback LayoutRecommendation
        """
        fallback_layout = "content"  # Safe default
        
        # Try to find a reasonable fallback
        if "content" in available_layouts:
            fallback_layout = "content"
        elif "title" in available_layouts and slide and slide.slide_type == "title":
            fallback_layout = "title"
        elif available_layouts:
            fallback_layout = list(available_layouts.keys())[0]
        
        return LayoutRecommendation(
            slide_type=fallback_layout,
            layout_id=available_layouts.get(fallback_layout, 0),
            confidence=0.2,
            reasoning="Fallback recommendation due to analysis failure",
            similar_slides=[],
            fallback_used=True
        )

    def create_slide_embedding(
        self, 
        slide: SlidePlan,
        slide_id: Optional[str] = None,
        source_file: Optional[str] = None
    ) -> SlideEmbedding:
        """
        Create an embedding for a slide.
        
        Args:
            slide: SlidePlan to create embedding for
            slide_id: Optional unique ID for the slide
            source_file: Optional source file name
            
        Returns:
            SlideEmbedding object with vector representation
        """
        # Generate slide ID if not provided
        if not slide_id:
            slide_id = f"slide_{slide.index}_{int(time.time())}"
        
        # Combine title and content for embedding (T-100: Support hierarchical bullets)
        content_text = slide.title
        bullet_texts = slide.get_bullet_texts()
        if bullet_texts:
            content_text += ": " + ", ".join(bullet_texts)
        
        # Create embedding using OpenAI
        try:
            response = self.client.embeddings.create(
                input=content_text,
                model=self.config.embedding_model
            )
            embedding_vector = response.data[0].embedding
            
            logger.debug(f"Created embedding for slide: {slide.title[:50]}...")
            
        except Exception as e:
            logger.error(f"Failed to create embedding for slide {slide_id}: {e}")
            # Return zero vector as fallback
            embedding_vector = [0.0] * self.config.vector_dimension
        
        return SlideEmbedding(
            slide_id=slide_id,
            title=slide.title,
            content_text=content_text,
            slide_type=slide.slide_type,
            layout_id=slide.layout_id or 1,
            embedding=embedding_vector,
            bullet_count=len(slide.get_bullet_texts()),
            has_image=bool(slide.image_query),
            has_chart=bool(slide.chart_data),
            source_file=source_file,
            created_at=datetime.utcnow().isoformat()
        )

    def recommend_layout(
        self, 
        slide: SlidePlan,
        available_layouts: Optional[Dict[str, int]] = None
    ) -> LayoutRecommendation:
        """
        Recommend a layout for a given slide using ML similarity.
        
        Args:
            slide: SlidePlan to recommend layout for
            available_layouts: Dict mapping layout types to layout IDs
            
        Returns:
            LayoutRecommendation with confidence score and reasoning
        """
        logger.debug(f"Getting layout recommendation for: {slide.title}")
        
        # If no historical data, use fallback
        if not self.embeddings_cache:
            return self._fallback_recommendation(slide, available_layouts)
        
        # Create embedding for the input slide
        query_embedding = self.create_slide_embedding(slide)
        
        # Find similar slides using k-NN
        similar_slides, similarities = self._find_similar_slides(
            query_embedding.embedding, 
            self.config.max_neighbors
        )
        
        if not similar_slides:
            return self._fallback_recommendation(slide, available_layouts)
        
        # Analyze similar slides to determine best layout
        layout_scores = self._calculate_layout_scores(similar_slides, similarities)
        
        # Get top recommendation
        if not layout_scores:
            return self._fallback_recommendation(slide, available_layouts)
        
        best_layout = max(layout_scores.items(), key=lambda x: x[1])
        layout_type, confidence = best_layout
        
        # Find corresponding layout ID
        layout_id = self._get_layout_id(layout_type, available_layouts)
        
        # Generate reasoning
        reasoning = self._generate_reasoning(slide, similar_slides, layout_type)
        
        # Check confidence threshold
        use_ml_recommendation = confidence >= self.config.confidence_threshold
        
        if not use_ml_recommendation:
            fallback_rec = self._fallback_recommendation(slide, available_layouts)
            fallback_rec.fallback_used = True
            return fallback_rec
        
        return LayoutRecommendation(
            slide_type=layout_type,
            layout_id=layout_id,
            confidence=confidence,
            reasoning=reasoning,
            similar_slides=[s.slide_id for s in similar_slides[:3]],
            fallback_used=False
        )

    def add_training_example(self, slide_embedding: SlideEmbedding) -> None:
        """
        Add a new training example to the vector store.
        
        Args:
            slide_embedding: SlideEmbedding to add to training data
        """
        self.embeddings_cache.append(slide_embedding)
        logger.debug(f"Added training example: {slide_embedding.slide_id}")

    def save_vector_store(self) -> None:
        """Save the vector store to disk."""
        try:
            vector_store_path = Path(self.config.vector_store_path)
            vector_store_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save as pickle for efficiency
            with open(vector_store_path, 'wb') as f:
                pickle.dump(self.embeddings_cache, f)
            
            logger.info(f"Saved vector store with {len(self.embeddings_cache)} embeddings to {vector_store_path}")
            
        except Exception as e:
            logger.error(f"Failed to save vector store: {e}")

    def load_vector_store(self) -> None:
        """Load the vector store from disk."""
        try:
            vector_store_path = Path(self.config.vector_store_path)
            
            if not vector_store_path.exists():
                logger.info("No existing vector store found, starting with empty cache")
                self.embeddings_cache = []
                return
            
            with open(vector_store_path, 'rb') as f:
                self.embeddings_cache = pickle.load(f)
            
            logger.info(f"Loaded vector store with {len(self.embeddings_cache)} embeddings")
            
        except Exception as e:
            logger.error(f"Failed to load vector store: {e}")
            self.embeddings_cache = []

    def get_store_stats(self) -> Dict[str, any]:
        """Get statistics about the vector store."""
        if not self.embeddings_cache:
            return {
                "total_embeddings": 0,
                "unique_layouts": 0,
                "layout_distribution": {},
                "source_files": []
            }
        
        layout_counts = {}
        source_files = set()
        
        for embedding in self.embeddings_cache:
            layout_counts[embedding.slide_type] = layout_counts.get(embedding.slide_type, 0) + 1
            if embedding.source_file:
                source_files.add(embedding.source_file)
        
        return {
            "total_embeddings": len(self.embeddings_cache),
            "unique_layouts": len(layout_counts),
            "layout_distribution": layout_counts,
            "source_files": list(source_files)
        }

    def _find_similar_slides(
        self, 
        query_embedding: List[float], 
        k: int
    ) -> Tuple[List[SlideEmbedding], List[float]]:
        """
        Find k most similar slides using cosine similarity.
        
        Args:
            query_embedding: Embedding vector to find similarities for
            k: Number of similar slides to return
            
        Returns:
            Tuple of (similar_slides, similarity_scores)
        """
        if not self.embeddings_cache:
            return [], []
        
        # Convert to numpy arrays for efficient computation
        query_vec = np.array(query_embedding)
        
        similarities = []
        for embedding in self.embeddings_cache:
            embed_vec = np.array(embedding.embedding)
            
            # Cosine similarity
            similarity = np.dot(query_vec, embed_vec) / (
                np.linalg.norm(query_vec) * np.linalg.norm(embed_vec)
            )
            similarities.append(similarity)
        
        # Get top k similar slides
        top_indices = np.argsort(similarities)[-k:][::-1]
        
        similar_slides = [self.embeddings_cache[i] for i in top_indices]
        similar_scores = [similarities[i] for i in top_indices]
        
        # Filter by similarity threshold
        filtered_slides = []
        filtered_scores = []
        
        for slide, score in zip(similar_slides, similar_scores):
            if score >= self.config.similarity_threshold:
                filtered_slides.append(slide)
                filtered_scores.append(score)
        
        return filtered_slides, filtered_scores

    def _calculate_layout_scores(
        self, 
        similar_slides: List[SlideEmbedding], 
        similarities: List[float]
    ) -> Dict[str, float]:
        """
        Calculate weighted scores for each layout type based on similar slides.
        
        Args:
            similar_slides: List of similar slide embeddings
            similarities: Corresponding similarity scores
            
        Returns:
            Dict mapping layout types to confidence scores
        """
        layout_scores = {}
        total_weight = sum(similarities)
        
        if total_weight == 0:
            return {}
        
        for slide, similarity in zip(similar_slides, similarities):
            layout_type = slide.slide_type
            weight = similarity / total_weight
            
            if layout_type not in layout_scores:
                layout_scores[layout_type] = 0.0
            
            layout_scores[layout_type] += weight
        
        return layout_scores

    def _get_layout_id(
        self, 
        layout_type: str, 
        available_layouts: Optional[Dict[str, int]]
    ) -> int:
        if available_layouts:
            if layout_type in available_layouts:
                logger.debug(f"Layout type '{layout_type}' found in available_layouts. Using ID: {available_layouts[layout_type]}")
                return available_layouts[layout_type]
            else:
                logger.warning(
                    f"Layout type '{layout_type}' not found in available_layouts (keys: {list(available_layouts.keys())}). "
                    f"Attempting fallback to 'content' layout from available_layouts."
                )
                if "content" in available_layouts:
                    logger.info(f"Using 'content' layout (ID: {available_layouts['content']}) as fallback for '{layout_type}'.")
                    return available_layouts["content"]
                else:
                    # If 'content' is also not in available_layouts, then consult internal defaults for the original type
                    logger.warning(f"'content' layout also not in available_layouts. Consulting internal default_mappings for '{layout_type}'.")
                    default_id = self.config.default_layout_mappings.get(layout_type)
                    if default_id is not None:
                        logger.info(f"Using internal default ID {default_id} for '{layout_type}'. This ID might not be valid for the current template.")
                        return default_id
                    else:
                        logger.warning(f"Layout type '{layout_type}' also not in internal default_mappings. Falling back to layout ID 0 (or first available).")
                        # Fallback to the first layout in available_layouts if any, else 0
                        if available_layouts: # Check if available_layouts is not empty
                            return next(iter(available_layouts.values()))
                        return 0 # Absolute fallback
        else:
            logger.warning("No available_layouts provided to _get_layout_id. Using internal default_mappings.")
            default_id = self.config.default_layout_mappings.get(layout_type)
            if default_id is not None:
                logger.info(f"Using internal default ID {default_id} for '{layout_type}'.")
                return default_id
            logger.warning(f"Layout type '{layout_type}' not in internal default_mappings. Falling back to layout ID 1 (generic content).")
            return 1 # Default to 1 (often a content slide) if no info at all

    def analyze_content_semantics(self, slide: SlidePlan) -> Dict[str, any]:
        """
        Analyze content semantics to extract layout-relevant features.
        
        This method identifies semantic patterns in slide content that help
        determine the most appropriate layout. Features include comparison
        patterns, list structures, visual cues, and content density.
        
        Args:
            slide: SlidePlan to analyze for semantic features
            
        Returns:
            Dictionary with semantic analysis results
        """
        analysis = {
            "comparison_signals": False,
            "list_structure": "simple",
            "visual_requirements": [],
            "content_density": "normal",
            "hierarchical_content": False,
            "process_flow": False,
            "numeric_data": False,
            "key_concepts": [],
            "layout_hints": []
        }
        
        # Combine all text content for analysis (T-100: Support hierarchical bullets)
        all_text = slide.title.lower()
        bullet_texts = slide.get_bullet_texts()
        if bullet_texts:
            all_text += " " + " ".join(bullet_texts).lower()
        
        # Check for comparison signals
        comparison_keywords = [
            "vs", "versus", "compared", "comparison", "against", "differences",
            "pros and cons", "advantages", "disadvantages", "before after",
            "old new", "current proposed", "option a", "option b"
        ]
        analysis["comparison_signals"] = any(keyword in all_text for keyword in comparison_keywords)
        
        # Analyze list structure complexity (T-100: Support hierarchical bullets)
        if bullet_texts:
            bullet_count = len(bullet_texts)
            avg_bullet_length = sum(len(bullet) for bullet in bullet_texts) / bullet_count
            
            if bullet_count > 6:
                analysis["list_structure"] = "dense"
            elif bullet_count > 3 and avg_bullet_length > 50:
                analysis["list_structure"] = "complex"
            elif any(":" in bullet for bullet in bullet_texts):
                analysis["list_structure"] = "structured"
            
            # Check for hierarchical content (nested concepts) - T-100: Support hierarchical bullets
            hierarchical_keywords = ["including", "such as", "for example", "specifically", "namely"]
            analysis["hierarchical_content"] = any(
                keyword in bullet.lower() for bullet in bullet_texts 
                for keyword in hierarchical_keywords
            )
            
            # Detect actual hierarchy in bullet structure
            if slide.bullet_hierarchy is not None:
                max_level = max(bullet.level for bullet in slide.bullet_hierarchy)
                if max_level > 0:
                    analysis["hierarchical_content"] = True
                    analysis["list_structure"] = "hierarchical"
        
        # Detect visual requirements
        visual_keywords = {
            "chart": ["chart", "graph", "data", "statistics", "metrics", "trends", "analysis"],
            "image": ["image", "photo", "picture", "diagram", "illustration", "visual"],
            "process": ["process", "workflow", "steps", "stages", "phases", "timeline"],
            "table": ["table", "matrix", "grid", "spreadsheet", "data table"]
        }
        
        for visual_type, keywords in visual_keywords.items():
            if any(keyword in all_text for keyword in keywords):
                analysis["visual_requirements"].append(visual_type)
        
        # Check for process flow indicators
        process_indicators = ["step", "phase", "stage", "first", "then", "next", "finally", "process"]
        analysis["process_flow"] = any(indicator in all_text for indicator in process_indicators)
        
        # Detect numeric data presence
        import re
        numbers_pattern = r'\b\d+(?:\.\d+)?(?:%|percent|million|billion|thousand)?\b'
        analysis["numeric_data"] = bool(re.search(numbers_pattern, all_text))
        
        # Extract key concepts (important nouns/topics)
        concept_keywords = [
            "strategy", "market", "revenue", "growth", "customer", "product", 
            "technology", "innovation", "performance", "analysis", "results",
            "team", "project", "goal", "target", "opportunity", "challenge"
        ]
        analysis["key_concepts"] = [concept for concept in concept_keywords if concept in all_text]
        
        # Generate layout hints based on analysis
        if analysis["comparison_signals"]:
            analysis["layout_hints"].append("two_column")
        if "chart" in analysis["visual_requirements"] or analysis["numeric_data"]:
            analysis["layout_hints"].append("content")  # Standard content can handle charts
        if analysis["list_structure"] == "dense":
            analysis["layout_hints"].append("two_column")
        if analysis["process_flow"]:
            analysis["layout_hints"].append("blank")  # More flexibility for custom layouts
        if analysis["hierarchical_content"]:
            analysis["layout_hints"].append("content")
        
        # Determine content density (T-100: Support hierarchical bullets)
        total_chars = len(slide.title) + sum(len(bullet) for bullet in bullet_texts)
        if total_chars > 500:
            analysis["content_density"] = "high"
        elif total_chars < 200:
            analysis["content_density"] = "low"
        
        logger.debug(f"Semantic analysis for '{slide.title}': {analysis}")
        return analysis

    def recommend_layout_with_llm(
        self,
        slide: SlidePlan,
        available_layouts: Optional[Dict[str, int]] = None
    ) -> LayoutRecommendation:
        """
        Recommend layout using LLM-based semantic analysis instead of ML similarity.
        
        Args:
            slide: SlidePlan to recommend layout for
            available_layouts: Dict mapping layout types to layout IDs
            
        Returns:
            LayoutRecommendation with LLM-based confidence and reasoning
        """
        logger.debug(f"Getting LLM-based layout recommendation for: {slide.title}")
        
        # Get semantic analysis
        semantic_analysis = self.analyze_content_semantics(slide)
        
        # Create LLM prompt for layout recommendation
        prompt = self._create_layout_recommendation_prompt(slide, semantic_analysis, available_layouts)
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",  # Use faster model for layout decisions
                messages=[
                    {
                        "role": "system", 
                        "content": "You are an expert presentation designer who selects optimal slide layouts based on content analysis. Provide structured JSON responses."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                temperature=0.1,  # Low temperature for consistent recommendations
                max_tokens=500
            )
            
            # Parse LLM response
            llm_response = response.choices[0].message.content
            recommendation_data = self._parse_llm_recommendation(llm_response, available_layouts)
            
            if recommendation_data:
                return LayoutRecommendation(
                    slide_type=recommendation_data["layout_type"],
                    layout_id=recommendation_data["layout_id"],
                    confidence=recommendation_data["confidence"],
                    reasoning=recommendation_data["reasoning"],
                    similar_slides=[],  # LLM doesn't use historical examples
                    fallback_used=False
                )
                
        except Exception as e:
            logger.warning(f"LLM layout recommendation failed for slide {slide.index}: {e}")
        
        # Fallback to rule-based recommendation
        return self._fallback_recommendation(slide, available_layouts)

    def _create_layout_recommendation_prompt(
        self,
        slide: SlidePlan,
        semantic_analysis: Dict[str, any],
        available_layouts: Optional[Dict[str, int]] = None
    ) -> str:
        """
        Create a structured prompt for LLM layout recommendation.
        
        Args:
            slide: SlidePlan to create prompt for
            semantic_analysis: Results from analyze_content_semantics
            available_layouts: Available layout options
            
        Returns:
            Formatted prompt string
        """
        # Get available layout types (T-99 Extended)
        if available_layouts:
            layout_options = list(available_layouts.keys())
        else:
            layout_options = [
                "title", "content", "two_column", "image", "chart", "section", "blank",
                "image_content", "content_dense", "three_column", "comparison"
            ]
        
        prompt = f"""Analyze this slide content and recommend the best layout:

SLIDE CONTENT:
Title: "{slide.title}"
Bullets: {len(slide.get_bullet_texts())} items
Content: {slide.get_bullet_texts() if slide.get_bullet_texts() else "No bullet content"}

SEMANTIC ANALYSIS:
- Comparison signals: {semantic_analysis.get('comparison_signals', False)}
- List structure: {semantic_analysis.get('list_structure', 'simple')}
- Visual requirements: {semantic_analysis.get('visual_requirements', [])}
- Content density: {semantic_analysis.get('content_density', 'normal')}
- Hierarchical content: {semantic_analysis.get('hierarchical_content', False)}
- Process flow: {semantic_analysis.get('process_flow', False)}
- Numeric data: {semantic_analysis.get('numeric_data', False)}
- Key concepts: {semantic_analysis.get('key_concepts', [])}

AVAILABLE LAYOUTS: {', '.join(layout_options)}

LAYOUT DESCRIPTIONS:
- title: For presentation titles and section headers
- content: Standard layout with title and bullet points area
- two_column: Side-by-side content areas for comparisons
- image: Layout optimized for visual content with text
- chart: Layout designed for data visualization
- section: Section divider with large title
- blank: Maximum flexibility for custom arrangements
- image_content: Hybrid layout combining images with content areas
- content_dense: Optimized layout for high-density content with better spacing
- three_column: Multi-column layout for maximum content organization
- comparison: Specialized layout for side-by-side comparisons

Please recommend the best layout and provide your response in this JSON format:
{{
    "layout_type": "recommended_layout_name",
    "confidence": 0.85,
    "reasoning": "Detailed explanation of why this layout works best for this content",
    "alternative_options": ["layout2", "layout3"]
}}

Focus on matching content characteristics to layout strengths. Consider content density, visual needs, and semantic patterns."""
        
        return prompt

    def _parse_llm_recommendation(
        self,
        llm_response: str,
        available_layouts: Optional[Dict[str, int]] = None
    ) -> Optional[Dict[str, any]]:
        """
        Parse LLM response into structured recommendation data.
        
        Args:
            llm_response: Raw response from LLM
            available_layouts: Available layout mappings
            
        Returns:
            Parsed recommendation data or None if parsing fails
        """
        try:
            # Extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', llm_response, re.DOTALL)
            if not json_match:
                logger.warning("No JSON found in LLM response")
                return None
            
            recommendation_json = json.loads(json_match.group())
            
            layout_type = recommendation_json.get("layout_type")
            confidence = recommendation_json.get("confidence", 0.5)
            reasoning = recommendation_json.get("reasoning", "LLM recommendation")
            
            if not layout_type:
                logger.warning("No layout_type in LLM recommendation")
                return None
            
            # Get layout ID
            layout_id = self._get_layout_id(layout_type, available_layouts)
            
            return {
                "layout_type": layout_type,
                "layout_id": layout_id,
                "confidence": min(max(confidence, 0.0), 1.0),  # Clamp to [0,1]
                "reasoning": f"LLM Analysis: {reasoning}"
            }
            
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to parse LLM recommendation: {e}")
            return None

    def _generate_reasoning(
        self, 
        slide: SlidePlan, 
        similar_slides: List[SlideEmbedding], 
        recommended_layout: str
    ) -> str:
        """
        Generate human-readable reasoning for the recommendation.
        
        Args:
            slide: Original slide
            similar_slides: Similar slides found
            recommended_layout: Recommended layout type
            
        Returns:
            Reasoning string
        """
        if not similar_slides:
            return "No similar slides found in training data"
        
        # Analyze patterns in similar slides
        patterns = []
        
        # Check for comparison keywords
        comparison_words = ["vs", "versus", "compared", "comparison", "against"]
        if any(word in slide.title.lower() for word in comparison_words):
            patterns.append("comparison keywords")
        
        # Check bullet count
        avg_bullets = np.mean([s.bullet_count for s in similar_slides])
        if slide.bullets and len(slide.bullets) > avg_bullets:
            patterns.append("high content density")
        
        # Check for chart/image presence
        has_visual = any(s.has_chart or s.has_image for s in similar_slides)
        if has_visual:
            patterns.append("visual content patterns")
        
        # Build reasoning string
        reasoning_parts = [
            f"Based on {len(similar_slides)} similar slides",
            f"with '{recommended_layout}' layout"
        ]
        
        if patterns:
            reasoning_parts.append(f"Detected patterns: {', '.join(patterns)}")
        
        return ". ".join(reasoning_parts)

    def _fallback_recommendation(
        self, 
        slide: SlidePlan,
        available_layouts: Optional[Dict[str, int]] = None
    ) -> LayoutRecommendation:
        """
        Provide fallback rule-based recommendation when ML fails.
        
        Args:
            slide: SlidePlan to recommend layout for
            available_layouts: Available layout mappings
            
        Returns:
            LayoutRecommendation using rule-based logic
        """
        # Simple rule-based logic
        layout_type = "content"  # Default
        reasoning = "Rule-based fallback: "
        
        # Check for comparison patterns
        comparison_words = ["vs", "versus", "compared", "comparison", "against"]
        if any(word in slide.title.lower() for word in comparison_words):
            layout_type = "two_column"
            reasoning += "detected comparison keywords"
        
        # Check for image/chart needs
        elif slide.image_query or slide.chart_data:
            layout_type = "content"  # Can accommodate visuals
            reasoning += "slide has visual content"
        
        # Check for title slides
        elif slide.slide_type == "title":
            layout_type = "title"
            reasoning += "title slide type"
        
        # Check bullet density (T-100: Support hierarchical bullets)
        elif slide.get_bullet_texts() and len(slide.get_bullet_texts()) > 5:
            layout_type = "content"
            reasoning += "high content density"
        
        else:
            reasoning += "default content layout"
        
        layout_id = self._get_layout_id(layout_type, available_layouts)
        
        return LayoutRecommendation(
            slide_type=layout_type,
            layout_id=layout_id,
            confidence=0.5,  # Medium confidence for rule-based
            reasoning=reasoning,
            similar_slides=[],
            fallback_used=True
        )