"""ML-assisted layout recommender for intelligent slide layout selection."""

import json
import logging
import pickle
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
        
        # Combine title and content for embedding
        content_text = slide.title
        if slide.bullets:
            content_text += ": " + ", ".join(slide.bullets)
        
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
            bullet_count=len(slide.bullets),
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
        """
        Get layout ID for a given layout type.
        
        Args:
            layout_type: Type of layout (e.g., "two_column")
            available_layouts: Available layout mappings
            
        Returns:
            Layout ID (integer)
        """
        if available_layouts and layout_type in available_layouts:
            return available_layouts[layout_type]
        
        # Default mappings if no available_layouts provided
        default_mappings = {
            "title": 0,
            "content": 1,
            "two_column": 3,
            "image": 5,
            "chart": 1,
            "section": 2,
            "blank": 6
        }
        
        return default_mappings.get(layout_type, 1)

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
        
        # Check bullet density
        elif slide.bullets and len(slide.bullets) > 5:
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