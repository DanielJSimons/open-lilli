"""Template ingestion system for building ML training data from historical presentations."""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from openai import OpenAI

from .layout_recommender import LayoutRecommender
from .models import IngestionResult, SlideEmbedding, VectorStoreConfig
from .regeneration_manager import RegenerationManager
from .template_parser import TemplateParser

logger = logging.getLogger(__name__)


class TemplateIngestionPipeline:
    """Pipeline for ingesting historical presentations into the ML training corpus."""

    def __init__(
        self, 
        openai_client: OpenAI,
        vector_config: Optional[VectorStoreConfig] = None
    ):
        """
        Initialize the ingestion pipeline.
        
        Args:
            openai_client: OpenAI client for embeddings
            vector_config: Vector store configuration
        """
        self.client = openai_client
        self.vector_config = vector_config or VectorStoreConfig()
        self.layout_recommender = LayoutRecommender(openai_client, vector_config)
        
        logger.info("Template ingestion pipeline initialized")

    def ingest_presentation_file(
        self, 
        pptx_path: Path,
        template_path: Optional[Path] = None
    ) -> Tuple[List[SlideEmbedding], List[str]]:
        """
        Ingest a single presentation file and extract slide embeddings.
        
        Args:
            pptx_path: Path to PowerPoint presentation
            template_path: Optional template file for layout detection
            
        Returns:
            Tuple of (slide_embeddings, errors)
        """
        logger.info(f"Ingesting presentation: {pptx_path}")
        
        embeddings = []
        errors = []
        
        try:
            # Use RegenerationManager to extract slide content
            if template_path:
                template_parser = TemplateParser(str(template_path))
                regen_manager = RegenerationManager(
                    template_parser, None, None  # We only need extraction
                )
            else:
                # Create a minimal mock for extraction
                regen_manager = RegenerationManager(None, None, None)
            
            # Extract slides from presentation
            outline, slides = regen_manager.extract_slides_from_presentation(pptx_path)
            
            logger.info(f"Extracted {len(slides)} slides from {pptx_path.name}")
            
            # Create embeddings for each slide
            for i, slide in enumerate(slides):
                try:
                    slide_id = f"{pptx_path.stem}_slide_{i}"
                    
                    # Create embedding
                    embedding = self.layout_recommender.create_slide_embedding(
                        slide, 
                        slide_id=slide_id,
                        source_file=pptx_path.name
                    )
                    
                    embeddings.append(embedding)
                    logger.debug(f"Created embedding for slide {i}: {slide.title[:50]}...")
                    
                except Exception as e:
                    error_msg = f"Failed to create embedding for slide {i} in {pptx_path.name}: {e}"
                    errors.append(error_msg)
                    logger.error(error_msg)
            
        except Exception as e:
            error_msg = f"Failed to process presentation {pptx_path}: {e}"
            errors.append(error_msg)
            logger.error(error_msg)
        
        return embeddings, errors

    def ingest_directory(
        self, 
        corpus_path: Path,
        template_path: Optional[Path] = None,
        file_pattern: str = "*.pptx"
    ) -> IngestionResult:
        """
        Ingest all presentations in a directory.
        
        Args:
            corpus_path: Path to directory containing presentations
            template_path: Optional template file for layout detection
            file_pattern: File pattern to match (default: "*.pptx")
            
        Returns:
            IngestionResult with statistics and errors
        """
        logger.info(f"Starting directory ingestion: {corpus_path}")
        start_time = time.time()
        
        # Find all presentation files
        pptx_files = list(corpus_path.glob(file_pattern))
        if not pptx_files:
            logger.warning(f"No files matching '{file_pattern}' found in {corpus_path}")
        
        logger.info(f"Found {len(pptx_files)} presentation files to process")
        
        # Track results
        total_slides = 0
        successful_embeddings = 0
        failed_embeddings = 0
        all_errors = []
        unique_layouts = set()
        initial_store_size = len(self.layout_recommender.embeddings_cache)
        
        # Process each file
        for i, pptx_file in enumerate(pptx_files):
            logger.info(f"Processing file {i+1}/{len(pptx_files)}: {pptx_file.name}")
            
            try:
                embeddings, errors = self.ingest_presentation_file(
                    pptx_file, template_path
                )
                
                # Add to training corpus
                for embedding in embeddings:
                    self.layout_recommender.add_training_example(embedding)
                    unique_layouts.add(embedding.slide_type)
                
                total_slides += len(embeddings) + len(errors)
                successful_embeddings += len(embeddings)
                failed_embeddings += len(errors)
                all_errors.extend(errors)
                
                logger.info(f"Processed {pptx_file.name}: {len(embeddings)} successful, {len(errors)} failed")
                
            except Exception as e:
                error_msg = f"Critical error processing {pptx_file.name}: {e}"
                all_errors.append(error_msg)
                logger.error(error_msg)
        
        # Save updated vector store
        try:
            self.layout_recommender.save_vector_store()
            logger.info("Vector store saved successfully")
        except Exception as e:
            error_msg = f"Failed to save vector store: {e}"
            all_errors.append(error_msg)
            logger.error(error_msg)
        
        # Calculate results
        processing_time = time.time() - start_time
        final_store_size = len(self.layout_recommender.embeddings_cache)
        
        result = IngestionResult(
            total_slides_processed=total_slides,
            successful_embeddings=successful_embeddings,
            failed_embeddings=failed_embeddings,
            unique_layouts_found=len(unique_layouts),
            vector_store_size=final_store_size,
            processing_time_seconds=processing_time,
            errors=all_errors
        )
        
        logger.info(f"Ingestion complete: {result.success_rate:.1f}% success rate, "
                   f"{final_store_size - initial_store_size} new embeddings added")
        
        return result

    def ingest_single_file(
        self, 
        pptx_path: Path,
        template_path: Optional[Path] = None
    ) -> IngestionResult:
        """
        Ingest a single presentation file.
        
        Args:
            pptx_path: Path to PowerPoint presentation
            template_path: Optional template file for layout detection
            
        Returns:
            IngestionResult with statistics and errors
        """
        logger.info(f"Ingesting single file: {pptx_path}")
        start_time = time.time()
        
        initial_store_size = len(self.layout_recommender.embeddings_cache)
        
        embeddings, errors = self.ingest_presentation_file(pptx_path, template_path)
        
        # Add to training corpus
        unique_layouts = set()
        for embedding in embeddings:
            self.layout_recommender.add_training_example(embedding)
            unique_layouts.add(embedding.slide_type)
        
        # Save vector store
        try:
            self.layout_recommender.save_vector_store()
        except Exception as e:
            error_msg = f"Failed to save vector store: {e}"
            errors.append(error_msg)
            logger.error(error_msg)
        
        processing_time = time.time() - start_time
        final_store_size = len(self.layout_recommender.embeddings_cache)
        
        result = IngestionResult(
            total_slides_processed=len(embeddings) + len(errors),
            successful_embeddings=len(embeddings),
            failed_embeddings=len(errors),
            unique_layouts_found=len(unique_layouts),
            vector_store_size=final_store_size,
            processing_time_seconds=processing_time,
            errors=errors
        )
        
        logger.info(f"Single file ingestion complete: {result.success_rate:.1f}% success rate")
        
        return result

    def validate_corpus_quality(
        self, 
        min_slides_per_layout: int = 10
    ) -> Dict[str, Any]:
        """
        Validate the quality of the ingested corpus.
        
        Args:
            min_slides_per_layout: Minimum slides needed per layout type
            
        Returns:
            Dictionary with validation results
        """
        store_stats = self.layout_recommender.get_store_stats()
        
        # Check layout distribution
        layout_dist = store_stats["layout_distribution"]
        sparse_layouts = {
            layout: count for layout, count in layout_dist.items()
            if count < min_slides_per_layout
        }
        
        # Calculate coverage metrics
        total_slides = store_stats["total_embeddings"]
        coverage_score = len(layout_dist) / max(1, len(sparse_layouts)) if sparse_layouts else 10.0
        
        validation_result = {
            "total_slides": total_slides,
            "unique_layouts": store_stats["unique_layouts"],
            "layout_distribution": layout_dist,
            "sparse_layouts": sparse_layouts,
            "coverage_score": min(10.0, coverage_score),
            "recommendations": []
        }
        
        # Generate recommendations
        if total_slides < 100:
            validation_result["recommendations"].append(
                f"Consider ingesting more presentations (current: {total_slides}, recommended: 100+)"
            )
        
        if sparse_layouts:
            validation_result["recommendations"].append(
                f"Some layouts have few examples: {list(sparse_layouts.keys())}"
            )
        
        if store_stats["unique_layouts"] < 5:
            validation_result["recommendations"].append(
                "Consider adding presentations with more diverse layout types"
            )
        
        return validation_result

    def get_corpus_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the current training corpus.
        
        Returns:
            Dictionary with corpus statistics
        """
        store_stats = self.layout_recommender.get_store_stats()
        
        # Calculate additional metrics
        embeddings = self.layout_recommender.embeddings_cache
        
        if not embeddings:
            return {
                "message": "No training data available",
                "recommendations": ["Run 'ai-ppt ingest' to build training corpus"]
            }
        
        # Analyze content patterns
        avg_bullets = sum(e.bullet_count for e in embeddings) / len(embeddings)
        has_images_pct = sum(1 for e in embeddings if e.has_image) / len(embeddings) * 100
        has_charts_pct = sum(1 for e in embeddings if e.has_chart) / len(embeddings) * 100
        
        return {
            "total_embeddings": store_stats["total_embeddings"],
            "unique_layouts": store_stats["unique_layouts"],
            "layout_distribution": store_stats["layout_distribution"],
            "source_files": len(store_stats["source_files"]),
            "avg_bullets_per_slide": round(avg_bullets, 1),
            "slides_with_images_pct": round(has_images_pct, 1),
            "slides_with_charts_pct": round(has_charts_pct, 1),
            "vector_store_path": self.vector_config.vector_store_path,
            "embedding_model": self.vector_config.embedding_model
        }