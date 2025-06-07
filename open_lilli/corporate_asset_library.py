"""Corporate asset library connector for brand-compliant image and icon sourcing.

This module implements T-53: Plug in internal DAM API; fallback to Unsplash disabled when --strict-brand.
"""

import logging
import os
import requests
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from urllib.parse import urljoin, quote_plus
import json
import hashlib

from .models import AssetLibraryConfig

logger = logging.getLogger(__name__)


class CorporateAssetLibrary:
    """Connects to corporate Digital Asset Management (DAM) systems for brand-compliant assets."""

    def __init__(self, config: Optional[AssetLibraryConfig] = None):
        """
        Initialize the corporate asset library connector.
        
        Args:
            config: Asset library configuration
        """
        self.config = config or AssetLibraryConfig()
        self.session = requests.Session()
        
        # Set up authentication if provided
        if self.config.api_key:
            self.session.headers.update({
                'Authorization': f'Bearer {self.config.api_key}',
                'User-Agent': 'OpenLilli-AssetLibrary/1.0'
            })
        
        # Cache directory for downloaded assets
        self.cache_dir = Path.home() / '.open_lilli' / 'asset_cache'
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"CorporateAssetLibrary initialized, strict mode: {self.config.brand_guidelines_strict}")

    def search_assets(
        self,
        query: str,
        asset_type: str = "image",
        limit: int = 10
    ) -> List[Dict]:
        """
        Search for assets in the corporate library.
        
        Args:
            query: Search query
            asset_type: Type of asset (image, icon, logo, etc.)
            limit: Maximum number of results
            
        Returns:
            List of asset metadata dictionaries
        """
        try:
            if not self.config.dam_api_url:
                logger.warning("No DAM API URL configured")
                return []
            
            # Construct search URL
            search_endpoint = urljoin(self.config.dam_api_url, 'search')
            
            params = {
                'q': query,
                'type': asset_type,
                'limit': limit,
                'format': 'json'
            }
            
            # Add asset type filter if specified in config
            if asset_type in self.config.preferred_asset_types:
                params['preferred'] = 'true'
            
            logger.debug(f"Searching corporate assets: {query} ({asset_type})")
            
            response = self.session.get(search_endpoint, params=params, timeout=10)
            response.raise_for_status()
            
            results = response.json()
            assets = results.get('assets', [])
            
            # Filter by size if specified
            if self.config.max_asset_size_mb > 0:
                assets = [
                    asset for asset in assets
                    if asset.get('size_mb', 0) <= self.config.max_asset_size_mb
                ]
            
            logger.info(f"Found {len(assets)} corporate assets for query: {query}")
            return assets
            
        except Exception as e:
            logger.error(f"Corporate asset search failed: {e}")
            return []

    def download_asset(
        self,
        asset_metadata: Dict,
        output_dir: Optional[Path] = None
    ) -> Optional[Path]:
        """
        Download an asset from the corporate library.
        
        Args:
            asset_metadata: Asset metadata from search results
            output_dir: Optional output directory (defaults to cache)
            
        Returns:
            Path to downloaded asset or None if failed
        """
        try:
            asset_url = asset_metadata.get('download_url') or asset_metadata.get('url')
            if not asset_url:
                logger.error("No download URL in asset metadata")
                return None
            
            # Determine output directory
            if output_dir is None:
                output_dir = self.cache_dir
            
            # Generate filename from asset metadata
            asset_id = asset_metadata.get('id', 'unknown')
            asset_name = asset_metadata.get('name', 'asset')
            asset_ext = asset_metadata.get('format', 'jpg')
            
            # Create safe filename
            safe_name = self._sanitize_filename(asset_name)
            filename = f"{safe_name}_{asset_id}.{asset_ext}"
            output_path = output_dir / filename
            
            # Check if already cached
            if output_path.exists():
                logger.debug(f"Using cached asset: {output_path}")
                return output_path
            
            # Download asset
            logger.debug(f"Downloading corporate asset: {asset_url}")
            
            response = self.session.get(asset_url, timeout=30)
            response.raise_for_status()
            
            # Verify file size
            content_length = len(response.content)
            size_mb = content_length / (1024 * 1024)
            
            if size_mb > self.config.max_asset_size_mb:
                logger.warning(f"Asset too large: {size_mb:.1f}MB > {self.config.max_asset_size_mb}MB")
                return None
            
            # Save asset
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(response.content)
            
            logger.info(f"Downloaded corporate asset: {output_path} ({size_mb:.1f}MB)")
            return output_path
            
        except Exception as e:
            logger.error(f"Asset download failed: {e}")
            return None

    def get_brand_approved_image(
        self,
        query: str,
        slide_index: int,
        fallback_allowed: bool = None
    ) -> Optional[Path]:
        """
        Get a brand-approved image for a slide.
        
        Args:
            query: Image search query
            slide_index: Slide index for filename
            fallback_allowed: Whether to allow external fallback (overrides config)
            
        Returns:
            Path to image file or None if not found
        """
        try:
            # Check if fallback is allowed
            if fallback_allowed is None:
                fallback_allowed = self.config.fallback_to_external
            
            # Search corporate library first
            assets = self.search_assets(query, asset_type="image", limit=5)
            
            if assets:
                # Try to download first suitable asset
                for asset in assets:
                    image_path = self.download_asset(asset)
                    if image_path:
                        return image_path
            
            # If strict brand mode, don't use external sources
            if self.config.brand_guidelines_strict:
                logger.warning(f"No brand-approved image found for '{query}' and strict mode enabled")
                return self._create_brand_placeholder(query, slide_index)
            
            # Use external fallback if allowed
            if fallback_allowed:
                logger.info(f"No corporate assets found for '{query}', using external fallback")
                return self._get_external_fallback_image(query, slide_index)
            
            # Create placeholder
            return self._create_brand_placeholder(query, slide_index)
            
        except Exception as e:
            logger.error(f"Brand image retrieval failed: {e}")
            return None

    def get_brand_icon(
        self,
        icon_name: str,
        size: str = "medium",
        color_variant: str = "primary"
    ) -> Optional[Path]:
        """
        Get a brand-approved icon.
        
        Args:
            icon_name: Name or description of icon
            size: Icon size (small, medium, large)
            color_variant: Color variant (primary, secondary, white, etc.)
            
        Returns:
            Path to icon file or None if not found
        """
        try:
            # Search for icons
            query = f"{icon_name} icon {size} {color_variant}"
            assets = self.search_assets(query, asset_type="icon", limit=3)
            
            if assets:
                # Prefer exact matches
                for asset in assets:
                    if (icon_name.lower() in asset.get('name', '').lower() and
                        size in asset.get('tags', [])):
                        icon_path = self.download_asset(asset)
                        if icon_path:
                            return icon_path
                
                # Try any matching icon
                for asset in assets:
                    icon_path = self.download_asset(asset)
                    if icon_path:
                        return icon_path
            
            logger.warning(f"No brand-approved icon found: {icon_name}")
            return None
            
        except Exception as e:
            logger.error(f"Brand icon retrieval failed: {e}")
            return None

    def validate_asset_compliance(self, asset_path: Path) -> Dict[str, Union[bool, str, List[str]]]:
        """
        Validate an asset against brand guidelines.
        
        Args:
            asset_path: Path to asset file
            
        Returns:
            Validation results dictionary
        """
        results = {
            'compliant': True,
            'issues': [],
            'warnings': [],
            'metadata': {}
        }
        
        try:
            if not asset_path.exists():
                results['compliant'] = False
                results['issues'].append('Asset file not found')
                return results
            
            # Check file size
            size_mb = asset_path.stat().st_size / (1024 * 1024)
            if size_mb > self.config.max_asset_size_mb:
                results['compliant'] = False
                results['issues'].append(f'File too large: {size_mb:.1f}MB > {self.config.max_asset_size_mb}MB')
            
            # Check file format
            allowed_formats = ['.jpg', '.jpeg', '.png', '.svg', '.gif']
            if asset_path.suffix.lower() not in allowed_formats:
                results['warnings'].append(f'Uncommon file format: {asset_path.suffix}')
            
            # If we have DAM API, check if asset is approved
            if self.config.dam_api_url:
                compliance_result = self._check_dam_compliance(asset_path)
                if not compliance_result.get('approved', True):
                    results['compliant'] = False
                    results['issues'].append('Asset not found in approved corporate library')
            
            results['metadata'] = {
                'size_mb': size_mb,
                'format': asset_path.suffix.lower(),
                'filename': asset_path.name
            }
            
        except Exception as e:
            logger.error(f"Asset compliance validation failed: {e}")
            results['compliant'] = False
            results['issues'].append(f'Validation error: {e}')
        
        return results

    def _check_dam_compliance(self, asset_path: Path) -> Dict:
        """Check if asset exists in corporate DAM system."""
        try:
            # Generate asset hash for lookup
            asset_hash = self._calculate_file_hash(asset_path)
            
            # Query DAM API for asset by hash
            check_endpoint = urljoin(self.config.dam_api_url, 'verify')
            params = {'hash': asset_hash}
            
            response = self.session.get(check_endpoint, params=params, timeout=5)
            
            if response.status_code == 200:
                return response.json()
            else:
                return {'approved': False, 'reason': 'Asset not found in DAM'}
                
        except Exception as e:
            logger.debug(f"DAM compliance check failed: {e}")
            return {'approved': True, 'reason': 'Could not verify'}

    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of file for verification."""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()

    def _get_external_fallback_image(
        self,
        query: str,
        slide_index: int
    ) -> Optional[Path]:
        """Get external image as fallback when corporate assets not available."""
        try:
            # This would integrate with the existing visual generator
            # For now, return None to indicate external fallback needed
            logger.debug(f"External fallback requested for: {query}")
            return None
            
        except Exception as e:
            logger.error(f"External fallback failed: {e}")
            return None

    def _create_brand_placeholder(self, query: str, slide_index: int) -> Path:
        """Create a brand-compliant placeholder image."""
        try:
            from PIL import Image, ImageDraw, ImageFont
            
            # Use corporate colors if available
            primary_color = "#1F497D"  # Default corporate blue
            background_color = "#FFFFFF"
            text_color = "#404040"
            
            # Create placeholder image
            width, height = 1200, 675  # 16:9 aspect ratio
            img = Image.new('RGB', (width, height), color=background_color)
            draw = ImageDraw.Draw(img)
            
            # Draw border with corporate color
            border_width = 8
            draw.rectangle(
                [0, 0, width-1, height-1], 
                outline=primary_color, 
                width=border_width
            )
            
            # Add text
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 48)
                small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
            except:
                font = ImageFont.load_default()
                small_font = ImageFont.load_default()
            
            # Main text
            main_text = "Corporate Asset Placeholder"
            bbox = draw.textbbox((0, 0), main_text, font=font)
            text_width = bbox[2] - bbox[0]
            text_x = (width - text_width) // 2
            text_y = height // 2 - 60
            
            draw.text((text_x, text_y), main_text, fill=primary_color, font=font)
            
            # Query text
            query_text = f"Query: {query}"
            bbox = draw.textbbox((0, 0), query_text, font=small_font)
            query_width = bbox[2] - bbox[0]
            query_x = (width - query_width) // 2
            query_y = text_y + 80
            
            draw.text((query_x, query_y), query_text, fill=text_color, font=small_font)
            
            # Brand compliance notice
            notice_text = "Brand-compliant placeholder • Contact marketing for approved assets"
            bbox = draw.textbbox((0, 0), notice_text, font=small_font)
            notice_width = bbox[2] - bbox[0]
            notice_x = (width - notice_width) // 2
            notice_y = height - 60
            
            draw.text((notice_x, notice_y), notice_text, fill=text_color, font=small_font)
            
            # Save placeholder
            filename = f"brand_placeholder_slide_{slide_index}_{self._sanitize_filename(query)}.png"
            output_path = self.cache_dir / filename
            
            img.save(output_path, 'PNG', quality=95)
            
            logger.info(f"Created brand placeholder: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to create brand placeholder: {e}")
            return None

    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename for safe file system usage."""
        import re
        
        # Replace problematic characters
        filename = re.sub(r'[^\w\s-]', '', filename)
        filename = re.sub(r'\s+', '_', filename)
        filename = filename.lower()
        
        # Limit length
        if len(filename) > 50:
            filename = filename[:50]
        
        return filename or 'asset'

    def get_library_status(self) -> Dict[str, Union[bool, str, int]]:
        """
        Get status of corporate asset library connection.
        
        Returns:
            Status information dictionary
        """
        status = {
            'connected': False,
            'api_url': self.config.dam_api_url,
            'strict_mode': self.config.brand_guidelines_strict,
            'fallback_enabled': self.config.fallback_to_external,
            'cache_size': 0,
            'last_error': None
        }
        
        try:
            # Check cache size
            if self.cache_dir.exists():
                cache_files = list(self.cache_dir.glob('*'))
                status['cache_size'] = len(cache_files)
            
            # Test API connection if configured
            if self.config.dam_api_url and self.config.api_key:
                health_endpoint = urljoin(self.config.dam_api_url, 'health')
                response = self.session.get(health_endpoint, timeout=5)
                
                if response.status_code == 200:
                    status['connected'] = True
                    health_data = response.json()
                    status['api_version'] = health_data.get('version', 'unknown')
                    status['total_assets'] = health_data.get('total_assets', 0)
                else:
                    status['last_error'] = f"API returned {response.status_code}"
            
        except Exception as e:
            status['last_error'] = str(e)
        
        return status

    def clear_cache(self) -> int:
        """
        Clear the asset cache directory.
        
        Returns:
            Number of files removed
        """
        removed_count = 0
        
        try:
            if self.cache_dir.exists():
                for file_path in self.cache_dir.glob('*'):
                    if file_path.is_file():
                        file_path.unlink()
                        removed_count += 1
                
            logger.info(f"Cleared asset cache: {removed_count} files removed")
            
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
        
        return removed_count

    def get_asset_recommendations(self, slide_content: str) -> List[Dict]:
        """
        Get asset recommendations based on slide content.
        
        Args:
            slide_content: Text content of the slide
            
        Returns:
            List of recommended assets
        """
        try:
            # Extract keywords from slide content
            keywords = self._extract_keywords(slide_content)
            
            recommendations = []
            
            # Search for each keyword
            for keyword in keywords[:3]:  # Limit to top 3 keywords
                assets = self.search_assets(keyword, limit=2)
                for asset in assets:
                    asset['keyword'] = keyword
                    asset['relevance_score'] = self._calculate_relevance(keyword, slide_content)
                    recommendations.append(asset)
            
            # Sort by relevance
            recommendations.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
            
            return recommendations[:5]  # Return top 5
            
        except Exception as e:
            logger.error(f"Asset recommendation failed: {e}")
            return []

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract relevant keywords from text."""
        import re
        
        # Simple keyword extraction
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
        
        # Business-relevant keywords
        business_keywords = {
            'growth', 'revenue', 'profit', 'market', 'strategy', 'customer',
            'product', 'service', 'analysis', 'performance', 'team', 'project',
            'innovation', 'technology', 'partnership', 'expansion', 'success'
        }
        
        # Filter for business keywords
        relevant_keywords = [word for word in words if word in business_keywords]
        
        # Add unique words that appear multiple times
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1
        
        frequent_words = [word for word, count in word_counts.items() if count > 1 and len(word) > 4]
        
        # Combine and deduplicate
        keywords = list(set(relevant_keywords + frequent_words))
        
        return keywords[:10]  # Return top 10

    def _calculate_relevance(self, keyword: str, content: str) -> float:
        """Calculate relevance score for keyword in content."""
        content_lower = content.lower()
        keyword_lower = keyword.lower()
        
        # Count occurrences
        count = content_lower.count(keyword_lower)
        
        # Basic relevance score
        score = min(count * 0.2, 1.0)
        
        # Boost if keyword appears in important positions
        if keyword_lower in content_lower[:100]:  # Beginning of content
            score += 0.3
        
        return score