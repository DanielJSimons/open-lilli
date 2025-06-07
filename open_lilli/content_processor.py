"""Content processing module for extracting and cleaning text input."""

import logging
import re
from pathlib import Path
from typing import Union

logger = logging.getLogger(__name__)


class ContentProcessor:
    """Handles input content analysis and structuring."""

    def __init__(self):
        """Initialize the content processor."""
        self.supported_extensions = {'.txt', '.md'}

    def extract_text(self, input_source: Union[str, Path]) -> str:
        """
        Extract and clean text from various input sources.
        
        Args:
            input_source: Path to file or raw text string
            
        Returns:
            Cleaned text string
            
        Raises:
            FileNotFoundError: If file path doesn't exist
            ValueError: If file type is not supported
        """
        # If it's a string that looks like a file path, convert to Path
        if isinstance(input_source, str) and self._looks_like_filepath(input_source):
            input_source = Path(input_source)
        
        # Handle file paths
        if isinstance(input_source, Path):
            return self._extract_from_file(input_source)
        
        # Handle raw text strings
        elif isinstance(input_source, str):
            return self._clean_text(input_source)
        
        else:
            raise ValueError(f"Unsupported input type: {type(input_source)}")

    def _looks_like_filepath(self, text: str) -> bool:
        """Check if a string looks like a file path."""
        # Simple heuristic: contains path separators and has an extension
        return ('/' in text or '\\' in text) and '.' in text.split('/')[-1]

    def _extract_from_file(self, file_path: Path) -> str:
        """Extract text from a file."""
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if file_path.suffix.lower() not in self.supported_extensions:
            raise ValueError(
                f"Unsupported file type: {file_path.suffix}. "
                f"Supported types: {', '.join(self.supported_extensions)}"
            )
        
        logger.info(f"Extracting text from file: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            logger.info(f"Successfully extracted {len(content)} characters from {file_path}")
            return self._clean_text(content)
            
        except UnicodeDecodeError:
            # Try with different encoding
            logger.warning(f"UTF-8 failed, trying latin-1 encoding for {file_path}")
            with open(file_path, 'r', encoding='latin-1') as f:
                content = f.read()
            
            return self._clean_text(content)

    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize text content.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        if not text or not text.strip():
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove common artifacts
        text = self._remove_artifacts(text)
        
        # Normalize line endings
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        # Remove excessive blank lines (more than 2 consecutive)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        logger.debug(f"Cleaned text: {len(text)} characters")
        return text.strip()

    def _remove_artifacts(self, text: str) -> str:
        """Remove common document artifacts and noise."""
        # Remove page numbers (simple pattern)
        text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
        
        # Remove email artifacts like ">"
        text = re.sub(r'^>\s*', '', text, flags=re.MULTILINE)
        
        # Remove excessive punctuation
        text = re.sub(r'[.]{4,}', '...', text)
        text = re.sub(r'[-]{4,}', '---', text)
        
        # Remove common OCR artifacts
        text = re.sub(r'\s+[|]\s+', ' ', text)  # OCR table artifacts
        
        return text

    def extract_sections(self, text: str) -> dict[str, str]:
        """
        Extract logical sections from text based on headers and structure.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary mapping section names to content
        """
        sections = {}
        current_section = "Introduction"
        current_content = []
        
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if this looks like a header
            if self._is_header(line):
                # Save previous section
                if current_content:
                    sections[current_section] = '\n'.join(current_content).strip()
                    current_content = []
                
                # Start new section
                current_section = self._clean_header(line)
            else:
                current_content.append(line)
        
        # Don't forget the last section
        if current_content:
            sections[current_section] = '\n'.join(current_content).strip()
        
        logger.info(f"Extracted {len(sections)} sections: {list(sections.keys())}")
        return sections

    def _is_header(self, line: str) -> bool:
        """Check if a line looks like a section header."""
        line = line.strip()
        
        # Markdown-style headers
        if line.startswith('#'):
            return True
        
        # All caps (likely header)
        if len(line) > 3 and line.isupper() and not line.isdigit():
            return True
        
        # Short lines that don't end with punctuation (likely headers)
        if (len(line) < 60 and 
            not line.endswith(('.', '!', '?', ':', ';', ',')) and
            not line.isdigit()):
            # Check if it contains typical header words
            header_words = {
                'overview', 'introduction', 'background', 'summary', 
                'conclusion', 'methodology', 'results', 'analysis',
                'objectives', 'goals', 'strategy', 'plan', 'approach'
            }
            if any(word in line.lower() for word in header_words):
                return True
        
        return False

    def _clean_header(self, header: str) -> str:
        """Clean and normalize header text."""
        # Remove markdown symbols
        header = re.sub(r'^#+\s*', '', header)
        
        # Remove common numbering
        header = re.sub(r'^\d+[\.\)]\s*', '', header)
        
        # Title case if all caps
        if header.isupper():
            header = header.title()
        
        return header.strip()

    def get_word_count(self, text: str) -> int:
        """Get approximate word count of text."""
        if not text:
            return 0
        return len(text.split())

    def get_reading_time(self, text: str, wpm: int = 200) -> int:
        """
        Estimate reading time in minutes.
        
        Args:
            text: Text to analyze
            wpm: Words per minute reading speed
            
        Returns:
            Estimated reading time in minutes
        """
        word_count = self.get_word_count(text)
        return max(1, round(word_count / wpm))