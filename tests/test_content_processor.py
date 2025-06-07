"""Tests for content processor."""

import tempfile
from pathlib import Path

import pytest

from open_lilli.content_processor import ContentProcessor


class TestContentProcessor:
    """Tests for ContentProcessor class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.processor = ContentProcessor()

    def test_extract_text_from_string(self):
        """Test extracting text from a plain string."""
        text = "This is a test document with some content."
        result = self.processor.extract_text(text)
        
        assert result == text

    def test_extract_text_from_file(self):
        """Test extracting text from a file."""
        content = "This is test file content.\nWith multiple lines."
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(content)
            temp_path = Path(f.name)
        
        try:
            result = self.processor.extract_text(temp_path)
            assert "This is test file content." in result
            assert "With multiple lines." in result
        finally:
            temp_path.unlink()

    def test_extract_text_file_not_found(self):
        """Test handling of missing files."""
        with pytest.raises(FileNotFoundError):
            self.processor.extract_text(Path("nonexistent.txt"))

    def test_extract_text_unsupported_extension(self):
        """Test handling of unsupported file types."""
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as f:
            temp_path = Path(f.name)
        
        try:
            with pytest.raises(ValueError, match="Unsupported file type"):
                self.processor.extract_text(temp_path)
        finally:
            temp_path.unlink()

    def test_clean_text_whitespace(self):
        """Test text cleaning for whitespace."""
        messy_text = "  This   has    excessive   whitespace.  \n\n\n\n  "
        result = self.processor._clean_text(messy_text)
        
        assert result == "This has excessive whitespace."

    def test_clean_text_artifacts(self):
        """Test removal of common artifacts."""
        text_with_artifacts = """
        This is content.
        
        15
        
        > This is quoted
        Some text........ with dots
        """
        
        result = self.processor._clean_text(text_with_artifacts)
        
        assert "This is quoted" not in result  # Email quote removed
        assert "..." in result  # Excessive dots normalized

    def test_extract_sections_markdown(self):
        """Test section extraction with markdown headers."""
        text = """
        # Introduction
        This is the intro content.
        
        # Methods
        This describes the methods used.
        
        # Results
        Here are the results.
        """
        
        sections = self.processor.extract_sections(text)
        
        assert len(sections) == 3
        assert "Introduction" in sections
        assert "Methods" in sections
        assert "Results" in sections
        assert "intro content" in sections["Introduction"]

    def test_extract_sections_caps_headers(self):
        """Test section extraction with ALL CAPS headers."""
        text = """
        OVERVIEW
        This is overview content.
        
        DETAILED ANALYSIS
        This is detailed analysis.
        """
        
        sections = self.processor.extract_sections(text)
        
        assert len(sections) == 2
        assert "Overview" in sections
        assert "Detailed Analysis" in sections

    def test_is_header_detection(self):
        """Test header detection logic."""
        assert self.processor._is_header("# Introduction")
        assert self.processor._is_header("OVERVIEW")
        assert self.processor._is_header("Background Information")
        
        # These should not be headers
        assert not self.processor._is_header("This is a long sentence that continues.")
        assert not self.processor._is_header("123")
        assert not self.processor._is_header("End with period.")

    def test_word_count(self):
        """Test word counting."""
        text = "This is a test with five words."
        count = self.processor.get_word_count(text)
        assert count == 8  # "This", "is", "a", "test", "with", "five", "words."

    def test_reading_time(self):
        """Test reading time estimation."""
        # 200 words should take 1 minute at 200 wpm
        text = " ".join(["word"] * 200)
        time_minutes = self.processor.get_reading_time(text, wpm=200)
        assert time_minutes == 1
        
        # 400 words should take 2 minutes
        text = " ".join(["word"] * 400)
        time_minutes = self.processor.get_reading_time(text, wpm=200)
        assert time_minutes == 2

    def test_empty_text_handling(self):
        """Test handling of empty or whitespace-only text."""
        assert self.processor.extract_text("") == ""
        assert self.processor.extract_text("   ") == ""
        assert self.processor.get_word_count("") == 0
        assert self.processor.get_reading_time("") == 1  # Minimum 1 minute

    def test_looks_like_filepath(self):
        """Test filepath detection heuristic."""
        assert self.processor._looks_like_filepath("/path/to/file.txt")
        assert self.processor._looks_like_filepath("./file.md")
        assert self.processor._looks_like_filepath("C:\\Windows\\file.txt")
        
        # These should not look like filepaths
        assert not self.processor._looks_like_filepath("Just some text")
        assert not self.processor._looks_like_filepath("No extension here")
        assert not self.processor._looks_like_filepath("Has.dots but no path")