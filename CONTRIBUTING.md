# Contributing to Open Lilli

Thank you for your interest in contributing to Open Lilli! This document provides guidelines and information for contributors.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Contributing Process](#contributing-process)
- [Code Style](#code-style)
- [Testing](#testing)
- [Documentation](#documentation)
- [Submitting Changes](#submitting-changes)

## Code of Conduct

This project follows a standard code of conduct. Please be respectful and professional in all interactions.

## Getting Started

### Prerequisites

- Python 3.11 or higher
- OpenAI API key
- Git

### Fork and Clone

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/yourusername/open-lilli.git
   cd open-lilli
   ```

## Development Setup

### 1. Create Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 2. Install Dependencies

```bash
# Install package in development mode
pip install -e .[dev]

# Install pre-commit hooks
pre-commit install
```

### 3. Environment Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env and add your OpenAI API key
OPENAI_API_KEY=your_api_key_here
```

### 4. Verify Setup

```bash
# Run tests
pytest

# Run linting
pre-commit run --all-files

# Test CLI
ai-ppt --help
```

## Contributing Process

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/bug-description
```

### 2. Make Changes

- Follow the existing code style and patterns
- Add tests for new functionality
- Update documentation as needed
- Ensure all tests pass

### 3. Commit Changes

```bash
git add .
git commit -m "descriptive commit message"
```

## Code Style

### Python Style Guide

- Follow PEP 8
- Use type hints for all function parameters and return values
- Maximum line length: 88 characters (Black default)
- Use descriptive variable and function names

### Code Formatting

We use automated code formatting tools:

- **Black**: Code formatting
- **isort**: Import sorting
- **flake8**: Linting
- **mypy**: Type checking

Run formatting:
```bash
# Format code
black .
isort .

# Check linting
flake8
mypy .
```

### Example Code Style

```python
from typing import List, Optional

from pydantic import BaseModel


class Example(BaseModel):
    """Example class with proper documentation."""
    
    name: str
    items: List[str] = []
    description: Optional[str] = None
    
    def process_items(self, filter_empty: bool = True) -> List[str]:
        """
        Process items with optional filtering.
        
        Args:
            filter_empty: Whether to filter out empty strings
            
        Returns:
            List of processed items
        """
        processed = [item.strip() for item in self.items]
        
        if filter_empty:
            processed = [item for item in processed if item]
        
        return processed
```

## Testing

### Test Structure

- Unit tests: `tests/test_*.py`
- Integration tests: `tests/test_*_integration.py`
- End-to-end tests: `tests/test_e2e.py`

### Writing Tests

```python
import pytest
from unittest.mock import Mock

from open_lilli.models import SlidePlan


class TestYourFeature:
    """Tests for your feature."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.test_slide = SlidePlan(
            index=0,
            slide_type="content",
            title="Test Slide",
            bullets=["Point 1", "Point 2"]
        )
    
    def test_basic_functionality(self):
        """Test basic functionality."""
        assert self.test_slide.title == "Test Slide"
        assert len(self.test_slide.bullets) == 2
    
    @pytest.mark.parametrize("slide_type,expected", [
        ("title", True),
        ("content", False),
    ])
    def test_slide_type_detection(self, slide_type, expected):
        """Test slide type detection."""
        slide = SlidePlan(index=0, slide_type=slide_type, title="Test", bullets=[])
        # Your test logic here
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_models.py

# Run with coverage
pytest --cov=open_lilli

# Run only unit tests
pytest -m "not e2e"

# Run end-to-end tests
pytest -m e2e
```

## Documentation

### Docstring Style

Use Google-style docstrings:

```python
def generate_outline(text: str, language: str = "en") -> Outline:
    """
    Generate a presentation outline from text.
    
    Args:
        text: Input text to analyze
        language: Language code for the presentation
        
    Returns:
        Generated outline with slide plans
        
    Raises:
        ValueError: If text is empty or invalid
        
    Example:
        >>> generator = OutlineGenerator(client)
        >>> outline = generator.generate_outline("Business content")
        >>> print(outline.slide_count)
        5
    """
```

### README Updates

When adding new features, update the README.md with:
- Feature description
- Usage examples
- Configuration options

### API Documentation

For new public APIs, add examples to the `examples/` directory.

## Submitting Changes

### 1. Ensure Quality

Before submitting, ensure:

- [ ] All tests pass (`pytest`)
- [ ] Code is formatted (`pre-commit run --all-files`)
- [ ] Type hints are present (`mypy .`)
- [ ] Documentation is updated
- [ ] Changes are covered by tests

### 2. Create Pull Request

1. Push your branch to your fork
2. Create a pull request on GitHub
3. Fill out the pull request template
4. Wait for review

### Pull Request Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Added tests for new functionality
- [ ] All tests pass
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes (or documented)
```

## Component Architecture

### Adding New Components

When adding new pipeline components:

1. Create the main class in `open_lilli/`
2. Add corresponding tests in `tests/`
3. Update the CLI if needed
4. Add integration to the main pipeline
5. Document the component

### Component Structure

```python
# open_lilli/new_component.py
class NewComponent:
    """Brief description of the component."""
    
    def __init__(self, config: ComponentConfig):
        """Initialize the component."""
        pass
    
    def process(self, input_data: InputType) -> OutputType:
        """Main processing method."""
        pass

# tests/test_new_component.py
class TestNewComponent:
    """Tests for NewComponent."""
    
    def test_basic_functionality(self):
        """Test basic functionality."""
        pass
```

## Issue Templates

### Bug Report

When reporting bugs, include:
- Python version
- Operating system
- Steps to reproduce
- Expected vs actual behavior
- Error messages/stack traces

### Feature Request

When requesting features:
- Use case description
- Proposed solution
- Alternative solutions considered
- Additional context

## Getting Help

- Check existing issues and discussions
- Ask questions in GitHub Discussions
- Review the documentation
- Look at existing code examples

## Recognition

Contributors will be recognized in:
- CONTRIBUTORS.md file
- Release notes for significant contributions
- GitHub contributor graphs

Thank you for contributing to Open Lilli!