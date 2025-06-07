# Open Lilli

AI-powered PowerPoint generation tool inspired by McKinsey Lilli, built with Python and OpenAI.

## Features

- Template-driven slide generation using corporate .pptx templates
- AI-powered content creation and structuring
- Multilingual support
- Visual generation (charts, images, icons)
- Self-critique and iterative refinement

## Quick Start

### Installation

```bash
# Clone and install
git clone <repository-url>
cd open-lilli
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e .[dev]
```

### Configuration

1. Copy `.env.example` to `.env`
2. Add your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

### Usage

```bash
ai-ppt generate \
  --template templates/corporate.pptx \
  --input content/script.txt \
  --lang en \
  --slides 10 \
  --output presentation.pptx
```

## Development

### Setup

```bash
# Install pre-commit hooks
pre-commit install

# Run tests
pytest

# Run linting
pre-commit run --all-files
```

### Architecture

The tool follows a modular pipeline architecture:

1. **ContentProcessor** - Extracts and cleans input text
2. **OutlineGenerator** - Creates structured outline using GPT-4
3. **TemplateParser** - Analyzes PowerPoint template layouts
4. **SlidePlanner** - Maps content to slide layouts
5. **ContentGenerator** - Generates polished slide content
6. **VisualGenerator** - Creates charts and sources images
7. **SlideAssembler** - Builds final PowerPoint presentation
8. **Reviewer** - Provides AI-powered quality feedback

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting
5. Submit a pull request

## License

MIT License - see LICENSE file for details.