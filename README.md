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

### Regenerating Slides

Use the `regenerate` command to update specific slides in an existing deck:

```bash
ai-ppt regenerate \
  --template templates/corporate.pptx \
  --input-pptx existing.pptx \
  --slides 2,4 \
  --output updated.pptx
```

This command extracts the selected slides, regenerates their content and visuals, and patches the presentation with the new versions.

### Auto-Refine Loop

Passing `--auto-refine` to `ai-ppt generate` enables iterative refinement. The tool reviews the output after each generation and regenerates failing slides until all quality gates pass or `--max-iterations` is reached.

### Corporate Asset Library

Configure a `CorporateAssetLibrary` using `AssetLibraryConfig`:

```python
from open_lilli.corporate_asset_library import CorporateAssetLibrary
from open_lilli.models import AssetLibraryConfig

asset_config = AssetLibraryConfig(
    dam_api_url="https://api.company.com/assets",
    api_key="YOUR_API_KEY",
    brand_guidelines_strict=True,
    fallback_to_external=False,
)
asset_library = CorporateAssetLibrary(asset_config)
```

Provide this via `VisualExcellenceConfig` or use `--strict-brand` on the CLI to enforce only corporate-approved assets.

### Ingesting Decks for ML Layouts

Build the training corpus for layout recommendations by ingesting existing presentations:

```bash
ai-ppt ingest --pptx historical_decks/ --template templates/corporate.pptx
```

This extracts slides, creates embeddings and updates `layouts.vec` for improved ML-based layout suggestions.

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
