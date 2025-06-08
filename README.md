# Open Lilli - AI-Powered PowerPoint Generation

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Open Lilli is a comprehensive AI-powered PowerPoint generation tool inspired by McKinsey's Lilli platform. It transforms text content into professional presentations using sophisticated AI models, template-driven design, and advanced features like ML layout recommendations, visual proofreading, and narrative flow analysis.

## 🌟 Key Features

- **🎯 End-to-End AI Pipeline**: Complete automation from content analysis to slide assembly
- **🎨 Template-Driven Design**: Leverages corporate PowerPoint templates and branding
- **🧠 AI Content Generation**: GPT-4 powered content creation with configurable tone and complexity
- **📊 Smart Visual Generation**: Automatic chart creation and image sourcing with brand alignment
- **🔍 Visual Proofreader**: T-79 compliant AI-powered design issue detection (90%+ accuracy)
- **🔄 Flow Intelligence**: T-80 compliant narrative flow analysis and transition generation
- **🚀 Engagement Tuner**: T-81 compliant verb diversity optimization and rhetorical questions
- **🤖 ML Layout Recommendations**: Machine learning-based slide layout selection
- **✨ Auto-Refine**: Iterative quality improvement with feedback loops
- **🌍 Multilingual Support**: Generate presentations in 26+ languages
- **⚡ Async Processing**: High-performance parallel slide generation

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Core Commands](#core-commands)
- [Configuration](#configuration)
- [Advanced Features](#advanced-features)
- [Template Management](#template-management)
- [Development](#development)
- [API Usage](#api-usage)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

## 🚀 Installation

### Prerequisites

- **Python 3.11 or higher**
- **OpenAI API key** (get one at [platform.openai.com](https://platform.openai.com))
- **PowerPoint template file** (.pptx format)

### Install from Source

```bash
# Clone the repository
git clone https://github.com/your-org/open-lilli.git
cd open-lilli

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install with development dependencies
pip install -e .[dev]

# Run setup wizard
ai-ppt setup
```

### Verify Installation

```bash
# Check installation
ai-ppt --version

# Test API connection
ai-ppt setup

# View available commands
ai-ppt --help
```

## ⚡ Quick Start

### 1. Initial Setup

```bash
# Create project directory
mkdir my-presentation && cd my-presentation

# Set up environment
ai-ppt setup

# Edit .env file with your API key
echo "OPENAI_API_KEY=your_api_key_here" > .env
```

### 2. Basic Presentation Generation

```bash
# Generate presentation from text file
ai-ppt generate \
  --template templates/corporate.pptx \
  --input content/quarterly-report.txt \
  --output q4-presentation.pptx \
  --slides 12 \
  --lang en

# With review and auto-refinement
ai-ppt generate \
  --template templates/corporate.pptx \
  --input content/strategy-doc.md \
  --output strategy-deck.pptx \
  --auto-refine \
  --max-iterations 3 \
  --review
```

### 3. Quick Demo

```bash
# Run complete demo with sample content
python examples/complete_demo.py
```

## 🎛️ Core Commands

### `generate` - Create New Presentations

The primary command for generating presentations from text content.

```bash
ai-ppt generate [OPTIONS]
```

**Required Options:**
- `--template, -t`: Path to PowerPoint template (.pptx)
- `--input, -i`: Path to input content file (.txt, .md)

**Common Options:**
- `--output, -o`: Output file path (default: presentation.pptx)
- `--slides`: Maximum slides to generate (default: 15)
- `--lang, -l`: Language code (default: en)
- `--tone`: Content tone (professional, casual, formal, friendly)
- `--complexity`: Content complexity (basic, intermediate, advanced)

**Quality & Enhancement:**
- `--review / --no-review`: Enable AI review (default: on)
- `--auto-refine`: Automatic iterative improvement
- `--max-iterations`: Max refinement cycles (default: 3)

**Content Control:**
- `--no-images`: Disable image generation
- `--no-charts`: Disable chart generation
- `--assets-dir`: Directory for generated assets

**Performance:**
- `--async`: Use async processing for speed
 - `--model`: OpenAI model (default: gpt-4.1)
- `--verbose, -v`: Detailed output

**Example:**
```bash
ai-ppt generate \
  --template templates/consulting.pptx \
  --input reports/market-analysis.txt \
  --output market-deck.pptx \
  --slides 10 \
  --tone professional \
  --complexity advanced \
  --auto-refine \
  --lang en \
  --verbose
```

### `regenerate` - Update Existing Slides

Selectively regenerate specific slides in an existing presentation.

```bash
ai-ppt regenerate [OPTIONS]
```

**Required Options:**
- `--template, -t`: Template file
- `--input-pptx, -p`: Existing presentation to update
- `--slides, -s`: Comma-separated slide indices (0-based)

**Optional:**
- `--feedback, -f`: Specific feedback to apply
- `--output, -o`: Output file (default: regenerated.pptx)

**Example:**
```bash
# Regenerate slides 2, 5, and 8 with feedback
ai-ppt regenerate \
  --template templates/corporate.pptx \
  --input-pptx existing-deck.pptx \
  --slides 1,4,7 \
  --feedback "Make content more concise and add data visualization" \
  --output updated-deck.pptx
```

### `analyze-template` - Inspect Templates

Analyze PowerPoint templates to understand available layouts and design elements.

```bash
ai-ppt analyze-template template.pptx
```

**Output:**
- Available slide layouts and their capabilities
- Template dimensions and theme colors
- Placeholder analysis for each layout type

### `proofread` - Visual Design Review (T-79)

AI-powered visual proofreading with 90%+ accuracy for design issues.

```bash
ai-ppt proofread [OPTIONS]
```

**Options:**
- `--input, -i`: Presentation file to review
- `--focus`: Specific issue types (capitalization, formatting, consistency, etc.)
- `--test-mode`: Run accuracy validation with seeded errors
- `--output, -o`: Detailed report file (JSON)

**Example:**
```bash
# Focus on capitalization issues
ai-ppt proofread \
  --input presentation.pptx \
  --focus capitalization \
  --output review-report.json \
  --verbose

# Test detection accuracy
ai-ppt proofread \
  --input presentation.pptx \
  --test-mode
```

### `analyze-flow` - Narrative Flow Analysis (T-80)

Analyze and enhance presentation narrative flow with transition generation.

```bash
ai-ppt analyze-flow [OPTIONS]
```

**Options:**
- `--input, -i`: Slides file (JSON or PPTX)
- `--target-coherence`: Coherence score target (default: 4.0/5.0)
- `--insert-transitions`: Add transitions to speaker notes
- `--output, -o`: Flow analysis report

**Example:**
```bash
ai-ppt analyze-flow \
  --input slides.json \
  --target-coherence 4.5 \
  --insert-transitions \
  --output flow-report.json
```

### `enhance-engagement` - Engagement Optimization (T-81)

Enhance presentation engagement through verb diversity and rhetorical techniques.

```bash
ai-ppt enhance-engagement [OPTIONS]
```

**Options:**
- `--input, -i`: Slides file to enhance
- `--target-diversity`: Target verb diversity ratio (default: 30%)
- `--baseline-ratio`: Baseline comparison (default: 15%)
- `--output, -o`: Enhanced slides and analysis

**Example:**
```bash
ai-ppt enhance-engagement \
  --input slides.json \
  --target-diversity 0.35 \
  --output enhanced-analysis.json \
  --verbose
```

### `ingest` - ML Training Data

Build training corpus for ML-based layout recommendations.

```bash
ai-ppt ingest [OPTIONS]
```

**Options:**
- `--pptx`: PowerPoint file or directory to ingest
- `--template, -t`: Optional template for layout detection
- `--vector-store`: Vector store file path (default: layouts.vec)
- `--batch-size`: Processing batch size (default: 50)

**Example:**
```bash
# Ingest historical presentations
ai-ppt ingest \
  --pptx historical-decks/ \
  --template templates/corporate.pptx \
  --vector-store corporate-layouts.vec
```

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in your project directory:

```bash
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4.1
OPENAI_TEMPERATURE=0.3

# Image Generation
UNSPLASH_ACCESS_KEY=your_unsplash_key_here
USE_DALLE=false

# Performance
MAX_CONCURRENT_SLIDES=5
ENABLE_ASYNC_GENERATION=true

# Logging
LOG_LEVEL=INFO
DEBUG_MODE=false
```

### Global Options

All commands support these global options:

- `--debug`: Show detailed error traces
- `--log-file`: Write debug logs to file
- `--verbose, -v`: Increase output verbosity

### Configuration Files

**Generation Config (`config.yaml`):**
```yaml
default_generation:
  max_slides: 15
  tone: professional
  complexity_level: intermediate
  include_images: true
  include_charts: true
  max_iterations: 3

template_mapping:
  corporate: templates/corporate.pptx
  consulting: templates/mckinsey-style.pptx
  academic: templates/university.pptx

quality_gates:
  min_coherence_score: 4.0
  min_engagement_score: 7.0
  max_validation_issues: 3
```

## 🔬 Advanced Features

### Auto-Refine Quality Gates

Open Lilli includes sophisticated quality validation:

```bash
# Enable auto-refine with custom criteria
ai-ppt generate \
  --template templates/corporate.pptx \
  --input content.txt \
  --auto-refine \
  --max-iterations 5
```

**Quality Gates:**
- Content coherence and flow
- Design consistency
- Template compliance
- Engagement metrics
- Visual balance

### ML-Powered Layout Selection

Train the system on your historical presentations:

```bash
# Build training corpus
ai-ppt ingest --pptx historical-decks/ --template templates/corp.pptx

# Use ML recommendations in generation
ai-ppt generate \
  --template templates/corp.pptx \
  --input content.txt \
  --enable-ml-layouts
```

### Corporate Asset Integration

```python
# Configure corporate asset library
from open_lilli.corporate_asset_library import CorporateAssetLibrary
from open_lilli.models import AssetLibraryConfig

config = AssetLibraryConfig(
    dam_api_url="https://api.company.com/assets",
    api_key="YOUR_DAM_API_KEY",
    brand_guidelines_strict=True,
    fallback_to_external=False
)

asset_library = CorporateAssetLibrary(config)
```

### Batch Processing

```bash
# Process multiple content files
for file in content/*.txt; do
  ai-ppt generate \
    --template templates/corporate.pptx \
    --input "$file" \
    --output "presentations/$(basename "$file" .txt).pptx" \
    --async
done
```

## 🎨 Template Management

### Template Requirements

**Supported Features:**
- Standard PowerPoint layouts (Title, Content, Two-Column, etc.)
- Custom corporate layouts
- Theme colors and fonts
- Master slide branding
- Placeholder types (title, content, image, chart)

### Template Analysis

```bash
# Analyze template capabilities
ai-ppt analyze-template templates/corporate.pptx
```

**Output Example:**
```
📁 File: templates/corporate.pptx
📐 Dimensions: 13.3" × 7.5"
🎨 Layouts: 11

Available Layouts:
1. title_slide (index 0)
   • 2 placeholders
   • Has title: True
   • Has content: False

2. content_slide (index 1)
   • 3 placeholders
   • Has title: True
   • Has content: True
   • Has image: False

Theme Colors:
• accent1: #0073E6
• accent2: #FF6B35
```

### Best Practices

**Template Design:**
- Include diverse layout types (title, content, two-column, image)
- Use consistent placeholder naming
- Define clear theme colors
- Test with sample content

**Corporate Branding:**
- Maintain brand color schemes
- Use approved fonts
- Include logo placeholders
- Follow design guidelines

## 🛠️ Development

### Development Setup

```bash
# Clone and setup development environment
git clone https://github.com/your-org/open-lilli.git
cd open-lilli

# Install development dependencies
pip install -e .[dev]

# Install pre-commit hooks
pre-commit install

# Run tests
pytest

# Run linting
pre-commit run --all-files
```

### Testing

```bash
# Run all tests
pytest

# Run specific test categories
pytest -m "unit"          # Unit tests only
pytest -m "e2e"           # End-to-end tests
pytest -m "not e2e"       # Skip E2E tests

# Run with coverage
pytest --cov=open_lilli --cov-report=html

# Test specific modules
pytest tests/test_slide_assembler.py -v
```

### Code Quality

```bash
# Format code
black .
isort .

# Type checking
mypy .

# Linting
flake8

# All quality checks
pre-commit run --all-files
```

## 📚 API Usage

### Python API

```python
import asyncio
from pathlib import Path
from openai import AsyncOpenAI

from open_lilli import (
    ContentProcessor, OutlineGenerator, SlidePlanner,
    ContentGenerator, VisualGenerator, SlideAssembler,
    TemplateParser, Reviewer
)
from open_lilli.models import GenerationConfig

async def generate_presentation():
    # Initialize components
    client = AsyncOpenAI(api_key="your-api-key")
    config = GenerationConfig(
        max_slides=10,
        tone="professional",
        include_images=True
    )
    
    # Process content
    processor = ContentProcessor()
    content = processor.extract_text("content.txt")
    
    # Generate outline
    outline_gen = OutlineGenerator(client)
    outline = await outline_gen.generate_outline_async(content, config)
    
    # Plan slides
    template_parser = TemplateParser("template.pptx")
    planner = SlidePlanner(template_parser)
    slides = planner.plan_slides(outline, config)
    
    # Generate content
    content_gen = ContentGenerator(client, template_parser=template_parser)
    enhanced_slides = await content_gen.generate_content_async(
        slides, config, outline.style_guidance, "en"
    )
    
    # Create visuals
    visual_gen = VisualGenerator("assets/", template_parser.palette)
    visuals = visual_gen.generate_visuals(enhanced_slides)
    
    # Review quality
    reviewer = Reviewer(client)
    feedback = await reviewer.review_presentation_async(enhanced_slides)
    
    # Assemble presentation
    assembler = SlideAssembler(template_parser)
    output_path = assembler.assemble(
        outline, enhanced_slides, visuals, "output.pptx"
    )
    
    return output_path

# Run the async function
presentation_path = asyncio.run(generate_presentation())
print(f"Generated: {presentation_path}")
```

### Custom Components

```python
from open_lilli.models import SlidePlan, GenerationConfig

class CustomContentGenerator:
    """Custom content generation with domain-specific logic."""
    
    def __init__(self, client, domain_knowledge: dict):
        self.client = client
        self.domain_knowledge = domain_knowledge
    
    def enhance_slide_content(self, slide: SlidePlan, config: GenerationConfig) -> SlidePlan:
        # Custom enhancement logic
        if slide.slide_type == "financial":
            slide.bullets = self.add_financial_context(slide.bullets)
        return slide
    
    def add_financial_context(self, bullets: list) -> list:
        # Add domain-specific enhancements
        enhanced = []
        for bullet in bullets:
            if "revenue" in bullet.lower():
                enhanced.append(f"{bullet} (YoY growth: +15%)")
            else:
                enhanced.append(bullet)
        return enhanced
```

## 🐛 Troubleshooting

### Common Issues

**API Key Problems:**
```bash
# Error: OPENAI_API_KEY not set
export OPENAI_API_KEY=your_key_here
# Or add to .env file

# Verify API key works
ai-ppt setup
```

**Template Issues:**
```bash
# Analyze template for compatibility
ai-ppt analyze-template templates/problematic.pptx

# Check for supported layouts
python -c "
from open_lilli.template_parser import TemplateParser
parser = TemplateParser('template.pptx')
print(parser.get_template_info())
"
```

**Generation Failures:**
```bash
# Run with debug output
ai-ppt generate \
  --template template.pptx \
  --input content.txt \
  --debug \
  --log-file debug.log \
  --verbose

# Check debug.log for detailed errors
```

**Performance Issues:**
```bash
# Use async mode for better performance
ai-ppt generate \
  --template template.pptx \
  --input content.txt \
  --async \
  --model gpt-3.5-turbo  # Faster but less capable

# Reduce slide count for testing
ai-ppt generate \
  --template template.pptx \
  --input content.txt \
  --slides 5
```

### Debug Mode

```bash
# Enable comprehensive debugging
export DEBUG_MODE=true
export LOG_LEVEL=DEBUG

ai-ppt generate \
  --template template.pptx \
  --input content.txt \
  --debug \
  --log-file full-debug.log \
  --verbose

# Review logs
tail -f full-debug.log
```

### Memory and Performance

```bash
# Monitor memory usage
pip install memory-profiler
python -m memory_profiler examples/complete_demo.py

# Profile performance
python -m cProfile -o profile.stats examples/complete_demo.py
python -c "
import pstats
p = pstats.Stats('profile.stats')
p.sort_stats('cumulative').print_stats(20)
"
```

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

### Quick Start for Contributors

```bash
# Fork and clone
git clone https://github.com/yourusername/open-lilli.git
cd open-lilli

# Setup development environment
pip install -e .[dev]
pre-commit install

# Create feature branch
git checkout -b feature/amazing-new-feature

# Make changes and test
pytest
pre-commit run --all-files

# Submit pull request
git push origin feature/amazing-new-feature
```

### Areas for Contribution

- **🎯 New AI Features**: Enhanced prompt engineering, multimodal generation
- **🎨 Template Support**: Additional PowerPoint features, design improvements
- **📊 Analytics**: Usage metrics, quality measurements
- **🌍 Internationalization**: Additional language support, cultural adaptations
- **🔧 Performance**: Optimization, caching, parallel processing
- **📚 Documentation**: Tutorials, examples, best practices

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Inspired by [McKinsey Lilli](https://www.mckinsey.com/capabilities/mckinsey-digital/our-insights/the-economic-potential-of-generative-ai-the-next-productivity-frontier) and other AI presentation tools
- Built with [OpenAI GPT models](https://openai.com/gpt-4) for content generation
- Uses [python-pptx](https://python-pptx.readthedocs.io/) for PowerPoint manipulation
- Template design principles from consulting industry best practices

## 📞 Support

- **Documentation**: This README and [CONTRIBUTING.md](CONTRIBUTING.md)
- **Issues**: [GitHub Issues](https://github.com/your-org/open-lilli/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-org/open-lilli/discussions)
- **Examples**: See `examples/` directory for usage patterns

---

**🎯 Ready to create professional presentations with AI? Get started with `ai-ppt setup` and see the magic happen!**