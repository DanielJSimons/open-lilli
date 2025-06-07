# Theme Color Extraction

The TemplateParser now supports extracting theme colors from PowerPoint templates, providing access to the standard Office color scheme (dk1, lt1, acc1-6) for use in visual generation and consistent styling.

## Overview

PowerPoint templates define a color scheme in the `ppt/theme/theme1.xml` file within the .pptx archive. This feature automatically extracts these colors and makes them available through the TemplateParser API.

## Supported Color Types

The parser extracts 8 standard theme colors:

- **dk1**: Dark color 1 (typically black for text)
- **lt1**: Light color 1 (typically white for backgrounds)  
- **acc1-6**: Six accent colors for charts, highlights, etc.

## Color Format Support

The parser handles three types of color definitions from the XML:

### 1. sRGB Colors (Explicit RGB)
```xml
<a:accent1>
    <a:srgbClr val="5B9BD5"/>
</a:accent1>
```
Converted to: `#5B9BD5`

### 2. System Colors
```xml
<a:dk1>
    <a:sysClr val="windowText" lastClr="000000"/>
</a:dk1>
```
Converted to: `#000000`

### 3. Preset Colors
```xml
<a:accent1>
    <a:prstClr val="darkBlue"/>
</a:accent1>
```
Converted to: `#000080`

## API Usage

### Basic Theme Color Access

```python
from open_lilli.template_parser import TemplateParser

# Load template
parser = TemplateParser("template.pptx")

# Access theme colors via palette
dark_color = parser.palette["dk1"]
light_color = parser.palette["lt1"] 
accent_color = parser.palette["acc1"]

# Get individual colors with fallback
primary_color = parser.get_theme_color("acc1")
unknown_color = parser.get_theme_color("unknown")  # Returns #000000
```

### Direct Theme Extraction

```python
# Extract colors directly from XML
theme_colors = parser.get_theme_colors()
print(theme_colors)
# Output: {'dk1': '#000000', 'lt1': '#FFFFFF', 'acc1': '#5B9BD5', ...}
```

### Template Information

```python
# Get complete template info including theme colors
info = parser.get_template_info()
theme_colors = info["theme_colors"]
```

## Error Handling

The parser gracefully handles various error conditions:

- **Missing theme file**: Returns empty dict, falls back to defaults
- **Invalid XML**: Returns empty dict, falls back to defaults  
- **Unsupported color formats**: Skips unsupported colors
- **Corrupt PPTX files**: Catches zipfile errors

When theme extraction fails, the parser falls back to a default color palette:

```python
default_palette = {
    "dk1": "#000000",      # Black
    "lt1": "#FFFFFF",      # White
    "acc1": "#1F497D",     # Dark blue
    "acc2": "#4F81BD",     # Medium blue
    "acc3": "#9BBB59",     # Green
    "acc4": "#F79646",     # Orange
    "acc5": "#8064A2",     # Purple
    "acc6": "#4BACC6",     # Light blue
}
```

## Integration with Visual Generation

Theme colors can be used throughout the pipeline for consistent styling:

### Chart Generation
```python
# Use theme colors for chart styling
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.bar(data.keys(), data.values(), color=parser.get_theme_color("acc1"))
ax.set_facecolor(parser.get_theme_color("lt1"))
```

### Custom Visual Elements
```python
# Use accent colors for infographics
colors = [parser.get_theme_color(f"acc{i}") for i in range(1, 7)]
```

## Implementation Details

### XML Structure
The parser looks for the color scheme in this XML structure:
```xml
<a:theme xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">
    <a:themeElements>
        <a:clrScheme name="Office">
            <a:dk1>...</a:dk1>
            <a:lt1>...</a:lt1>
            <a:accent1>...</a:accent1>
            <!-- ... more accent colors ... -->
        </a:clrScheme>
    </a:themeElements>
</a:theme>
```

### Namespace Handling
The parser correctly handles the DrawingML namespace:
```python
namespaces = {
    'a': 'http://schemas.openxmlformats.org/drawingml/2006/main'
}
```

### Preset Color Mapping
Common preset colors are mapped to hex values:
```python
preset_to_hex = {
    'black': '#000000',
    'white': '#FFFFFF', 
    'red': '#FF0000',
    'green': '#008000',
    'blue': '#0000FF',
    'yellow': '#FFFF00',
    'darkBlue': '#000080',
    'darkGreen': '#008000',
    'darkRed': '#800000'
    # ... more mappings
}
```

## Testing

The feature includes comprehensive tests covering:

- Standard Office theme color extraction
- Preset color conversion
- Missing theme file handling
- Invalid XML handling
- Integration with TemplateParser

Run tests with:
```bash
pytest tests/test_theme_color_extraction.py -v
```

## Examples

See `examples/theme_color_demo.py` for a complete demonstration of the theme color extraction functionality.

## Future Enhancements

Potential improvements for future versions:

1. **HSL Color Support**: Handle HSL color definitions
2. **Color Modifications**: Support color transformations (tint, shade, etc.)
3. **Custom Color Schemes**: Allow override of default fallback colors
4. **Color Validation**: Validate extracted colors for accessibility
5. **Multiple Themes**: Support templates with multiple theme variants