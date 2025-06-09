# Template-Driven Content Planning Architecture

## Core Principle
**The template defines HOW content should be distributed, not our rules.**

## Current Flawed Approach
```
1. Generate content with our structure (title, bullets, etc.)
2. Apply our rules about when to use subtitles/footers
3. Force content into template placeholders
4. Hide "unused" placeholders
```

## New Template-First Approach
```
1. Analyze SELECTED template layout comprehensively
2. Understand template's intended content structure from placeholders
3. Generate/adapt content to FIT the template's design intent
4. Populate ALL placeholders according to template purpose
```

## Implementation Strategy

### Phase 1: Template Intent Analysis
For each selected layout, analyze:
- **Placeholder Purpose**: What is each placeholder designed for?
- **Content Relationships**: How do placeholders relate to each other?
- **Visual Hierarchy**: What's the intended information flow?
- **Semantic Intent**: What type of content does this layout expect?

### Phase 2: Content Adaptation
Instead of forcing pre-generated content into templates:
- **Analyze original content intent**
- **Adapt content structure to match template expectations**
- **Generate appropriate content for each placeholder type**
- **Ensure content coherence across all placeholders**

### Phase 3: Template-Driven Population
- **Respect every placeholder in the selected layout**
- **Generate content specifically for subtitle/footer/header placeholders**
- **Use LLM to create appropriate content for each placeholder type**
- **Never leave placeholders empty unless template intends it**

## Example: "Meet the Team" Template
```
Template has:
- Title placeholder: "Meet Our Team"
- Subtitle placeholder: Team tagline/description
- 3 image placeholders: Team member photos
- 3 text placeholders: Member names/roles
- Footer placeholder: Contact info

Content adaptation:
- Generate team tagline for subtitle
- Prepare member info for text areas
- Request team photos for image areas
- Add department contact for footer
```

## Technical Implementation

### 1. Enhanced Layout Analysis
```python
def analyze_template_intent(layout_index, content_context):
    """Analyze what the template expects for content."""
    layout = template.slide_layouts[layout_index]
    
    # Deep analysis of placeholder relationships
    placeholder_analysis = analyze_placeholder_semantics(layout)
    
    # LLM-based intent understanding
    template_intent = llm_analyze_template_purpose(layout, placeholder_analysis)
    
    return template_intent
```

### 2. Content Structure Adaptation
```python
def adapt_content_to_template(original_content, template_intent):
    """Reshape content to match template expectations."""
    
    # Use LLM to understand how to distribute content
    content_mapping = llm_map_content_to_placeholders(
        original_content, 
        template_intent
    )
    
    return content_mapping
```

### 3. Template-Driven Population
```python
def populate_template_completely(slide, content_mapping, template_intent):
    """Fill ALL placeholders according to template design."""
    
    for placeholder in slide.placeholders:
        placeholder_purpose = template_intent.get_purpose(placeholder)
        content_for_placeholder = content_mapping.get_content(placeholder_purpose)
        
        if content_for_placeholder:
            populate_placeholder(placeholder, content_for_placeholder)
        else:
            # Generate appropriate content for this placeholder type
            generated_content = llm_generate_content_for_placeholder(
                placeholder_purpose, 
                original_content_context
            )
            populate_placeholder(placeholder, generated_content)
```

## Benefits

1. **True Template Respect**: Templates work as designed
2. **Better Visual Harmony**: Content fits template's visual intent
3. **No Empty Placeholders**: Every element serves its purpose
4. **Semantic Coherence**: Content matches template semantics
5. **Professional Results**: Presentations look intentionally designed

## Implementation Priority

1. **Template Intent Analysis**: Understand what each layout expects
2. **Content Adaptation Logic**: Reshape content to fit templates
3. **LLM-Driven Content Generation**: Create content for specific placeholder types
4. **Complete Template Population**: Fill all placeholders appropriately