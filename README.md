# Globule

> Advanced AI-powered thought management with modular frontends and schema-driven layouts

![Project Status: Active Development](https://img.shields.io/badge/status-active_development-green)
![Features: CLI Mirroring](https://img.shields.io/badge/feature-cli_mirroring-blue)
![Features: Frontend Modularity](https://img.shields.io/badge/feature-frontend_modularity-blue)
![Features: Layout Templates](https://img.shields.io/badge/feature-layout_templates-blue)

Globule transforms scattered thoughts into organized knowledge through AI-powered processing, intelligent layout management, and flexible interface options. The system features a modular architecture with scriptable CLI commands, visual canvas layouts, and reusable templates.

## The Core Experience

Globule's magic is in its simplicity. Capture any thought, and let the AI handle the rest.

#### 1. Capture Instantly

No need to think about folders or filenames. Just capture the thought.

```bash
$ globule add "The concept of 'progressive overload' in fitness could apply to creative stamina."

$ globule add "A core theme for my next post: discipline isn't about restriction, it's about freedom."
```

#### 2. Synthesize with Ease

When you're ready to write, tell Globule what you're thinking about.

```bash
$ globule draft "my next blog post"
```

Globule's intelligent engine understands these thoughts are related and presents them in a clean, two-pane interface, ready for you to weave them together into a coherent first draft.

## Your Files, Your Computer

Globule organizes your thoughts into a clean, human-readable folder structure right on your local machine. A thought about creative philosophy might be saved as:

`~/globule/philosophy/creativity/applying-progressive-overload.md`

You can browse and edit these files with any tool. No proprietary formats, no lock-in. A single database file, `globule.db`, lives alongside your notes, holding the semantic connections that make the magic possible.

## Getting Started

Globule Phase 2 is now functional! Here's how to get started:

```bash
# Clone and install from source
git clone https://github.com/asavschaeffer/globule
cd globule
pip install -e .

# Learn how Globule works with the Glass Engine tutorial
globule tutorial --mode=interactive

# Start capturing your thoughts
globule add "Your first thought here"

# Draft content from your captured thoughts
globule draft "your topic"
```

## 🏗️ Advanced Architecture

Globule now features a sophisticated modular architecture with multiple interface options and intelligent layout management:

### 🎯 Frontend Modularity

Choose the interface that fits your workflow - all sharing the same powerful backend:

#### 🖥️ Terminal UI (Interactive)
```bash
globule draft "project planning"                    # Default TUI
globule draft "project planning" --frontend=tui     # Explicit
```
Visual canvas with drag-and-drop, real-time AI processing, schema-aware styling.

#### 🌐 Web Interface  
```bash
globule draft "project planning" --frontend=web --port=8000
# Opens http://localhost:8000 with browser-based interface
```
Responsive grid layouts, template management, cross-platform browser access.

#### ⌨️ CLI Scripting
```bash
globule draft "project planning" --frontend=cli --output=results.md
```
Non-interactive batch processing, perfect for automation and LLM chains.

### ⚡ CLI Mirroring System

Key interactive functionality exposed as scriptable commands:

```bash
# Natural language search with AI SQL generation
globule nlsearch "valet maria honda parked yesterday"
globule nlsearch "meetings about budget" --output=meetings.md

# Draft management  
globule add-to-draft "Key insight from analysis"
globule export-draft summary.md --metadata
globule draft-stats

# Frontend discovery
globule frontends  # Show available interfaces and capabilities
```

### 📐 Configuration-Driven Canvas Layouts

Schemas control their own visual presentation across all frontends:

```json
{
  "title": "Valet Schema",
  "canvas_config": {
    "layout": { "type": "widget", "position": "top-left", "size": "small" },
    "tui_style": { "border": "round", "border_color": "$accent" },
    "web_style": { "className": "valet-widget bg-green-50 rounded-lg p-4" }
  }
}
```

**Benefits:**
- Same positioning logic across TUI and Web
- Schema-specific styling (valet=green, academic=blue, technical=orange)
- User customizable via JSON configuration
- DRY architecture: define once, render everywhere

### 🎨 Template System

Save successful canvas layouts as reusable templates:

```bash
# Create default templates
globule skeleton-create-defaults

# List available templates  
globule skeleton-list
# → valet_dashboard: search top-left, stats top-right, summary center
# → research_layout: main center, sources right, notes bottom-left

# Apply template to new content
globule skeleton-apply valet_dashboard --output=dashboard.md

# Template statistics and usage tracking
globule skeleton-stats
```

**Template Features:**
- **Structure without content**: Positions and schemas, not specific data
- **Placeholder substitution**: `{query}` and `{content}` filled dynamically  
- **Cross-platform sharing**: JSON files in `~/.globule/skeletons/`
- **Usage tracking**: Popular templates rise to the top

### 🔄 Unified Workflow Example

```bash
# 1. Capture thoughts (unchanged)
globule add "Valet Maria reported BMW parking issue"
globule add "Need dashboard for valet operations"

# 2. Interactive exploration
globule draft "valet operations" --frontend=tui
# → Drag search results to canvas, organize visually

# 3. Save successful layout as template (via TUI interface)
# → Creates "valet_dashboard" with positioned modules

# 4. Apply template to new searches  
globule nlsearch "valet issues this week" | \
  globule skeleton-apply valet_dashboard --output=weekly_report.md

# 5. Automate with CLI mirroring
#!/bin/bash
for topic in "parking" "incidents" "scheduling"; do
  globule skeleton-apply valet_dashboard \
    --query="$topic operations" --output="$topic_dashboard.md"
done
```

## 🎨 Phase 4: Multi-Modal Intelligence

Globule now processes more than just text! Phase 4 introduces **processor extensions** that intelligently handle different content types through specialized analysis pipelines.

### 📸 Image Processing
```bash
# Capture image thoughts directly
globule add "/path/to/vacation/sunset.jpg"
globule add "https://example.com/diagram.png" 
globule add "data:image/jpeg;base64,/9j/4AAQSkZJRg..."

# AI analyzes images automatically:
# ✓ Scene and object detection
# ✓ EXIF metadata extraction  
# ✓ Thumbnail generation
# ✓ Multi-modal LLM descriptions
```

### 🧠 Intelligent Content Routing
Globule's **ProcessorRouter** automatically selects the best processing method:

```bash
# Text content → Text processor (existing behavior)
globule add "Meeting notes about quarterly planning"

# Image content → Image processor (new!)  
globule add "/photos/whiteboard_brainstorm.jpg"

# Mixed search across all content types
globule nlsearch "project diagrams from last week"
```

### 🔍 Enhanced Search Capabilities
```bash
# Search by content domain
globule search --domain=image --category=media

# Find processor-specific content
globule nlsearch "photos of code architecture" --processor=image

# Get processing statistics
globule stats --processors
```

### 🚀 Extensible Architecture
Phase 4 provides a foundation for future content types:
- **Phase 5**: Audio processing (speech-to-text, music analysis)
- **Phase 6**: Video processing (scene detection, transcription)  
- **Phase 7**: Document processing (PDF analysis, OCR)

**Key Benefits:**
- **Backward Compatible**: All existing functionality preserved
- **Performance Optimized**: Concurrent processing with <5ms routing overhead
- **Gracefully Degrading**: System works even when specialized processors fail
- **Developer Friendly**: [Processor Development Guide](docs/guides/processor-development-guide.md)

## The Glass Engine: Transparent Software

Globule features the **Glass Engine** - a revolutionary tutorial system that shows you exactly how the software works while you learn to use it. No black boxes, no guesswork, complete transparency.

We're still updating this at the moment.

**Choose your learning style:**

- 🎓 **New to Globule?** → `globule tutorial --mode=interactive` (guided hands-on learning)
- 🎪 **Want to see capabilities?** → `globule tutorial --mode=demo` (professional showcase)  
- 🔧 **Need technical details?** → `globule tutorial --mode=debug` (raw system analysis)

The Glass Engine embodies our philosophy: *"Let the user see exactly how the pistons fire while teaching them to drive."*

**📚 Learn more:** [Glass Engine Guide](docs/glass-engine-guide.md) | [Quick Start](docs/glass-engine-quick-start.md)

## The Vision: Where We're Going

The initial version of Globule is focused on the core experience of capture and synthesis. But this is just the foundation for a much larger vision.

-   **Empowering Workflows:** Soon, you'll be able to teach Globule about *your* specific types of information (like `Recipes` or `Code Snippets`), enabling custom formatting and perfect integration with tools like Obsidian.
-   **Personalized Organization:** You will be able to tune Globule's brain, defining your own templates for how files and folders are named and organized, making the semantic filesystem truly your own.

Our ultimate goal is to build a new foundational layer for personal computing—one that understands context, not just commands.

## Contributing

This project is currently in a design-heavy phase. If you are interested in the architecture, design philosophy, and the future of semantic computing, we welcome you to explore our **[Project Wiki](https://github.com/asavschaeffer/globule/wiki)** where the system is being designed in the open.