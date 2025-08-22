# Globule CLI Reference

Complete reference for Globule's command-line interface. All commands follow the clean GlobuleAPI architecture for consistent behavior and reliability.

## Overview

The Globule CLI provides scriptable access to all core functionality:
- **Thought Management**: Add, search, and organize your thoughts
- **AI Analysis**: Semantic search, clustering, and natural language queries
- **Canvas Layouts**: Template-based content organization
- **Developer Tools**: Glass Engine tutorials and system diagnostics

## Global Options

All commands support these global flags:

```bash
--verbose, -v    # Enable detailed output and debug logging
--help          # Show command help and usage examples
```

## Core Commands

### `globule add`

Add a new thought to your collection.

```bash
# Basic usage
globule add "Your thought here"

# Examples
globule add "Machine learning could improve our search relevance"
globule add "Meeting notes: Q4 planning discussion with Sarah"
globule add "Technical debt: refactor the authentication module"
```

**What happens:**
1. Text is processed through the orchestration engine
2. AI generates semantic embeddings via Ollama
3. Content is parsed for structured metadata  
4. Thought is stored with intelligent file organization
5. Returns unique globule ID for reference

### `globule search`

Perform semantic search using natural language queries.

```bash
# Basic search
globule search "artificial intelligence"

# Limit results  
globule search "project ideas" --limit 5

# Verbose output with metadata
globule search "technical documentation" --verbose

# Examples
globule search "budget meetings"
globule search "customer feedback" --limit 10
globule search "code review notes" --verbose
```

**Output Format:**
```
Found 3 similar thoughts:

1. Machine learning could improve our search relevance...
   ID: a1b2c3d4
   Created: 2025-08-21T10:30:22

2. AI ethics considerations for our recommendation engine...
   ID: e5f6g7h8  
   Created: 2025-08-20T15:45:10
```

### `globule draft`

Interactive drafting interface with multiple frontend options.

```bash
# Open TUI for interactive drafting
globule draft "blog post ideas"

# Web interface  
globule draft "quarterly report" --frontend web --port 8080

# CLI-only mode (immediate output)
globule draft "meeting summary" --frontend cli --output summary.md

# Additional options
globule draft "research notes" --limit 50 --host 0.0.0.0
```

**Frontend Options:**
- `tui` (default): Rich terminal interface with canvas
- `web`: Browser-based interface  
- `cli`: Direct output without interaction

### `globule cluster`

Discover semantic clusters and themes in your thoughts.

```bash
# Basic clustering analysis
globule cluster

# Verbose output with details
globule cluster --verbose  

# Export results to file
globule cluster --export analysis.json

# Example output
globule cluster --verbose
```

**Example Output:**
```
Analyzing semantic clusters in your thoughts...

Discovered 3 semantic clusters:

1. Technical Architecture (8 thoughts)
   Description: Software design and system architecture discussions
   Keywords: api, database, microservices, scalability
   Confidence: 0.87

2. Project Management (5 thoughts)  
   Description: Planning and coordination activities
   Keywords: timeline, budget, stakeholders, deadlines
   Confidence: 0.92
```

## Advanced Commands

### `globule nlsearch`

Natural language database queries with AI-powered SQL generation.

```bash
# Ask questions about your data
globule nlsearch "How many thoughts mention Python?"
globule nlsearch "What are my most recent technical notes?"
globule nlsearch "Show thoughts from last week about meetings"

# Complex queries
globule nlsearch "Count thoughts by category this month"
globule nlsearch "Find the longest thought I've written"
```

**What happens:**
1. Natural language question is converted to SQL
2. Query is executed against your globule database
3. Results are formatted as a readable table

**Example Output:**
```
Result for: 'How many thoughts contain the word python?'

┌───────────────┬───────┐
│ Description   │ Count │
├───────────────┼───────┤
│ Python Thoughts│   12  │
└───────────────┴───────┘
```

### `globule skeleton`

Manage canvas layout templates for consistent organization.

#### `globule skeleton list`
```bash
# Show all available templates
globule skeleton list

# Example output:
# Available Skeletons:
# - research_layout: Academic research organization template
# - project_dashboard: Project management overview template  
# - technical_analysis: Technical documentation structure
```

#### `globule skeleton apply`
```bash
# Apply a template to organize content
globule skeleton apply research_layout

# Generated Modules:
# - main_content: Primary research findings
# - sources: Reference materials and citations
# - methodology: Research approach and methods
```

#### `globule skeleton stats`
```bash
# Show template statistics
globule skeleton stats

# Skeleton Stats:
# - Total Templates: 5
# - Most Used: research_layout (12 uses)
# - Average Modules: 3.2
```

#### `globule skeleton create-defaults`
```bash
# Create default template collection
globule skeleton create-defaults

# Created 3 default skeletons: research_layout, project_dashboard, technical_analysis
```

## System Commands

### `globule reconcile`

Reconcile files on disk with the database for consistency.

```bash
globule reconcile

# Example output:
# Starting file-database reconciliation...
# Reconciliation Complete:
#   Files Processed: 142
#   Files Added: 3  
#   Files Updated: 1
#   Errors: 0
```

### `globule tutorial`

Run the Glass Engine tutorial system in different modes.

```bash
# Interactive mode (board meeting ready)
globule tutorial --mode interactive

# Demo mode (quick professional showcase)  
globule tutorial --mode demo

# Debug mode (developer deep dive)
globule tutorial --mode debug
```

**Tutorial Modes:**
- **Interactive**: Guided walkthrough with user control, perfect for presentations
- **Demo**: Automated professional showcase highlighting AI capabilities
- **Debug**: Raw technical output with performance metrics for developers

## Usage Examples

### Daily Workflow
```bash
# Morning: Add thoughts from overnight
globule add "Idea for improving customer onboarding flow"
globule add "Bug report: search sometimes returns empty results"

# Midday: Find related information  
globule search "customer onboarding" --limit 5

# Evening: Analyze patterns
globule cluster --verbose --export daily_insights.json
```

### Research Workflow
```bash
# Gather information
globule search "machine learning ethics" --limit 20

# Organize with templates
globule skeleton apply research_layout

# Draft synthesis  
globule draft "ML ethics summary" --frontend tui
```

### Data Analysis
```bash
# Query your thought patterns
globule nlsearch "What topics do I think about most?"
globule nlsearch "How many ideas did I capture this month?"

# Export insights
globule cluster --export thought_analysis.json
```

## Advanced Usage

### Scripting and Automation

```bash
#!/bin/bash
# Daily summary script

echo "Generating daily thought summary..."

# Search for today's thoughts  
globule search "today" --limit 10 > daily_thoughts.txt

# Analyze patterns
globule cluster --export daily_patterns.json

# Generate summary draft
globule draft "daily summary" --frontend cli --output summary.md

echo "Summary ready: summary.md"
```

### Integration with Other Tools

```bash
# Pipe to other commands
globule search "project ideas" | grep -i "urgent"

# Export for external analysis
globule nlsearch "all technical thoughts" | jq '.[] | .content' > tech_thoughts.txt

# Combine with system tools
globule cluster --export clusters.json && python analyze_clusters.py clusters.json
```

## Configuration

Globule uses intelligent defaults but can be configured:

```bash
# Use different embedding models (via Ollama)
export GLOBULE_EMBEDDING_MODEL="llama2"

# Custom database location
export GLOBULE_DB_PATH="/custom/path/globule.db"

# Default output directory
export GLOBULE_OUTPUT_DIR="./outputs"
```

## Error Handling

All commands provide clear error messages and appropriate exit codes:

```bash
# Check command success
if globule search "nonexistent topic"; then
    echo "Search completed successfully"
else
    echo "Search failed with code $?"
fi
```

**Exit Codes:**
- `0`: Success
- `1`: Command failed
- `2`: Invalid arguments
- `3`: System error (database, network, etc.)

## Performance Tips

1. **Use specific queries**: More specific searches are faster and more relevant
2. **Limit results**: Use `--limit` to control output size  
3. **Export for analysis**: Use `--export` for large datasets rather than verbose output
4. **Batch operations**: Add multiple thoughts at once rather than individual commands

## Getting Help

```bash
# Command-specific help
globule search --help
globule draft --help
globule skeleton --help

# Show all available commands
globule --help

# Verbose output for debugging
globule search "query" --verbose
```

The Globule CLI provides powerful scriptable access to AI-powered thought management while maintaining the simplicity and intelligence that makes Globule unique.