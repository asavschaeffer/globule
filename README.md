# Globule

> Turn your scattered thoughts into structured drafts. Effortlessly.

![Project Status: Design](https://img.shields.io/badge/status-design-lightgrey)

We jot down ideas in notebooks, send ourselves emails, and save links across a dozen apps. These fragments of inspiration are disconnected and often lost. Globule is a local-first, AI-powered system designed to automatically organize this chaos.

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

Globule is currently in the architectural design phase. The following represents the intended installation process for the initial release.

```bash
# Installation (via pip)
pip install globule-cli

# First-time setup
globule init
```

## The Vision: Where We're Going

The initial version of Globule is focused on the core experience of capture and synthesis. But this is just the foundation for a much larger vision.

-   **Empowering Workflows:** Soon, you'll be able to teach Globule about *your* specific types of information (like `Recipes` or `Code Snippets`), enabling custom formatting and perfect integration with tools like Obsidian.
-   **Personalized Organization:** You will be able to tune Globule's brain, defining your own templates for how files and folders are named and organized, making the semantic filesystem truly your own.

Our ultimate goal is to build a new foundational layer for personal computing—one that understands context, not just commands.

## Contributing

This project is currently in a design-heavy phase. If you are interested in the architecture, design philosophy, and the future of semantic computing, we welcome you to explore our **[Project Wiki](https://github.com/asavschaeffer/globule/wiki)** where the system is being designed in the open.
