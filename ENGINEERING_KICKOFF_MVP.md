# Engineering Kickoff: The Globule MVP

**Subject: Building the Cornerstone of a New Paradigm**

**To:** The Globule Engineering Team

**From:** Leadership

**Date:** 2025-07-23

---

This document marks the official kickoff for the implementation of Globule. We have completed an exhaustive phase of research and design, culminating in the comprehensive wiki you have all contributed to. That wiki represents our cathedral—a grand, ambitious vision for a new paradigm in personal computing.

Today, our task is to lay the first stone. And it must be perfect.

This memo outlines our plan for the Minimum Viable Product (MVP). It is the result of a critical, focused review of our own vision, with the express purpose of casting out all complexity that does not directly serve the core, magical experience for our first users. We will build a product that is small, potent, and revolutionary from day one. We will build the foundation, and we will build it right.

## 1. The Mission: What We Are Building First

Our vision is vast, but our MVP mission is surgically precise. We are not building a Semantic OS. We are not building a schema-driven workflow engine. We are building a tool that solves one, deeply felt problem:

**The chaos of unrealized inspiration.**

Our users have brilliant thoughts scattered across a dozen apps and notebooks. The friction of organizing this chaos is so high that most of it becomes a digital graveyard. Globule will be its resurrection.

**The MVP Promise:** Effortlessly turn a day's scattered thoughts into a focused first draft.

This is the entire product. This is the magic we are selling. The user experience is as follows:

1.  Throughout the day, the user captures any thought, idea, or quote using a simple CLI command: `globule add "..."`. They do not think about where it goes. There is zero friction.

2.  When they are ready to write, they sit down and issue another simple command: `globule draft "A post about creative discipline"`.

3.  Globule presents them with a clean, two-pane terminal interface. On the left, it has *already* found and clustered the handful of related thoughts they captured. On the right is a blank canvas.

That's it. That moment—of seeing your scattered ideas intelligently gathered and waiting for you—is the core "wow" moment. It must be fast, intuitive, and feel like magic. Everything we build for the MVP must serve this singular experience.

## 2. The Art of Focus: What We Are NOT Building (Yet)

A product is defined by the features it lacks. For our MVP, we will be ruthless in our focus. The following systems, though detailed in our LLDs, are to be deferred. We must have the discipline to build a tool that does one thing perfectly, rather than a platform that does ten things poorly.

-   **The Schema Definition Engine is DEFERRED.**
    -   **Reasoning:** The MVP is about capturing *unstructured* thoughts. A complex engine for defining schemas, complete with transpilers and sandboxed validators, is a powerful solution for a problem our initial users do not have. It is the second step, not the first.
    -   **MVP Implementation:** We will use a single, internal, hard-coded Pydantic model for a `Globule` (e.g., content, metadata). There will be no user-facing schema system.

-   **The Advanced Configuration System is DEFERRED.**
    -   **Reasoning:** A three-tier cascade with hot-reloading and context overrides is over-engineering for a single-user CLI tool. We must earn the right to ask users to configure our product.
    -   **MVP Implementation:** A single `config.yaml` in the user's config directory with 3-4 simple, essential keys (`storage_path`, `default_model`).

-   **The Multi-Strategy Orchestration Engine is SIMPLIFIED.**
    -   **Reasoning:** The various strategies (parallel, sequential, iterative) and complex disagreement detection are fascinating architectural explorations, but they add immense complexity for a subtle benefit in the MVP. The core value is getting *both* semantic and structural data, not a perfect, nuanced synthesis of the two.
    -   **MVP Implementation:** Implement a single, fixed `ParallelStrategy`. The engine calls the Embedding and Parsing services concurrently. That is all.

-   **The Full-Featured Synthesis Engine is FOCUSED.**
    -   **Reasoning:** Multiple palette views, deep "ripples of relevance," and other power-user features dilute the core magic. The initial "wow" is seeing your *recent*, *related* thoughts clustered by meaning.
    -   **MVP Implementation:** The Palette will *only* show semantic clusters of recent globules (e.g., last 7 days). The Canvas will provide basic text editing and the core AI actions (expand, summarize). "Explore Mode" is a feature for version 2.

By deferring these, we can channel all our energy into perfecting the core loop.

## 3. The Blueprint: How We Build the Foundation

Our wiki contains a brilliant and exhaustive set of Low-Level Designs. We will now use them as a reference library, not a verbatim instruction manual. We will extract the architectural essence required for the MVP, ensuring that what we build is a stable, scalable foundation for the full cathedral.

Our foundation rests on four, non-negotiable architectural pillars:

1.  **The Dual-Intelligence Abstraction:** The core idea of separate Semantic and Structural services is fundamental. We will build abstract base classes (`EmbeddingProvider`, `ParsingProvider`) for these services. The initial concrete implementations will use Ollama, but this abstraction ensures we can support any model or API in the future.
    *   **Reference:** `3_Core_Components/33_Semantic_Embedding_Service/` and `3_Core_Components/34_Structural_Parsing_Service/`
    *   **Action:** Implement the `OllamaEmbeddingProvider` and `OllamaParser`. Ignore all other providers and advanced features for now.

2.  **The Storage Manager Abstraction (DAL):** All application logic MUST interact with the database and filesystem through a single `StorageManager` interface. This is our Data Access Layer.
    *   **Reference:** `3_Core_Components/36_Intelligent_Storage_Manager/`
    *   **Action:** Implement the `SQLiteStorageManager`. Adhere to the core schema for the `globules` table, but defer complex virtual tables (beyond the basic vector index) and generated columns. The key is the abstraction, not the initial implementation's feature set.

3.  **The Core Data Model:** The `ProcessedGlobule` data structure is the lifeblood of our system. It must be robustly defined in Pydantic from day one, establishing a clear contract for data flowing between components.
    *   **Reference:** `3_Core_Components/35_Orchestration_Engine/30_LLD_Orchestration_Engine.md` (Data Structures section)
    *   **Action:** Implement the `EnrichedInput` and `ProcessedGlobule` data classes as defined. This is not a place to cut corners.

4.  **The Asynchronous TUI Foundation:** The `globule draft` experience must be fluid and responsive. There is no excuse for a frozen UI.
    *   **Reference:** `3_Core_Components/37_Interactive_Synthesis_Engine/`
    *   **Action:** Build the TUI using the `Textual` framework. All I/O operations (database queries, AI service calls) MUST be `async` and handled in background workers to never block the main UI thread.

By focusing on these four pillars, we ensure that the MVP is not a throwaway prototype. It is a robust, scalable foundation upon which the entire vision can be built.

## 4. The First Floor: Our Path After Launch

Once we have shipped the MVP and validated the core experience, we have a clear and exciting path forward. The foundational pillars we are building directly enable our most requested future enhancements.

-   **First, we will introduce Schema-Driven Outputs.** We will build the `Schema Definition Engine` we deferred. This will allow users to teach Globule how to format its output, such as adding YAML frontmatter to notes for perfect integration with Obsidian. This transforms Globule from a standalone tool into a powerful hub in a user's existing workflow.

-   **Second, we will enable Configurable Organization.** We will enhance the `Configuration System` and `StorageManager` to use user-defined templates for file and directory naming. This will give users complete control over their semantic filesystem, fulfilling the promise of a truly personalized knowledge base.

These are not distant dreams; they are the immediate, logical next steps that our MVP architecture is designed to support.

## 5. The North Star: The Cathedral We Are Building

I want to end by reminding everyone of the grand vision. We are not just building a note-taking app. We are building the first stone of a cathedral.

Our work will lead to a system that can truly augment human thought. A system that doesn't just store information, but understands it. A system that can surface a forgotten idea from years ago at the exact moment it becomes relevant. A system that helps us see the hidden connections in our own thinking.

Every line of code we write for this focused MVP is a step toward that North Star. Let's build this cornerstone with the care, focus, and quality it deserves.

Let's begin.