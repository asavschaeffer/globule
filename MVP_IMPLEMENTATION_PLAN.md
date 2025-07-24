# Globule MVP: Engineering Implementation Plan

**Status:** Ready for Implementation

## 1. Objective

To build and deliver the Globule MVP as defined in the `ENGINEERING_KICKOFF_MVP.md` memo, focusing on the core user loop of frictionless capture and semantic synthesis.

This document outlines the technical steps, phases, and priorities for the engineering team. Our goal is to build a stable, well-architected foundation that delivers the core "magic" of Globule from its first release.

## 2. Core User Stories

These are the only user-facing features we will build for the MVP. All technical work must serve these stories.

-   **Story 1: Frictionless Capture**
    -   As a user, I can run `globule add "<my thought>"` from my terminal to instantly capture a piece of text.
    -   *Acceptance Criteria:* The command returns in under 500ms. The text is passed to the backend for processing. The user is not required to choose a name or location.

-   **Story 2: Initiating Synthesis**
    -   As a user, I can run `globule draft "<a topic>"` to launch an interactive drafting session based on a topic.
    -   *Acceptance Criteria:* A full-screen Textual TUI application launches.

-   **Story 3: Viewing Clustered Thoughts**
    -   As a user, upon launching the TUI, I can see a list of my recent thoughts that are semantically related to my draft topic.
    -   *Acceptance Criteria:* The thoughts are grouped into clusters based on meaning. The UI is responsive and does not block while loading.

-   **Story 4: Drafting from Globules**
    -   As a user, I can select a thought from a cluster to add its content to a text editor canvas on the other side of the screen.
    -   *Acceptance Criteria:* The content appears in the editor. I can navigate between clusters and globules using the keyboard.

-   **Story 5: AI-Assisted Writing**
    -   As a user, I can select text in the editor and trigger a simple AI action (e.g., "expand" or "summarize").
    -   *Acceptance Criteria:* The selected text is replaced by the AI-generated output.

## 3. Phased Technical Breakdown

We will build the MVP in three distinct, sequential phases. Each phase builds upon the last and results in a testable, partially functional application.

### Phase 1: The Walking Skeleton

**Goal:** Prove the core, end-to-end plumbing works. A user can `add` a globule, and `draft` to see its content. No complex AI, no interactivity.

**Architectural Focus:** Establish the core abstractions and service communication.

| Component | Task | Details |
| :--- | :--- | :--- |
| **Storage** | Implement `SQLiteStorageManager` | Create the `globules` table with basic columns (id, content, embedding, parsed_data, file_path). Implement `store_globule()` and a simple `get_recent_globules()`. No vector index yet. |
| **Providers** | Implement `OllamaEmbeddingProvider` | Implement the `embed()` method to call a local Ollama instance. |
| | Mock `OllamaParser` | The `parse()` method should return a hard-coded, empty dictionary. We are only testing the interface. |
| **Orchestration** | Implement `ParallelStrategy` | The orchestrator should call the embedding service and the (mocked) parsing service concurrently. It then calls the `StorageManager` to save the result. |
| **CLI** | Implement `globule add` | A simple `click` or `argparse` command that takes text and passes it to the Orchestration Engine. |
| **TUI** | Basic `Textual` App | The `globule draft` command launches a Textual app. On load, it calls `StorageManager.get_recent_globules()` and displays the raw content of each in a simple, non-interactive list. |

**Outcome of Phase 1:** A developer can run `globule add "test"` and `globule draft "test"` and see the word "test" appear in the terminal UI. This validates our entire service architecture and data flow.

### Phase 2: Core Intelligence

**Goal:** Inject the "brains" of the operation. Make the system understand meaning.

**Architectural Focus:** Integrate AI services and vector search.

| Component | Task | Details |
| :--- | :--- | :--- |
| **Providers** | Implement `OllamaParser` | The `parse()` method should now call the LLM to extract a `title` and a simple list of `keywords`. |
| **Orchestration** | Enhance `ParallelStrategy` | The orchestrator now receives real data from both services and stores it. |
| **Storage** | Implement Vector Search | Add the `sqlite-vec` virtual table to the database. Implement a `search_by_embedding()` method in the `SQLiteStorageManager`. |
| **TUI** | Implement Semantic Clustering | When `globule draft` is run, its topic is embedded. This vector is used to call `search_by_embedding()`. The results are then clustered using a simple algorithm (e.g., basic K-Means on the embeddings). |
| | Update Palette Widget | The Palette now displays the *clusters* of globules, with each globule title visible in a nested view. |

**Outcome of Phase 2:** `globule draft` now shows genuinely related thoughts, grouped by theme. The core magic is now functional.

### Phase 3: The Interactive Experience

**Goal:** Transform the TUI from a display into a usable drafting tool.

**Architectural Focus:** Build out the front-end application logic.

| Component | Task | Details |
| :--- | :--- | :--- |
| **TUI** | Implement Interactive Palette | The user can now navigate the clusters and globules with arrow keys. Pressing `Enter` on a globule fires an event. |
| | Implement Canvas Editor | The right-hand pane is now a proper `TextArea` widget from Textual. |
| | Connect Palette to Canvas | When the `Enter` key event is fired, the content of the selected globule is appended to the `TextArea`. |
| | Implement AI Co-Pilot | Implement the `expand` and `summarize` actions. These take the currently selected text in the `TextArea`, construct a prompt, call the `OllamaParser`, and replace the selection with the result. |
| | Implement Save/Export | A `Ctrl+S` command saves the content of the `TextArea` to a local Markdown file. |

**Outcome of Phase 3:** The MVP is feature-complete. The core user loop is fully functional and delivers the promised "magic."

## 4. Developer Setup

To contribute, a new engineer will need the following:

1.  **Python Environment:**
    -   Clone the repository.
    -   Create a virtual environment: `python -m venv .venv`
    -   Install dependencies: `pip install -r requirements.txt`

2.  **Local AI Models:**
    -   Install Docker.
    -   Run `docker-compose up -d` from the project root. This will start an Ollama container.
    -   Pull the required models: `docker exec -it ollama ollama pull mxbai-embed-large` and `docker exec -it ollama ollama pull llama3.2:3b`.

3.  **Running Tests:**
    -   The test suite can be run with: `pytest`

This plan provides a clear, incremental path to a successful MVP launch. It prioritizes architectural integrity while focusing relentlessly on the core user experience. Let's get to building.