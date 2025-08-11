# REVISED Globule MVP (Ollie) - Build Plan

**Status: 2025-08-11**

## 1. Assessment

The existing codebase has a highly advanced backend and virtually no frontend. The core services (Storage, Orchestration, AI providers) are implemented to a post-MVP level, while the user interfaces (CLI, TUI) are placeholders.

The original `MVP_BUILD_PLAN.md` is therefore obsolete. The new priority is to build the user-facing layers to expose the powerful backend that already exists.

## 2. Revised Objective

The sole focus is now to implement the user interfaces as defined in the original plan. All backend work is considered complete for the MVP.

**Goal:** Connect a functional CLI and TUI to the existing backend services.

## 3. Revised Implementation Plan

Work will focus on two components in the `src/globule/interfaces/` and `src/globule/tui/` directories.

### Phase 1: Implement the Command-Line Interface (CLI)

**File:** `src/globule/interfaces/cli/main.py`

**Objective:** Create a functional `globule add` command that allows a user to capture a thought, have it processed, and stored.

**Tasks:**

1.  **Structure the CLI App:** Use `typer` to structure the main application.
2.  **Implement `add` Command:**
    - Create a `add(text: str)` function.
    - Instantiate the `OrchestrationEngine`.
    - **Crucially, implement the "Adaptive Input" logic:**
        - Use the `SchemaManager` to detect if the input text matches a schema trigger (e.g., contains a URL).
        - If a schema is detected, prompt the user for confirmation using a simple `input()` with a timeout.
        - Create an `EnrichedInput` object based on the text and any schema context.
    - Pass the `EnrichedInput` object to the `orchestration_engine.process_globule()` method.
    - Print a confirmation to the user, including the file path where the globule was stored.
3.  **Implement `draft` Command:**
    - Create a `draft()` command that simply launches the Textual TUI application.

### Phase 2: Implement the Textual User Interface (TUI)

**File:** `src/globule/tui/app.py`

**Objective:** Build the two-pane interactive drafting table for synthesizing thoughts.

**Tasks:**

1.  **Scaffold the App Layout:**
    - Create a `GlobuleApp(App)` class.
    - Use a `HorizontalLayout` to create the two main panes: `Palette` and `Canvas`.

2.  **Implement the `Palette` Pane (Left):**
    - Create a `Palette(Widget)` class.
    - On application start (`on_mount`), do the following:
        - Instantiate the `SQLiteStorageManager`.
        - Call `storage.get_recent_globules()` to fetch initial data.
        - Instantiate the `SemanticClusteringService` (`src/globule/services/clustering/semantic_clustering.py`).
        - Use the clustering service to group the fetched globules.
    - Display the clusters and their child globules in a `textual.widgets.Tree` widget.
    - **Implement Keyboard Interactions:**
        - **Enter Key (`on_key` event for "enter"):** When a globule is selected in the Tree, emit a custom message containing the `Globule` object to the parent `GlobuleApp`. This is "Build Mode".
        - **Tab Key (`on_key` event for "tab"):** When a globule is selected, use its embedding to perform a semantic search via `storage.search_by_embedding()`. Display these new results in the Tree. This is "Explore Mode".

3.  **Implement the `Canvas` Pane (Right):**
    - Create a `Canvas(Widget)` class containing a `textual.widgets.TextArea`.
    - **Implement Message Handling:**
        - Write a message handler (`on_message`) to listen for the custom message from the `Palette`.
        - When a `Globule` is received, append its `content` to the `TextArea`.
    - **Implement AI Co-pilot (Placeholder):**
        - On a key binding (e.g., `ctrl+a`), take the selected text from the `TextArea`.
        - Call the `OrchestrationEngine` to perform a simple action like "summarize".
        - Replace the selected text with the result.

## 4. End Goal

The result of this revised plan will be a functional MVP that aligns with the original user story. The powerful, existing backend will be made accessible through a simple but effective CLI and TUI. No new backend features are to be added.
