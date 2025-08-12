### **Project Charter: Evolving Globule from Prototype to Cathedral**

**To:** The Globule Engineering Team (Asa & Gemini)
**From:** Project Architect (Asa)
**Date:** August 12, 2025
**Subject:** Six-Phase Refactoring Roadmap – `headless-core` Integration Branch

#### **1. Introduction: From Foundation Stone to Finished Cathedral**

This document outlines the strategic vision and tactical roadmap for the architectural evolution of Globule. We begin this journey from a position of strength: we have a functional, working prototype that proves the viability of our core ideas. This existing codebase is not a liability to be discarded, but rather the essential foundation stone upon which we will construct a more elegant, powerful, and enduring structure.

The architectural vision for Globule, as detailed in our foundational documents, is not merely an application but a "Cathedral of Recursive Understanding". It represents a new paradigm for human-computer interaction, built on principles of **Collaborative Intelligence**, a **Semantic Filesystem**, and **Progressive Discovery**.

Our task now is to deliberately and methodically elevate our functional prototype to fully embody this profound architectural vision. This is not a cleanup of "ugly" code; it is the professional engineering process of transforming a brilliant first draft into a masterpiece.

#### **2. Core Engineering Mandates**

These principles are the non-negotiable ground rules that apply to every phase of this refactor. The **`headless-core`** objective drives every decision.

  * **Integration Branch:** `headless-core` is the umbrella branch for all six phases. All work merges here; `main` remains pristine and only receives merges from `headless-core` at tagged milestones.
  * **Contracts-First Policy:** All new components will begin with immutable **Pydantic models** and **Abstract Base Classes (ABCs)**. These contracts must be merged with comprehensive tests *before* any dependent UI or service implementation work begins.
  * **Adapters at Boundaries:** All external dependencies (Ollama, SQLite, etc.) will be accessed via **provider-agnostic adapters**. No vendor-specific logic will leak into the core application.
  * **One-Way Dependencies:** The dependency flow is strict and unidirectional: `UI` → `Engine` → `Adapters` → `Providers`. No reverse dependencies are permitted.
  * **Test Coverage:** Unit test coverage for all new code must be **≥ 90%**. Contract tests and headless integration tests are required before any merge to `headless-core`.
  * **Backward Compatibility:** We will preserve existing CLI/TUI behavior at each phase boundary. Any breaking change must be explicitly versioned and documented.
  * **Rollback Ready:** Every change must be reversible to the previous milestone tag.
  * **Milestone Tags:** We will tag the completion of each phase (e.g., `phase-0-complete`). Releases to `main` will be cut exclusively from these tags.
  * **Non-Goals:** This refactoring cycle will **not** tackle visual redesigns of the TUI, performance tuning beyond documented budgets, introducing new AI providers without adapters, or speculative features not listed in the roadmap.

#### **3. Guiding Methodology: Iterative and Test-Driven Refactoring**

We will not be undertaking a high-risk, "big bang" rewrite. Instead, we will follow a disciplined and incremental refactoring process, guided by the core mandates listed above to ensure stability, quality, and alignment with our goals at every step.

#### **4. The Phased Roadmap & Definition of Done**

We will proceed in six distinct phases. Each phase will have its own detailed plan and must satisfy the following **Definition of Done (DoD)** template before it can be merged and tagged.

##### **Phase DoD Template:**

  * **[ ] Contracts:** All new or changed Pydantic models and ABCs are merged and validated by contract tests.
  * **[ ] Tests:** Unit tests achieve ≥ 90% coverage for new modules; headless integration tests covering core flows are passing; golden I/O samples are updated.
  * **[ ] Frontend Isolation:** UI contains zero business logic; only engine-level, frontend-agnostic APIs are used.
  * **[ ] Backward Compatibility:** Existing behaviors are preserved or changes are clearly documented in the changelog.
  * **[ ] Docs:** An Architecture Decision Record (ADR) is recorded for significant choices; the `ARCHITECTURE.md` diagram is updated; a public changelog entry is drafted.

#### **5. Git & Branching Strategy**

  * **Umbrella Branch:** `headless-core` is the long-lived integration branch.
  * **Phase Branches:**
      * `feature/phase-0-foundations`
      * `feature/phase-1-decoupling`
      * ...and so on for all six phases.
  * **Commits:** The `type(scope): message` convention will be enforced (e.g., `feat(core): add IOrchestrationEngine`).
  * **Merge Discipline:** Phase branches will be rebased on the latest `headless-core` before merge. Merges into `headless-core` will use `--no-ff` to preserve a clean, narrative branch topology.
  * **Tags & Releases:** Phase completions will be tagged (e.g., `phase-0-complete`). Releases to `main` will be cut from these tags.

#### **6. The Destination: An Architecture of Elegance and Power**

Upon completion of this roadmap, the Globule codebase will be a direct reflection of the vision laid out in our architectural documents. It will be modular, maintainable, extensible, robust, and a true Cathedral for amplified intelligence.