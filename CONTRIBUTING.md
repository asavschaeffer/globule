# Contributing to Globule

Thank you for your interest in contributing to Globule! This document outlines the development process, commit conventions, and branching strategy to ensure a smooth and collaborative workflow.

## Branching Strategy

We follow a phased development model centered around a main integration branch.

- **`main`**: This branch is always stable and represents the latest production-ready release. Direct pushes are prohibited.
- **`headless-core`**: This is the primary integration branch for the ongoing refactoring effort. All feature branches are merged into this branch.
- **Feature Branches**: All new work is done on feature branches, named according to the phase or feature being developed.
  - Format: `feature/phase-X-short-description` (e.g., `feature/phase-0-foundations`)
  - Branches are created from the latest `headless-core`.

## Development Workflow

1.  Create a feature branch from `headless-core`.
2.  Make your changes, adhering to the coding standards and testing requirements.
3.  Ensure all tests and CI checks pass.
4.  Rebase your feature branch on the latest `headless-core` before creating a pull request.
5.  Create a pull request from your feature branch to `headless-core`.
6.  Once the PR is reviewed and approved, it will be merged using a non-fast-forward merge (`--no-ff`).

## Commit Message Convention

We adhere to the [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) specification. This helps in automating changelog generation and keeps the project history clean and understandable.

The format is `type(scope): message`.

- **Types**: `feat`, `fix`, `refactor`, `test`, `docs`, `chore`, `style`, `ci`, `perf`
- **Scope**: The part of the codebase affected (e.g., `core`, `models`, `tui`, `ci`).

**Examples:**
- `feat(models): add Pydantic models for Globule and ProcessedGlobule`
- `test(interfaces): add compliance tests for dummy provider implementations`
- `docs(adr): create ADR-0001 for contracts-first architecture`

## Coding Standards

- **Formatting**: We use `black` for code formatting.
- **Linting**: We use `ruff` for linting.
- **Type Checking**: We use `mypy` for static type analysis.

Please ensure your contributions are formatted and pass all linter and type checks before submitting.
