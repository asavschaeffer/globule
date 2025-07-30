"""
Orchestration Engine for Globule processing.

Coordinates the processing pipeline, managing the flow between embedding,
parsing, and storage services.
"""

from .engine import OrchestrationEngine

__all__ = ['OrchestrationEngine']