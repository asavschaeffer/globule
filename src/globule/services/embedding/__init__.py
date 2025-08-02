"""
Embedding Services.

Handles semantic embedding generation for text using various providers.
"""

from .ollama_provider import OllamaEmbeddingProvider

__all__ = ['OllamaEmbeddingProvider']