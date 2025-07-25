"""
Semantic Clustering Module for Phase 2 Intelligence.

This module provides intelligent clustering capabilities that automatically
discover themes and group related thoughts based on semantic similarity.
"""

from .semantic_clustering import (
    SemanticCluster,
    ClusteringAnalysis, 
    SemanticClusteringEngine
)

__all__ = [
    'SemanticCluster',
    'ClusteringAnalysis',
    'SemanticClusteringEngine'
]