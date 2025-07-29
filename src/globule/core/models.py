"""
Core data models for Globule.

These models define the contracts for data flowing between components,
following the specifications in the LLD documents.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Set
from datetime import datetime
from pathlib import Path
from enum import Enum
import numpy as np


@dataclass
class EnrichedInput:
    """Input received from Adaptive Input Module"""
    original_text: str                    # Raw user input
    enriched_text: str                    # Text after schema processing  
    detected_schema_id: Optional[str]     # e.g., "link_curation", "task_entry"
    schema_config: Optional[Dict[str, Any]]  # Schema-specific settings
    additional_context: Dict[str, Any]    # User corrections, clarifications
    source: str                          # "cli", "api", "tui"
    timestamp: datetime = field(default_factory=datetime.now)
    session_id: Optional[str] = None     # For context tracking
    verbosity: str = "concise"           # "silent", "concise", "verbose"


@dataclass
class FileDecision:
    """File organization recommendation"""
    semantic_path: Path                  # e.g., /writing/fantasy/dragons/
    filename: str                        # e.g., dragon-lore-fire-breathing.md
    metadata: Dict[str, Any]             # Additional file metadata
    confidence: float                    # 0.0-1.0
    alternative_paths: List[Path]        # Other considered paths


@dataclass
class Interpretation:
    """Represents one possible interpretation of content"""
    type: str                           # "literal", "semantic", "contextual"
    confidence: float
    data: Dict[str, Any]
    source: str                         # Which service generated this


@dataclass
class ProcessedGlobule:
    """Output sent to Intelligent Storage Manager"""
    # Core content
    text: str                            # Original text
    embedding: Optional[np.ndarray] = None        # Final embedding vector (1024-d)
    embedding_confidence: float = 0.0    # 0.0-1.0
    
    # Structured data from parsing
    parsed_data: Dict[str, Any] = field(default_factory=dict)  # Entities, categories, metadata
    parsing_confidence: float = 0.0      # 0.0-1.0
    
    # File organization
    file_decision: Optional[FileDecision] = None  # Suggested path and metadata
    
    # Processing metadata
    processing_time_ms: Dict[str, float] = field(default_factory=dict)  # Breakdown by stage
    orchestration_strategy: str = "parallel"      # "parallel", "sequential", "iterative"
    confidence_scores: Dict[str, float] = field(default_factory=dict)   # Per-component confidence
    
    # Disagreement handling
    interpretations: List[Interpretation] = field(default_factory=list)  # Multiple possible interpretations
    has_nuance: bool = False             # Sarcasm, metaphor detected
    
    # Context
    semantic_neighbors: List[str] = field(default_factory=list)  # UUIDs of related content
    processing_notes: List[str] = field(default_factory=list)    # Warnings, info for debugging
    
    # Persistence
    id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    modified_at: datetime = field(default_factory=datetime.now)


class UIMode(Enum):
    """Current interaction mode of the UI"""
    BUILD = "build"      # Adding globules to canvas
    EXPLORE = "explore"  # Discovering related content
    EDIT = "edit"       # Editing canvas content


@dataclass
class GlobuleCluster:
    """Represents a semantic grouping of related globules"""
    id: str
    globules: List[ProcessedGlobule]
    centroid: Optional[np.ndarray] = None
    label: str = "Untitled Cluster"      # Auto-generated or user-defined
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class SynthesisState:
    """Complete state of the synthesis session"""
    # UI State
    current_mode: UIMode = UIMode.BUILD
    selected_cluster_id: Optional[str] = None
    selected_globule_id: Optional[str] = None
    
    # Palette State
    visible_clusters: List[GlobuleCluster] = field(default_factory=list)
    cluster_view_mode: str = "semantic"  # semantic, temporal, alphabetical
    expanded_clusters: Set[str] = field(default_factory=set)
    
    # Canvas State
    canvas_content: str = ""
    cursor_position: int = 0
    selection_start: Optional[int] = None
    selection_end: Optional[int] = None
    incorporated_globules: Set[str] = field(default_factory=set)
    
    # Discovery State
    discovery_query: Optional[str] = None
    discovery_results: List[ProcessedGlobule] = field(default_factory=list)
    discovery_depth: int = 1  # Ripples of relevance depth


# Dataclass models with validation
@dataclass
class MetadataOutput:
    """Validated metadata output from parsing service"""
    domain: str = "general"
    timestamp: Optional[str] = None
    category: str = "note"
    title: str = "Untitled"
    entities: List[Dict[str, str]] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate fields after initialization"""
        if not isinstance(self.domain, str) or len(self.domain.strip()) == 0:
            raise ValueError("Domain must be a non-empty string")
        if self.timestamp is not None and not isinstance(self.timestamp, str):
            raise ValueError("Timestamp must be a string or None")
        if not isinstance(self.category, str) or len(self.category.strip()) == 0:
            raise ValueError("Category must be a non-empty string")
        if not isinstance(self.title, str) or len(self.title.strip()) == 0:
            raise ValueError("Title must be a non-empty string")
        if not isinstance(self.entities, list):
            raise ValueError("Entities must be a list")
        if not isinstance(self.keywords, list):
            raise ValueError("Keywords must be a list")


@dataclass
class EmbeddingResult:
    """Result of embedding generation"""
    embedding: List[float]  # Vector as list for JSON serialization
    model: str
    dimension: int
    generation_time_ms: float
    cached: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate fields after initialization"""
        if not isinstance(self.embedding, list) or len(self.embedding) == 0:
            raise ValueError("Embedding must be a non-empty list")
        if not all(isinstance(x, (int, float)) for x in self.embedding):
            raise ValueError("All embedding values must be numeric")
        if not isinstance(self.model, str) or len(self.model.strip()) == 0:
            raise ValueError("Model must be a non-empty string")
        if not isinstance(self.dimension, int) or self.dimension <= 0:
            raise ValueError("Dimension must be a positive integer")
        if not isinstance(self.generation_time_ms, (int, float)) or self.generation_time_ms < 0:
            raise ValueError("Generation time must be non-negative")
        if len(self.embedding) != self.dimension:
            raise ValueError(f"Embedding length ({len(self.embedding)}) must match dimension ({self.dimension})")


@dataclass
class ParsingResult:
    """Result of structural parsing"""
    data: Dict[str, Any]
    confidence: float
    processing_time_ms: float
    model: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate fields after initialization"""
        if not isinstance(self.data, dict):
            raise ValueError("Data must be a dictionary")
        if not isinstance(self.confidence, (int, float)) or not (0.0 <= self.confidence <= 1.0):
            raise ValueError("Confidence must be a float between 0.0 and 1.0")
        if not isinstance(self.processing_time_ms, (int, float)) or self.processing_time_ms < 0:
            raise ValueError("Processing time must be non-negative")
        if not isinstance(self.model, str) or len(self.model.strip()) == 0:
            raise ValueError("Model must be a non-empty string")
        if not isinstance(self.metadata, dict):
            raise ValueError("Metadata must be a dictionary")