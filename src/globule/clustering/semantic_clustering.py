"""
Semantic Clustering Engine for Phase 2 Intelligence.

This module implements intelligent clustering of thoughts and ideas using
semantic similarity and machine learning techniques. It automatically
discovers themes, groups related content, and provides meaningful cluster
labels based on content analysis.

Features:
- K-means clustering on semantic embeddings
- Automatic optimal cluster number detection  
- Intelligent cluster naming using content analysis
- Temporal clustering (recent vs historical)
- Dynamic re-clustering as content grows
- Cross-domain theme detection

Author: Globule Team
Version: 2.0.0
"""

import asyncio
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import Counter
import json


from globule.core.models import ProcessedGlobule
from globule.storage.sqlite_manager import SQLiteStorageManager


@dataclass
class SemanticCluster:
    """
    Represents a semantic cluster of related thoughts.
    
    Contains the cluster metadata, representative content,
    and intelligence about the common themes.
    """
    id: str
    label: str
    description: str
    size: int
    centroid: np.ndarray
    member_ids: List[str]
    keywords: List[str]
    domains: List[str]
    confidence_score: float
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    representative_samples: List[str] = field(default_factory=list)
    theme_analysis: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert cluster to dictionary for serialization."""
        return {
            "id": self.id,
            "label": self.label,
            "description": self.description,
            "size": self.size,
            "member_ids": self.member_ids,
            "keywords": self.keywords,
            "domains": self.domains,
            "confidence_score": self.confidence_score,
            "created_at": self.created_at.isoformat(),
            "last_updated": self.last_updated.isoformat(),
            "representative_samples": self.representative_samples,
            "theme_analysis": self.theme_analysis
        }


@dataclass
class ClusteringAnalysis:
    """
    Complete clustering analysis results.
    
    Contains all discovered clusters plus metadata about
    the clustering process and quality metrics.
    """
    clusters: List[SemanticCluster]
    total_globules: int
    clustering_method: str
    optimal_k: int
    silhouette_score: float
    processing_time_ms: float
    analysis_timestamp: datetime = field(default_factory=datetime.now)
    cross_cluster_relationships: Dict[str, List[str]] = field(default_factory=dict)
    temporal_patterns: Dict[str, Any] = field(default_factory=dict)
    quality_metrics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert analysis to dictionary for serialization."""
        return {
            "clusters": [cluster.to_dict() for cluster in self.clusters],
            "total_globules": self.total_globules,
            "clustering_method": self.clustering_method,
            "optimal_k": self.optimal_k,
            "silhouette_score": self.silhouette_score,
            "processing_time_ms": self.processing_time_ms,
            "analysis_timestamp": self.analysis_timestamp.isoformat(),
            "cross_cluster_relationships": self.cross_cluster_relationships,
            "temporal_patterns": self.temporal_patterns,
            "quality_metrics": self.quality_metrics
        }


class SemanticClusteringEngine:
    """
    Intelligent semantic clustering engine for Phase 2.
    
    Discovers themes and groups related thoughts using advanced machine learning
    techniques combined with content analysis and domain knowledge.
    """

    def __init__(self, storage_manager: SQLiteStorageManager):
        """Initialize the clustering engine."""
        self.storage = storage_manager
        self.logger = logging.getLogger(__name__)
        
        # Clustering parameters
        self.min_cluster_size = 2
        self.max_clusters = 20
        self.min_similarity_threshold = 0.3
        self.temporal_window_days = 30

    async def analyze_semantic_clusters(
        self, 
        min_globules: int = 5,
        force_recalculation: bool = False
    ) -> ClusteringAnalysis:
        """
        Perform comprehensive semantic clustering analysis.
        
        Args:
            min_globules: Minimum number of globules required for clustering
            force_recalculation: Force recalculation even if recent results exist
            
        Returns:
            ClusteringAnalysis with discovered clusters and metadata
        """
        start_time = datetime.now()
        
        try:
            # Get all globules with embeddings
            globules = await self._get_clusterable_globules()
            
            if len(globules) < min_globules:
                self.logger.warning(f"Insufficient globules for clustering: {len(globules)} < {min_globules}")
                return self._create_empty_analysis(len(globules))
            
            self.logger.info(f"Starting semantic clustering analysis on {len(globules)} globules")
            
            # Extract embeddings and prepare data
            embeddings_matrix, globule_map = self._prepare_clustering_data(globules)
            
            # Determine optimal number of clusters
            optimal_k = self._find_optimal_clusters(embeddings_matrix)
            
            # Perform clustering
            cluster_labels = self._perform_clustering(embeddings_matrix, optimal_k)
            
            # Calculate silhouette score for quality assessment
            silhouette = self._silhouette_score(embeddings_matrix, cluster_labels)
            
            # Create semantic clusters with intelligent labeling
            semantic_clusters = await self._create_semantic_clusters(
                globules, cluster_labels, embeddings_matrix, globule_map
            )
            
            # Analyze cross-cluster relationships
            cross_relationships = self._analyze_cross_cluster_relationships(semantic_clusters)
            
            # Analyze temporal patterns
            temporal_patterns = self._analyze_temporal_patterns(globules, cluster_labels)
            
            # Calculate quality metrics
            quality_metrics = self._calculate_quality_metrics(
                semantic_clusters, embeddings_matrix, cluster_labels
            )
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            analysis = ClusteringAnalysis(
                clusters=semantic_clusters,
                total_globules=len(globules),
                clustering_method="kmeans_with_intelligent_labeling",
                optimal_k=optimal_k,
                silhouette_score=silhouette,
                processing_time_ms=processing_time,
                cross_cluster_relationships=cross_relationships,
                temporal_patterns=temporal_patterns,
                quality_metrics=quality_metrics
            )
            
            self.logger.info(f"Clustering analysis completed: {len(semantic_clusters)} clusters, silhouette={silhouette:.3f}")
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Clustering analysis failed: {e}")
            raise

    async def _get_clusterable_globules(self) -> List[ProcessedGlobule]:
        """Get all globules suitable for clustering."""
        # Get recent globules with good quality embeddings
        all_globules = await self.storage.get_recent_globules(limit=1000)
        
        clusterable = []
        for globule in all_globules:
            if (globule.embedding is not None and 
                globule.embedding_confidence > 0.5 and
                len(globule.text.strip()) > 10):  # Reasonable content length
                clusterable.append(globule)
        
        return clusterable

    def _prepare_clustering_data(
        self, 
        globules: List[ProcessedGlobule]
    ) -> Tuple[np.ndarray, Dict[int, ProcessedGlobule]]:
        """Prepare embeddings matrix and globule mapping for clustering."""
        embeddings = []
        globule_map = {}
        
        for i, globule in enumerate(globules):
            if globule.embedding is not None:
                # Normalize embeddings for consistent clustering
                normalized_embedding = globule.embedding / np.linalg.norm(globule.embedding)
                embeddings.append(normalized_embedding)
                globule_map[i] = globule
        
        embeddings_matrix = np.vstack(embeddings)
        
        # Optional: Apply dimensionality reduction for very high-dimensional spaces
        # For now, we'll work directly with the embeddings
        
        return embeddings_matrix, globule_map

    def _find_optimal_clusters(self, embeddings_matrix: np.ndarray) -> int:
        """
        Find optimal number of clusters using elbow method and silhouette analysis.
        
        Combines multiple heuristics to determine the best cluster count.
        """
        n_samples = embeddings_matrix.shape[0]
        
        # Determine reasonable range for k
        min_k = max(2, min(3, n_samples // 4))
        max_k = min(self.max_clusters, n_samples // 2)
        
        if min_k >= max_k:
            return min_k
        
        k_range = range(min_k, max_k + 1)
        
        inertias = []
        silhouette_scores = []
        
        for k in k_range:
            try:
                cluster_labels, centroids, inertia = self._kmeans(embeddings_matrix, k)
                
                inertias.append(inertia)
                
                if len(set(cluster_labels)) > 1:  # Need at least 2 clusters for silhouette
                    sil_score = self._silhouette_score(embeddings_matrix, cluster_labels)
                    silhouette_scores.append(sil_score)
                else:
                    silhouette_scores.append(0)
                    
            except Exception as e:
                self.logger.warning(f"Clustering failed for k={k}: {e}")
                inertias.append(float('inf'))
                silhouette_scores.append(0)
        
        # Find optimal k using combination of methods
        optimal_k = self._select_optimal_k(k_range, inertias, silhouette_scores)
        
        self.logger.info(f"Optimal cluster count determined: k={optimal_k}")
        return optimal_k

    def _select_optimal_k(
        self, 
        k_range: range, 
        inertias: List[float], 
        silhouette_scores: List[float]
    ) -> int:
        """Select optimal k using multiple criteria."""
        
        # Method 1: Highest silhouette score
        best_silhouette_idx = np.argmax(silhouette_scores)
        silhouette_k = list(k_range)[best_silhouette_idx]
        
        # Method 2: Elbow method (simplified)
        # Look for the point where inertia reduction slows down significantly
        if len(inertias) > 2:
            deltas = [inertias[i] - inertias[i + 1] for i in range(len(inertias) - 1)]
            elbow_idx = np.argmax(deltas)
            elbow_k = list(k_range)[elbow_idx]
        else:
            elbow_k = silhouette_k
        
        # Method 3: Conservative approach - prefer fewer clusters if quality is similar
        conservative_k = min(silhouette_k, elbow_k)
        
        # Final decision: use silhouette if it's significantly better, otherwise conservative
        if silhouette_scores[best_silhouette_idx] > 0.3:  # Good silhouette score
            return silhouette_k
        else:
            return conservative_k

    def _perform_clustering(self, embeddings_matrix: np.ndarray, k: int) -> np.ndarray:
        """Perform the actual clustering using custom K-means."""
        try:
            cluster_labels, centroids, inertia = self._kmeans(embeddings_matrix, k)
            return cluster_labels
            
        except Exception as e:
            self.logger.error(f"K-means clustering failed: {e}")
            # Fallback to simple distance-based clustering
            try:
                cluster_labels = self._simple_clustering(embeddings_matrix, k)
                return cluster_labels
            except Exception as e2:
                self.logger.error(f"Simple clustering also failed: {e2}")
                # Last resort: assign everything to one cluster
                return np.zeros(embeddings_matrix.shape[0], dtype=int)

    async def _create_semantic_clusters(
        self,
        globules: List[ProcessedGlobule],
        cluster_labels: np.ndarray,
        embeddings_matrix: np.ndarray,
        globule_map: Dict[int, ProcessedGlobule]
    ) -> List[SemanticCluster]:
        """Create SemanticCluster objects with intelligent labeling."""
        
        clusters = []
        unique_labels = set(cluster_labels)
        
        for cluster_id in unique_labels:
            # Get globules in this cluster
            cluster_indices = np.where(cluster_labels == cluster_id)[0]
            cluster_globules = [globule_map[i] for i in cluster_indices]
            
            if len(cluster_globules) < self.min_cluster_size:
                continue  # Skip very small clusters
            
            # Calculate cluster centroid
            cluster_embeddings = embeddings_matrix[cluster_indices]
            centroid = np.mean(cluster_embeddings, axis=0)
            
            # Generate intelligent cluster metadata
            label, description = await self._generate_cluster_label(cluster_globules)
            keywords = self._extract_cluster_keywords(cluster_globules)
            domains = self._analyze_cluster_domains(cluster_globules)
            confidence = self._calculate_cluster_confidence(cluster_globules, cluster_embeddings)
            representative_samples = self._select_representative_samples(cluster_globules)
            theme_analysis = self._analyze_cluster_themes(cluster_globules)
            
            cluster = SemanticCluster(
                id=f"cluster_{cluster_id}",
                label=label,
                description=description,
                size=len(cluster_globules),
                centroid=centroid,
                member_ids=[g.id for g in cluster_globules],
                keywords=keywords,
                domains=domains,
                confidence_score=confidence,
                representative_samples=representative_samples,
                theme_analysis=theme_analysis
            )
            
            clusters.append(cluster)
        
        # Sort clusters by size (largest first) 
        clusters.sort(key=lambda c: c.size, reverse=True)
        
        return clusters

    async def _generate_cluster_label(self, globules: List[ProcessedGlobule]) -> Tuple[str, str]:
        """Generate intelligent label and description for a cluster."""
        
        # Analyze common themes in the cluster
        all_keywords = []
        all_domains = []
        all_categories = []
        
        for globule in globules:
            if globule.parsed_data:
                all_keywords.extend(globule.parsed_data.get('keywords', []))
                all_domains.append(globule.parsed_data.get('domain', 'other'))
                all_categories.append(globule.parsed_data.get('category', 'note'))
        
        # Find most common keywords, domains, categories
        top_keywords = [word for word, count in Counter(all_keywords).most_common(3)]
        top_domain = Counter(all_domains).most_common(1)[0][0] if all_domains else 'mixed'
        top_category = Counter(all_categories).most_common(1)[0][0] if all_categories else 'thoughts'
        
        # Generate label based on content analysis
        if top_keywords:
            # Use top keywords to create meaningful label
            if len(top_keywords) >= 2:
                label = f"{top_keywords[0].title()} & {top_keywords[1].title()}"
            else:
                label = f"{top_keywords[0].title()} {top_category.title()}"
        else:
            # Fallback to domain/category based label
            label = f"{top_domain.title()} {top_category.title()}"
        
        # Generate description
        size = len(globules)
        description = f"A cluster of {size} {top_category}s primarily focused on {top_domain} themes"
        
        if top_keywords:
            description += f", with key concepts: {', '.join(top_keywords[:3])}"
        
        return label, description

    def _extract_cluster_keywords(self, globules: List[ProcessedGlobule]) -> List[str]:
        """Extract the most representative keywords for a cluster."""
        all_keywords = []
        
        for globule in globules:
            if globule.parsed_data and 'keywords' in globule.parsed_data:
                all_keywords.extend(globule.parsed_data['keywords'])
        
        # Return top 5 most common keywords
        keyword_counts = Counter(all_keywords)
        return [word for word, count in keyword_counts.most_common(5)]

    def _analyze_cluster_domains(self, globules: List[ProcessedGlobule]) -> List[str]:
        """Analyze the domains represented in a cluster."""
        domains = []
        
        for globule in globules:
            if globule.parsed_data and 'domain' in globule.parsed_data:
                domains.append(globule.parsed_data['domain'])
        
        # Return unique domains, sorted by frequency
        domain_counts = Counter(domains)
        return [domain for domain, count in domain_counts.most_common()]

    def _calculate_cluster_confidence(
        self, 
        globules: List[ProcessedGlobule], 
        embeddings: np.ndarray
    ) -> float:
        """Calculate confidence score for cluster quality."""
        
        # Factor 1: Embedding coherence (how similar are the embeddings?)
        if len(embeddings) > 1:
            centroid = np.mean(embeddings, axis=0)
            similarities = [np.dot(emb, centroid) for emb in embeddings]
            embedding_coherence = np.mean(similarities)
        else:
            embedding_coherence = 1.0
        
        # Factor 2: Content quality (parsing confidence)
        parsing_confidences = [g.parsing_confidence for g in globules if g.parsing_confidence > 0]
        content_quality = np.mean(parsing_confidences) if parsing_confidences else 0.5
        
        # Factor 3: Size factor (larger clusters are generally more reliable)
        size_factor = min(1.0, len(globules) / 10)  # Plateau at 10 members
        
        # Combine factors
        confidence = (embedding_coherence * 0.5 + content_quality * 0.3 + size_factor * 0.2)
        
        return min(1.0, max(0.0, confidence))

    def _select_representative_samples(self, globules: List[ProcessedGlobule]) -> List[str]:
        """Select representative text samples from the cluster."""
        
        # Sort by parsing confidence and select top examples
        sorted_globules = sorted(globules, key=lambda g: g.parsing_confidence, reverse=True)
        
        # Take up to 3 representative samples
        samples = []
        for globule in sorted_globules[:3]:
            # Truncate long texts for readability
            sample = globule.text[:100] + "..." if len(globule.text) > 100 else globule.text
            samples.append(sample)
        
        return samples

    def _analyze_cluster_themes(self, globules: List[ProcessedGlobule]) -> Dict[str, Any]:
        """Perform deeper thematic analysis of the cluster."""
        
        # Analyze temporal patterns
        creation_dates = [g.created_at for g in globules if g.created_at]
        
        temporal_analysis = {}
        if creation_dates:
            temporal_analysis = {
                "earliest": min(creation_dates).isoformat(),
                "latest": max(creation_dates).isoformat(),
                "span_days": (max(creation_dates) - min(creation_dates)).days,
                "recent_activity": sum(1 for d in creation_dates if d > datetime.now() - timedelta(days=7))
            }
        
        # Analyze content characteristics
        content_analysis = {
            "avg_length": np.mean([len(g.text) for g in globules]),
            "total_words": sum(len(g.text.split()) for g in globules),
            "sentiment_distribution": self._analyze_sentiment_distribution(globules),
            "category_distribution": self._analyze_category_distribution(globules)
        }
        
        return {
            "temporal": temporal_analysis,
            "content": content_analysis,
            "cluster_density": self._calculate_cluster_density(globules),
            "cross_domain_score": self._calculate_cross_domain_score(globules)
        }

    def _analyze_sentiment_distribution(self, globules: List[ProcessedGlobule]) -> Dict[str, int]:
        """Analyze sentiment distribution in the cluster."""
        sentiments = []
        
        for globule in globules:
            if globule.parsed_data and 'metadata' in globule.parsed_data:
                sentiment = globule.parsed_data['metadata'].get('sentiment', 'neutral')
                sentiments.append(sentiment)
        
        return dict(Counter(sentiments))

    def _analyze_category_distribution(self, globules: List[ProcessedGlobule]) -> Dict[str, int]:
        """Analyze category distribution in the cluster."""
        categories = []
        
        for globule in globules:
            if globule.parsed_data:
                category = globule.parsed_data.get('category', 'note')
                categories.append(category)
        
        return dict(Counter(categories))

    def _calculate_cluster_density(self, globules: List[ProcessedGlobule]) -> float:
        """Calculate how tightly clustered the content is."""
        # For now, return a placeholder based on size
        # In future versions, this could use embedding distances
        size = len(globules)
        return min(1.0, size / 20)  # Density increases with size, plateaus at 20

    def _calculate_cross_domain_score(self, globules: List[ProcessedGlobule]) -> float:
        """Calculate how much this cluster spans multiple domains."""
        domains = set()
        
        for globule in globules:
            if globule.parsed_data:
                domain = globule.parsed_data.get('domain', 'other')
                domains.add(domain)
        
        # Higher score = more cross-domain (potentially more interesting)
        return len(domains) / max(1, len(set(['creative', 'technical', 'personal', 'academic', 'business', 'philosophy'])))

    def _analyze_cross_cluster_relationships(self, clusters: List[SemanticCluster]) -> Dict[str, List[str]]:
        """Analyze relationships between different clusters."""
        relationships = {}
        
        for i, cluster_a in enumerate(clusters):
            related_clusters = []
            
            for j, cluster_b in enumerate(clusters):
                if i != j:
                    # Calculate centroid similarity
                    similarity = np.dot(cluster_a.centroid, cluster_b.centroid)
                    
                    # Also check keyword overlap
                    keyword_overlap = len(set(cluster_a.keywords) & set(cluster_b.keywords))
                    
                    # Check domain overlap
                    domain_overlap = len(set(cluster_a.domains) & set(cluster_b.domains))
                    
                    # Combined relationship score
                    relationship_score = similarity * 0.6 + (keyword_overlap / 5) * 0.3 + (domain_overlap / 3) * 0.1
                    
                    if relationship_score > 0.3:  # Threshold for "related"
                        related_clusters.append(cluster_b.id)
            
            relationships[cluster_a.id] = related_clusters
        
        return relationships

    def _analyze_temporal_patterns(
        self, 
        globules: List[ProcessedGlobule], 
        cluster_labels: np.ndarray
    ) -> Dict[str, Any]:
        """Analyze temporal patterns in the clustering."""
        
        # Group by time periods
        now = datetime.now()
        recent_window = now - timedelta(days=7)
        medium_window = now - timedelta(days=30)
        
        temporal_stats = {
            "recent_clusters": set(),
            "active_clusters": set(),
            "historical_clusters": set(),
            "temporal_distribution": {}
        }
        
        for i, globule in enumerate(globules):
            if globule.created_at:
                cluster_id = f"cluster_{cluster_labels[i]}"
                
                if globule.created_at > recent_window:
                    temporal_stats["recent_clusters"].add(cluster_id)
                elif globule.created_at > medium_window:
                    temporal_stats["active_clusters"].add(cluster_id)
                else:
                    temporal_stats["historical_clusters"].add(cluster_id)
        
        # Convert sets to lists for JSON serialization
        temporal_stats["recent_clusters"] = list(temporal_stats["recent_clusters"])
        temporal_stats["active_clusters"] = list(temporal_stats["active_clusters"])
        temporal_stats["historical_clusters"] = list(temporal_stats["historical_clusters"])
        
        return temporal_stats

    def _calculate_quality_metrics(
        self,
        clusters: List[SemanticCluster],
        embeddings_matrix: np.ndarray,
        cluster_labels: np.ndarray
    ) -> Dict[str, float]:
        """Calculate various quality metrics for the clustering."""
        
        metrics = {}
        
        # Basic metrics
        metrics["num_clusters"] = len(clusters)
        metrics["avg_cluster_size"] = np.mean([c.size for c in clusters]) if clusters else 0
        metrics["largest_cluster_size"] = max([c.size for c in clusters]) if clusters else 0
        metrics["smallest_cluster_size"] = min([c.size for c in clusters]) if clusters else 0
        
        # Confidence metrics
        metrics["avg_cluster_confidence"] = np.mean([c.confidence_score for c in clusters]) if clusters else 0
        metrics["high_confidence_clusters"] = sum(1 for c in clusters if c.confidence_score > 0.7)
        
        # Domain diversity
        all_domains = set()
        for cluster in clusters:
            all_domains.update(cluster.domains)
        metrics["domain_diversity"] = len(all_domains)
        
        # Cross-domain clusters (potentially interesting insights)
        metrics["cross_domain_clusters"] = sum(1 for c in clusters if len(c.domains) > 1)
        
        return metrics

    def _kmeans(self, X: np.ndarray, k: int, max_iters: int = 300, tol: float = 1e-4) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Custom K-means implementation using numpy.
        
        Args:
            X: Data matrix (n_samples, n_features)
            k: Number of clusters
            max_iters: Maximum iterations
            tol: Convergence tolerance
            
        Returns:
            (cluster_labels, centroids, inertia)
        """
        n_samples, n_features = X.shape
        
        # Initialize centroids randomly
        np.random.seed(42)  # For reproducibility
        centroids = X[np.random.choice(n_samples, k, replace=False)]
        
        for iteration in range(max_iters):
            # Assign points to nearest centroid
            distances = np.sqrt(((X - centroids[:, np.newaxis])**2).sum(axis=2))
            labels = np.argmin(distances, axis=0)
            
            # Update centroids
            new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])
            
            # Handle empty clusters by reinitializing
            for i in range(k):
                if np.sum(labels == i) == 0:
                    new_centroids[i] = X[np.random.choice(n_samples)]
            
            # Check convergence
            if np.allclose(centroids, new_centroids, atol=tol):
                break
                
            centroids = new_centroids
        
        # Calculate inertia (within-cluster sum of squares)
        distances = np.sqrt(((X - centroids[labels])**2).sum(axis=1))
        inertia = np.sum(distances**2)
        
        return labels, centroids, inertia
    
    def _silhouette_score(self, X: np.ndarray, labels: np.ndarray) -> float:
        """
        Custom silhouette score implementation using numpy.
        
        Args:
            X: Data matrix (n_samples, n_features)
            labels: Cluster labels
            
        Returns:
            Average silhouette score
        """
        n_samples = len(X)
        unique_labels = np.unique(labels)
        
        if len(unique_labels) <= 1:
            return 0.0
        
        silhouette_scores = []
        
        for i in range(n_samples):
            # Current point and its cluster
            point = X[i]
            current_cluster = labels[i]
            
            # Calculate a(i): mean distance to other points in same cluster
            same_cluster_mask = (labels == current_cluster) & (np.arange(n_samples) != i)
            if np.sum(same_cluster_mask) > 0:
                a_i = np.mean(np.sqrt(np.sum((X[same_cluster_mask] - point)**2, axis=1)))
            else:
                a_i = 0
            
            # Calculate b(i): minimum mean distance to points in other clusters
            b_i = float('inf')
            for other_cluster in unique_labels:
                if other_cluster != current_cluster:
                    other_cluster_mask = labels == other_cluster
                    if np.sum(other_cluster_mask) > 0:
                        mean_dist = np.mean(np.sqrt(np.sum((X[other_cluster_mask] - point)**2, axis=1)))
                        b_i = min(b_i, mean_dist)
            
            # Calculate silhouette coefficient for this point
            if max(a_i, b_i) > 0:
                s_i = (b_i - a_i) / max(a_i, b_i)
            else:
                s_i = 0
            
            silhouette_scores.append(s_i)
        
        return np.mean(silhouette_scores)
    
    def _simple_clustering(self, X: np.ndarray, k: int) -> np.ndarray:
        """
        Simple distance-based clustering fallback.
        
        Groups points by proximity using a greedy approach.
        """
        n_samples = X.shape[0]
        labels = np.zeros(n_samples, dtype=int)
        
        if k >= n_samples:
            return np.arange(n_samples)
        
        # Select initial cluster centers spread apart
        centers_idx = [0]  # Start with first point
        
        for _ in range(k - 1):
            # Find point farthest from existing centers
            max_min_dist = -1
            next_center = 0
            
            for i in range(n_samples):
                if i not in centers_idx:
                    # Find minimum distance to existing centers
                    min_dist = min(np.sqrt(np.sum((X[i] - X[c])**2)) for c in centers_idx)
                    if min_dist > max_min_dist:
                        max_min_dist = min_dist
                        next_center = i
            
            centers_idx.append(next_center)
        
        # Assign points to nearest center
        for i in range(n_samples):
            distances = [np.sqrt(np.sum((X[i] - X[c])**2)) for c in centers_idx]
            labels[i] = np.argmin(distances)
        
        return labels

    def _create_empty_analysis(self, total_globules: int) -> ClusteringAnalysis:
        """Create empty analysis when clustering is not possible."""
        return ClusteringAnalysis(
            clusters=[],
            total_globules=total_globules,
            clustering_method="insufficient_data",
            optimal_k=0,
            silhouette_score=0.0,
            processing_time_ms=0.0,
            cross_cluster_relationships={},
            temporal_patterns={},
            quality_metrics={"reason": "insufficient_globules_for_clustering"}
        )

    async def get_cluster_by_id(self, cluster_id: str) -> Optional[SemanticCluster]:
        """Retrieve a specific cluster by ID (would need caching/storage)."""
        # For now, this would need to re-run clustering
        # In production, we'd cache clustering results
        analysis = await self.analyze_semantic_clusters()
        
        for cluster in analysis.clusters:
            if cluster.id == cluster_id:
                return cluster
        
        return None

    async def find_globule_cluster(self, globule_id: str) -> Optional[SemanticCluster]:
        """Find which cluster a specific globule belongs to."""
        analysis = await self.analyze_semantic_clusters()
        
        for cluster in analysis.clusters:
            if globule_id in cluster.member_ids:
                return cluster
        
        return None