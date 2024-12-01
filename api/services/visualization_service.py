"""
Visualization Service Implementation
Provides visualization generation and data formatting for embeddings analysis.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import logging
from scipy.cluster import hierarchy
from sklearn.manifold import TSNE
from collections import Counter

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class VisualizationData:
    """Container for processed visualization data."""
    chart_type: str             
    data: List[Dict[str, Any]]  
    config: Dict[str, Any]      
    metadata: Dict[str, Any]    

    def to_dict(self) -> Dict[str, Any]:
        """Convert visualization data to dictionary format.
        Ensures all nested structures are JSON serializable."""
        def make_serializable(obj):
            if isinstance(obj, (dict, Counter)):
                return {str(k): make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple, set)):
                return [make_serializable(x) for x in obj]
            elif isinstance(obj, np.ndarray):
                return make_serializable(obj.tolist())
            elif isinstance(obj, (int, float, str, bool)) or obj is None:
                return obj
            return str(obj)

        return {
            'chart_type': self.chart_type,
            'data': make_serializable(self.data),
            'config': make_serializable(self.config),
            'metadata': make_serializable(self.metadata)
        }

class VisualizationService:
    """Service for generating visualization data for PDF analysis."""
    
    def __init__(self):
        """Initialize visualization service with default configuration."""
        self.color_palette = [
            "#2196F3", "#4CAF50", "#FFC107", "#E91E63", 
            "#9C27B0", "#00BCD4", "#F44336", "#3F51B5"
        ]
        
        self.default_configs = {
            'relevance': {
                'xAxis': 'Relevance Level',
                'yAxis': 'Number of Cards',
                'colors': {
                    'high': self.color_palette[0],
                    'medium': self.color_palette[2],
                    'low': self.color_palette[6]
                }
            },
            'embedding': {
                'xAxis': 't-SNE Dimension 1',
                'yAxis': 't-SNE Dimension 2',
                'colors': {
                    'high': self.color_palette[0],
                    'medium': self.color_palette[2],
                    'low': self.color_palette[6]
                }
            },
            'similarity': {
                'colorRange': ['#FFFFFF', self.color_palette[0]],
                'cellSize': 30,
                'legendTitle': 'Similarity Score'
            }
        }

    def generate_relevance_distribution(
        self,
        results: List[Dict[str, Any]]
    ) -> VisualizationData:
        """Generate visualization data for relevance distribution."""
        try:
            # Count relevance tags
            tag_counts = Counter(result['tag'] for result in results)
            
            # Calculate percentages
            total = len(results)
            data = [
                {
                    'tag': tag,
                    'count': count,
                    'percentage': round(count / total * 100, 1)
                }
                for tag, count in sorted(tag_counts.items())
            ]
            
            return VisualizationData(
                chart_type="bar",
                data=data,
                config=self.default_configs['relevance'],
                metadata={
                    'totalCards': total,
                    'distribution': dict(tag_counts),
                    'averageRelevance': np.mean([
                        1.0 if r['tag'] == 'high' else 0.5 if r['tag'] == 'medium' else 0.0 
                        for r in results
                    ])
                }
            )
        except Exception as e:
            logger.error(f"Error generating relevance distribution: {str(e)}")
            raise

    def generate_embedding_visualization(
        self,
        embeddings: np.ndarray,
        metadata: List[Dict[str, Any]]
    ) -> VisualizationData:
        """Generate t-SNE visualization of embeddings."""
        try:
            # Perform dimensionality reduction
            tsne = TSNE(
                n_components=2,
                perplexity=min(30, len(embeddings) - 1),
                random_state=42
            )
            coords = tsne.fit_transform(embeddings)
            
            # Prepare visualization data
            data = [
                {
                    'x': float(coords[i, 0]),
                    'y': float(coords[i, 1]),
                    'relevance': metadata[i].get('relevance', 'unknown'),
                    'cardId': metadata[i].get('card_id', f'card_{i}'),
                    'similarity': metadata[i].get('similarity', 0.0)
                }
                for i in range(len(metadata))
            ]
            
            # Estimate clusters
            n_clusters = self._estimate_clusters(coords)
            cluster_info = self._analyze_clusters(coords, [m['relevance'] for m in metadata], n_clusters)
            
            return VisualizationData(
                chart_type="scatter",
                data=data,
                config=self.default_configs['embedding'],
                metadata={
                    'clusters': n_clusters,
                    'clusterAnalysis': cluster_info,
                    'dimensions': embeddings.shape,
                    'variance': np.var(coords, axis=0).tolist()
                }
            )
        except Exception as e:
            logger.error(f"Error generating embedding visualization: {str(e)}")
            raise

    def _estimate_clusters(self, coords: np.ndarray) -> int:
        """Estimate optimal number of clusters in the data."""
        try:
            # Perform hierarchical clustering
            linkage = hierarchy.linkage(coords, method='ward')
            
            # Use elbow method to estimate optimal clusters
            last = linkage[-10:, 2]
            acceleration = np.diff(last, 2)
            k = acceleration.argmax() + 2
            
            return int(min(k, len(coords) // 5))  # Limit max clusters
        except Exception as e:
            logger.error(f"Error estimating clusters: {str(e)}")
            return 1

    def _analyze_clusters(
        self,
        coords: np.ndarray,
        relevance_tags: List[str],
        n_clusters: int
    ) -> Dict[str, Any]:
        """Analyze cluster composition and characteristics."""
        try:
            from sklearn.cluster import KMeans
            
            # Perform clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(coords)
            
            # Analyze cluster composition
            cluster_analysis = []
            for i in range(n_clusters):
                cluster_mask = cluster_labels == i
                cluster_info = {
                    'id': i,
                    'size': int(np.sum(cluster_mask)),
                    'center': kmeans.cluster_centers_[i].tolist(),
                    'composition': Counter([tag for tag, is_member in zip(relevance_tags, cluster_mask) if is_member])
                }
                cluster_analysis.append(cluster_info)
            
            return {
                'n_clusters': n_clusters,
                'clusters': cluster_analysis,
                'silhouette_score': float(np.mean([
                    np.mean([
                        np.linalg.norm(coords[i] - center) 
                        for i, label in enumerate(cluster_labels) 
                        if label == cluster_id
                    ]) 
                    for cluster_id, center in enumerate(kmeans.cluster_centers_)
                ]))
            }
        except Exception as e:
            logger.error(f"Error analyzing clusters: {str(e)}")
            return {'error': str(e)}

    def _calculate_similarity_stats(self, similarity_matrix: np.ndarray) -> Dict[str, float]:
        """Calculate statistics for similarity matrix."""
        try:
            # Remove self-similarities from diagonal
            mask = ~np.eye(similarity_matrix.shape[0], dtype=bool)
            similarities = similarity_matrix[mask]
            
            return {
                'mean': float(np.mean(similarities)),
                'std': float(np.std(similarities)),
                'min': float(np.min(similarities)),
                'max': float(np.max(similarities)),
                'median': float(np.median(similarities)),
                'q1': float(np.percentile(similarities, 25)),
                'q3': float(np.percentile(similarities, 75))
            }
        except Exception as e:
            logger.error(f"Error calculating similarity stats: {str(e)}")
            return {}

    def get_visualization_config(self) -> Dict[str, Any]:
        """Get complete visualization configuration."""
        return {
            'colorPalette': self.color_palette,
            'supportedCharts': ['bar', 'scatter'],
            'defaultConfigs': self.default_configs,
            'defaultSettings': {
                'fontSize': 12,
                'padding': 20,
                'aspectRatio': 16/9,
                'animation': True,
                'legend': True,
                'tooltip': True
            },
            'responsive': True
        }