"""
WiFi-based Localization System

This module implements WiFi-based indoor localization using graph techniques:
1. Preprocessing and filtering of WiFi signals
2. Graph construction for localization
3. Graph clustering for localization
4. Matching live scans to stored graphs
5. Localization and navigation output
"""

import numpy as np
import igraph as ig
from typing import Dict, Any, List, Tuple, Optional, Union
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import scipy.stats as stats
import pygmtools as pygm
import networkx as nx
from scipy import sparse
import app.graph_manager as gm

# Constants
DEFAULT_RSSI_THRESHOLD = -85  # Default RSSI threshold in dBm
EWMA_ALPHA = 0.3  # Alpha value for Exponential Weighted Moving Average filter
ZSCORE_THRESHOLD = 3.0  # Z-score threshold for outlier detection


# 1. Preprocessing and Filtering
def filter_by_rssi(wifi_data: List[Dict[str, Any]], 
                  rssi_threshold: float = DEFAULT_RSSI_THRESHOLD) -> List[Dict[str, Any]]:
    """
    Filter WiFi data based on RSSI threshold
    
    Args:
        wifi_data: List of WiFi connection dictionaries
        rssi_threshold: RSSI threshold in dBm (default: -85 dBm)
        
    Returns:
        Filtered WiFi data
    """
    return [wifi for wifi in wifi_data if wifi['SIGNAL'] >= rssi_threshold]


def apply_ewma_filter(current_rssi: float, 
                     previous_rssi: Optional[float] = None, 
                     alpha: float = EWMA_ALPHA) -> float:
    """
    Apply Exponential Weighted Moving Average (EWMA) filter to smooth RSSI values
    
    Args:
        current_rssi: Current RSSI value
        previous_rssi: Previous filtered RSSI value (None if first reading)
        alpha: EWMA coefficient (0 < alpha < 1)
        
    Returns:
        Filtered RSSI value
    """
    if previous_rssi is None:
        return current_rssi
    
    return alpha * current_rssi + (1 - alpha) * previous_rssi


def smooth_rssi_values(wifi_data: List[Dict[str, Any]], 
                      previous_values: Dict[str, float] = None) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    """
    Apply EWMA filter to smooth RSSI values in WiFi data
    
    Args:
        wifi_data: List of WiFi connection dictionaries
        previous_values: Dictionary of previous RSSI values by BSSID
        
    Returns:
        Tuple of (smoothed WiFi data, updated previous values dictionary)
    """
    if previous_values is None:
        previous_values = {}
    
    smoothed_data = []
    
    for wifi in wifi_data:
        bssid = wifi['BSSID']
        current_rssi = wifi['SIGNAL']
        
        # Apply EWMA filter
        previous_rssi = previous_values.get(bssid)
        smoothed_rssi = apply_ewma_filter(current_rssi, previous_rssi)
        
        # Update previous values
        previous_values[bssid] = smoothed_rssi
        
        # Create new WiFi entry with smoothed RSSI
        smoothed_wifi = wifi.copy()
        smoothed_wifi['SIGNAL'] = smoothed_rssi
        smoothed_data.append(smoothed_wifi)
    
    return smoothed_data, previous_values


def remove_duplicates(wifi_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Remove duplicate BSSIDs by keeping the strongest signal
    
    Args:
        wifi_data: List of WiFi connection dictionaries
        
    Returns:
        Deduplicated WiFi data
    """
    # Group by BSSID
    bssid_groups = defaultdict(list)
    for wifi in wifi_data:
        bssid_groups[wifi['BSSID']].append(wifi)
    
    # Keep the strongest signal for each BSSID
    deduplicated = []
    for bssid, entries in bssid_groups.items():
        strongest = max(entries, key=lambda x: x['SIGNAL'])
        deduplicated.append(strongest)
    
    return deduplicated


def detect_outliers_zscore(wifi_data: List[Dict[str, Any]], 
                          threshold: float = ZSCORE_THRESHOLD) -> List[Dict[str, Any]]:
    """
    Detect and remove outliers using Z-score
    
    Args:
        wifi_data: List of WiFi connection dictionaries
        threshold: Z-score threshold (default: 3.0)
        
    Returns:
        WiFi data with outliers removed
    """
    if len(wifi_data) <= 1:
        return wifi_data
    
    # Extract RSSI values
    rssi_values = np.array([wifi['SIGNAL'] for wifi in wifi_data])
    
    # Calculate Z-scores
    z_scores = np.abs(stats.zscore(rssi_values))
    
    # Filter out outliers
    filtered_data = [wifi for i, wifi in enumerate(wifi_data) if z_scores[i] <= threshold]
    
    return filtered_data


def detect_spatial_outliers(position_data: List[Dict[str, Any]], 
                           eps: float = 2.0, 
                           min_samples: int = 2) -> List[Dict[str, Any]]:
    """
    Detect and remove spatial outliers using DBSCAN
    
    Args:
        position_data: List of position dictionaries with WiFi data
        eps: DBSCAN epsilon parameter (maximum distance between points)
        min_samples: DBSCAN min_samples parameter
        
    Returns:
        Position data with spatial outliers removed
    """
    if len(position_data) <= min_samples:
        return position_data
    
    # Extract position coordinates
    positions = np.array([[pos['Position']['X'], pos['Position']['Y'], pos['Position']['Z']] 
                         for pos in position_data])
    
    # Standardize features
    scaler = StandardScaler()
    positions_scaled = scaler.fit_transform(positions)
    
    # Apply DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(positions_scaled)
    
    # Filter out outliers (cluster label -1)
    filtered_data = [pos for i, pos in enumerate(position_data) if clusters[i] != -1]
    
    return filtered_data


def preprocess_wifi_data(position_data: List[Dict[str, Any]], 
                        rssi_threshold: float = DEFAULT_RSSI_THRESHOLD,
                        apply_smoothing: bool = True,
                        remove_duplicates_flag: bool = True,
                        detect_outliers: bool = True) -> List[Dict[str, Any]]:
    """
    Preprocess WiFi data with all filtering steps
    
    Args:
        position_data: List of position dictionaries with WiFi data
        rssi_threshold: RSSI threshold in dBm
        apply_smoothing: Whether to apply RSSI smoothing
        remove_duplicates_flag: Whether to remove duplicate BSSIDs
        detect_outliers: Whether to detect and remove outliers
        
    Returns:
        Preprocessed position data
    """
    # Create a deep copy to avoid modifying the original data
    processed_data = []
    previous_values = {}
    
    for position in position_data:
        position_copy = position.copy()
        wifi_data = position_copy['WiFi']
        
        # 1. Filter by RSSI threshold
        wifi_data = filter_by_rssi(wifi_data, rssi_threshold)
        
        # 2. Apply EWMA smoothing if enabled
        if apply_smoothing:
            wifi_data, previous_values = smooth_rssi_values(wifi_data, previous_values)
        
        # 3. Remove duplicates if enabled
        if remove_duplicates_flag:
            wifi_data = remove_duplicates(wifi_data)
        
        # 4. Detect outliers if enabled
        if detect_outliers:
            wifi_data = detect_outliers_zscore(wifi_data)
        
        position_copy['WiFi'] = wifi_data
        processed_data.append(position_copy)
    
    # 5. Detect spatial outliers
    if detect_outliers and len(processed_data) > 2:
        processed_data = detect_spatial_outliers(processed_data)
    
    return processed_data


# 2. Graph Construction for Localization
def create_anchor_graph(position_data: List[Dict[str, Any]]) -> ig.Graph:
    """
    Create a graph with anchor nodes for positioning
    
    Args:
        position_data: List of position dictionaries with WiFi data
        
    Returns:
        iGraph graph with position nodes as anchors
    """
    g = ig.Graph()
    
    position_nodes = set()
    wifi_nodes = set()
    edges = []
    rssi_values = []
    
    for entry in position_data:
        position = entry['Position']
        wifi_connections = entry['WiFi']
        
        position_str = f"({position['X']}, {position['Y']}, {position['Z']})"
        
        if position_str not in position_nodes:
            g.add_vertex(name=position_str, type='position')
            position_nodes.add(position_str)
        
        for wifi in wifi_connections:
            bssid = wifi['BSSID']
            signal = wifi['SIGNAL']
            
            if bssid is not None and bssid not in wifi_nodes:
                g.add_vertex(name=bssid, type='wifi')
                wifi_nodes.add(bssid)
            
            if bssid is not None:
                edges.append((position_str, bssid))
                rssi_values.append(signal)
    
    g.add_edges(edges)
    g.es['rssi'] = rssi_values
    g.es['weight'] = [abs(rssi) for rssi in rssi_values]  # Use absolute RSSI for weight
    
    return g


def create_live_scan_graph(wifi_data: List[Dict[str, Any]]) -> ig.Graph:
    """
    Create a star graph from live WiFi scan
    
    Args:
        wifi_data: List of WiFi connection dictionaries
        
    Returns:
        iGraph graph with user position at center
    """
    g = ig.Graph()
    
    # Add user position node at center
    g.add_vertex(name="user_position", type='position')
    
    # Add WiFi nodes and connect to user position
    edges = []
    rssi_values = []
    
    for wifi in wifi_data:
        bssid = wifi['BSSID']
        signal = wifi['SIGNAL']
        
        if bssid is not None:
            g.add_vertex(name=bssid, type='wifi')
            edges.append(("user_position", bssid))
            rssi_values.append(signal)
    
    g.add_edges(edges)
    g.es['rssi'] = rssi_values
    g.es['weight'] = [abs(rssi) for rssi in rssi_values]  # Use absolute RSSI for weight
    
    return g


# 3. Graph Clustering for Localization
def apply_louvain_clustering(g: ig.Graph) -> ig.Graph:
    """
    Apply Louvain method for community detection
    
    Args:
        g: iGraph graph
        
    Returns:
        Graph with community assignments
    """
    # Convert edge weights to be suitable for clustering (higher weight = stronger connection)
    # For RSSI, we need to invert the values since more negative = weaker signal
    max_weight = max(g.es['weight']) if g.es and 'weight' in g.es.attributes() else 1
    weights = [max_weight - w + 1 for w in g.es['weight']]
    
    # Apply Louvain method
    communities = g.community_multilevel(weights=weights)
    
    # Add community membership as vertex attribute
    g.vs['community'] = communities.membership
    
    return g


def apply_spectral_clustering(g: ig.Graph, k: int = 5) -> ig.Graph:
    """
    Apply spectral clustering using the graph Laplacian
    
    Args:
        g: iGraph graph
        k: Number of clusters
        
    Returns:
        Graph with cluster assignments
    """
    # Ensure k is not larger than the number of vertices
    k = min(k, len(g.vs) - 1)
    
    # Get adjacency matrix
    adj_matrix = g.get_adjacency_sparse()
    
    # Compute the normalized Laplacian
    n = adj_matrix.shape[0]
    diags = 1.0 / np.sqrt(adj_matrix.sum(axis=1).A.flatten())
    D_sqrt_inv = sparse.spdiags(diags, 0, n, n)
    L_norm = sparse.identity(n) - D_sqrt_inv @ adj_matrix @ D_sqrt_inv
    
    # Compute the k smallest eigenvectors
    eigenvalues, eigenvectors = sparse.linalg.eigsh(L_norm, k=k, which='SM')
    
    # Apply k-means to the eigenvectors
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=k)
    clusters = kmeans.fit_predict(eigenvectors)
    
    # Add cluster membership as vertex attribute
    g.vs['spectral_cluster'] = clusters
    
    return g


def apply_dbscan_clustering(g: ig.Graph, eps: float = 0.5, min_samples: int = 2) -> ig.Graph:
    """
    Apply DBSCAN for spatial clustering
    
    Args:
        g: iGraph graph
        eps: DBSCAN epsilon parameter
        min_samples: DBSCAN min_samples parameter
        
    Returns:
        Graph with cluster assignments
    """
    # Get positions of WiFi nodes (if available) or use graph layout
    layout = g.layout("kk")
    positions = np.array(layout.coords)
    
    # Apply DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(positions)
    
    # Add cluster membership as vertex attribute
    g.vs['dbscan_cluster'] = clusters
    
    return g


def cluster_graph(g: ig.Graph, 
                 apply_louvain: bool = True,
                 apply_spectral: bool = True,
                 apply_dbscan: bool = True,
                 spectral_k: int = 5) -> ig.Graph:
    """
    Apply multiple clustering techniques to the graph
    
    Args:
        g: iGraph graph
        apply_louvain: Whether to apply Louvain method
        apply_spectral: Whether to apply spectral clustering
        apply_dbscan: Whether to apply DBSCAN
        spectral_k: Number of clusters for spectral clustering
        
    Returns:
        Graph with cluster assignments
    """
    if apply_louvain:
        g = apply_louvain_clustering(g)
    
    if apply_spectral:
        g = apply_spectral_clustering(g, k=spectral_k)
    
    if apply_dbscan:
        g = apply_dbscan_clustering(g)
    
    return g


# 4. Matching Live Scan to Stored Graphs
def convert_to_networkx(g: ig.Graph) -> nx.Graph:
    """
    Convert iGraph graph to NetworkX graph for compatibility with pygmtools
    
    Args:
        g: iGraph graph
        
    Returns:
        NetworkX graph
    """
    nx_graph = nx.Graph()
    
    # Add nodes with attributes
    for v in g.vs:
        nx_graph.add_node(v.index, name=v['name'], type=v['type'])
    
    # Add edges with attributes
    for e in g.es:
        source, target = e.tuple
        nx_graph.add_edge(source, target, rssi=e['rssi'], weight=e['weight'])
    
    return nx_graph


def calculate_graph_similarity(live_graph: ig.Graph, 
                              stored_graph: ig.Graph,
                              method: str = 'rrwm') -> float:
    """
    Calculate similarity between live scan graph and stored graph
    
    Args:
        live_graph: iGraph graph of live scan
        stored_graph: iGraph graph of stored reference
        method: Graph matching method ('rrwm', 'sm', 'ipfp')
        
    Returns:
        Similarity score (0-1)
    """
    # Extract WiFi nodes only
    live_wifi_nodes = [v for v in live_graph.vs if v['type'] == 'wifi']
    stored_wifi_nodes = [v for v in stored_graph.vs if v['type'] == 'wifi']
    
    # Check for common BSSIDs
    live_bssids = set(v['name'] for v in live_wifi_nodes)
    stored_bssids = set(v['name'] for v in stored_wifi_nodes)
    common_bssids = live_bssids.intersection(stored_bssids)
    
    # If no common BSSIDs, return 0 similarity
    if not common_bssids:
        return 0.0
    
    # Calculate Jaccard similarity (intersection over union)
    jaccard = len(common_bssids) / len(live_bssids.union(stored_bssids))
    
    # If too few common nodes, just use Jaccard similarity
    if len(common_bssids) < 3:
        return jaccard
    
    # For more common nodes, calculate a more sophisticated similarity
    # Get signal strength similarity for common BSSIDs
    signal_similarity = 0.0
    count = 0
    
    for bssid in common_bssids:
        # Find the nodes in both graphs
        live_node = next((v for v in live_wifi_nodes if v['name'] == bssid), None)
        stored_node = next((v for v in stored_wifi_nodes if v['name'] == bssid), None)
        
        if live_node is None or stored_node is None:
            continue
            
        # Get the edges connecting to these nodes
        live_edges = live_graph.es.select(_source=live_graph.vs.find(name=bssid).index) or \
                     live_graph.es.select(_target=live_graph.vs.find(name=bssid).index)
        stored_edges = stored_graph.es.select(_source=stored_graph.vs.find(name=bssid).index) or \
                       stored_graph.es.select(_target=stored_graph.vs.find(name=bssid).index)
        
        if not live_edges or not stored_edges:
            continue
            
        # Compare signal strengths (RSSI values)
        live_rssi = live_edges[0]['rssi'] if 'rssi' in live_edges[0].attributes() else live_edges[0]['weight']
        stored_rssi = stored_edges[0]['rssi'] if 'rssi' in stored_edges[0].attributes() else stored_edges[0]['weight']
        
        # Calculate similarity between RSSI values (closer values = higher similarity)
        # Normalize the difference to a similarity score between 0 and 1
        rssi_diff = abs(live_rssi - stored_rssi)
        max_diff = 100.0  # Maximum expected difference in RSSI values
        rssi_similarity = max(0, 1 - (rssi_diff / max_diff))
        
        signal_similarity += rssi_similarity
        count += 1
    
    # Calculate average signal similarity
    avg_signal_similarity = signal_similarity / count if count > 0 else 0
    
    # Combine Jaccard similarity and signal similarity
    combined_score = 0.7 * jaccard + 0.3 * avg_signal_similarity
    
    return combined_score


def find_best_match(live_scan: List[Dict[str, Any]], 
                   stored_positions: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], float]:
    """
    Find the best matching position for a live scan
    
    Args:
        live_scan: List of WiFi connections from live scan
        stored_positions: List of stored position data
        
    Returns:
        Tuple of (best matching position, similarity score)
    """
    # Create live scan graph
    live_graph = create_live_scan_graph(live_scan)
    
    best_match = None
    best_score = 0.0
    
    # Compare with each stored position
    for position in stored_positions:
        # Create graph for this position
        position_graph = create_anchor_graph([position])
        
        # Calculate similarity
        score = calculate_graph_similarity(live_graph, position_graph)
        
        # Update best match if better
        if score > best_score:
            best_score = score
            best_match = position
    
    return best_match, best_score


# 5. Localization & Navigation Output
def estimate_position(live_scan: List[Dict[str, Any]], 
                     stored_positions: List[Dict[str, Any]],
                     similarity_threshold: float = 0.6,
                     blend_threshold: float = 0.4) -> Dict[str, Any]:
    """
    Estimate user position based on WiFi scan
    
    Args:
        live_scan: List of WiFi connections from live scan
        stored_positions: List of stored position data
        similarity_threshold: Threshold for strong match
        blend_threshold: Threshold for transition area
        
    Returns:
        Estimated position data
    """
    # Preprocess live scan
    processed_scan = filter_by_rssi(live_scan)
    processed_scan = remove_duplicates(processed_scan)
    
    # Find best match
    best_match, best_score = find_best_match(processed_scan, stored_positions)
    
    # If no match found or score too low, return None
    if best_match is None or best_score < blend_threshold:
        return None
    
    # If strong match, return the position
    if best_score >= similarity_threshold:
        return {
            'Position': best_match['Position'],
            'Score': best_score,
            'Type': 'exact_match'
        }
    
    # If in transition area, blend multiple positions
    # Find all positions with scores above blend_threshold
    matches = []
    for position in stored_positions:
        position_graph = create_anchor_graph([position])
        live_graph = create_live_scan_graph(processed_scan)
        score = calculate_graph_similarity(live_graph, position_graph)
        
        if score >= blend_threshold:
            matches.append((position, score))
    
    # Blend positions based on scores
    if matches:
        total_weight = sum(score for _, score in matches)
        blended_x = sum(pos['Position']['X'] * score for pos, score in matches) / total_weight
        blended_y = sum(pos['Position']['Y'] * score for pos, score in matches) / total_weight
        blended_z = sum(pos['Position']['Z'] * score for pos, score in matches) / total_weight
        
        return {
            'Position': {
                'X': blended_x,
                'Y': blended_y,
                'Z': blended_z
            },
            'Score': best_score,
            'Type': 'blended_match',
            'NumMatches': len(matches)
        }
    
    # Fallback to best match
    return {
        'Position': best_match['Position'],
        'Score': best_score,
        'Type': 'fallback_match'
    }


def plot_localization_result(stored_graph: ig.Graph, 
                            live_graph: ig.Graph, 
                            estimated_position: Dict[str, Any],
                            output_path: str):
    """
    Plot the localization result
    
    Args:
        stored_graph: iGraph graph of stored positions
        live_graph: iGraph graph of live scan
        estimated_position: Estimated position data
        output_path: Path to save the plot
    """
    # Create a combined graph
    combined_graph = stored_graph.copy()
    
    # Add live scan nodes and edges
    for v in live_graph.vs:
        if v['type'] == 'wifi' and v['name'] not in [node['name'] for node in combined_graph.vs]:
            combined_graph.add_vertex(name=v['name'], type='wifi')
    
    # Add estimated position node
    if estimated_position:
        pos = estimated_position['Position']
        pos_str = f"ESTIMATED({pos['X']:.2f}, {pos['Y']:.2f}, {pos['Z']:.2f})"
        combined_graph.add_vertex(name=pos_str, type='estimated')
    
    # Create layout
    layout = combined_graph.layout("kk")
    
    # Plot graph
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Define colors based on node type
    colors = []
    for v in combined_graph.vs:
        if v['type'] == 'position':
            colors.append('blue')
        elif v['type'] == 'wifi':
            colors.append('red')
        elif v['type'] == 'estimated':
            colors.append('green')
        else:
            colors.append('gray')
    
    # Plot graph
    ig.plot(
        combined_graph,
        target=ax,
        layout=layout,
        vertex_label=combined_graph.vs["name"],
        vertex_color=colors,
        vertex_size=20,
        mark_groups=True
    )
    
    plt.savefig(output_path)
    plt.close()
