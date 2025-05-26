import pygmtools as pygm
pygm.BACKEND = 'pytorch'  # or 'numpy', depending on your setup

import networkx as nx
from igraph import Graph
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple, Optional

class WiFiSignal:
    def __init__(self, bssid: str, rssi: float, ssid: str = None):
        self.bssid = bssid
        self.rssi = rssi
        self.ssid = ssid

def rssi_similarity(rssi1, rssi2):
    """Calculate similarity between two RSSI values"""
    if rssi1 is not None and rssi2 is not None:
        return 1 / (1 + abs(rssi1 - rssi2))
    else: 
        return 0

def compute_similarity(db_signals: List[Dict], scan_signals: List[WiFiSignal]):
    """Compute similarity between database signals and scanned signals"""
    best_score = 0.0
    best_x, best_y, best_z = None, None, None
    
    for position_data in db_signals:
        current_score = 0.0
        matches = 0
        
        wifi_devices = position_data.get('WiFi', [])
        position = position_data.get('Position', {})
        
        for scan_signal in scan_signals:
            for device in wifi_devices:
                if scan_signal.bssid == device.get('BSSID'):
                    print(f"RSSI SCANNED {scan_signal.bssid}: {scan_signal.rssi}, RSSI STORED: {device.get('RSSI')}")
                    current_score += rssi_similarity(scan_signal.rssi, device.get('RSSI'))
                    matches += 1
        
        # Normalize score by number of matches to avoid bias toward locations with more APs
        if matches > 0:
            normalized_score = current_score / matches
            if normalized_score > best_score:
                best_score = normalized_score
                best_x = position.get('X')
                best_y = position.get('Y')
                best_z = position.get('Z')
    
    return best_score, best_x, best_y, best_z

def compute_best_position_basic(wifi_signals: List[Dict], scanned_signals: List[WiFiSignal]):
    """Find best position using basic RSSI similarity"""
    best_score = -1
    best_position = None

    score, x, y, z = compute_similarity(wifi_signals, scanned_signals)

    if score > best_score:
        best_score = score
        best_position = {"x": x, "y": y, "z": z, "score": score}

    return best_position

def graph_to_pygm_format(graph: nx.Graph):
    """Convert NetworkX graph to pygmtools format"""
    nodes = list(graph.nodes())
    A = nx.to_numpy_array(graph, nodelist=nodes, weight='weight')
    return A, nodes

def networkx_to_tensor(graph: nx.Graph):
    """Convert NetworkX graph to tensor format for pygmtools"""
    A, nodes = graph_to_pygm_format(graph)
    # Add batch dimension for pygmtools
    A_tensor = torch.tensor(A, dtype=torch.float32).unsqueeze(0)
    return A_tensor, nodes

def compute_rrwm_to_graphs(live_scan: nx.Graph, stored_graph: nx.Graph):
    """Compute RRWM matching between two NetworkX graphs and return center node match"""
    A1, nodes1 = graph_to_pygm_format(live_scan)
    A2, nodes2 = graph_to_pygm_format(stored_graph)
    
    # Convert to tensors with batch dimension
    A1_tensor = torch.tensor(A1, dtype=torch.float32).unsqueeze(0)
    A2_tensor = torch.tensor(A2, dtype=torch.float32).unsqueeze(0)
    
    # RRWM matching
    X = pygm.rrwm(A1_tensor, A2_tensor)
    X_np = X[0].detach().cpu().numpy()
    
    # Find center nodes
    center_idx1 = None
    center_idx2 = None
    
    for i, node in enumerate(nodes1):
        if live_scan.nodes[node].get('node_type') == 'center':
            center_idx1 = i
            break
    
    for i, node in enumerate(nodes2):
        if stored_graph.nodes[node].get('node_type') == 'center':
            center_idx2 = i
            break
    
    # Fallback to highest degree if center not found
    if center_idx1 is None:
        degrees1 = [live_scan.degree(node) for node in nodes1]
        center_idx1 = int(np.argmax(degrees1))
    
    if center_idx2 is None:
        degrees2 = [stored_graph.degree(node) for node in nodes2]
        center_idx2 = int(np.argmax(degrees2))
    
    center_match_score = float(X_np[center_idx1, center_idx2])
    total_matching_score = X.sum().item()
    
    print(f"### Center Node Match Score -- {center_match_score}")
    print(f"### Total Similarity Score -- {total_matching_score}")
    print(f"### Matched Center Node: {nodes2[center_idx2]}")
    
    return total_matching_score, center_match_score, nodes2[center_idx2]

def igraph_to_numpy(graph: Graph):
    """Convert igraph to numpy adjacency matrix"""
    adj_matrix = np.array(graph.get_adjacency(attribute='weight').data, dtype=float)
    nodes = [v['name'] if 'name' in v.attributes() else str(v.index) for v in graph.vs]
    return adj_matrix, nodes

def match_graphs(live_graph: Graph, stored_graph: Graph):
    """Match two igraph graphs using RRWM"""
    A1, nodes1 = igraph_to_numpy(live_graph)
    A2, nodes2 = igraph_to_numpy(stored_graph)

    # RRWM takes batched matrices; add a batch dimension
    A1_tensor = torch.tensor(A1, dtype=torch.float32).unsqueeze(0)  # shape: (1, N, N)
    A2_tensor = torch.tensor(A2, dtype=torch.float32).unsqueeze(0)  # shape: (1, M, M)

    # RRWM returns a match matrix: shape (1, N, M)
    X = pygm.rrwm(A1_tensor, A2_tensor)
    score = X.sum().item()  # total match strength

    return score, nodes1, nodes2

def compute_best_position_graph(live_graph: Graph, stored_graphs: List[Tuple[Graph, Dict]]):
    """Find best position using graph matching"""
    best_score = -1
    best_position = None

    for stored_graph, position_info in stored_graphs:
        score, live_nodes, stored_nodes = match_graphs(live_graph, stored_graph)
        
        if score > best_score:
            best_score = score
            best_position = {
                "x": position_info.get('x'),
                "y": position_info.get('y'), 
                "z": position_info.get('z'),
                "score": score
            }

    return best_position

def igraph_to_features(graph: Graph):
    """Extract features from igraph"""
    # Use degree and weighted degree as features
    features = []
    for v in graph.vs:
        degree = v.degree()
        # Get weighted degree if weights exist
        weighted_degree = sum([graph.es[e]['weight'] if 'weight' in graph.es[e].attributes() else 1.0 
                              for e in graph.incident(v.index)])
        features.append([degree, weighted_degree])
    
    features = np.array(features, dtype=float)
    adj = np.array(graph.get_adjacency(attribute='weight').data, dtype=float)
    nodes = [v['name'] if 'name' in v.attributes() else str(v.index) for v in graph.vs]
    
    return features, adj, nodes

def build_node_affinity(X1: np.ndarray, X2: np.ndarray):
    """Build node affinity matrix using cosine similarity"""
    # Ensure non-zero features for cosine similarity
    X1_norm = X1 + 1e-8
    X2_norm = X2 + 1e-8
    sim = cosine_similarity(X1_norm, X2_norm)
    return sim

def build_affinity_matrix(X1: np.ndarray, X2: np.ndarray):
    """Build full affinity matrix for RRWM"""
    node_aff = build_node_affinity(X1, X2)  # shape: (n1, n2)
    n1, n2 = node_aff.shape
    
    # Construct K: (n1*n2, n1*n2) with diagonal node similarities
    K = np.zeros((n1 * n2, n1 * n2), dtype=float)
    for i in range(n1):
        for j in range(n2):
            K[i * n2 + j, i * n2 + j] = node_aff[i, j]
    
    return torch.tensor(K, dtype=torch.float32).unsqueeze(0)  # shape (1, n1*n2, n1*n2)

def match_graphs_return_position(live_graph: Graph, stored_graph: Graph):
    """Match graphs and return position with score"""
    try:
        x1_np, _, nodes1 = igraph_to_features(live_graph)
        x2_np, _, nodes2 = igraph_to_features(stored_graph)

        n1, n2 = x1_np.shape[0], x2_np.shape[0]
        
        if n1 == 0 or n2 == 0:
            return 0.0, None

        K = build_affinity_matrix(x1_np, x2_np)  # shape (1, n1*n2, n1*n2)
        n1_tensor = torch.tensor([n1])
        n2_tensor = torch.tensor([n2])

        X = pygm.rrwm(K, n1=n1_tensor, n2=n2_tensor)
        X = X[0].detach().cpu().numpy()  # shape (n1, n2)

        # Find center node in live_graph (node with type 'center')
        center_index_live = None
        for i, v in enumerate(live_graph.vs):
            # Fixed: Use dictionary-style access or check if attribute exists
            try:
                if 'type' in v.attributes() and v['type'] == 'center':
                    center_index_live = i
                    break
            except (KeyError, TypeError):
                continue
        
        if center_index_live is None:
            # Fallback: use highest degree node
            degrees = [v.degree() for v in live_graph.vs]
            center_index_live = int(np.argmax(degrees))

        # Find center node in stored_graph (node with type 'center')
        center_index_stored = None
        for i, v in enumerate(stored_graph.vs):
            # Fixed: Use dictionary-style access or check if attribute exists
            try:
                if 'type' in v.attributes() and v['type'] == 'center':
                    center_index_stored = i
                    break
            except (KeyError, TypeError):
                continue
        
        if center_index_stored is None:
            # Fallback: use highest degree node
            degrees = [v.degree() for v in stored_graph.vs]
            center_index_stored = int(np.argmax(degrees))

        # Get the matching score between the two center nodes
        center_match_score = float(X[center_index_live, center_index_stored])
        total_score = float(X.sum())
        
        # Return the center node (position node) from stored graph
        center_node_name = nodes2[center_index_stored]

        return total_score, {
            "total_score": total_score,
            "center_match_score": center_match_score,
            "matched_center_node": center_node_name
        }
        
    except Exception as e:
        print(f"Error in graph matching: {e}")
        return 0.0, None

def create_wifi_graph_networkx(wifi_signals: List[WiFiSignal], center_name: str = "center") -> nx.Graph:
    """Create NetworkX graph from WiFi signals with center node"""
    G = nx.Graph()
    
    # Add center node
    G.add_node(center_name, node_type='center')
    
    # Add WiFi nodes and connect to center
    for signal in wifi_signals:
        wifi_node = f"wifi_{signal.bssid}"
        G.add_node(wifi_node, node_type='wifi', bssid=signal.bssid)
        
        # Edge weight based on RSSI (stronger signal = higher weight)
        weight = max(0, 100 + signal.rssi)  # Convert RSSI to positive weight
        G.add_edge(center_name, wifi_node, weight=weight, rssi=signal.rssi)
    
    return G

def create_wifi_graph_igraph(wifi_signals: List[WiFiSignal], center_name: str = "center") -> Graph:
    """Create igraph from WiFi signals with center node"""
    g = Graph()
    
    # Add vertices
    vertices = [center_name]
    for signal in wifi_signals:
        vertices.append(f"wifi_{signal.bssid}")
    
    g.add_vertices(vertices)
    
    # Set vertex names and types
    g.vs['name'] = vertices
    g.vs[0]['type'] = 'center'
    for i in range(1, len(vertices)):
        g.vs[i]['type'] = 'wifi'
        g.vs[i]['bssid'] = wifi_signals[i-1].bssid
    
    # Add edges from center to all WiFi nodes
    edges = []
    edge_weights = []
    for i, signal in enumerate(wifi_signals):
        edges.append((0, i+1))  # Connect center (0) to wifi node (i+1)
        weight = max(0, 100 + signal.rssi)  # Convert RSSI to positive weight
        edge_weights.append(weight)
    
    g.add_edges(edges)
    g.es['weight'] = edge_weights
    
    return g


# Additional helper function for safe vertex attribute access
def get_vertex_attribute(vertex, attr_name, default=None):
    """Safely get vertex attribute from igraph vertex"""
    try:
        if attr_name in vertex.attributes():
            return vertex[attr_name]
        return default
    except (KeyError, TypeError):
        return default