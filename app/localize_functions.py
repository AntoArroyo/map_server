import pygmtools as pygm
pygm.BACKEND = 'pytorch'  # or 'numpy', depending on your setup


# Assume `G1` and `G2` are networkx graphs with BSSID nodes and RSSI-weighted edges
import networkx as nx
from igraph import Graph
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity

def rssi_similarity(rssi1, rssi2):
    # Smaller difference means more similar (using inverse of distance)
    if rssi1 is not None and rssi2 is not None:
        return 1 / (1 + abs(rssi1 - rssi2))
    else: 
        return 0

def compute_similarity(db_signals: dict, scan_signals: dict):
    score = 0.0
    x, y, z = None, None, None
    
    for position in db_signals:
        for signal in scan_signals:
            for device in position.get('WiFi'):
                if signal.bssid == device.get('BSSID'):
                    print(f" RSSI SCANNED {signal.bssid} RSSI STORED {device.get('BSSID')}")
                    score += rssi_similarity(signal.rssi, device.get('RSSI'))
                    x = position.get('Position').get('X')
                    y = position.get('Position').get('Y')
                    z = position.get('Position').get('Z')
            
    return score, x, y, z

def compute_best_position_basic(wifi_signals: list, scanned_signals: list):
    best_score = -1
    best_position = None

    # similarity between the positions and wifi received and stored
    score, x, y, z = compute_similarity(wifi_signals, scanned_signals )

    if score > best_score:
        best_score = score
        best_position = {"x": x, "y": y, "z": z}

    return best_position


def graph_to_pygm_format(graph):
    A = nx.to_numpy_array(graph, weight='weight')
    return A, list(graph.nodes())


# G1: live scan graph
# G2: stored graph for location X
def compute_rrwm_to_graphs(live_scan, stored_graph): 


    A1, nodes1 = graph_to_pygm_format(live_scan)
    A2, nodes2 = graph_to_pygm_format(stored_graph)

    X = pygm.rrwm(A1, A2)  # matching probability matrix
    print(f" ### Prob MATRIX -- {X}")
    matching_score = X.sum().item()  # total similarity
    print(f" ### TOTAL SIMILARITY -- {matching_score}")
    

def match_graphs(live_graph, stored_graph):
    A1, nodes1 = igraph_to_numpy(live_graph)
    A2, nodes2 = igraph_to_numpy(stored_graph)

    # RRWM takes batched matrices; add a batch dimension
    A1 = A1[None, :, :]  # shape: (1, N, N)
    A2 = A2[None, :, :]  # shape: (1, M, M)

    # RRWM returns a match matrix: shape (1, N, M)
    X = pygm.rrwm(A1, A2)
    score = X.sum().item()  # total match strength

    return score
def compute_best_position_graph():
    best_score = -1
    best_position = None

    # similarity between the positions and wifi received and stored
    score, node = match_graphs(wifi_signals, scanned_signals)

    if score > best_score:
        best_score = score
        best_position = {"x": x, "y": y, "z": z}

    return best_position

def igraph_to_features(graph: Graph):
    # Dummy feature: use degree or any scalar (replace with RSSI vector for real use)
    features = np.array([[v.degree()] for v in graph.vs], dtype=float)
    adj = np.array(graph.get_adjacency(attribute='weight').data, dtype=float)
    return features, adj, list(graph.vs["name"])

def build_node_affinity(X1, X2):
    # Use cosine similarity to compute node affinities
    sim = cosine_similarity(X1, X2)
    return sim

def build_affinity_matrix(X1, X2):
    node_aff = build_node_affinity(X1, X2)  # shape: (n1, n2)
    n1, n2 = node_aff.shape
    # Construct K: (n1*n2, n1*n2) with diagonal node similarities only
    K = np.zeros((n1 * n2, n1 * n2), dtype=float)
    for i in range(n1):
        for j in range(n2):
            K[i * n2 + j, i * n2 + j] = node_aff[i, j]
    return torch.tensor(K, dtype=torch.float32).unsqueeze(0)  # shape (1, n1*n2, n1*n2)

def match_graphs_return_position(live_graph: Graph, stored_graph: Graph):
    X1_np, _, nodes1 = igraph_to_features(live_graph)
    X2_np, _, nodes2 = igraph_to_features(stored_graph)

    n1, n2 = X1_np.shape[0], X2_np.shape[0]

    K = build_affinity_matrix(X1_np, X2_np)  # shape (1, n1*n2, n1*n2)
    n1_tensor = torch.tensor([n1])
    n2_tensor = torch.tensor([n2])

    X = pygm.rrwm(K, n1=n1_tensor, n2=n2_tensor)
    X = X[0].detach().cpu().numpy()  # shape (n1, n2)

    center_index_live = nodes1.index("LIVE_SCAN")  # Use actual name of center node
    best_match_index = np.argmax(X[center_index_live])
    matched_node = nodes2[best_match_index]
    score = X.sum()

    return score, matched_node