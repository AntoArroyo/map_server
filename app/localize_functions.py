import igraph as ig
import ast
import numpy as np
from typing import List, Dict, Tuple

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

# Normalize function for RSSI
def normalize_rssi(rssi: float) -> float:
    rssi = min(max(rssi, -90), -30) 
    return (rssi + 90) / 60


    
def get_estimated_position(
    scan_data: List[Tuple[str, float]], 
    g: ig.Graph, 
    sigma: float = 0.1, 
    penalty_per_missing: float = 0.01, 
    penalty_extra: float = 0.005,
    score_threshold: float = 0.2,
    top_n_signals: int = 10,
    dynamic_sigma: bool = True,
    common_bssid_threshold: int = 5
):
    """
    Localizes the device based on scan data using a graph of positions and Wi-Fi nodes.

    Parameters:
        scan_data: List of (BSSID, RSSI)
        g: igraph.Graph with vertices of type 'position' and 'wifi', and edges containing 'rssi'
        sigma: Base standard deviation for Gaussian similarity
        penalty_per_missing: Penalty per missing expected BSSID
        penalty_extra: Penalty per unexpected observed BSSID
        score_threshold: Minimum acceptable score for considering a position
        top_n_signals: Number of strongest signals to consider
        dynamic_sigma: Whether to reduce sigma dynamically if many signals match

    Returns:
        The name of the most likely position node, and optionally the average (x, y) coordinates.
    """

    # Use only top-N strongest signals
    scan_data = sorted(scan_data, key=lambda x: x[1], reverse=True)[:top_n_signals]
    scan_dict = dict(scan_data)
    position_nodes = [v for v in g.vs if v["type"] == "position"]

    best_match = None
    best_score = float('-inf')
    best_scores = {}

    for pos_v in position_nodes:
        pos_idx = pos_v.index
        pos_name = pos_v["name"]
        edges = g.incident(pos_idx, mode="OUT")

        stored_bssids = {}
        for e_idx in edges:
            edge = g.es[e_idx]
            wifi_v = g.vs[edge.target]
            if wifi_v["type"] == "wifi":
                stored_bssids[wifi_v["name"]] = edge["rssi"]

        expected = set(stored_bssids)
        observed = set(scan_dict)

        common = expected & observed
        missing = expected - observed
        extra = observed - expected

        # Skip if not common BSSIDs
        if not common:
            continue

        # Adjust sigma if many BSSIDs are in common
        sigma_eff = sigma
        if dynamic_sigma and len(common) > common_bssid_threshold:
            sigma_eff *= 0.7

        # Compute RSSI similarity with Gaussian weighting and RSSI importance
        score = sum(
            np.exp(-((scan_dict[b] - stored_bssids[b]) ** 2) / (2 * sigma_eff ** 2)) * (scan_dict[b])
            for b in common
        )

        # Penalize for missing expected and extra unexpected BSSIDs
        score -= len(missing) * penalty_per_missing
        score -= len(extra) * penalty_extra

        if score > score_threshold:
            best_scores[pos_name] = score

        if score > best_score:
            best_score = score
            best_match = pos_name

    if best_match:
        print(f"[INFO] Best node: {best_match} with score: {best_score:.2f}")

    if best_scores:
        print(f"[INFO] Nodes above threshold: {best_scores}")

    if not best_scores and not best_match:
        return None, None

    # Filter best_scores to those within 90% of best_score
    filtered_scores = {
        name: score for name, score in best_scores.items()
        if score >= best_score * 0.9
    }

    coords = []
    for name in filtered_scores:
        try:
            x, y, _ = ast.literal_eval(name)  # Assuming name = "(x, y, z)"
            coords.append((x, y))
            if filtered_scores[name] > 3.5:
                print(f"[INFO] Possible 100% node match found: {name}")
                return name, None
        except Exception:
            print(f"[WARNING] Could not parse coordinates from name: {name}")
            continue

    if not coords:
        return best_match, None  # Fallback to best_match even if we can't average

    x_avg = sum(x for x, _ in coords) / len(coords)
    y_avg = sum(y for _, y in coords) / len(coords)

    print(f"[INFO] Estimated position: ({x_avg:.2f}, {y_avg:.2f})")

    return best_match, (round(x_avg, 2), round(y_avg, 2))



def build_scan_graph(scan_data):
    """
    Crea un grafo estrella para el escaneo:
    - Un vértice central llamado 'scan_center'
    - Un vértice por cada BSSID con arista ponderada por RSSI
    """
    g = ig.Graph()
    g.add_vertex(name="scan_center", type="scan")

    for bssid, rssi in scan_data:
        g.add_vertex(name=bssid, type="wifi")
        g.add_edge("scan_center", bssid, rssi=rssi)

    return g

def get_position_subgraph(map_g, pos_vertex):
    """
    Devuelve un subgrafo que contiene:
    - El nodo de posición
    - Todos los vértices wifi conectados y las aristas RSSI
    """
    neighbors = map_g.neighbors(pos_vertex, mode="OUT")
    nodes = [pos_vertex] + neighbors
    return map_g.subgraph(nodes)



def graph_similarity(scan_g, pos_sub_g, sigma=0.2, alpha=0.02, betha=0.02):
    def get_edges(g, center):
        return {
            g.vs[edge.target if edge.source == center else edge.source]["name"]: edge["rssi"]
            for edge in g.es[g.incident(center, mode="OUT")]
        }

    scan_center = scan_g.vs.find(type="scan").index
    scan_edges = get_edges(scan_g, scan_center)

    pos_center = [v.index for v in pos_sub_g.vs if v["type"] == "position"][0]
    pos_edges = get_edges(pos_sub_g, pos_center)

    common = set(scan_edges) & set(pos_edges)
    missing = set(pos_edges) - set(scan_edges)
    extra = set(scan_edges) - set(pos_edges)

    if not common:
        return float("-inf")

    # similitud gaussiana por AP (RSSI en [0,1])
    similarities = [
        np.exp(-((scan_edges[b] - pos_edges[b])**2) / (2*sigma**2))
        for b in common
    ]

    # score normalizado entre 0 y 1
    score = sum(similarities) / len(common)

    # penalizaciones ajustadas
    score -= alpha * len(missing)
    score -= betha * len(extra)

    return score


def localize_by_graph_matching(scan_data, map_g, best_score_threshold=0.65):
    scan_g = build_scan_graph(scan_data)
    best_pos_list = []
    best_score = float("-inf")
    best_pos = None
    try:
        for v in map_g.vs.select(type="position"):
            sub = get_position_subgraph(map_g, v.index)
            s = graph_similarity(scan_g, sub)
            if s > best_score:
                #print(f"BEST Score -- {s}")
                best_score = s
                best_pos = v["name"]
            if s > best_score_threshold:
                best_pos_list.append(v["name"])
        len_best_positions = len(best_pos_list)
        print(f"Best position list len -- {len_best_positions}") 
        print(f"Best position list -- {best_pos_list}") 
        if len_best_positions > 1:
            x_coords = 0
            y_coords = best_score_threshold
            z_coords = 0
           # print(f"LIST BEST POSITION -- {best_pos_list}")
            for pos in best_pos_list:
                (x, y, z) = parse_coordinate(pos)
                x_coords += x
                y_coords += y
                z_coords += z
            #print(f"x -- {x_coords} y -- {y_coords}")
            #print(f"LEN -- {len_best_positions}")
            best_pos = (x_coords/len_best_positions, y_coords/len_best_positions, z_coords)
            best_pos = str(best_pos)
        return best_pos, best_score
    except Exception as e:
        print(f"Error while localize with graph matching -- {e}")
    # In case only 1 pos
    best_pos_parsed = parse_coordinate(best_pos)
    return best_pos_parsed, best_score


def parse_coordinate(coord_str: str):
    """
    Parse a string like "(-4.353343681658524, 29.05150388982781, 0.0)"
    into a tuple of floats (-4.353343681658524, 29.05150388982781, 0.0).
    """
    # Remove any whitespace and enclosing parentheses
    stripped = coord_str.strip().strip("()")
    # Split by comma
    parts = stripped.split(",")
    # Convert each piece to float
    return tuple(float(p) for p in parts)