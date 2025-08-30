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
    dynamic_sigma: bool = True
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

        if not common:
            continue

        # Adjust sigma if many BSSIDs are in common
        sigma_eff = sigma
        if dynamic_sigma and len(common) > 5:
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