from typing import Dict, Any, List
import igraph as ig
import matplotlib.pyplot as plt
from app.xml_manager import read_xml_from_file
import numpy as np
import os

def create_wifi_graph(data: List[Dict[str, Any]]) -> ig.Graph:
    g = ig.Graph()
    
    position_map = {}  # Maps position tuples to vertex indices
    wifi_map = {}  # Maps BSSID to vertex indices
    edges = []
    rssi_values = []
    
    # Ensure data is a list
    if isinstance(data, dict):
        data = [data]
        
    for entry in data:
        position = entry['Position']
        wifi_connections = entry['WiFi']
        
        position_tuple = (position['X'], position['Y'], position['Z'])
        
        # Add position node if not already in the graph
        if position_tuple not in position_map:
            v_idx = g.add_vertex(name=str(position_tuple), type='position').index
            position_map[position_tuple] = v_idx
        else:
            v_idx = position_map[position_tuple]
        
        for wifi in wifi_connections:
            bssid = wifi['BSSID']
            signal = wifi['SIGNAL']
            
            if bssid is not None:
                # Add Wi-Fi node if not already in the graph
                if bssid not in wifi_map:
                    wifi_idx = g.add_vertex(name=bssid, type='wifi').index
                    wifi_map[bssid] = wifi_idx
                else:
                    wifi_idx = wifi_map[bssid]
                
                edges.append((v_idx, wifi_idx))
                rssi_values.append(signal)
    
    g.add_edges(edges)
    g.es['rssi'] = rssi_values  # Explicitly name the attribute as 'rssi'
    g.es['weight'] = rssi_values  # Keep 'weight' for compatibility with existing code
    
    g = g.simplify(combine_edges=dict(rssi="mean", weight="mean"))

    return g


def filter_close_points( data, threshold) -> list:
        """
        Filter out points that are too close to each other.
        Compares every point to ALL existing filtered with the threshold.
        The distance is calculated using matrix or vector norm.
        """
        
        filtered_points = []
        positions = []
        filtered_data = []
        for entry in data:
            position = entry['Position']
            x = float(position['X'])
            y = float(position['Y'])
            z = float(position['Z'])
            point = (x,y,z)
            positions.append(point)

        for point in positions:
            if all(np.linalg.norm(np.array(point[:3]) - np.array(p[:3])) >= threshold for p in filtered_points):
                filtered_points.append(point)

        for entry in data:
            position = entry['Position']
            x = float(position['X'])
            y = float(position['Y'])
            z = float(position['Z'])
            point = (x,y,z)
            if point in filtered_points:
                
                # remove the wifi ssh used for connecting the robot
                wifis = entry['WiFi']
                entry['WiFi'] = [network for network in wifis if network['BSSID'] != "02:2b:47:cb:e0:cb"]
                
                filtered_data.append(entry)
        
           
        return filtered_data



def plot_graph(g: ig.Graph, output_path: str):
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    layout = g.layout("kk")
    _, ax = plt.subplots(figsize=(10, 10))
    # Format vertex labels: if it's a position (tuple or string with coords), keep only 2 decimals
    labels = []
    for v in g.vs:
        if v["type"] == "position":
            try:
                # assume stored as "(x, y, z)" string
                coords = v["name"].strip("()").split(",")
                formatted = "(" + ", ".join(f"{float(c):.2f}" for c in coords) + ")"
                labels.append(formatted)
            except Exception:
                labels.append(str(v["name"]))
        else:
            labels.append(str(v["name"]))

    ig.plot(
        g,
        target=ax,
        layout=layout,
        vertex_label=labels,
        vertex_color=["blue" if v["type"] == "position" else "red" for v in g.vs],
        edge_width=[max(0.5, w * 5.0) for w in g.es["rssi"]],
        edge_label=[f"{w:.2f}" for w in g.es["rssi"]],
        vertex_size=0.5,
        mark_groups=True
    )
    plt.savefig(output_path)
    plt.close()


############# For testing #############
def main():
    positions = read_xml_from_file("../", "wireless_data_amcl.xml")
    filtered_positions = filter_close_points(positions, 3)
   # print(filtered_positions)
    grafo = create_wifi_graph(filtered_positions)
    #print(grafo)
    
    plot_graph(grafo, "output_graph/out")


if __name__ == "__main__":
    main()