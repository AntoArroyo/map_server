from typing import Dict, Any, List
import igraph as ig
import matplotlib.pyplot as plt
import xml_manager

def create_wifi_graph(data: List[Dict[str, Any]]) -> ig.Graph:
    g = ig.Graph()
    
    position_nodes = set()
    wifi_nodes = set()
    edges = []
    rssi_values = []

    # Ensure data is a list
    if isinstance(data, dict):
        data = [data]
        
    for entry in data:
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
    g.es['rssi'] = rssi_values  # Explicitly name the attribute as 'rssi'
    g.es['weight'] = rssi_values  # Keep 'weight' for compatibility with existing code
    
    return g

def plot_graph(g: ig.Graph, output_path: str):
    layout = g.layout("kk")
    fig, ax = plt.subplots(figsize=(10, 10))
    ig.plot(
        g,
        target=ax,
        layout=layout,
        vertex_label=g.vs["name"],
        vertex_color=["blue" if v["type"] == "position" else "red" for v in g.vs],
        edge_width=[w / 10.0 for w in g.es["rssi"]],  # Use 'rssi' instead of 'weight'
        edge_label=g.es["rssi"],  # Add edge labels showing RSSI values
        vertex_size=20,
        mark_groups=True
    )
    plt.savefig(output_path)
    plt.close()

def main():
    map_data = xml_manager.read_xml("uploads/", "example_data.xml")
    print(map_data)
    
    graph = create_wifi_graph(map_data)
    print(graph)
    plot_graph(graph, 'graph_output.png')

if __name__ == "__main__":
    main()
