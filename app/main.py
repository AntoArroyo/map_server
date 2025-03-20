from typing import Union, List, Dict, Any
from fastapi import FastAPI, File, UploadFile, HTTPException
import os
from app.xml_manager import read_xml
import app.graph_manager as g
import app.wifi_localization as wl
from pydantic import BaseModel

app = FastAPI()

# Directory to store uploaded files
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# In-memory storage for processed WiFi data
processed_maps = {}

class WiFiScan(BaseModel):
    wifi_data: List[Dict[str, Any]]

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.put("/upload_graph/")
async def upload_file(file: UploadFile = File(...)):
    # Log the content type for debugging
    print(f"Received file with content type: {file.content_type}")
    
    if file.content_type != "text/xml":
        raise HTTPException(status_code=400, detail="Invalid file type. Only XML files are allowed.")
    
    file_location = os.path.join(UPLOAD_DIR, file.filename)
    
    try:
        with open(file_location, "wb") as f:
            f.write(await file.read())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving file: {e}")
    
    return {"info": f"file '{file.filename}' saved at '{file_location}'"}


@app.get("/plot_graph/{graph_name}")
async def plot_graph(graph_name: str):
    filename = graph_name + ".xml"
    try:
        current_graph = read_xml(UPLOAD_DIR, filename)
        graph = g.create_wifi_graph(current_graph)
        g.plot_graph(graph, "outputPlot")
        return {"info": f"Plotted graph '{graph_name}'"}, 
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File not found")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Error reading map: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")


@app.post("/process_graph/{graph_name}")
async def process_graph(graph_name: str, 
                     rssi_threshold: float = -85.0, 
                     apply_smoothing: bool = True,
                     remove_duplicates: bool = True,
                     detect_outliers: bool = True):
    """
    Process and filter a WiFi map for improved localization
    """
    filename = graph_name + ".xml"
    try:
        # Read the map
        current_graph = read_xml(UPLOAD_DIR, filename)
        
        # Apply preprocessing and filtering
        processed_data = wl.preprocess_wifi_data(
            current_graph,
            rssi_threshold=rssi_threshold,
            apply_smoothing=apply_smoothing,
            remove_duplicates_flag=remove_duplicates,
            detect_outliers=detect_outliers
        )
        
        # Store processed data in memory
        processed_maps[graph_name] = processed_data
        
        # Create and plot the graph
        graph = g.create_wifi_graph(processed_data)
        clustered_graph = wl.cluster_graph(graph)
        g.plot_graph(clustered_graph, f"processed_{graph_name}")
        
        return {
            "info": f"Processed and plotted graph '{graph_name}'",
            "original_points": len(current_graph),
            "processed_points": len(processed_data),
            "wifi_connections": sum(len(pos['WiFi']) for pos in processed_data)
        }
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File not found")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Error reading map: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")


@app.post("/localize/{graph_name}")
async def localize_position(graph_name: str, scan: WiFiScan, 
                           similarity_threshold: float = 0.6,
                           blend_threshold: float = 0.4):
    """
    Localize user position based on WiFi scan against a processed map
    """
    try:
        # Check if map exists and is processed
        if graph_name not in processed_maps:
            # Try to load and process the map
            filename = graph_name + ".xml"
            current_graph = read_xml(UPLOAD_DIR, filename)
            processed_maps[graph_name] = wl.preprocess_wifi_data(current_graph)
        
        # Get stored positions
        stored_positions = processed_maps[graph_name]
        
        # Estimate position
        estimated_position = wl.estimate_position(
            scan.wifi_data,
            stored_positions,
            similarity_threshold=similarity_threshold,
            blend_threshold=blend_threshold
        )
        
        if estimated_position is None:
            return {"info": "Could not determine position with sufficient confidence"}
        
        # Create graphs for visualization
        stored_graph = g.create_wifi_graph(stored_positions)
        live_graph = wl.create_live_scan_graph(scan.wifi_data)
        
        # Plot localization result
        wl.plot_localization_result(
            stored_graph,
            live_graph,
            estimated_position,
            f"localization_{graph_name}"
        )
        
        return {
            "position": estimated_position['Position'],
            "confidence": estimated_position['Score'],
            "match_type": estimated_position['Type']
        }
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Graph '{graph_name}' not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Localization error: {e}")


@app.post("/cluster_graph/{graph_name}")
async def cluster_graph(graph_name: str,
                     apply_louvain: bool = True,
                     apply_spectral: bool = True,
                     apply_dbscan: bool = True,
                     spectral_k: int = 5):
    """
    Apply clustering algorithms to a WiFi map
    """
    try:
        # Check if map exists and is processed
        if graph_name not in processed_maps:
            # Try to load and process the map
            filename = graph_name + ".xml"
            current_graph = read_xml(UPLOAD_DIR, filename)
            processed_maps[graph_name] = wl.preprocess_wifi_data(current_graph)
        
        # Get stored positions
        stored_positions = processed_maps[graph_name]
        
        # Create graph
        graph = g.create_wifi_graph(stored_positions)
        
        # Apply clustering
        clustered_graph = wl.cluster_graph(
            graph,
            apply_louvain=apply_louvain,
            apply_spectral=apply_spectral,
            apply_dbscan=apply_dbscan,
            spectral_k=spectral_k
        )
        
        # Plot clustered graph
        g.plot_graph(clustered_graph, f"clustered_{graph_name}")
        
        # Extract clustering information
        clustering_info = {}
        
        if apply_louvain and 'community' in clustered_graph.vs.attributes():
            communities = set(clustered_graph.vs['community'])
            clustering_info['louvain'] = {
                'num_communities': len(communities),
                'community_sizes': {comm: clustered_graph.vs['community'].count(comm) for comm in communities}
            }
        
        if apply_spectral and 'spectral_cluster' in clustered_graph.vs.attributes():
            clusters = set(clustered_graph.vs['spectral_cluster'])
            clustering_info['spectral'] = {
                'num_clusters': len(clusters),
                'cluster_sizes': {cluster: clustered_graph.vs['spectral_cluster'].count(cluster) for cluster in clusters}
            }
        
        if apply_dbscan and 'dbscan_cluster' in clustered_graph.vs.attributes():
            clusters = set(clustered_graph.vs['dbscan_cluster'])
            clustering_info['dbscan'] = {
                'num_clusters': len(clusters),
                'cluster_sizes': {cluster: clustered_graph.vs['dbscan_cluster'].count(cluster) for cluster in clusters}
            }
        
        return {
            "info": f"Clustered graph '{graph_name}'",
            "clustering": clustering_info
        }
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Graph '{graph_name}' not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Clustering error: {e}")
