import os
import app.graph_manager as g
import app.wifi_localization as wl
import app.models as models
import app.database as database
from app.xml_manager import read_xml
from pydantic import BaseModel
from sqlalchemy.orm import Session
from typing import Union, List, Dict, Any
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from app.crud import save_positions_from_list
from app.schemas import PositionCreate, PositionDeleteRequest, WiFiScanPayload
from app.auth import authenticate_user, create_access_token, ACCESS_TOKEN_EXPIRE_MINUTES, get_current_admin
from fastapi.security import OAuth2PasswordRequestForm
from datetime import timedelta
from app.localize_functions import compute_best_position_basic, match_graphs_return_position, match_graphs

app = FastAPI()

# Creates all the tables
database.create_tables()

def get_db():
    db = database.SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Directory to store uploaded files
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# In-memory storage for processed WiFi data
processed_maps_data = {}
processed_maps_graphs = {}

class WiFiScan(BaseModel):
    wifi_data: List[Dict[str, Any]]

@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/token")
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=400, detail="Incorrect username or password")
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user["username"]}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}


@app.put("/upload_data/{map_name}")
async def upload_file(map_name: str, file: UploadFile = File(...),
                      db: Session = Depends(get_db), current_user: dict = Depends(get_current_admin)):
    # Log the content type for debugging
    print(f"Received file with content type: {file.content_type}")

    if file.content_type != "text/xml":
        raise HTTPException(status_code=400, detail="Invalid file type. Only XML files are allowed.")

    try:
        file_content = await file.read()  # Read the file content directly
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading file: {e}")

    # Process the XML content directly
    positions_data = read_xml(file_content)  # Assuming read_xml() can handle raw XML content
    filtered_positions = g.filter_close_points(positions_data, 3)
    

    save_positions_from_list(db, filtered_positions, map_name, file.filename)   
    
    
    processed_maps_data[map_name] = filtered_positions     # saves it to memory directly
    
       
    return {"info": f"Data stored in database for {map_name} successfully"}

@app.get("/load_map/{map_name}")
async def load_map(map_name: str, db: Session = Depends(get_db), 
                   current_user: dict = Depends(get_current_admin)):
    """
    Load data from the database for the specified map name and create a graph.
    """
    try:
        # Retrieve the map
        map_entry = db.query(models.Map).filter(models.Map.name == map_name).first()
        if not map_entry:
            raise HTTPException(status_code=404, detail=f"No map found with name '{map_name}'")

        # Retrieve associated positions
        positions = db.query(models.Position).filter(models.Position.map_id == map_entry.id).all()
        if not positions:
            raise HTTPException(status_code=404, detail=f"No data found for map '{map_name}'")

        # Create graph data structure
        graph_data = []
        for position in positions:
            graph_data.append({
                "Position": {
                    "X": position.X,
                    "Y": position.Y,
                    "Z": position.Z,
                    "timestamp": position.timestamp
                },
                "WiFi": [{ "BSSID": wifi.bssid, "SIGNAL": wifi.rssi } for wifi in position.wifi_signals],
                "Bluetooth": [{ "Address": bt.address, "RSSI": bt.rssi } for bt in position.bluetooth_signals]
            })

        processed_maps_data[map_name] = graph_data
        # Create the graph
        graph = g.create_wifi_graph(graph_data)

        # Save or update graph in processed_maps
        if processed_maps_data.get(map_name) is None:
            processed_maps_graphs[map_name] = graph
            return {
                "info": f"Graph created and stored for map '{map_name}'",
                "graph_summary": graph.summary()
            }
        else:
            # Optional: Replace the existing graph
            processed_maps_graphs[map_name] = graph
            return {
                "info": f"Graph for map '{map_name}' was updated",
                "graph_summary": graph.summary()
            }

    except HTTPException:
        raise  # Re-raise known HTTP exceptions as-is
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading graph: {e}")




@app.get("/plot_graph/{graph_name}")
async def plot_graph(graph_name: str, current_user: dict = Depends(get_current_admin)):
    try:
        
        g.plot_graph(processed_maps_graphs.get(graph_name), f"output_graphs/{graph_name}_plot.png")
        return {"info": f"Plotted graph '{graph_name}'"}, 
    except KeyError as e:
        raise HTTPException(status_code=404, detail=f"Key Error: {e}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Error reading map: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")

@app.post("/delete_map/{map_name}")
async def delete_map(map_name: str, db: Session = Depends(get_db),
                     current_user: dict = Depends(get_current_admin)):
    try:
        # Retrieve the map
        map_entry = db.query(models.Map).filter(models.Map.name == map_name).first()
        if not map_entry:
            raise HTTPException(status_code=404, detail=f"No map found with name '{map_name}'")

        # Retrieve all positions for the map
        positions = db.query(models.Position).filter(models.Position.map_id == map_entry.id).all()
        if not positions:
            raise HTTPException(status_code=404, detail=f"No data found for map '{map_name}'")

        # Delete associated WiFi and Bluetooth signals first
        for position in positions:
            db.query(models.WiFiSignal).filter(models.WiFiSignal.position_id == position.id).delete()
            db.query(models.BluetoothSignal).filter(models.BluetoothSignal.position_id == position.id).delete()

        # Delete the positions
        db.query(models.Position).filter(models.Position.map_id == map_entry.id).delete()

        # Delete the map
        db.delete(map_entry)

        db.commit()
        return {"info": f"Map '{map_name}' and all associated data have been deleted."}

    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error deleting map: {e}")

@app.post("/add_entry/{map_name}")
async def add_entry(map_name: str, entry: PositionCreate, db: Session = Depends(get_db),
                    current_user: dict = Depends(get_current_admin)):
    try:
        # Retrieve the map
        map_entry = db.query(models.Map).filter(models.Map.name == map_name).first()
        if not map_entry:
            raise HTTPException(status_code=404, detail=f"No map found with name '{map_name}'")

        # Create and store the position
        new_position = models.Position(
            X=entry.pos_x,
            Y=entry.pos_y,
            Z=entry.pos_z,
            timestamp=entry.timestamp,
            map_id=map_entry.id
        )
        db.add(new_position)
        db.flush()  # Get ID before adding related records

        # Store Wi-Fi signals
        for wifi in entry.wifi_signals:
            new_wifi = models.WiFiSignal(
                position_id=new_position.id,
                bssid=wifi.bssid,
                rssi=wifi.rssi
            )
            db.add(new_wifi)

        # Store Bluetooth signals (if any)
        for bt in entry.bluetooth_signals:
            new_bt = models.BluetoothSignal(
                position_id=new_position.id,
                address=bt.address,
                rssi=bt.rssi
            )
            db.add(new_bt)

        db.commit()
        return {"info": f"Entry successfully added to map '{map_name}'."}

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error adding entry: {e}")

@app.delete("/delete_entry/{map_name}")
async def delete_entry(map_name: str, entry: PositionDeleteRequest,
                       db: Session = Depends(get_db), current_user: dict = Depends(get_current_admin)):
    try:
        # Find the map
        map_entry = db.query(models.Map).filter(models.Map.name == map_name).first()
        if not map_entry:
            raise HTTPException(status_code=404, detail=f"No map found with name '{map_name}'")

        # Find the position
        position = db.query(models.Position).filter(
            models.Position.map_id == map_entry.id,
            models.Position.X == entry.pos_x,
            models.Position.Y == entry.pos_y,
            models.Position.Z == entry.pos_z
        ).first()

        if not position:
            raise HTTPException(status_code=404, detail="No matching position entry found.")

        # Delete related Wi-Fi signals
        db.query(models.WiFiSignal).filter(models.WiFiSignal.position_id == position.id).delete()

        # Delete related Bluetooth signals
        db.query(models.BluetoothSignal).filter(models.BluetoothSignal.position_id == position.id).delete()

        # Delete the position
        db.delete(position)
        db.commit()

        return {"info": "Entry successfully deleted."}

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error deleting entry: {e}")


@app.post("/localize_basic/{map_name}")
async def localize_basic(map_name: str, payload: WiFiScanPayload):

    print(f"Received Wi-Fi data from device {payload.device_id}")
    
    scanned_signals = payload.wifi_signals
    for signal in scanned_signals:
        print(f" SINGAL PAYLOAD - {signal.bssid} --  RSSI {signal.rssi}")
    
    positon = compute_best_position_basic(processed_maps_data[map_name], payload.wifi_signals)

    return {"estimated_position": positon, "map": map_name}

@app.post("/localize/{map_name}")
async def localize(map_name: str, payload: WiFiScanPayload):

    if processed_maps_graphs.get(map_name) is None:
        raise HTTPException(status_code=404, detail=f"No map found with name '{map_name}'")
    
    print(f"Received Wi-Fi data from device {payload.device_id}")
    
    scanned_signals = payload.wifi_signals
    for signal in scanned_signals:
        print(f" SINGAL PAYLOAD - {signal.bssid} --  RSSI {signal.rssi}")
    wifi_list = []
    for entry in scanned_signals:
        wifi_list.append({
            "BSSID": entry.bssid,
            "SIGNAL": entry.rssi
        })
    
    
    graph_dict = []
    graph_dict.append({
        "Position": {
            "X": -1,
            "Y": -1,
            "Z": -1
        },
        "WiFi": wifi_list
    })    
        
    scanned_graph = g.create_wifi_graph(graph_dict)
    #compute_rrwm_to_graphs(scanned_graph, processed_maps_graphs[map_name])
    score, node = match_graphs_return_position(scanned_graph, processed_maps_graphs[map_name])

    return {"estimated_position": node, "score": str(score), "map": map_name}



#############################################################################################################################
"""
@app.post("/process_graph/{graph_name}")
async def process_graph(graph_name: str, 
                     rssi_threshold: float = -85.0, 
                     apply_smoothing: bool = True,
                     remove_duplicates: bool = True,
                     detect_outliers: bool = True):
    """"""
    Process and filter a WiFi map for improved localization
    """"""
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
    """"""
    Localize user position based on WiFi scan against a processed map
    """"""
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
    """"""
    Apply clustering algorithms to a WiFi map
    """"""
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
"""