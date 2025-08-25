import os
import app.graph_manager as g
import app.models as models
import app.database as database
from app.xml_manager import read_xml
from pydantic import BaseModel
from sqlalchemy.orm import Session, joinedload
from typing import List, Dict, Any
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from app.crud import save_positions_from_list
from app.schemas import PositionCreate, PositionDeleteRequest, WiFiScanPayload
from app.auth import authenticate_user, create_access_token, ACCESS_TOKEN_EXPIRE_MINUTES, get_current_admin
from fastapi.security import OAuth2PasswordRequestForm
from datetime import timedelta
from app.localize_functions import compute_best_position_basic, get_estimated_position
from cachetools import TTLCache

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

# In-memory storage for processed WiFi data, with limit 10 maps and 30 min
processed_maps_data = TTLCache(maxsize=10, ttl=1800)  # used for comute basic withou graphs
processed_maps_graphs = TTLCache(maxsize=10, ttl=1800)

class WiFiScan(BaseModel):
    wifi_data: List[Dict[str, Any]]

@app.get("/")
def read_root():
    return {"Server Status" : "Running!"}


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
    """
    Upload_data endpoint, gets a map name as parameters and the file containing the positions, wifi and bluetooth signals
    process the data and stores it in the database

    Args:
        map_name (str): Map name.
        file (UploadFile): File containing position, wifi and bluetooth data.
        db (Session, optional): database
        current_user (dict, optional): check if the current user is admin.

    Raises:
        HTTPException: 400 Invalid file type. Only XML files are allowed.
        HTTPException: 400 Invalid name, already exists a map with that name.
        HTTPException: 500 Error reading file.

    Returns:
        (dict): info: Data store in database {map_name} successfully
    """
    # Log the content type for debugging
    print(f"Received file with content type: {file.content_type}")

    if file.content_type != "text/xml":
        raise HTTPException(status_code=400, detail="Invalid file type. Only XML files are allowed.")
    
    db_map = db.query(models.Map).filter(models.Map.name == map_name).first()
    if db_map:
        raise HTTPException(status_code=400, detail="Invalid name, already exists a map with that name.")
    try:
        file_content = await file.read()  # Read the file content directly
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading file: {e}")

    # Process the XML content directly
    positions_data = read_xml(file_content) 
    filtered_positions = g.filter_close_points(positions_data, 2)
    
    

    save_positions_from_list(db, filtered_positions, map_name, file.filename)   
        
       
    return {"info": f"Data stored in database for {map_name} successfully"}

@app.get("/load_map/{map_name}")
async def load_map(map_name: str, db: Session = Depends(get_db), 
                   current_user: dict = Depends(get_current_admin)):
    """
    Load data from the database for the specified map name and create a graph.

    Raises:
        HTTPException: 404 No map found with name {map_name}
        HTTPException: 404 No positions data found for map '{map_name}
        HTTPException: 500 Error loading graph

    Returns:
        (dict): Graph summary
    """
    try:
        db_map = db.query(models.Map).filter(models.Map.name == map_name).first()
        if not db_map:
            raise HTTPException(status_code=404, detail=f"No map found with name '{map_name}'")

        if processed_maps_graphs.get(map_name):
            return {
                "info" : f"Map {map_name} already loaded.",
                "graph_summary": processed_maps_graphs[map_name].summary()
                    }
        
        positions = db.query(models.Position).options(
            joinedload(models.Position.wifi_signals),
            joinedload(models.Position.bluetooth_signals)
        ).filter(models.Position.map_id == db_map.id).all()
        if not positions:
            raise HTTPException(status_code=404, detail=f"No positions data found for map '{map_name}'")

        graph_data = [
            {
                "Position": {
                    "X": p.X, "Y": p.Y, "Z": p.Z, "timestamp": p.timestamp
                },
                "WiFi": [{ "BSSID": wifi.bssid, "SIGNAL": wifi.rssi } for wifi in p.wifi_signals],
                "Bluetooth": [{ "Address": bt.address, "RSSI": bt.rssi } for bt in p.bluetooth_signals]
            }
            for p in positions
        ]

        processed_maps_data[map_name] = graph_data
        graph = g.create_wifi_graph(graph_data)
        
        processed_maps_graphs[map_name] = graph

        return {
            "info": f"Graph for map '{map_name}' was loaded",
            "graph_summary": graph.summary()
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading graph: {e}")


@app.get("/plot_graph/{graph_name}")
async def plot_graph(graph_name: str, current_user: dict = Depends(get_current_admin)):
    """
    Plots graph and creates the png with the image to the current path output_graphs/
    Args:
        graph_name (str): graph name.
        current_user (dict, optional): Checks if the user is admin.

    Raises:
        HTTPException: 500 Key Error
        HTTPException: 500 Error reading map
        HTTPException: 500 Unexpected error

    Returns:
        dict: Plotted graph {graph_name}
    """
    try:
        
        g.plot_graph(processed_maps_graphs.get(graph_name), f"output_graphs/{graph_name}_plot.png")
        return {"info": f"Plotted graph '{graph_name}'"}, 
    except KeyError as e:
        raise HTTPException(status_code=500, detail=f"Key Error: {e}")
    except ValueError as e:
        raise HTTPException(status_code=500, detail=f"Error reading map: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")

@app.post("/delete_map/{map_name}")
async def delete_map(map_name: str, db: Session = Depends(get_db),
                     current_user: dict = Depends(get_current_admin)):
    """
    Deletes map from the database if it is stored

    Args:
        map_name (str): Map name to delete
        db (Session, optional): database
        current_user (dict, optional): Check if the user is admin.

    Raises:
        HTTPException: 404 No map found with name '{map_name}'
        HTTPException: 404 No data found for map '{map_name}'
        HTTPException: 500 Error deleting map

    Returns:
        dict: Map '{map_name}' and all associated data have been deleted.
    """
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
    """
    Adds Position entry to the database to the current map associated to the name

    Args:
        map_name (str): Map name
        entry (PositionCreate): Entry to add
        db (Session, optional): Database. Defaults to Depends(get_db).
        current_user (dict, optional): Check if the user is admin. Defaults to Depends(get_current_admin).

    Raises:
        HTTPException: 400 No map found with name '{map_name}'
        HTTPException: 500 Error adding entry

    Returns:
        dict: Entry successfully added to map '{map_name}'
    """
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
    """
    Deletes an entry from the position table in the database
    Args:
        map_name (str): Map name
        entry (PositionDeleteRequest): Entry to delete.
        db (Session, optional): Databasse. Defaults to Depends(get_db).
        current_user (dict, optional): Check if the current user is admin. Defaults to Depends(get_current_admin).

    Raises:
        HTTPException: 404 No map found with name '{map_name}'
        HTTPException: 404 No matching position entry found.
        HTTPException: 500 Error deleting entry

    Returns:
        dict: {info : Entry successfully deleted.}
    """
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
    """
    Basic localization based on BSSID comparisson without using graphs
    Args:
        map_name (str): Map name
        payload (WiFiScanPayload): Wifi payload scanned

    Returns:
        dict: {estiamted_positon : postion, map : map_name}
    """
    print(f"Received Wi-Fi data from device {payload.device_id}")
    
    scanned_signals = payload.wifi_signals
    for signal in scanned_signals:
        print(f" SINGAL PAYLOAD - {signal.bssid} --  RSSI {signal.rssi}")
    
    positon = compute_best_position_basic(processed_maps_data[map_name], payload.wifi_signals)

    return {"estimated_position": positon, "map": map_name}
 
@app.post("/localize_basic_graph/{map_name}")
async def localize_basic_graph(map_name: str, payload: WiFiScanPayload):
    """
    Localize endpoint using graphs, with the position and wifi APs as nodes and the RSSI values as weighted edges

    Args:
        map_name (str): Map name
        payload (WiFiScanPayload): Wifi payload scanned

    Raises:
        HTTPException: 404 No map found with name '{map_name}'

    Returns:
        Best node containg the best postion match and Calculated Node containing the aproximatly position
        dict: {Best Node: (x,y, z), Calculated Node: (x, y)}
    """
    if processed_maps_graphs.get(map_name) is None:
        raise HTTPException(status_code=404, detail=f"No map found with name '{map_name}'")
    
    print(f"Received Wi-Fi data from device {payload.device_id}")
    
    wifi_list = []
    for wifi_values in payload.wifi_signals:
        wifi_list.append((wifi_values.bssid, wifi_values.rssi))
    
    node, node_calculated = get_estimated_position(wifi_list, processed_maps_graphs[map_name])  
    
    if not node and not node_calculated:
        return {"No match found, try to move again"}
    
    msg_dict = {}
    if node:
        msg_dict["Best Node"] = node
    if node_calculated:        
        msg_dict["Calculated Node"] = node_calculated
    
    return msg_dict

