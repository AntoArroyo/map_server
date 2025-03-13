from typing import Union
from fastapi import FastAPI, File, UploadFile, HTTPException
import os
from xml_manager import read_xml
import graph_manager as g

app = FastAPI()

# Directory to store uploaded files
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

@app.put("/upload_map/")
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


@app.get("/plot_map/{map_name}")
async def plot_map(map_name: str):
    filename = map_name + ".xml"
    try:
        current_map = read_xml(UPLOAD_DIR, filename)
        graph = g.create_wifi_graph(current_map)
        g.plot_graph(graph, "outputPlot")
        return {"info": f"Plotted map '{map_name}'"}, 
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File not found")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Error reading map: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")
    
