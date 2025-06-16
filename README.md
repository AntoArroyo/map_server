# 📡 Wi-Fi Localization & Mapping API

This project provides a FastAPI-based backend to collect, manage, and use Wi-Fi and Bluetooth RSSI data for indoor positioning. Devices can upload scan data in XML format, and the server builds signal graphs to estimate user location.

---

## 🚀 Features

- 📦 Upload and parse XML files with Wi-Fi and Bluetooth data
- 🧠 Filter and save positions to a PostgreSQL database
- 🗺️ Load maps and build signal graphs
- 🎯 Localize device positions based on Wi-Fi scans
- 🧹 Add or delete specific entries from a map
- 📊 Plot graphs of signal data
- 🔐 Admin authentication using OAuth2 + JWT tokens

---

## 📁 Project Structure

```
.
├── app/
│   ├── auth.py               # Authentication with JWT
│   ├── crud.py               # CRUD operations
│   ├── database.py           # DB session and setup
│   ├── graph_manager.py      # Wi-Fi graph creation and plotting
│   ├── localize_functions.py # Position estimation logic
│   ├── models.py             # SQLAlchemy models
│   ├── schemas.py            # Pydantic schemas
│   ├── wifi_localization.py  # Localization helpers
│   ├── xml_manager.py        # XML file parsing
├── main.py                   # FastAPI app entry point
├── requirements.txt          # Project dependencies
├── Dockerfile                # (optional) Container setup
```

---

## ⚙️ Installation

1. **Clone the repository**

```bash
git clone https://https://github.com/AntoArroyo/map_server
cd map_server
```

2. **Create a virtual environment**

```bash
python3 -m venv venv
source venv/bin/activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Run the FastAPI server**

```bash
uvicorn main:app --reload
```

---

## 🐳 Docker 

To run the project with docker use docker compose:

```bash
docker compose up --build
```

---

## 🦗 Locust 

To stress test the server, user the tool locust with the file provided for the endpoint "localize" or the one you want to test 

```bash
locust -f tests/locust.py --host http://localhost:8000
```

---

## 🔐 Authentication

The app uses OAuth2 with JWT:

- `POST /token`: Get a token by sending valid username and password.
- Use the returned token as a `Bearer` token in all other endpoints.

Example:

```bash
curl -X POST "http://localhost:8000/token" -d "username=admin&password=admin"
```

---

## 📤 Upload Map Data

```http
PUT /upload_data/{map_name}
```

- Upload an XML file with Wi-Fi/Bluetooth data
- Requires `Authorization: Bearer <token>`

---

## 🗺️ Load Map and Create Graph

```http
GET /load_map/{map_name}
```

- Loads data from the DB and generates an in-memory graph

---

## 🧭 Localize Device

```http
POST /localize/{map_name}
```

- Provide live Wi-Fi scan (`device_id`, list of `bssid` + `rssi`)
- Returns estimated position

Example payload:

```json
{
  "device_id": "mydevice",
  "wifi_signals": [
    { "bssid": "00:11:22:33:44:55", "rssi": -45 },
    { "bssid": "66:77:88:99:AA:BB", "rssi": -60 }
  ]
}
```

---

## ➕ Add Entry

```http
POST /add_entry/{map_name}
```

- Adds a manual position with Wi-Fi and Bluetooth data

---

## 🗑️ Delete Entry

```http
DELETE /delete_entry/{map_name}
```

- Deletes a specific position from the map

---

## 🔄 Delete Map

```http
POST /delete_map/{map_name}
```

- Removes an entire map and all its entries from the DB

---

## 📊 Plot Graph

```http
GET /plot_graph/{graph_name}
```

- Saves a plot of the loaded graph to an image (e.g., `outputPlot.png`)


