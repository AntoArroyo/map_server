from pydantic import BaseModel
from typing import List, Optional

class WiFiSignalBase(BaseModel):
    bssid: str  
    rssi: float

class BluetoothSignalBase(BaseModel):
    address: str
    rssi: float

class PositionBase(BaseModel):
    pos_x: float
    pos_y: float
    pos_z: float
    map_id: int
    wifi_signals: List[WiFiSignalBase]
    bluetooth_signals: List[BluetoothSignalBase]

class MapBase(BaseModel):
    name: str

class PositionCreate(BaseModel):
    pos_x: float
    pos_y: float
    pos_z: float
    map_name: str  
    wifi_signals: List[WiFiSignalBase] = []
    bluetooth_signals: List[BluetoothSignalBase] = []
    timestamp: Optional[str] = None  

class PositionDeleteRequest(BaseModel):
    pos_x: float
    pos_y: float
    pos_z: float

class Position(PositionBase):
    id: int
    timestamp: Optional[str]

    class Config:
        orm_mode = True  # Enable ORM mode for SQLAlchemy compatibility
