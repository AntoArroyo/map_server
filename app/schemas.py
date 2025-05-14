from pydantic import BaseModel, ConfigDict
from typing import List, Optional
import datetime

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


model_config = ConfigDict(from_attributes=True)
# Enable ORM mode for SQLAlchemy compatibility


class User(BaseModel):
    username: str
    email :str
    disable: bool
    
class WiFiScanPayload(BaseModel):
    device_id: str
    wifi_signals: List[WiFiSignalBase]
    
class LocalizationRequest(BaseModel):
    map_name: str
    wifi_signals: List[WiFiSignalBase]