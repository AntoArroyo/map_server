from pydantic import BaseModel, ConfigDict
from typing import List, Optional
import datetime

class WiFiSignalBase(BaseModel):
    bssid: str  
    rssi: float

class BluetoothSignalBase(BaseModel):
    address: str
    rssi: float

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
    sm_id: Optional[str]
    sm_name: Optional[str]

class PositionDeleteRequest(BaseModel):
    pos_x: float
    pos_y: float
    pos_z: float

class Position(PositionCreate):
    id: int
    timestamp: Optional[str]

class SuperMap(BaseModel):
    id: int
    name: str

model_config = ConfigDict(from_attributes=True)
# Enable ORM mode for SQLAlchemy compatibility


class User(BaseModel):
    username: str
    disable: bool
    
class WiFiScanPayload(BaseModel):
    device_id: str
    wifi_signals: List[WiFiSignalBase]
    
class LocalizationRequest(BaseModel):
    map_name: str
    wifi_signals: List[WiFiSignalBase]