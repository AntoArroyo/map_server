from pydantic import ConfigDict
from sqlalchemy import Column, Integer, Float, String, ForeignKey, DateTime
from sqlalchemy.orm import relationship
from datetime import datetime
from .database import Base

model_config = ConfigDict(from_attributes=True)

class Position(Base):
    __tablename__ = "POSITIONS"

    id = Column(Integer, primary_key=True, index=True)
    X = Column(Float)
    Y = Column(Float)
    Z = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)
    map_id = Column(Integer, ForeignKey("MAPS.id"), nullable=False)

    wifi_signals = relationship("WiFiSignal", back_populates="position", cascade="all, delete")
    bluetooth_signals = relationship("BluetoothSignal", back_populates="position", cascade="all, delete")

class Map(Base):
    __tablename__ = "MAPS"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, nullable=False, index=True)
    filename = Column(String, unique=True, nullable=False, index=True)


class WiFiSignal(Base):
    __tablename__ = "WIFI_SIGNALS"

    id = Column(Integer, primary_key=True, index=True)
    bssid = Column(String, index=True)
    rssi = Column(Float)
    position_id = Column(Integer, ForeignKey("POSITIONS.id"))
    map_id = Column(Integer, ForeignKey("MAPS.id"))
    map_name = Column(String, ForeignKey("MAPS.name"))
    position = relationship("Position", back_populates="wifi_signals")

class BluetoothSignal(Base):
    __tablename__ = "BLUETOOTH_SIGNALS"

    id = Column(Integer, primary_key=True, index=True)
    address = Column(String, index=True)
    rssi = Column(Float)
    position_id = Column(Integer, ForeignKey("POSITIONS.id"))

    position = relationship("Position", back_populates="bluetooth_signals")
