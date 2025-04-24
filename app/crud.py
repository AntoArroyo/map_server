from sqlalchemy.orm import Session
from . import models, schemas


def create_map(db: Session, map_name: str, filename: str):
    db_map = models.Map(name=map_name, filename=filename)
    db.add(db_map)
    db.commit()
    db.refresh(db_map)
    
    
    
    return db_map
    



def create_position(db: Session, point: dict, map_id: int):
    db_position = models.Position(
        X=point["Position"]["X"],
        Y=point["Position"]["Y"],
        Z=point["Position"]["Z"],
        map_id=map_id
    )
    db.add(db_position)
    db.commit()
    db.refresh(db_position)

    for wifi in point["WiFi"]:
        db.add(models.WiFiSignal(bssid=wifi["BSSID"], rssi=wifi["SIGNAL"], position_id=db_position.id))

    for bt in point["Bluetooth"]:
        db.add(models.BluetoothSignal(address=bt["Address"], rssi=bt["RSSI"], position_id=db_position.id))

    db.commit()
    return db_position




def get_all_positions(db: Session, map_id: int):
    return db.query(models.Position).filter(models.Position.map_id == map_id).all()

def create_wifi_signal(db: Session, wifi: schemas.WiFiSignalBase, position_id: int):
    db_wifi = models.WiFiSignal(
        bssid=wifi["BSSID"],
        rssi=wifi["SIGNAL"],
        position_id=position_id
    )
    
    db.add(db_wifi)
    db.commit()
    db.refresh(db_wifi)
    return db_wifi

def create_bt_signal(db: Session, bt: schemas.BluetoothSignalBase, position_id: int):
    db_bt = models.BluetoothSignal(
        address=bt["Address"],
        rssi=bt["RSSI"],
        position_id=position_id
    )
    
    db.add(db_bt)
    db.commit()
    db.refresh(db_bt)
    return db_bt

def save_positions_from_list(db: Session, positions_data: list, map_name: str, filename: str):
    map_data = create_map(db, map_name, filename)
    for pos_dict in positions_data:
        position = create_position(db, pos_dict, map_data.id)
        for wifi in pos_dict.get("WiFi"):
            create_wifi_signal(db, wifi, position.id)
        for bt in pos_dict.get("Bluetooth"):
            create_bt_signal(db, bt, position.id)