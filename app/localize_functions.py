def rssi_similarity(rssi1, rssi2):
    # Smaller difference means more similar (using inverse of distance)
    if rssi1 is not None and rssi2 is not None:
        return 1 / (1 + abs(rssi1 - rssi2))
    else: 
        return 0

def compute_similarity(db_signals: dict, scan_signals: dict):
    score = 0.0
    x, y, z = None, None, None
    
    for position in db_signals:
        for signal in scan_signals:
            for device in position.get('WiFi'):
                if signal.bssid == device.get('BSSID'):
                    print(f" RSSI SCANNED {signal.bssid} RSSI STORED {device.get('BSSID')}")
                    score += rssi_similarity(signal.rssi, device.get('RSSI'))
                    x = position.get('Position').get('X')
                    y = position.get('Position').get('Y')
                    z = position.get('Position').get('Z')
            
    return score, x, y, z

def compute_best_position_basic(wifi_signals: list, scanned_signals: list):
    best_score = -1
    best_position = None

    # similarity between the positions and wifi received and stored
    score, x, y, z = compute_similarity(wifi_signals, scanned_signals )

    if score > best_score:
        best_score = score
        best_position = {"x": x, "y": y, "z": z}

    return best_position