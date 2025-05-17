def rssi_similarity(rssi1, rssi2):
    # Smaller difference means more similar (using inverse of distance)
    return 1 / (1 + abs(rssi1 - rssi2))

def compute_similarity(scan_signals: dict, db_signals: dict):
    score = 0.0
    x, y, z = None, None, None
    
    for position in db_signals:
        for signal in scan_signals:
            for device in position.get('WiFi'):
                if signal.bssid in device.get('BSSID'):
                    score += rssi_similarity(signal.rssi, db_signals[signal.mac_address])
                    x = position.get('Position').get('X')
                    y = position.get('Position').get('Y')
                    z = position.get('Position').get('Z')
            
    return score, x, y, z

def compute_best_position_basic(wifi_signals: list, scanned_signals: list):
    best_score = -1
    best_position = None

    # similarity between the positions and wifi received and stored
    score, x, y, z = compute_similarity(scanned_signals, wifi_signals)

    if score > best_score:
        best_score = score
        best_position = {"x": x, "y": y, "z": z}

    return best_position