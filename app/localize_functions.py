
def rssi_similarity(rssi1, rssi2):
    # Smaller difference means more similar (using inverse of distance)
    return 1 / (1 + abs(rssi1 - rssi2))

def compute_similarity(scan_signals, db_signals):
    score = 0.0
    db_dict = {entry[0]: entry[1] for entry in db_signals}  # mac: rssi
    for signal in scan_signals:
        if signal.mac_address in db_dict:
            score += rssi_similarity(signal.rssi, db_dict[signal.mac_address])
    return score

def compute_best_position(positions: dict, scanned_signals: dict):
    best_score = -1
    best_position = None

    for pos_id, x, y, z in positions:
        
        positions_signals = {}
        
        
        # similarity between the positions and wifi received and stored
        score = compute_similarity(scanned_signals, positions_signals)

        if score > best_score:
            best_score = score
            best_position = {"x": x, "y": y, "z": z}

    return best_position