import os
import xml.etree.ElementTree as ET

def read_xml(filepath: str, filename: str):
    file_location = os.path.join(filepath, filename)
    
    if not os.path.exists(file_location):
        raise FileNotFoundError(f"The file {file_location} does not exist.")
    
    try:
        tree = ET.parse(file_location)
        root = tree.getroot()
    except ET.ParseError as e:
        raise ValueError(f"Error parsing XML file: {e}")
    
    # The root element is 'Positions' which contains multiple 'Position' elements
    positions = []
    
    for position_elem in root.findall('Position'):
        position_data = {
            "Position": {},
            "Orientation": {},
            "WiFi": [],
            "Bluetooth": [],
            "Timestamp": None
        }
        
        # Extract Position coordinates
        x_elem = position_elem.find('X')
        y_elem = position_elem.find('Y')
        z_elem = position_elem.find('Z')
        
        if x_elem is not None and y_elem is not None and z_elem is not None:
            position_data["Position"] = {
                "X": float(x_elem.text),
                "Y": float(y_elem.text),
                "Z": float(z_elem.text)
            }
        
        # Extract Orientation
        orientation = position_elem.find('Orientation')
        if orientation is not None:
            x_elem = orientation.find('X')
            y_elem = orientation.find('Y')
            z_elem = orientation.find('Z')
            w_elem = orientation.find('W')
            
            if x_elem is not None and y_elem is not None and z_elem is not None and w_elem is not None:
                position_data["Orientation"] = {
                    "X": float(x_elem.text),
                    "Y": float(y_elem.text),
                    "Z": float(z_elem.text),
                    "W": float(w_elem.text)
                }
        
        # Extract WiFi Networks
        wifi = position_elem.find('WiFi')
        if wifi is not None:
            for network in wifi.findall('Network'):
                ssid_elem = network.find('SSID')
                bssid_elem = network.find('BSSID')
                signal_elem = network.find('SIGNAL')
                
                if ssid_elem is not None and bssid_elem is not None and signal_elem is not None:
                    position_data["WiFi"].append({
                        "SSID": ssid_elem.text if ssid_elem.text is not None else "",
                        "BSSID": bssid_elem.text,
                        "SIGNAL": float(signal_elem.text)
                    })
        
        # Extract Bluetooth Devices
        bluetooth = position_elem.find('Bluetooth')
        if bluetooth is not None:
            for device in bluetooth.findall('Device'):
                n_elem = device.find('n')  # Using 'n' as it appears in the XML
                address_elem = device.find('Address')
                rssi_elem = device.find('RSSI')
                
                if n_elem is not None and address_elem is not None and rssi_elem is not None:
                    position_data["Bluetooth"].append({
                        "Name": n_elem.text,
                        "Address": address_elem.text,
                        "RSSI": int(rssi_elem.text)
                    })
        
        # Extract Timestamp if present
        timestamp_elem = position_elem.find('Timestamp')
        if timestamp_elem is not None and timestamp_elem.text is not None:
            position_data["Timestamp"] = int(timestamp_elem.text)
        
        positions.append(position_data)
    
    return positions
