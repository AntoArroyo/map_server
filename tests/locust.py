from locust import HttpUser, task, between

class FastAPIUser(HttpUser):
    wait_time = between(1, 5)

    @task
    def get_endpoint(self):
        """
        Localize endpoint stress test
        """
        json_wifis =  {
                        "device_id": "mydevice",
                        "wifi_signals": [

                        { "bssid": "24:81:3b:2b:bb:e4", "rssi": -60 },
                        { "bssid": "24:81:3b:2b:bb:e0", "rssi": -60 },
                        { "bssid": "24:81:3b:2b:bb:e2", "rssi": -60 },
                        { "bssid": "24:81:3b:2b:bb:e3", "rssi": -60 },
                        { "bssid": "24:e1:24:f3:15:9a", "rssi": -63 },
                        { "bssid": "72:c7:14:31:82:d0", "rssi": -67 },
                        { "bssid": "24:81:3b:51:09:c0", "rssi": -67 },
                        { "bssid": "24:81:3b:51:09:c2", "rssi": -67 },
                        { "bssid": "24:81:3b:51:09:c3", "rssi": -67 }
                    ]
}
        self.client.post("/localize/LAB", json=json_wifis)
