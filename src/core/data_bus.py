import json
import socket
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        return super().default(obj)


class DataBus:
    def __init__(self, host='127.0.0.1', port=5005):
        self.host = host
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def broadcast(self, data_dict):
        try:
            message = json.dumps(data_dict, cls=NumpyEncoder).encode('utf-8')
            self.sock.sendto(message, (self.host, self.port))
        except Exception:
            pass

    def close(self):
        self.sock.close()