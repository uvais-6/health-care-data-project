import flwr as fl
import numpy as np
import pickle
import hashlib
import socket
import time
from cryptography.fernet import Fernet

from model import create_model
from dataset import load_data

# ---------------- SECURITY SETUP ---------------- #

# Access Control Tokens
ALLOWED_TOKENS = ["hospitalA", "hospitalB", "hospitalC"]

def authenticate(token):
    return token in ALLOWED_TOKENS

# Encryption Setup
key = Fernet.generate_key()
cipher = Fernet(key)

def encrypt_data(data):
    return cipher.encrypt(data)

# Hashing
def hash_weights(weights):
    flat = np.concatenate([w.flatten() for w in weights])
    return hashlib.sha256(flat.tobytes()).hexdigest()

# Differential Privacy Noise
def add_noise(weights, noise_level=0.01):
    noisy = []
    for w in weights:
        noise = np.random.normal(0, noise_level, w.shape)
        noisy.append(w + noise)
    return noisy

# ---------------- CLIENT CLASS ---------------- #

class Client(fl.client.NumPyClient):
    def __init__(self, cid):
        self.cid = int(cid)
        print(f"Access Granted for Client {self.cid}")

        self.model = create_model()
        self.X_train, self.X_test, self.y_train, self.y_test = load_data(self.cid)

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        print(f"Client {self.cid} Training...")

        self.model.set_weights(parameters)
        self.model.fit(self.X_train, self.y_train, epochs=2, verbose=0)

        weights = self.model.get_weights()

        # Encryption Demo
        serialized = pickle.dumps(weights)
        encrypted = encrypt_data(serialized)
        print("Encryption Applied | Encrypted Size:", len(encrypted))

        # Hashing
        weight_hash = hash_weights(weights)
        print("Weight Hash Generated:", weight_hash[:15], "...")

        # Differential Privacy
        weights = add_noise(weights)
        print("Differential Privacy Noise Added")

        return weights, len(self.X_train), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, acc = self.model.evaluate(self.X_test, self.y_test, verbose=0)
        print(f"Client {self.cid} Accuracy: {acc}")
        return loss, len(self.X_test), {"accuracy": acc}

# ---------------- AUTH + START ---------------- #

token = "hospitalC"   # change for other clients


def wait_for_server(host: str, port: int, timeout: int = 30) -> bool:
    """Wait until a TCP server is reachable or timeout (seconds) is reached."""
    start = time.time()
    while True:
        try:
            with socket.create_connection((host, port), timeout=3):
                print(f"Server reachable at {host}:{port}")
                return True
        except OSError:
            if time.time() - start > timeout:
                print(f"Timed out waiting for server at {host}:{port}")
                return False
            print("Waiting for server...", end="\r")
            time.sleep(1)


if authenticate(token):
    host = "localhost"
    port = 9090
    if wait_for_server(host, port, timeout=30):
        try:
            fl.client.start_numpy_client(
                server_address=f"{host}:{port}",
                client=Client("2")
            )
        except Exception as e:
            print("Client failed to start:", e)
    else:
        print("Server not available. Start the server and retry.")
else:
    print("Access Denied")
