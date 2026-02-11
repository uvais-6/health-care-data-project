import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

def load_data(client_id=0, total_clients=3):
    data = pd.read_csv("heart.csv")

    X = data.drop("target", axis=1)
    y = data["target"]

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split dataset into clients
    split_size = len(X) // total_clients
    start = client_id * split_size
    end = start + split_size

    X_client = X[start:end]
    y_client = y[start:end]

    return train_test_split(X_client, y_client, test_size=0.2)
