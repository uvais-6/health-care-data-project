import flwr as fl

strategy = fl.server.strategy.FedAvg()

fl.server.start_server(
    server_address="localhost:9090",
    config=fl.server.ServerConfig(num_rounds=3),
    strategy=strategy
)
