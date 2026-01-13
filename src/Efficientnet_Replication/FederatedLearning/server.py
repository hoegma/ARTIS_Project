import flwr as fl

def weighted_metrics(metrics):
    acc = sum(m["accuracy"] for _, m in metrics) / len(metrics)
    auc = sum(m["auc"] for _, m in metrics) / len(metrics)
    return {"accuracy": acc, "auc": auc}

strategy = fl.server.strategy.FedAvg(
    fraction_fit=0.6,
    fraction_evaluate=0.6,
    min_fit_clients=2,
    min_evaluate_clients=2,
    min_available_clients=5,
    evaluate_metrics_aggregation_fn=weighted_metrics
)
