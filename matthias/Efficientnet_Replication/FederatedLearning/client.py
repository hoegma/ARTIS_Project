import flwr as fl
from model import create_model

class FLClient(fl.client.NumPyClient):
    def __init__(self, train_ds, val_ds):
        self.model = create_model()
        self.train_ds = train_ds
        self.val_ds = val_ds

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        self.model.fit(self.train_ds, epochs=3, verbose=0)
        return self.model.get_weights(), 1, {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, acc, auc = self.model.evaluate(self.val_ds, verbose=0)
        return loss, 1, {"accuracy": acc, "auc": auc}
