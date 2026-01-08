import os
import sys
import time
import gc
import pickle
import numpy as np
import tensorflow as tf
import flwr as fl

from cryptography.fernet import Fernet
from typing import List, Tuple, Dict
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras import layers, models
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays

# -------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
LOCAL_EPOCHS = 3
FEDERATED_ROUNDS = 10
NUM_CLIENTS = 5
SERVER_ADDRESS = "localhost:8000"

BASE_PATH = "./real-vs-fake"
TRAIN_PATH = f"{BASE_PATH}/train"
VAL_PATH = f"{BASE_PATH}/valid"
TEST_PATH = f"{BASE_PATH}/test"

AUTOTUNE = tf.data.AUTOTUNE

# -------------------------------------------------------------
# 🔐 ENCRYPTION SETUP (STATIC SHARED KEY)
# -------------------------------------------------------------
# Generate once with Fernet.generate_key() and reuse
FERNET_KEY = b"REPLACE_WITH_YOUR_GENERATED_KEY_HERE"
cipher = Fernet(FERNET_KEY)

def encrypt_weights(weights: List[np.ndarray]) -> bytes:
    serialized = pickle.dumps(weights)
    return cipher.encrypt(serialized)

def decrypt_weights(encrypted: bytes) -> List[np.ndarray]:
    decrypted = cipher.decrypt(encrypted)
    return pickle.loads(decrypted)

# -------------------------------------------------------------
# GPU SETUP
# -------------------------------------------------------------
def setup_gpu():
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✓ GPU enabled ({len(gpus)} GPU)")
    else:
        print("⚠ No GPU detected")

# -------------------------------------------------------------
# MODEL
# -------------------------------------------------------------
def create_model():
    base_model = EfficientNetB0(
        weights="imagenet",
        include_top=False,
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
    )
    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(2, activation="softmax", dtype='float32')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
    )
    return model

# -------------------------------------------------------------
# DATA
# -------------------------------------------------------------
def load_datasets():
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        TRAIN_PATH, image_size=IMG_SIZE, batch_size=BATCH_SIZE,
        label_mode="categorical", shuffle=True, seed=42
    )
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        VAL_PATH, image_size=IMG_SIZE, batch_size=BATCH_SIZE,
        label_mode="categorical"
    )
    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        TEST_PATH, image_size=IMG_SIZE, batch_size=BATCH_SIZE,
        shuffle=False, label_mode="categorical"
    )

    preprocess = lambda x, y: (preprocess_input(x), y)

    return (
        train_ds.map(preprocess).prefetch(AUTOTUNE),
        val_ds.map(preprocess).prefetch(AUTOTUNE),
        test_ds.map(preprocess).prefetch(AUTOTUNE)
    )

# -------------------------------------------------------------
# 🔐 FLOWER CLIENT
# -------------------------------------------------------------
class FakeImageClient(fl.client.NumPyClient):
    def __init__(self, cid, train_ds, num_samples):
        self.cid = cid
        self.train_ds = train_ds
        self.num_samples = num_samples
        self.model = create_model()

    def get_parameters(self, config):
        weights = self.model.get_weights()
        encrypted = encrypt_weights(weights)
        return ndarrays_to_parameters([np.frombuffer(encrypted, dtype=np.uint8)])

    def fit(self, parameters, config):
        server_round = config.get("server_round", "?")
        print(f"\n[Client {self.cid}] Round {server_round} training")

        encrypted_blob = parameters_to_ndarrays(parameters)[0].tobytes()
        weights = decrypt_weights(encrypted_blob)
        self.model.set_weights(weights)

        history = self.model.fit(self.train_ds, epochs=LOCAL_EPOCHS, verbose=1)

        updated_weights = self.model.get_weights()
        encrypted_update = encrypt_weights(updated_weights)

        return (
            ndarrays_to_parameters([np.frombuffer(encrypted_update, dtype=np.uint8)]),
            self.num_samples,
            {"loss": history.history["loss"][-1]}
        )

    def evaluate(self, parameters, config):
        return 0.0, self.num_samples, {}

# -------------------------------------------------------------
# 🔐 SERVER STRATEGY
# -------------------------------------------------------------
class EncryptedFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, val_ds, **kwargs):
        super().__init__(**kwargs)
        self.val_ds = val_ds
        self.global_model = create_model()

    def aggregate_fit(self, server_round, results, failures):
        decrypted_results = []

        for client_proxy, fit_res in results:
            encrypted_blob = parameters_to_ndarrays(fit_res.parameters)[0].tobytes()
            weights = decrypt_weights(encrypted_blob)
            decrypted_results.append(
                (client_proxy, fl.common.FitRes(
                    parameters=ndarrays_to_parameters(weights),
                    num_examples=fit_res.num_examples,
                    metrics=fit_res.metrics
                ))
            )

        aggregated = super().aggregate_fit(server_round, decrypted_results, failures)
        if aggregated[0] is None:
            return aggregated

        weights = parameters_to_ndarrays(aggregated[0])
        self.global_model.set_weights(weights)

        loss, acc, auc = self.global_model.evaluate(self.val_ds, verbose=0)
        print(f"[Server] Round {server_round} | Val Acc: {acc:.4f} | AUC: {auc:.4f}")

        encrypted = encrypt_weights(weights)
        encrypted_params = ndarrays_to_parameters(
            [np.frombuffer(encrypted, dtype=np.uint8)]
        )

        return encrypted_params, aggregated[1], aggregated[2]

# -------------------------------------------------------------
# SERVER
# -------------------------------------------------------------
def start_server():
    setup_gpu()
    _, val_ds, _ = load_datasets()

    def on_fit_config(server_round):
        return {"server_round": server_round}

    strategy = EncryptedFedAvg(
        val_ds=val_ds,
        fraction_fit=1.0,
        min_fit_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
        on_fit_config_fn=on_fit_config,
        initial_parameters=ndarrays_to_parameters(
            [np.frombuffer(encrypt_weights(create_model().get_weights()), dtype=np.uint8)]
        )
    )

    fl.server.start_server(
        server_address=SERVER_ADDRESS,
        config=fl.server.ServerConfig(num_rounds=FEDERATED_ROUNDS),
        strategy=strategy
    )

# -------------------------------------------------------------
# CLIENT
# -------------------------------------------------------------
def start_client(cid: int):
    setup_gpu()
    train_ds, _, _ = load_datasets()
    client_ds = train_ds.shard(NUM_CLIENTS, cid)
    num_samples = sum(1 for _ in client_ds) * BATCH_SIZE

    client = FakeImageClient(cid, client_ds, num_samples)
    fl.client.start_numpy_client(SERVER_ADDRESS, client)

# -------------------------------------------------------------
# MAIN
# -------------------------------------------------------------
if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "server":
        start_server()
    elif mode == "client":
        start_client(int(sys.argv[2]))
