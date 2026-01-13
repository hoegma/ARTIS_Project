import os
import sys
import time
import pickle
import numpy as np
import tensorflow as tf
import flwr as fl

from cryptography.fernet import Fernet
from typing import List
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
# 🔐 ENCRYPTION (runtime injected key)
# -------------------------------------------------------------
cipher: Fernet | None = None

def set_cipher(key: bytes):
    global cipher
    cipher = Fernet(key)

def encrypt_weights(weights: List[np.ndarray]) -> bytes:
    assert cipher is not None, "Cipher not initialized"
    return cipher.encrypt(pickle.dumps(weights))

def decrypt_weights(blob: bytes) -> List[np.ndarray]:
    assert cipher is not None, "Cipher not initialized"
    return pickle.loads(cipher.decrypt(blob))

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
        input_shape=(224, 224, 3)
    )
    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(2, activation="softmax", dtype="float32")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
    )
    return model

# -------------------------------------------------------------
# DATA
# -------------------------------------------------------------
def load_datasets():
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        TRAIN_PATH, image_size=IMG_SIZE,
        batch_size=BATCH_SIZE, label_mode="categorical",
        shuffle=True, seed=42
    )
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        VAL_PATH, image_size=IMG_SIZE,
        batch_size=BATCH_SIZE, label_mode="categorical"
    )

    preprocess = lambda x, y: (preprocess_input(x), y)
    return (
        train_ds.map(preprocess).prefetch(AUTOTUNE),
        val_ds.map(preprocess).prefetch(AUTOTUNE)
    )

# -------------------------------------------------------------
# CLIENT
# -------------------------------------------------------------
class FakeImageClient(fl.client.NumPyClient):
    def __init__(self, cid, train_ds, num_samples):
        self.cid = cid
        self.train_ds = train_ds
        self.num_samples = num_samples
        self.model = create_model()

    def fit(self, parameters, config):
        global cipher

        # 🔑 Receive encryption key once
        if cipher is None:
            key = config["fernet_key"].encode()
            set_cipher(key)
            print(f"[Client {self.cid}] Encryption key received")

            # First round → unencrypted weights (parameters is already a list of arrays)
            self.model.set_weights(parameters)
        else:
            # Subsequent rounds → decrypt weights
            encrypted = parameters_to_ndarrays(parameters)[0].tobytes()
            self.model.set_weights(decrypt_weights(encrypted))

        # Train the model
        self.model.fit(self.train_ds, epochs=LOCAL_EPOCHS, verbose=1)

        # Get updated weights
        updated = self.model.get_weights()

        # Return format: ALWAYS return (NDArrays, int, Dict)
        if cipher is None:
            # First round: return plain weights
            return updated, self.num_samples, {}
        else:
            # Subsequent rounds: return encrypted weights wrapped as single array
            encrypted = encrypt_weights(updated)
            encrypted_array = np.frombuffer(encrypted, dtype=np.uint8)
            # Return as list containing single array (still NDArrays type)
            return [encrypted_array], self.num_samples, {}

# -------------------------------------------------------------
# SERVER STRATEGY
# -------------------------------------------------------------
class EncryptedFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, val_ds, fernet_key: bytes, **kwargs):
        super().__init__(**kwargs)
        self.val_ds = val_ds
        self.fernet_key = fernet_key
        self.global_model = create_model()
        set_cipher(fernet_key)

    def aggregate_fit(self, rnd, results, failures):
        decrypted_results = []

        for client, fit_res in results:
            if rnd == 1:
                decrypted_results.append((client, fit_res))
            else:
                blob = parameters_to_ndarrays(fit_res.parameters)[0].tobytes()
                weights = decrypt_weights(blob)
                decrypted_results.append(
                    (client, fl.common.FitRes(
                        parameters=ndarrays_to_parameters(weights),
                        num_examples=fit_res.num_examples,
                        metrics=fit_res.metrics
                    ))
                )

        aggregated = super().aggregate_fit(rnd, decrypted_results, failures)
        if aggregated[0] is None:
            return aggregated

        self.global_model.set_weights(parameters_to_ndarrays(aggregated[0]))
        loss, acc, auc = self.global_model.evaluate(self.val_ds, verbose=0)
        print(f"[Server] Round {rnd} | Val Acc: {acc:.4f} | AUC: {auc:.4f}")

        if rnd == 1:
            return aggregated

        encrypted = encrypt_weights(parameters_to_ndarrays(aggregated[0]))
        return (
            ndarrays_to_parameters([np.frombuffer(encrypted, dtype=np.uint8)]),
            aggregated[1],
            aggregated[2]
        )

# -------------------------------------------------------------
# SERVER
# -------------------------------------------------------------
def start_server():
    setup_gpu()
    _, val_ds = load_datasets()

    # 🔑 Generate key ONCE
    fernet_key = Fernet.generate_key()
    print("🔐 Server generated encryption key")

    def on_fit_config(server_round):
        return {"fernet_key": fernet_key.decode()}

    strategy = EncryptedFedAvg(
        val_ds=val_ds,
        fernet_key=fernet_key,
        fraction_fit=1.0,
        min_fit_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
        on_fit_config_fn=on_fit_config,
        initial_parameters=ndarrays_to_parameters(create_model().get_weights())
    )

    fl.server.start_server(
        server_address=SERVER_ADDRESS,
        config=fl.server.ServerConfig(num_rounds=FEDERATED_ROUNDS),
        strategy=strategy
    )


# -------------------------------------------------------------
# CLIENT
# -------------------------------------------------------------
def start_client(cid):
    setup_gpu()
    train_ds, _ = load_datasets()
    client_ds = train_ds.shard(NUM_CLIENTS, cid)
    num_samples = sum(1 for _ in client_ds) * BATCH_SIZE

    fl.client.start_numpy_client(
        server_address=SERVER_ADDRESS,
        client=FakeImageClient(cid, client_ds, num_samples)
    )


# -------------------------------------------------------------
# MAIN
# -------------------------------------------------------------
if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "server":
        start_server()
    elif mode == "client":
        start_client(int(sys.argv[2]))
