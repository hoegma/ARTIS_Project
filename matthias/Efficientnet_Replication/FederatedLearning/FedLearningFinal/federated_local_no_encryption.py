# # chmod +x run_federated.sh
# # ./run_federated.sh

import os
import sys
import time
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras import layers, models
import flwr as fl
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from typing import List, Tuple, Dict
import gc

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
# GPU Setup (CRITICAL FOR MULTI-PROCESS)
# -------------------------------------------------------------
def setup_gpu():
    """Configures GPU to allow multiple processes to share memory"""
    # Force TensorFlow to only see the GPU if it's actually there
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                # This prevents one client from taking 100% of VRAM
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✓ GPU enabled: {len(gpus)} GPU(s)")
        except RuntimeError as e:
            print(f"GPU error: {e}")
    else:
        print("⚠ No GPU detected by TensorFlow. Check CUDA/cuDNN installation.")

# -------------------------------------------------------------
# Create Model & Datasets (Same logic as yours)
# -------------------------------------------------------------
def create_model():
    """Create EfficientNetB0 model with custom top layers"""
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

def load_datasets():
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(TRAIN_PATH, image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode="categorical", shuffle=True, seed=42)
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(VAL_PATH, image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode="categorical")
    test_ds = tf.keras.preprocessing.image_dataset_from_directory(TEST_PATH, image_size=IMG_SIZE, batch_size=BATCH_SIZE, shuffle=False, label_mode="categorical")

    preprocess = lambda x, y: (preprocess_input(x), y)
    return (train_ds.map(preprocess, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE),
            val_ds.map(preprocess, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE),
            test_ds.map(preprocess, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE))

# -------------------------------------------------------------
# FLOWER CLIENT (Updated with timing and verbose)
# -------------------------------------------------------------
class FakeImageClient(fl.client.NumPyClient):
    def __init__(self, client_id, train_ds, num_samples):
        self.client_id = client_id
        self.train_ds = train_ds
        self.num_samples = num_samples
        self.model = create_model()

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        server_round = config.get("server_round", "?")
        print(f"\n[Client {self.client_id}] Round {server_round} - Training...")
        
        start_time = time.time()
        self.model.set_weights(parameters)
        
        # Verbose=1 shows progress bar per client
        history = self.model.fit(self.train_ds, epochs=LOCAL_EPOCHS, verbose=1)
        
        duration = time.time() - start_time
        print(f"[Client {self.client_id}] Round {server_round} finished in {duration:.2f}s")
        
        return self.model.get_weights(), self.num_samples, {"loss": history.history['loss'][-1], "accuracy": history.history['accuracy'][-1]}

    def evaluate(self, parameters, config):
        return 0.0, self.num_samples, {}

# -------------------------------------------------------------
# FLOWER STRATEGY (Updated with timing and round config)
# -------------------------------------------------------------
class WeightedFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, val_ds, **kwargs):
        super().__init__(**kwargs)
        self.val_ds = val_ds
        self.global_model = create_model()
        self.history = {'round': [], 'val_acc': [], 'duration': []}

    def aggregate_fit(self, server_round, results, failures):
        agg_res = super().aggregate_fit(server_round, results, failures)
        if agg_res[0] is not None:
            print(f"\n[Server] Round {server_round} aggregated. Evaluating...")
            start_eval = time.time()
            self.global_model.set_weights(parameters_to_ndarrays(agg_res[0]))
            loss, acc, auc = self.global_model.evaluate(self.val_ds, verbose=0)
            
            self.history['round'].append(server_round)
            self.history['val_acc'].append(acc)
            
            print(f"--- Round {server_round} Results ---")
            print(f"Val Acc: {acc:.4f} | Val AUC: {auc:.4f} | Eval Time: {time.time()-start_eval:.2f}s")
        return agg_res

# -------------------------------------------------------------
# EXECUTION LOGIC
# -------------------------------------------------------------
# def start_server():
#     setup_gpu()
#     _, val_ds, test_ds = load_datasets()
    
#     # This sends the round number to clients so they can print it
#     def on_fit_config(server_round: int):
#         return {"server_round": server_round}

#     strategy = WeightedFedAvg(
#         val_ds=val_ds,
#         fraction_fit=1.0,
#         min_fit_clients=NUM_CLIENTS,
#         min_available_clients=NUM_CLIENTS,
#         on_fit_config_fn=on_fit_config,
#         initial_parameters=ndarrays_to_parameters(create_model().get_weights())
#     )
#     fl.server.start_server(server_address=SERVER_ADDRESS, config=fl.server.ServerConfig(num_rounds=FEDERATED_ROUNDS), strategy=strategy)

# -------------------------------------------------------------
# EXECUTION LOGIC
# -------------------------------------------------------------
def start_server():
    setup_gpu()
    _, val_ds, test_ds = load_datasets()
    
    # 1. Define the folder and file name
    SAVE_DIR = "saved_models"
    MODEL_NAME = "final_federated_model.h5"
    
    # 2. Create the folder if it doesn't exist
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
        print(f"Created directory: {SAVE_DIR}")

    def on_fit_config(server_round: int):
        return {"server_round": server_round}

    strategy = WeightedFedAvg(
        val_ds=val_ds,
        fraction_fit=1.0,
        min_fit_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
        on_fit_config_fn=on_fit_config,
        initial_parameters=ndarrays_to_parameters(create_model().get_weights())
    )
    
    # This call blocks until all FEDERATED_ROUNDS are finished
    fl.server.start_server(
        server_address=SERVER_ADDRESS, 
        config=fl.server.ServerConfig(num_rounds=FEDERATED_ROUNDS), 
        strategy=strategy
    )

    # 3. After training, save the final weights from the strategy's global_model
    print("\n--- Training Complete. Saving Global Model ---")
    save_path = os.path.join(SAVE_DIR, MODEL_NAME)
    strategy.global_model.save(save_path)
    print(f"✓ Model successfully saved to: {save_path}")

def start_client(client_id):
    setup_gpu()
    train_ds, _, _ = load_datasets()
    client_ds = train_ds.shard(num_shards=NUM_CLIENTS, index=client_id)
    # Count samples for weighting
    num_samples = sum(1 for _ in client_ds) * BATCH_SIZE 
    
    client = FakeImageClient(client_id, client_ds, num_samples)
    fl.client.start_numpy_client(server_address=SERVER_ADDRESS, client=client)

if __name__ == "__main__":
    mode = sys.argv[1].lower()
    if mode == "server":
        start_server()
    elif mode == "client":
        start_client(int(sys.argv[2]))

# chmod +x run_federated.sh
# ./run_federated.sh
