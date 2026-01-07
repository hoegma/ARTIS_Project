# Federated Learning with Flower Framework - LOCAL MODE
# Run server and clients as separate processes on your laptop

#i created a bash file to run the multiple clients and server
#to run use this command in a single terminal:
# open bash terminal and run this to open the folder that holds the script: 
#  cd \matthias\\Efficientnet_Replication\\FederatedLearning\\FedLearningFinal
#  chmod +x run_federated.sh
#  ./run_federated.sh

import os
import sys
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras import layers, models
from tensorflow.keras import backend as K
import flwr as fl
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from typing import List, Tuple, Dict
import gc

# -------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------
IMG_SIZE = (224, 224)
BATCH_SIZE = 128
LOCAL_EPOCHS = 3
FEDERATED_ROUNDS = 10
NUM_CLIENTS = 5
SERVER_ADDRESS = "localhost:8080"

# Update these paths to your local dataset location
BASE_PATH = "./real-vs-fake"  # Change this to your dataset path
TRAIN_PATH = f"{BASE_PATH}/train"
VAL_PATH = f"{BASE_PATH}/valid"
TEST_PATH = f"{BASE_PATH}/test"

AUTOTUNE = tf.data.AUTOTUNE

# -------------------------------------------------------------
# GPU Setup
# -------------------------------------------------------------
def setup_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✓ GPU enabled: {len(gpus)} GPU(s)")
        except RuntimeError as e:
            print(f"GPU error: {e}")
    else:
        print("⚠ No GPU - using CPU")

# -------------------------------------------------------------
# Create Model
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

# -------------------------------------------------------------
# Load Datasets
# -------------------------------------------------------------
def load_datasets():
    """Load training, validation, and test datasets"""
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        TRAIN_PATH,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        shuffle=True,
        seed=42
    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        VAL_PATH,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical"
    )

    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        TEST_PATH,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False,
        label_mode="categorical"
    )

    # Preprocessing
    train_ds = train_ds.map(lambda x, y: (preprocess_input(x), y), num_parallel_calls=AUTOTUNE)
    val_ds = val_ds.map(lambda x, y: (preprocess_input(x), y), num_parallel_calls=AUTOTUNE)
    test_ds = test_ds.map(lambda x, y: (preprocess_input(x), y), num_parallel_calls=AUTOTUNE)

    # Optimize
    train_ds = train_ds.prefetch(AUTOTUNE)
    val_ds = val_ds.prefetch(AUTOTUNE)
    test_ds = test_ds.prefetch(AUTOTUNE)

    return train_ds, val_ds, test_ds

# -------------------------------------------------------------
# Partition Data for Clients
# -------------------------------------------------------------
def get_client_data(train_ds, client_id: int, num_clients: int):
    """Get data partition for a specific client"""
    client_ds = train_ds.shard(num_shards=num_clients, index=client_id)
    client_samples = sum(int(batch_x.shape[0]) for batch_x, _ in client_ds)
    client_ds = client_ds.prefetch(AUTOTUNE)
    
    print(f"  Client {client_id}: {client_samples} samples")
    return client_ds, client_samples

# -------------------------------------------------------------
# FLOWER CLIENT
# -------------------------------------------------------------
class FakeImageClient(fl.client.NumPyClient):
    """Flower client for federated fake image detection"""
    
    def __init__(self, client_id: int, train_ds, num_samples: int):
        self.client_id = client_id
        self.train_ds = train_ds
        self.num_samples = num_samples
        self.model = create_model()
        print(f"[Client {client_id}] Initialized with {num_samples} samples")
    
    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        """Return current model parameters"""
        return self.model.get_weights()
    
    def fit(self, parameters: List[np.ndarray], config: Dict) -> Tuple[List[np.ndarray], int, Dict]:
        """Train model on local data"""
        print(f"\n[Client {self.client_id}] Starting local training (Epoch {config.get('server_round', '?')})...")
        
        # Update model with global parameters
        self.model.set_weights(parameters)
        
        # Train on local data
        history = self.model.fit(
            self.train_ds,
            epochs=LOCAL_EPOCHS,
            verbose=0
        )
        
        # Get metrics
        final_loss = history.history['loss'][-1]
        final_acc = history.history['accuracy'][-1]
        
        print(f"[Client {self.client_id}] Training complete - Loss: {final_loss:.4f}, Acc: {final_acc:.4f}")
        
        # Return updated weights and metrics
        updated_weights = self.model.get_weights()
        metrics = {
            "loss": final_loss,
            "accuracy": final_acc
        }
        
        # Cleanup
        del history
        gc.collect()
        
        return updated_weights, self.num_samples, metrics
    
    def evaluate(self, parameters: List[np.ndarray], config: Dict) -> Tuple[float, int, Dict]:
        """Evaluate model (optional)"""
        return 0.0, self.num_samples, {}

# -------------------------------------------------------------
# FLOWER STRATEGY (Server-side)
# -------------------------------------------------------------
class WeightedFedAvg(fl.server.strategy.FedAvg):
    """Custom FedAvg strategy with evaluation"""
    
    def __init__(self, val_ds, test_ds, **kwargs):
        super().__init__(**kwargs)
        self.val_ds = val_ds
        self.test_ds = test_ds
        self.global_model = create_model()
        self.history = {
            'round': [],
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_auc': []
        }
    
    def aggregate_fit(self, server_round, results, failures):
        """Aggregate client updates"""
        if failures:
            print(f"⚠ {len(failures)} clients failed")
        
        print(f"\n{'='*60}")
        print(f"[Server] Round {server_round}/{FEDERATED_ROUNDS} - Aggregating {len(results)} client updates")
        print(f"{'='*60}")
        
        # Extract metrics
        train_losses = [fit_res.metrics["loss"] for _, fit_res in results]
        train_accs = [fit_res.metrics["accuracy"] for _, fit_res in results]
        
        # Perform weighted aggregation
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )
        
        # Evaluate global model
        if aggregated_parameters is not None:
            print(f"[Server] Evaluating global model on validation set...")
            global_weights = parameters_to_ndarrays(aggregated_parameters)
            self.global_model.set_weights(global_weights)
            
            val_results = self.global_model.evaluate(self.val_ds, verbose=0)
            val_loss, val_acc, val_auc = val_results
            
            # Store metrics
            self.history['round'].append(server_round)
            self.history['train_loss'].append(np.mean(train_losses))
            self.history['train_acc'].append(np.mean(train_accs))
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['val_auc'].append(val_auc)
            
            print(f"\n{'='*60}")
            print(f"Round {server_round} Summary:")
            print(f"  Avg Client Train Loss: {np.mean(train_losses):.4f}")
            print(f"  Avg Client Train Acc:  {np.mean(train_accs):.4f}")
            print(f"  Global Val Loss:       {val_loss:.4f}")
            print(f"  Global Val Acc:        {val_acc:.4f}")
            print(f"  Global Val AUC:        {val_auc:.4f}")
            print(f"{'='*60}\n")
            
            # Save checkpoint
            if server_round % 2 == 0:
                self.global_model.save(f"federated_model_round_{server_round}.keras")
                print(f"✓ Checkpoint saved: federated_model_round_{server_round}.keras\n")
        
        return aggregated_parameters, aggregated_metrics

# -------------------------------------------------------------
# SERVER MODE
# -------------------------------------------------------------
def start_server():
    """Start the Flower server"""
    setup_gpu()
    
    print("\n" + "="*60)
    print("FLOWER FEDERATED LEARNING SERVER")
    print("="*60)
    print(f"Server Address: {SERVER_ADDRESS}")
    print(f"Expected Clients: {NUM_CLIENTS}")
    print(f"Federated Rounds: {FEDERATED_ROUNDS}")
    print(f"Local Epochs: {LOCAL_EPOCHS}")
    print("="*60 + "\n")
    
    # Load validation and test datasets
    print("Loading validation and test datasets...")
    _, val_ds, test_ds = load_datasets()
    
    # Initialize model
    initial_model = create_model()
    initial_parameters = ndarrays_to_parameters(initial_model.get_weights())
    
    # Create strategy
    strategy = WeightedFedAvg(
        val_ds=val_ds,
        test_ds=test_ds,
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        min_fit_clients=NUM_CLIENTS,
        min_evaluate_clients=0,
        min_available_clients=NUM_CLIENTS,
        initial_parameters=initial_parameters,
    )
    
    print("✓ Server ready! Waiting for clients to connect...\n")
    
    # Start server
    fl.server.start_server(
        server_address=SERVER_ADDRESS,
        config=fl.server.ServerConfig(num_rounds=FEDERATED_ROUNDS),
        strategy=strategy,
    )
    
    # Final evaluation
    print("\n" + "="*60)
    print("FINAL EVALUATION ON TEST SET")
    print("="*60)
    
    final_model = strategy.global_model
    test_results = final_model.evaluate(test_ds, verbose=1)
    test_loss, test_acc, test_auc = test_results
    
    print(f"\nFinal Test Results:")
    print(f"  Test Loss:     {test_loss:.4f}")
    print(f"  Test Accuracy: {test_acc*100:.2f}%")
    print(f"  Test AUC:      {test_auc:.4f}")
    
    # Save final model
    final_model.save("federated_final_model.h5")
    with open("federated_history.pkl", "wb") as f:
        pickle.dump(strategy.history, f)
    
    print("\n✓ Training Complete!")
    print("✓ Model saved: federated_final_model.h5")
    print("✓ History saved: federated_history.pkl")

# -------------------------------------------------------------
# CLIENT MODE
# -------------------------------------------------------------
def start_client(client_id: int):
    """Start a Flower client"""
    setup_gpu()
    
    print("\n" + "="*60)
    print(f"FLOWER CLIENT {client_id}")
    print("="*60)
    print(f"Connecting to: {SERVER_ADDRESS}")
    print("="*60 + "\n")
    
    # Load training data
    print("Loading training dataset...")
    train_ds, _, _ = load_datasets()
    
    # Get client's data partition
    client_ds, num_samples = get_client_data(train_ds, client_id, NUM_CLIENTS)
    
    # Create client
    client = FakeImageClient(
        client_id=client_id,
        train_ds=client_ds,
        num_samples=num_samples
    )
    
    print(f"\n✓ Client {client_id} ready! Connecting to server...\n")
    
    # Start client
    fl.client.start_numpy_client(
        server_address=SERVER_ADDRESS,
        client=client
    )
    
    print(f"\n✓ Client {client_id} finished training!")

# -------------------------------------------------------------
# MAIN ENTRY POINT
# -------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Server: python federated_local.py server")
        print("  Client: python federated_local.py client <client_id>")
        print(f"  Example: python federated_local.py client 0")
        print(f"           python federated_local.py client 1")
        print(f"           ... (up to client {NUM_CLIENTS-1})")
        sys.exit(1)
    
    mode = sys.argv[1].lower()
    
    if mode == "server":
        start_server()
    elif mode == "client":
        if len(sys.argv) < 3:
            print("Error: Client ID required")
            print(f"Usage: python federated_local.py client <0-{NUM_CLIENTS-1}>")
            sys.exit(1)
        client_id = int(sys.argv[2])
        if client_id < 0 or client_id >= NUM_CLIENTS:
            print(f"Error: Client ID must be between 0 and {NUM_CLIENTS-1}")
            sys.exit(1)
        start_client(client_id)
    else:
        print(f"Error: Unknown mode '{mode}'")
        print("Use 'server' or 'client'")
        sys.exit(1)