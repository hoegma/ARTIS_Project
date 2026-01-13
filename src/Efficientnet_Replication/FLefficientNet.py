# Federated Learning Implementation for Fake Image Detection
# Based on: "Fake Image Detection Using Deep Learning"

import os
import numpy as np
import pickle
import datetime
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras import layers, models

# -------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
LOCAL_EPOCHS = 3  # Epochs per client per round
FEDERATED_ROUNDS = 10  # Number of federated rounds
NUM_CLIENTS = 5  # Number of federated clients

BASE_PATH = "140k_real_fake_face/real_vs_fake/real-vs-fake"
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
# Create Model (same architecture as paper)
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
# Load Full Dataset
# -------------------------------------------------------------
def load_full_dataset():
    """Load the complete training dataset"""
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
# Partition Data for Federated Clients
# -------------------------------------------------------------
def partition_data_for_clients(train_ds, num_clients):
    """
    Split training data among clients (simulating different organizations)
    
    In real federated learning:
    - Each client has their own local data
    - Data is NOT shared or transferred
    - This function simulates that by partitioning
    """
    # Convert to list for partitioning
    all_batches = list(train_ds.as_numpy_iterator())
    total_batches = len(all_batches)
    
    print(f"Total training batches: {total_batches}")
    print(f"Partitioning among {num_clients} clients...")
    
    # Divide batches among clients
    batches_per_client = total_batches // num_clients
    client_datasets = []
    client_sample_counts = []  # Track number of samples per client
    
    for i in range(num_clients):
        start_idx = i * batches_per_client
        if i == num_clients - 1:
            # Last client gets remaining batches
            end_idx = total_batches
        else:
            end_idx = (i + 1) * batches_per_client
        
        client_batches = all_batches[start_idx:end_idx]
        num_samples = len(client_batches) * BATCH_SIZE  # Approximate sample count
        
        # Create dataset from batches
        client_ds = tf.data.Dataset.from_generator(
            lambda: iter(client_batches),
            output_signature=(
                tf.TensorSpec(shape=(None, IMG_SIZE[0], IMG_SIZE[1], 3), dtype=tf.float32),
                tf.TensorSpec(shape=(None, 2), dtype=tf.float32)
            )
        )
        client_ds = client_ds.prefetch(AUTOTUNE)
        client_datasets.append(client_ds)
        client_sample_counts.append(num_samples)
        
        print(f"  Client {i+1}: {len(client_batches)} batches (~{num_samples} images)")
    
    return client_datasets, client_sample_counts

# -------------------------------------------------------------
# Federated Learning: Weight Aggregation
# -------------------------------------------------------------
def aggregate_weights(client_weights_list, client_sample_counts):
    """
    Federated Averaging (FedAvg) - WEIGHTED average based on client data size
    
    This is the PROPER way to do federated averaging:
    - Each client's contribution is weighted by their dataset size
    - Clients with more data have more influence
    - This prevents small datasets from biasing the global model
    
    Formula: w_global = Σ(n_k / n_total) * w_k
    where:
        n_k = number of samples for client k
        n_total = total samples across all clients
        w_k = weights from client k
    """
    total_samples = sum(client_sample_counts)
    
    # Initialize with zeros
    avg_weights = [np.zeros_like(w) for w in client_weights_list[0]]
    
    # Weighted sum of client weights
    for client_weights, num_samples in zip(client_weights_list, client_sample_counts):
        # Calculate weight for this client
        client_weight = num_samples / total_samples
        
        for i, w in enumerate(client_weights):
            avg_weights[i] += client_weight * w
    
    return avg_weights

# -------------------------------------------------------------
# Client Training Function
# -------------------------------------------------------------
def train_client(client_id, client_dataset, global_weights, local_epochs):
    """
    Train a single client on their local data
    
    In real federated learning:
    - This happens on the client's device/server
    - Data never leaves the client
    - Only model weights are sent to server
    """
    print(f"  Training Client {client_id}...")
    
    # Create fresh model with global weights
    model = create_model()
    model.set_weights(global_weights)
    
    # Train on local data
    history = model.fit(
        client_dataset,
        epochs=local_epochs,
        verbose=0
    )
    
    # Get updated weights (this is what gets sent to server)
    updated_weights = model.get_weights()
    
    # Get training metrics
    final_loss = history.history['loss'][-1]
    final_acc = history.history['accuracy'][-1]
    
    print(f"    Client {client_id} - Loss: {final_loss:.4f}, Acc: {final_acc:.4f}")
    
    return updated_weights, final_loss, final_acc

# -------------------------------------------------------------
# Federated Learning Main Loop
# -------------------------------------------------------------
def federated_learning(client_datasets, client_sample_counts, val_ds, test_ds):
    """
    Main federated learning loop
    
    Process:
    1. Initialize global model
    2. For each round:
       a. Send global weights to all clients
       b. Each client trains locally
       c. Clients send updated weights back
       d. Server aggregates weights (WEIGHTED by dataset size)
       e. Evaluate global model
    3. Final evaluation
    """
    print("\n" + "="*60)
    print("FEDERATED LEARNING - FAKE IMAGE DETECTION")
    print("="*60)
    print(f"Clients: {NUM_CLIENTS}")
    print(f"Federated Rounds: {FEDERATED_ROUNDS}")
    print(f"Local Epochs per Round: {LOCAL_EPOCHS}")
    print(f"\nClient Data Distribution:")
    total = sum(client_sample_counts)
    for i, count in enumerate(client_sample_counts):
        percentage = (count / total) * 100
        print(f"  Client {i+1}: {count} samples ({percentage:.1f}%)")
    print("="*60 + "\n")
    
    # Initialize global model
    global_model = create_model()
    global_weights = global_model.get_weights()
    
    # Track metrics
    history = {
        'round': [],
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_auc': []
    }
    
    # Federated training rounds
    for round_num in range(1, FEDERATED_ROUNDS + 1):
        print(f"\n{'='*60}")
        print(f"FEDERATED ROUND {round_num}/{FEDERATED_ROUNDS}")
        print(f"{'='*60}")
        
        # Store weights from all clients
        client_weights_list = []
        round_losses = []
        round_accs = []
        
        # Train each client
        for client_id in range(1, NUM_CLIENTS + 1):
            client_weights, loss, acc = train_client(
                client_id=client_id,
                client_dataset=client_datasets[client_id - 1],
                global_weights=global_weights,
                local_epochs=LOCAL_EPOCHS
            )
            client_weights_list.append(client_weights)
            round_losses.append(loss)
            round_accs.append(acc)
        
        # Aggregate weights (WEIGHTED Federated Averaging)
        print(f"\n  Aggregating weights (weighted by dataset size)...")
        global_weights = aggregate_weights(client_weights_list, client_sample_counts)
        
        # Update global model
        global_model.set_weights(global_weights)
        
        # Evaluate on validation set
        print(f"  Evaluating global model on validation set...")
        val_results = global_model.evaluate(val_ds, verbose=0)
        val_loss, val_acc, val_auc = val_results
        
        # Average client metrics
        avg_train_loss = np.mean(round_losses)
        avg_train_acc = np.mean(round_accs)
        
        # Store metrics
        history['round'].append(round_num)
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(avg_train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_auc'].append(val_auc)
        
        print(f"\n  Round {round_num} Summary:")
        print(f"    Avg Client Train Loss: {avg_train_loss:.4f}")
        print(f"    Avg Client Train Acc:  {avg_train_acc:.4f}")
        print(f"    Global Val Loss:       {val_loss:.4f}")
        print(f"    Global Val Acc:        {val_acc:.4f}")
        print(f"    Global Val AUC:        {val_auc:.4f}")
        
        # Save checkpoint
        if round_num % 2 == 0:  # Save every 2 rounds
            global_model.save(f"federated_model_round_{round_num}.keras")
            print(f"    ✓ Checkpoint saved")
    
    return global_model, history

# -------------------------------------------------------------
# Main Function
# -------------------------------------------------------------
def main():
    setup_gpu()
    
    print("\n=== Loading Dataset ===")
    train_ds, val_ds, test_ds = load_full_dataset()
    
    print("\n=== Partitioning Data for Federated Clients ===")
    client_datasets, client_sample_counts = partition_data_for_clients(train_ds, NUM_CLIENTS)
    
    print("\n=== Starting Federated Learning ===")
    global_model, history = federated_learning(client_datasets, client_sample_counts, val_ds, test_ds)
    
    print("\n" + "="*60)
    print("FINAL EVALUATION ON TEST SET")
    print("="*60)
    test_results = global_model.evaluate(test_ds, verbose=1)
    test_loss, test_acc, test_auc = test_results
    
    print(f"\nFinal Test Results:")
    print(f"  Test Loss:     {test_loss:.4f}")
    print(f"  Test Accuracy: {test_acc*100:.2f}%")
    print(f"  Test AUC:      {test_auc:.4f}")
    
    # Save final model and history
    global_model.save("federated_final_model.h5")
    with open("federated_history.pkl", "wb") as f:
        pickle.dump(history, f)
    
    print("\n✓ Training Complete!")
    print("✓ Model saved as 'federated_final_model.h5'")

if __name__ == "__main__":
    main()