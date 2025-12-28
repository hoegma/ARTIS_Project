import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.efficientnet import preprocess_input

AUTOTUNE = tf.data.AUTOTUNE
BATCH_SIZE = 32

def load_dataset(path):
    ds = tf.keras.preprocessing.image_dataset_from_directory(
        path,
        image_size=(224, 224),
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        shuffle=True,
        seed=42
    )
    ds = ds.map(lambda x, y: (preprocess_input(x), y))
    return ds.prefetch(AUTOTUNE)


def non_iid_split(dataset, num_clients):
    """
    Non-IID partitioning via class skew
    """
    data = list(dataset.as_numpy_iterator())

    class0 = [b for b in data if np.argmax(b[1][0]) == 0]
    class1 = [b for b in data if np.argmax(b[1][0]) == 1]

    np.random.shuffle(class0)
    np.random.shuffle(class1)

    ratios = np.linspace(0.9, 0.1, num_clients)
    client_datasets = []

    for r in ratios:
        n0 = max(1, int(r * len(class0) / num_clients))
        n1 = max(1, int((1 - r) * len(class1) / num_clients))

        batches = class0[:n0] + class1[:n1]
        class0 = class0[n0:]
        class1 = class1[n1:]

        ds = tf.data.Dataset.from_generator(
            lambda b=batches: iter(b),
            output_signature=(
                tf.TensorSpec(shape=(None, 224, 224, 3), dtype = tf.float32),
                tf.TensorSpec(shape=(None, 2), dtype = tf.float32)
            )
        ).prefetch(AUTOTUNE)

        client_datasets.append(ds)

    return client_datasets
