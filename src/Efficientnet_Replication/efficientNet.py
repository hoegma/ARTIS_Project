# Reimplementing the paper "Fake Image Detection Using Deep Learning"
# DOI: https://doi.org/10.31449/inf.v47i7.4741

import os
import glob
import pickle
import datetime
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras import layers, models
from tensorflow.keras import mixed_precision


# -------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 30

BASE_PATH = "140k_real_fake_face/real_vs_fake/real-vs-fake"

TRAIN_PATH = f"{BASE_PATH}/train"
VAL_PATH   = f"{BASE_PATH}/valid"
TEST_PATH  = f"{BASE_PATH}/test"

LOG_DIR = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

AUTOTUNE = tf.data.AUTOTUNE

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"



# -------------------------------------------------------------
# Load Dataset
# -------------------------------------------------------------
def load_dataset(path_to_train, path_to_test, path_to_validation):
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        path_to_train,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical"
    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        path_to_validation,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical"
    )

    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        path_to_test,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False,
        label_mode="categorical"
    )

    return train_ds, val_ds, test_ds

# -------------------------------------------------------------
# Load EfficientNetB0 Model
# -------------------------------------------------------------
def load_efficientnet_model():
    base_model = EfficientNetB0(
        weights="imagenet",      
        include_top=False,       
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
    )
    base_model.trainable = False # Freeze weights

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(2, activation="softmax", dtype='float32')
    ])

    return model

# -------------------------------------------------------------
# Compile Model
# -------------------------------------------------------------
def compile_model(model):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
    )

    return model


# -------------------------------------------------------------
# Create Learning Schedule
# -------------------------------------------------------------
def get_learning_schedule(epoch, lr = None):
    if epoch <= 2:
        return 0.001
    elif epoch > 2 and epoch <= 15:
        return 0.0001
    else:
        return 0.00001
    
# -------------------------------------------------------------
# CALLBACKS: LR + Checkpoint + EarlyStopping
# -------------------------------------------------------------
def get_callbacks():
    lr_callback = tf.keras.callbacks.LearningRateScheduler(get_learning_schedule)

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath="best_model_model3.keras",
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=False,
        mode="min",
        verbose=1
    )

    earlystop_callback = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=6,
        restore_best_weights=True,
        verbose=1
    )

    return [lr_callback, checkpoint_callback, earlystop_callback]

def main():

    print("=== Read Dataset ===")
    train_ds, val_ds, test_ds = load_dataset(TRAIN_PATH, TEST_PATH, VAL_PATH)

    print("Classes:", train_ds.class_names)

    # Preprocessing for EfficientNet
    train_ds = train_ds.map(lambda x, y: (preprocess_input(x), y))
    val_ds   = val_ds.map(lambda x, y: (preprocess_input(x), y))
    test_ds  = test_ds.map(lambda x, y: (preprocess_input(x), y))

    # Performance optimizations
    train_ds = train_ds.shuffle(500).prefetch(AUTOTUNE)
    val_ds   = val_ds.prefetch(AUTOTUNE)
    test_ds  = test_ds.prefetch(AUTOTUNE)

    model = load_efficientnet_model()
    model = compile_model(model)

    callbacks = get_callbacks()

    # Train
    print("=== Starting Training ===")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    # Save training history

    try:
        with open("training_history.pkl", "wb") as f:
            pickle.dump(history.history, f)
    except:
        pass

    # Evaluate best model on test set
    print("=== Evaluating BEST Model on Test Set ===")
    best_model = tf.keras.models.load_model("best_model_model3.keras")
    best_model.evaluate(test_ds)

    # Optional: save final model
    best_model.save("efficientnetb0_fake_real_model3_final.h5")

if __name__ == "__main__":
    main()