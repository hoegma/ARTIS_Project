# Reimplementing the paper "Machine Learning Approach for Fake Face Image Forensics"
# DOI: https://ubjbas.ub.edu.sa/home/vol1/iss1/5/

import os
import glob
import pickle
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models
from tensorflow.keras.applications.vgg16 import preprocess_input


IMG_SIZE = (384, 384)
BATCH_SIZE = 32

# -------------------------------------------------------------
# 1. Load Dataset
# -------------------------------------------------------------
def load_dataset(fake_path, real_path):
    fake_images = glob.glob(os.path.join(fake_path, "*"))
    real_images = glob.glob(os.path.join(real_path, "*"))

    all_images = fake_images + real_images
    labels = [0] * len(fake_images) + [1] * len(real_images)

    return all_images, labels

# -------------------------------------------------------------
# 2. Feature Extraction Preprocessing
# -------------------------------------------------------------

def preprocess_image(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMG_SIZE)
    image = image / 255.0
    return image

# -------------------------------------------------------------
# 3. Create VGG16 Feature Extractor 
# -------------------------------------------------------------
def create_vgg16_feature_extractor():
    base_model = VGG16(weights="imagenet", include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    base_model.trainable = False

    model = models.Model(inputs=base_model.input, outputs=base_model.output)
    return model

# -------------------------------------------------------------
# 4. Extract Feature
# -------------------------------------------------------------
def extract_features(model, image_paths):
    features = []
    for path in image_paths:
        img = preprocess_image(path)
        img = tf.expand_dims(img, axis=0)
        feat = model(img)
        feat = tf.reshape(feat, [-1])
        features.append(feat.numpy())
    return features


# -------------------------------------------------------------
# 6. MAIN
# -------------------------------------------------------------
def main():
    FAKE_PATH = "dataset/real_and_fake_face/training_fake"
    REAL_PATH = "dataset/real_and_fake_face/training_real"

    all_paths, labels = load_dataset(FAKE_PATH, REAL_PATH)

    feature_model = create_vgg16_feature_extractor()

    print("Extract Features...")
    features = extract_features(feature_model, all_paths)

    with open("1_vgg16_features_384x384.pkl", "wb") as f:
        pickle.dump({"features": features, "labels": labels}, f)

    print("Successfully finished feature extraction and stored in file.")

if __name__ == "__main__":
    main()
