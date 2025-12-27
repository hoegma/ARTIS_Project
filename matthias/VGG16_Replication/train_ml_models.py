import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

import matplotlib.pyplot as plt

FEATURE_PATH = "1_vgg16_features_384x384.pkl"

# -----------------------------
# 1. Load Features
# -----------------------------
def load_features(path=FEATURE_PATH):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return np.array(data["features"]), np.array(data["labels"])

# -----------------------------
# 2. ANN Modell
# -----------------------------
def build_ann(input_dim):
    model = models.Sequential()
    model.add(layers.Dense(256, activation='relu', input_shape=(input_dim,)))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(2, activation='softmax'))
    optimizer = optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
    return model

# -----------------------------
# 3. Pipeline
# -----------------------------
def main():
    # Load features
    X, y = load_features()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Dictionary für Ergebnisse
    results = {}

    # -----------------------------
    # Logistic Regression
    # -----------------------------
    lr = LogisticRegression(penalty="l2", fit_intercept=True, solver="lbfgs", max_iter=150)
    lr.fit(X_train_scaled, y_train)
    lr_pred = lr.predict(X_test_scaled)
    results['LR'] = lr_pred

    # -----------------------------
    # KNN
    # -----------------------------
    knn = KNeighborsClassifier(n_neighbors=35, weights="uniform", metric="euclidean")
    knn.fit(X_train_scaled, y_train)
    knn_pred = knn.predict(X_test_scaled)
    results['KNN'] = knn_pred

    # -----------------------------
    # Decision Tree
    # -----------------------------
    dt = DecisionTreeClassifier(criterion="gini", min_samples_split=6, min_samples_leaf=3, max_features=25)
    dt.fit(X_train_scaled, y_train)
    dt_pred = dt.predict(X_test_scaled)
    results['DT'] = dt_pred

    # -----------------------------
    # Random Forest
    # -----------------------------
    rf = RandomForestClassifier(n_estimators=500, bootstrap=True, min_samples_split=5, min_samples_leaf=10)
    rf.fit(X_train_scaled, y_train)
    rf_pred = rf.predict(X_test_scaled)
    results['RF'] = rf_pred

    # -----------------------------
    # ANN
    # -----------------------------
    encoder = OneHotEncoder(sparse_output=False)
    y_train_oh = encoder.fit_transform(y_train.reshape(-1,1))
    y_test_oh = encoder.transform(y_test.reshape(-1,1))

    ann = build_ann(X_train_scaled.shape[1])
    ann.fit(X_train_scaled, y_train_oh, epochs=100, batch_size=100, validation_split=0.2, verbose=0)
    ann_pred_prob = ann.predict(X_test_scaled)
    ann_pred = np.argmax(ann_pred_prob, axis=1)
    results['ANN'] = ann_pred

    # -----------------------------
    # Calculate Metrics
    # -----------------------------
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-score']
    metric_values = {m: [] for m in metrics}

    for model_name, preds in results.items():
        metric_values['Accuracy'].append(accuracy_score(y_test, preds))
        metric_values['Precision'].append(precision_score(y_test, preds))
        metric_values['Recall'].append(recall_score(y_test, preds))
        metric_values['F1-score'].append(f1_score(y_test, preds))

    # -----------------------------
    # Plot
    # -----------------------------
    model_names = list(results.keys())
    for metric in metrics:
        plt.figure(figsize=(8,5))
        plt.bar(model_names, metric_values[metric], color='skyblue')
        plt.title(f'{metric} Comparison')
        plt.ylabel(metric)
        plt.ylim(0, 1)
        for i, v in enumerate(metric_values[metric]):
            plt.text(i, v + 0.02, f"{v:.2f}", ha='center', fontweight='bold')
        plt.show()

if __name__ == "__main__":
    main()
