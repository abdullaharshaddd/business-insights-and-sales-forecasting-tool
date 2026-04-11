"""
churn_model.py
==============
Business Insights & Sales Forecasting Tool
Phase 3 — Model Architecture (Neural Network Classifier)

Defines the Deep Neural Network architecture for the Customer Churn
Prediction task.

Architecture
------------
Input ──► Dense(128, ReLU) ──► Dropout(0.3)
       ──► Dense(64,  ReLU) ──► Dropout(0.3)
       ──► Dense(32,  ReLU)
       ──► Dense(1,   Sigmoid)

Loss      : Binary Cross-Entropy
Optimizer : Adam (lr=0.001)
Metric    : AUC-ROC, Accuracy, Precision, Recall
"""

import os
import yaml
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC, Precision, Recall

# ── Config ─────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG_PATH = os.path.join(BASE_DIR, "config", "config.yaml")

with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)

model_cfg = cfg["model"]
hp        = cfg["final_hp"]


# ── Architecture Builder ────────────────────────────────────────────────────

def build_churn_model(
    input_dim: int,
    layers: list      = None,
    dropout: float    = None,
    learning_rate: float = None,
) -> tf.keras.Model:
    """
    Build and compile the Customer Churn DNN.

    Parameters
    ----------
    input_dim     : Number of input features
    layers        : List of hidden layer sizes (default from config)
    dropout       : Dropout rate (default from config)
    learning_rate : Adam learning rate (default from config)

    Returns
    -------
    Compiled Keras model
    """
    layers_cfg    = layers        or hp["layers"]
    dropout_rate  = dropout       or hp["dropout"]
    lr            = learning_rate or hp["learning_rate"]

    # ── Input ───────────────────────────────────────────────────────────
    inputs = Input(shape=(input_dim,), name="input_features")
    x = inputs

    # ── Hidden Layers ───────────────────────────────────────────────────
    for i, units in enumerate(layers_cfg):
        x = Dense(units, activation="relu", name=f"dense_{i+1}")(x)
        x = BatchNormalization(name=f"bn_{i+1}")(x)
        if i < len(layers_cfg) - 1:          # no dropout before last hidden
            x = Dropout(dropout_rate, name=f"dropout_{i+1}")(x)

    # ── Output ──────────────────────────────────────────────────────────
    outputs = Dense(1, activation="sigmoid", name="output_churn")(x)

    # ── Compile ─────────────────────────────────────────────────────────
    model = Model(inputs=inputs, outputs=outputs, name="ChurnDNN")
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            AUC(name="auc_roc", curve="ROC"),
            Precision(name="precision"),
            Recall(name="recall"),
        ],
    )
    return model


# ── Baseline: Scikit-learn models for comparison ─────────────────────────────

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble    import RandomForestClassifier, GradientBoostingClassifier


def get_baseline_models():
    """
    Return a dictionary of baseline Scikit-learn classifiers for comparison.
    These serve as benchmarks against the neural network.
    """
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=42, class_weight="balanced"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=10, random_state=42, class_weight="balanced"
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42
        ),
    }


if __name__ == "__main__":
    # ── Quick sanity check ───────────────────────────────────────────────
    model = build_churn_model(input_dim=12)
    model.summary()
    print("\nBaseline models:", list(get_baseline_models().keys()))
