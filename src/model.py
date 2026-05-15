"""
Neural network architecture for NBA game prediction.

Architecture overview:
  Input  → Dense(128) → BatchNorm → ReLU → Dropout
         → Dense(64)  → BatchNorm → ReLU → Dropout
         → Dense(32)  → ReLU → Dropout
         → Dense(1)   → Sigmoid

Design choices explained:
  - BatchNormalization: stabilises training; helps with varying feature scales
  - Dropout: reduces over-fitting (common when dataset is ~2,000–5,000 games)
  - Sigmoid output: gives a probability — useful for calibration plots
  - Binary cross-entropy loss: standard for binary classification

We also expose a simple Logistic Regression baseline so you can compare
whether the neural network actually adds value over a linear model.
"""

import tensorflow as tf
from tensorflow import keras
from sklearn.linear_model import LogisticRegression

import config


def build_nn(input_dim: int) -> keras.Model:
    """
    Build and compile the neural network.

    Args:
        input_dim: number of input features

    Returns:
        Compiled Keras model
    """
    tf.random.set_seed(config.RANDOM_SEED)

    inputs = keras.Input(shape=(input_dim,), name="features")

    x = keras.layers.Dense(config.HIDDEN_UNITS[0])(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.Dropout(config.DROPOUT_RATE)(x)

    x = keras.layers.Dense(config.HIDDEN_UNITS[1])(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.Dropout(config.DROPOUT_RATE)(x)

    x = keras.layers.Dense(config.HIDDEN_UNITS[2])(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.Dropout(config.DROPOUT_RATE / 2)(x)

    output = keras.layers.Dense(1, activation="sigmoid", name="home_win_prob")(x)

    model = keras.Model(inputs=inputs, outputs=output, name="NBA_Predictor")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.AUC(name="auc"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )
    return model


def build_baseline() -> LogisticRegression:
    """
    Logistic Regression baseline — a linear model that's fast to train.
    Comparing NN vs LR shows whether the network learns non-linear patterns.
    """
    return LogisticRegression(
        max_iter=1000,
        random_state=config.RANDOM_SEED,
        C=1.0,           # inverse regularisation strength
        solver="lbfgs",
    )


def get_callbacks(monitor: str = "val_auc") -> list:
    """
    Training callbacks:
      - EarlyStopping  : stops when validation metric plateaus (prevents over-fitting)
      - ReduceLROnPlateau: halves learning rate when stuck
      - ModelCheckpoint: saves the best weights automatically
    """
    return [
        keras.callbacks.EarlyStopping(
            monitor=monitor,
            mode="max",
            patience=config.PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor=monitor,
            mode="max",
            factor=0.5,
            patience=config.PATIENCE // 2,
            min_lr=1e-6,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=config.MODEL_SAVE_PATH,
            monitor=monitor,
            mode="max",
            save_best_only=True,
            verbose=0,
        ),
    ]
