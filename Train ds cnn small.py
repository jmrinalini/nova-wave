"""
train_ds_cnn.py

Defines and trains a DS-CNN (Depthwise-Separable CNN) wake-word model
for "Nova" using the MFCC features produced by extract_mfcc.py.

This architecture is built entirely from standard Keras layers —
no pretrained weights are loaded or used anywhere in this script.

Expected input (from extract_mfcc.py):
    dataset/processed/X_train.npy, y_train.npy
    dataset/processed/X_val.npy,   y_val.npy
    dataset/processed/X_test.npy,  y_test.npy

Usage:
    python train_ds_cnn.py

Requires:
    pip install tensorflow numpy
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
PROCESSED_DIR = os.path.join("dataset", "processed")
MODEL_OUT_DIR = "models"
MODEL_NAME = "nova_ds_cnn.keras"

EPOCHS = 40
BATCH_SIZE = 32
LEARNING_RATE = 1e-3


def load_data():
    X_train = np.load(os.path.join(PROCESSED_DIR, "X_train.npy"))
    y_train = np.load(os.path.join(PROCESSED_DIR, "y_train.npy"))
    X_val = np.load(os.path.join(PROCESSED_DIR, "X_val.npy"))
    y_val = np.load(os.path.join(PROCESSED_DIR, "y_val.npy"))
    X_test = np.load(os.path.join(PROCESSED_DIR, "X_test.npy"))
    y_test = np.load(os.path.join(PROCESSED_DIR, "y_test.npy"))
    return X_train, y_train, X_val, y_val, X_test, y_test


def build_ds_cnn(input_shape, num_classes=2):
    """
    A compact DS-CNN for keyword spotting, following the general
    Google KWS reference pattern: one regular conv "stem" layer,
    followed by several depthwise-separable conv blocks, then a
    global pooling + dense classifier head.

    input_shape: (time_frames, n_mfcc, 1)
    """
    inputs = layers.Input(shape=input_shape)

    # Stem: regular conv to learn initial low-level features
    x = layers.Conv2D(64, (10, 4), strides=(2, 2), padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Depthwise-separable blocks — this is what makes it "DS-CNN"
    for filters in [64, 64, 64]:
        x = layers.DepthwiseConv2D((3, 3), padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Conv2D(filters, (1, 1), padding="same")(x)  # pointwise
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs, name="nova_ds_cnn")
    return model


def main():
    print("Loading processed features...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()

    # Add channel dimension: (samples, time, mfcc) -> (samples, time, mfcc, 1)
    X_train = X_train[..., np.newaxis]
    X_val = X_val[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    print(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}, Test shape: {X_test.shape}")

    input_shape = X_train.shape[1:]
    model = build_ds_cnn(input_shape, num_classes=2)
    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=8, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6
        ),
    ]

    print("\nTraining...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
    )

    print("\nEvaluating on held-out test set...")
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {test_acc:.4f}  |  Test loss: {test_loss:.4f}")

    # Check per-class confidence, not just accuracy — this is the
    # check that would have caught the earlier low-confidence problem.
    preds = model.predict(X_test)
    nova_mask = y_test == 1
    unknown_mask = y_test == 0
    if nova_mask.any():
        avg_nova_confidence = preds[nova_mask, 1].mean()
        print(f"Average confidence on true Nova clips: {avg_nova_confidence:.4f}")
    if unknown_mask.any():
        avg_unknown_confidence = preds[unknown_mask, 0].mean()
        print(f"Average confidence on true negative clips: {avg_unknown_confidence:.4f}")

    os.makedirs(MODEL_OUT_DIR, exist_ok=True)
    model.save(os.path.join(MODEL_OUT_DIR, MODEL_NAME))
    print(f"\nModel saved to {os.path.join(MODEL_OUT_DIR, MODEL_NAME)}")


if __name__ == "__main__":
    main()