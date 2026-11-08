"""
quantize_model.py

Converts the trained nova_ds_cnn.keras model to a quantized INT8
TFLite model suitable for deployment on the ESP32 (TFLite Micro).

Uses a representative dataset (drawn from your training data) for
full integer quantization, which is required for TFLite Micro and
gives the smallest, fastest model.

Usage:
    python quantize_model.py

Requires:
    pip install tensorflow numpy
"""

import os
import numpy as np
import tensorflow as tf

MODEL_PATH = os.path.join("models", "nova_ds_cnn.keras")
PROCESSED_DIR = os.path.join("dataset", "processed")
OUT_PATH = os.path.join("models", "nova_ds_cnn_int8.tflite")

NUM_REPRESENTATIVE_SAMPLES = 200


def representative_dataset_gen():
    X_train = np.load(os.path.join(PROCESSED_DIR, "X_train.npy"))
    X_train = X_train[..., np.newaxis].astype(np.float32)

    n = min(NUM_REPRESENTATIVE_SAMPLES, len(X_train))
    indices = np.random.choice(len(X_train), n, replace=False)

    for i in indices:
        sample = X_train[i:i+1]
        yield [sample]


def main():
    print(f"Loading model from {MODEL_PATH}...")
    model = tf.keras.models.load_model(MODEL_PATH)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Full integer quantization for TFLite Micro
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    print("Converting and quantizing to INT8...")
    tflite_model = converter.convert()

    os.makedirs("models", exist_ok=True)
    with open(OUT_PATH, "wb") as f:
        f.write(tflite_model)

    size_kb = len(tflite_model) / 1024
    print(f"\nSaved quantized model to {OUT_PATH}")
    print(f"Model size: {size_kb:.2f} KB")

    if size_kb > 256:
        print(f"WARNING: model size ({size_kb:.2f} KB) exceeds your 256KB RAM budget.")
        print("Consider reducing filters/blocks in the DS-CNN and retraining.")
    else:
        print(f"Model fits comfortably within your 256KB RAM budget.")


if __name__ == "__main__":
    main()