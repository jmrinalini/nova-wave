"""
test_tflite_accuracy.py

Diagnostic: runs the quantized nova_ds_cnn_int8.tflite model against
your existing X_test.npy / y_test.npy (the same held-out set that
scored 94.97% Nova confidence in the Keras model) and reports
accuracy + confidence. This isolates whether a problem is in the
TFLite quantization/conversion itself, or elsewhere (e.g. live mic
preprocessing).

Usage:
    python test_tflite_accuracy.py

Requires:
    pip install tensorflow numpy
"""

import os
import numpy as np
import tensorflow as tf

MODEL_PATH = os.path.join("models", "nova_ds_cnn_int8.tflite")
PROCESSED_DIR = os.path.join("dataset", "processed")


def main():
    print(f"Loading TFLite model from {MODEL_PATH}...")
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print("\n--- INPUT TENSOR DETAILS ---")
    print(input_details)
    print("\n--- OUTPUT TENSOR DETAILS ---")
    print(output_details)

    X_test = np.load(os.path.join(PROCESSED_DIR, "X_test.npy"))
    y_test = np.load(os.path.join(PROCESSED_DIR, "y_test.npy"))
    X_test = X_test[..., np.newaxis].astype(np.float32)

    in_scale, in_zero_point = input_details[0]["quantization"]
    out_scale, out_zero_point = output_details[0]["quantization"]
    print(f"\nInput quantization: scale={in_scale}, zero_point={in_zero_point}")
    print(f"Output quantization: scale={out_scale}, zero_point={out_zero_point}")

    correct = 0
    nova_confidences = []
    negative_confidences = []

    n_samples = len(X_test)
    print(f"\nRunning inference on {n_samples} test samples...")

    for i in range(n_samples):
        sample = X_test[i:i+1]

        if in_scale != 0:
            sample_q = (sample / in_scale + in_zero_point).astype(input_details[0]["dtype"])
        else:
            sample_q = sample.astype(input_details[0]["dtype"])

        interpreter.set_tensor(input_details[0]["index"], sample_q)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]["index"])[0]

        if out_scale != 0:
            probs = (output.astype(np.float32) - out_zero_point) * out_scale
        else:
            probs = output.astype(np.float32)

        pred_class = np.argmax(probs)
        true_class = y_test[i]

        if pred_class == true_class:
            correct += 1

        if true_class == 1:
            nova_confidences.append(probs[1])
        else:
            negative_confidences.append(probs[0])

    accuracy = correct / n_samples
    print(f"\n{'='*50}")
    print(f"TFLite model accuracy on test set: {accuracy:.4f}")
    if nova_confidences:
        print(f"Average confidence on true Nova clips: {np.mean(nova_confidences):.4f}")
    if negative_confidences:
        print(f"Average confidence on true negative clips: {np.mean(negative_confidences):.4f}")
    print(f"{'='*50}")

    print("\nFirst 5 raw output examples (for debugging):")
    for i in range(min(5, n_samples)):
        sample = X_test[i:i+1]
        if in_scale != 0:
            sample_q = (sample / in_scale + in_zero_point).astype(input_details[0]["dtype"])
        else:
            sample_q = sample.astype(input_details[0]["dtype"])
        interpreter.set_tensor(input_details[0]["index"], sample_q)
        interpreter.invoke()
        raw_output = interpreter.get_tensor(output_details[0]["index"])[0]
        print(f"  Sample {i}: true_label={y_test[i]}, raw_output={raw_output}")


if __name__ == "__main__":
    main()