import numpy as np
import librosa
import tensorflow as tf

AUDIO_FILE = "debug_nova.wav"
MODEL_FILE = "models/nova_ds_cnn_int8.tflite"
SAMPLE_RATE = 16000
DURATION = 1
NUM_SAMPLES = SAMPLE_RATE * DURATION
N_MFCC = 13


# ---------------------------------------------------------
# 1. Load audio exactly like training
# ---------------------------------------------------------

audio, sr = librosa.load(
    AUDIO_FILE,
    sr=16000,
    mono=True
)

print("\n========== AUDIO ==========")
print("Sample rate:", sr)
print("Original samples:", len(audio))
print("Original duration:", len(audio) / sr)

# Pad or truncate exactly as training
if len(audio) < NUM_SAMPLES:
    audio = np.pad(
        audio,
        (0, NUM_SAMPLES - len(audio))
    )
elif len(audio) > NUM_SAMPLES:
    audio = audio[:NUM_SAMPLES]

print("Samples used by model:", len(audio))
print("Duration used by model:", len(audio) / sr)


# ---------------------------------------------------------
# 2. Extract MFCC exactly as training
# ---------------------------------------------------------

mfcc = librosa.feature.mfcc(
    y=audio,
    sr=16000,
    n_mfcc=13
)

print("\n========== MFCC ==========")
print("MFCC shape before transpose:", mfcc.shape)
print("MFCC min:", mfcc.min())
print("MFCC max:", mfcc.max())
print("MFCC mean:", mfcc.mean())
print("MFCC std:", mfcc.std())


# ---------------------------------------------------------
# 3. Convert to model shape
# ---------------------------------------------------------

# librosa gives:
# (13, 32)
#
# Model expects:
# (32, 13, 1)

features = mfcc.T

print("MFCC shape after transpose:", features.shape)

features = features.astype(np.float32)

features = np.expand_dims(features, axis=-1)

features = np.expand_dims(features, axis=0)

print("Final model input shape:", features.shape)


# ---------------------------------------------------------
# 4. Load TFLite model
# ---------------------------------------------------------

interpreter = tf.lite.Interpreter(
    model_path=MODEL_FILE
)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("\n========== TFLITE INPUT ==========")

print("Input shape:", input_details[0]["shape"])
print("Input dtype:", input_details[0]["dtype"])

input_scale, input_zero_point = \
    input_details[0]["quantization"]

print("Input quantization scale:",
      input_scale)

print("Input quantization zero point:",
      input_zero_point)


# ---------------------------------------------------------
# 5. Quantize MFCC input for INT8 model
# ---------------------------------------------------------

if input_details[0]["dtype"] == np.int8:

    quantized_input = (
        features / input_scale
        + input_zero_point
    )

    quantized_input = np.round(
        quantized_input
    )

    quantized_input = np.clip(
        quantized_input,
        -128,
        127
    )

    quantized_input = \
        quantized_input.astype(np.int8)

else:

    quantized_input = features.astype(
        input_details[0]["dtype"]
    )


print("\n========== QUANTIZED INPUT ==========")

print("Quantized input min:",
      quantized_input.min())

print("Quantized input max:",
      quantized_input.max())

print("Quantized input mean:",
      quantized_input.mean())


# ---------------------------------------------------------
# 6. Run TFLite inference
# ---------------------------------------------------------

interpreter.set_tensor(
    input_details[0]["index"],
    quantized_input
)

interpreter.invoke()


# ---------------------------------------------------------
# 7. Get raw TFLite output
# ---------------------------------------------------------

raw_output = interpreter.get_tensor(
    output_details[0]["index"]
)

print("\n========== RAW TFLITE OUTPUT ==========")
print(raw_output)

print("Output dtype:",
      raw_output.dtype)

print("Output shape:",
      raw_output.shape)


# ---------------------------------------------------------
# 8. Dequantize output
# ---------------------------------------------------------

output_scale, output_zero_point = \
    output_details[0]["quantization"]

print("\n========== OUTPUT QUANTIZATION ==========")

print("Output scale:",
      output_scale)

print("Output zero point:",
      output_zero_point)


if output_details[0]["dtype"] == np.int8:

    dequantized_output = (
        raw_output.astype(np.float32)
        - output_zero_point
    ) * output_scale

else:

    dequantized_output = \
        raw_output.astype(np.float32)


print("\n========== DEQUANTIZED OUTPUT ==========")
print(dequantized_output)


# ---------------------------------------------------------
# 9. Probabilities
# ---------------------------------------------------------

values = dequantized_output[0]

print("\n========== PREDICTION ==========")

print("Class 0:", values[0])
print("Class 1:", values[1])

print("\nIMPORTANT:")
print("You must confirm which class index corresponds to NOVA.")
print("If class 0 = Nova:")

print("Nova probability:", values[0])
print("Negative probability:", values[1])

print("\nIf class 1 = Nova:")

print("Negative probability:", values[0])
print("Nova probability:", values[1])