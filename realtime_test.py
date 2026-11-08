"""
realtime_test.py

Listens through your laptop microphone in real time and runs the
quantized "Nova" model on rolling 1-second windows, printing a
detection whenever confidence crosses the threshold.

This is a stand-in for the ESP32 hardware demo -- same model,
same MFCC pipeline, just running on your PC mic instead of the
INMP441 + ESP32-S3.

Usage:
    python realtime_test.py

Requires:
    pip install tensorflow numpy sounddevice librosa
"""

import time
import numpy as np
import sounddevice as sd
import librosa
import tensorflow as tf

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
MODEL_PATH = "models/nova_ds_cnn_int8.tflite"

SAMPLE_RATE = 16000
CLIP_LENGTH = SAMPLE_RATE * 1  # 1 second window
N_MFCC = 13

# How often to check (seconds). 0.5 = check twice a second using a
# sliding window, giving faster response than waiting a full second.
CHECK_INTERVAL = 0.5

CONFIDENCE_THRESHOLD = 0.80  # tune this after watching real output


def load_tflite_model(path):
    interpreter = tf.lite.Interpreter(model_path=path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return interpreter, input_details, output_details


def extract_mfcc(audio):
    mfcc = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=N_MFCC)
    return mfcc.T  # (time_frames, N_MFCC)


def quantize_input(mfcc, input_details):
    """Convert float32 MFCC features to the int8 range the model expects."""
    scale, zero_point = input_details[0]["quantization"]
    if scale == 0:
        # Model wasn't actually quantized on the input side; just cast.
        return mfcc.astype(np.int8)
    quantized = (mfcc / scale + zero_point).astype(np.int8)
    return quantized


def dequantize_output(output, output_details):
    scale, zero_point = output_details[0]["quantization"]
    if scale == 0:
        return output.astype(np.float32)
    return (output.astype(np.float32) - zero_point) * scale


def run_inference(interpreter, input_details, output_details, mfcc):
    mfcc_input = mfcc[np.newaxis, ..., np.newaxis]  # (1, time, mfcc, 1)
    mfcc_input = quantize_input(mfcc_input, input_details)

    interpreter.set_tensor(input_details[0]["index"], mfcc_input)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]["index"])[0]

    probs = dequantize_output(output, output_details)
    return probs  # [prob_negative, prob_nova] assuming class order 0,1


def main():
    print(f"Loading model from {MODEL_PATH}...")
    interpreter, input_details, output_details = load_tflite_model(MODEL_PATH)
    print("Model loaded. Listening... (Ctrl+C to stop)\n")

    audio_buffer = np.zeros(CLIP_LENGTH, dtype=np.float32)

    def callback(indata, frames, time_info, status):
        nonlocal audio_buffer
        new_audio = indata[:, 0]
        audio_buffer = np.roll(audio_buffer, -len(new_audio))
        audio_buffer[-len(new_audio):] = new_audio

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        callback=callback,
        blocksize=int(SAMPLE_RATE * CHECK_INTERVAL),
    )

    with stream:
        try:
            while True:
                time.sleep(CHECK_INTERVAL)
                mfcc = extract_mfcc(audio_buffer.copy())

                # Pad/trim MFCC time-frame dimension to match training shape
                # (adjust target_frames if your training shape differs)
                target_frames = 32
                if mfcc.shape[0] > target_frames:
                    mfcc = mfcc[:target_frames]
                elif mfcc.shape[0] < target_frames:
                    pad = target_frames - mfcc.shape[0]
                    mfcc = np.pad(mfcc, ((0, pad), (0, 0)), mode="constant")

                probs = run_inference(interpreter, input_details, output_details, mfcc)
                nova_confidence = probs[1] if len(probs) > 1 else probs[0]

                bar = "#" * int(nova_confidence * 30)
                print(f"\rNova confidence: {nova_confidence:.3f} {bar:<30}", end="")

                if nova_confidence >= CONFIDENCE_THRESHOLD:
                    print(f"\n>>> NOVA DETECTED (confidence: {nova_confidence:.3f}) <<<\n")

        except KeyboardInterrupt:
            print("\n\nStopped listening.")


if __name__ == "__main__":
    main()