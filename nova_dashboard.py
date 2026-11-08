"""
nova_dashboard.py

A small live dashboard for demoing the Nova wake-word model.
Click "Record & Test", speak for 1 second, and see the confidence
for each class update in real time, with a history log below.

Uses the same working press-to-record pipeline as test_single_shot.py
(confirmed accurate against your test set), just wrapped in a GUI.

Usage:
    python nova_dashboard.py

Requires:
    pip install tensorflow numpy sounddevice librosa
"""

import threading
import time
import numpy as np
import sounddevice as sd
import librosa
import tensorflow as tf
import tkinter as tk
from tkinter import ttk

MODEL_PATH = "models/nova_ds_cnn_int8.tflite"
SAMPLE_RATE = 16000
CLIP_LENGTH = SAMPLE_RATE * 1  # 1 second
N_MFCC = 13
TARGET_FRAMES = 32


def load_model():
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    return interpreter, interpreter.get_input_details(), interpreter.get_output_details()


def extract_mfcc(audio):
    mfcc = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=N_MFCC).T
    if mfcc.shape[0] > TARGET_FRAMES:
        mfcc = mfcc[:TARGET_FRAMES]
    elif mfcc.shape[0] < TARGET_FRAMES:
        pad = TARGET_FRAMES - mfcc.shape[0]
        mfcc = np.pad(mfcc, ((0, pad), (0, 0)), mode="constant")
    return mfcc


def classify(interpreter, input_details, output_details, mfcc):
    sample = mfcc[np.newaxis, ..., np.newaxis].astype(np.float32)
    in_scale, in_zero_point = input_details[0]["quantization"]
    sample_q = (sample / in_scale + in_zero_point).astype(input_details[0]["dtype"]) \
        if in_scale != 0 else sample.astype(input_details[0]["dtype"])

    interpreter.set_tensor(input_details[0]["index"], sample_q)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]["index"])[0]

    out_scale, out_zero_point = output_details[0]["quantization"]
    probs = (output.astype(np.float32) - out_zero_point) * out_scale \
        if out_scale != 0 else output.astype(np.float32)
    return probs  # [prob_negative, prob_nova]


class NovaDashboard:
    def __init__(self, root):
        self.root = root
        root.title("Nova - Wake Word Dashboard")
        root.geometry("420x480")
        root.configure(bg="#1e1e2e")

        self.status_var = tk.StringVar(value="Loading model...")
        tk.Label(root, text="Nova Wake Word Tester", font=("Segoe UI", 16, "bold"),
                 bg="#1e1e2e", fg="white").pack(pady=(20, 5))

        tk.Label(root, textvariable=self.status_var, font=("Segoe UI", 9),
                 bg="#1e1e2e", fg="#a0a0a0").pack(pady=(0, 15))

        self.record_btn = tk.Button(
            root, text="Record (1s) and Test", font=("Segoe UI", 12, "bold"),
            bg="#3b82f6", fg="white", activebackground="#2563eb",
            relief="flat", padx=20, pady=10, command=self.on_record
        )
        self.record_btn.pack(pady=10)

        bars_frame = tk.Frame(root, bg="#1e1e2e")
        bars_frame.pack(pady=15, fill="x", padx=30)

        self.nova_label = tk.Label(bars_frame, text="hey_nova: 0%", font=("Segoe UI", 11),
                                    bg="#1e1e2e", fg="white", anchor="w")
        self.nova_label.pack(fill="x", pady=3)
        self.nova_bar = ttk.Progressbar(bars_frame, length=350, maximum=100)
        self.nova_bar.pack(pady=(0, 10))

        self.neg_label = tk.Label(bars_frame, text="not_nova: 0%", font=("Segoe UI", 11),
                                   bg="#1e1e2e", fg="white", anchor="w")
        self.neg_label.pack(fill="x", pady=3)
        self.neg_bar = ttk.Progressbar(bars_frame, length=350, maximum=100)
        self.neg_bar.pack(pady=(0, 10))

        tk.Label(root, text="History", font=("Segoe UI", 11, "bold"),
                 bg="#1e1e2e", fg="white").pack(pady=(15, 5))

        self.history_box = tk.Listbox(root, height=8, font=("Consolas", 9),
                                       bg="#2a2a3c", fg="white", relief="flat")
        self.history_box.pack(fill="both", expand=True, padx=20, pady=(0, 20))

        self.interpreter = None
        self.input_details = None
        self.output_details = None
        threading.Thread(target=self.load, daemon=True).start()

    def load(self):
        self.interpreter, self.input_details, self.output_details = load_model()
        self.status_var.set("Model loaded. Ready.")

    def on_record(self):
        if self.interpreter is None:
            self.status_var.set("Model still loading, wait a moment...")
            return
        self.record_btn.config(state="disabled")
        threading.Thread(target=self.record_and_classify, daemon=True).start()

    def record_and_classify(self):
        self.status_var.set("Recording... speak now!")
        audio = sd.rec(CLIP_LENGTH, samplerate=SAMPLE_RATE, channels=1, dtype="float32")
        sd.wait()
        audio = audio[:, 0]

        self.status_var.set("Classifying...")
        mfcc = extract_mfcc(audio)
        probs = classify(self.interpreter, self.input_details, self.output_details, mfcc)

        neg_pct = probs[0] * 100
        nova_pct = probs[1] * 100

        self.nova_bar["value"] = nova_pct
        self.nova_label.config(text=f"hey_nova: {nova_pct:.1f}%")
        self.neg_bar["value"] = neg_pct
        self.neg_label.config(text=f"not_nova: {neg_pct:.1f}%")

        result = "NOVA DETECTED" if nova_pct > neg_pct else "not detected"
        timestamp = time.strftime("%H:%M:%S")
        self.history_box.insert(0, f"[{timestamp}] {result}  (nova={nova_pct:.1f}%, other={neg_pct:.1f}%)")

        self.status_var.set("Ready.")
        self.record_btn.config(state="normal")


if __name__ == "__main__":
    root = tk.Tk()
    app = NovaDashboard(root)
    root.mainloop()