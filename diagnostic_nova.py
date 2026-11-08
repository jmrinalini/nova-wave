import os
import time
import threading
import numpy as np
import sounddevice as sd
import soundfile as sf
import librosa
import tensorflow as tf
import tkinter as tk
from tkinter import ttk


# ============================================================
# CONFIGURATION — MUST MATCH TRAINING
# ============================================================

MODEL_PATH = "models/nova_ds_cnn_int8.tflite"

SAMPLE_RATE = 16000
DURATION = 3.0
CLIP_LENGTH = int(SAMPLE_RATE * DURATION)

N_MFCC = 13
TARGET_FRAMES = 32

DEBUG_WAV = "debug_nova.wav"


# ============================================================
# LOAD TFLITE MODEL
# ============================================================

def load_model():

    interpreter = tf.lite.Interpreter(
        model_path=MODEL_PATH
    )

    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    return interpreter, input_details, output_details


# ============================================================
# MFCC — SAME AS evaluate.py
# ============================================================

def extract_mfcc(audio):

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCC
    )

    mfcc = mfcc.T

    # Make sure we have exactly 32 frames
    if mfcc.shape[0] > TARGET_FRAMES:

        mfcc = mfcc[:TARGET_FRAMES]

    elif mfcc.shape[0] < TARGET_FRAMES:

        pad = TARGET_FRAMES - mfcc.shape[0]

        mfcc = np.pad(
            mfcc,
            ((0, pad), (0, 0)),
            mode="constant"
        )

    return mfcc.astype(np.float32)


# ============================================================
# TFLITE CLASSIFICATION
# ============================================================

def classify(interpreter, input_details, output_details, mfcc):

    # Add batch + channel dimensions
    sample = mfcc[
        np.newaxis,
        ...,
        np.newaxis
    ].astype(np.float32)

    # --------------------------------------------------------
    # INPUT QUANTIZATION
    # --------------------------------------------------------

    input_scale, input_zero = input_details[0]["quantization"]

    if input_scale != 0:

        sample_quantized = (
            sample / input_scale
            + input_zero
        ).astype(
            input_details[0]["dtype"]
        )

    else:

        sample_quantized = sample.astype(
            input_details[0]["dtype"]
        )

    # --------------------------------------------------------
    # INFERENCE
    # --------------------------------------------------------

    start = time.perf_counter()

    interpreter.set_tensor(
        input_details[0]["index"],
        sample_quantized
    )

    interpreter.invoke()

    elapsed_ms = (
        time.perf_counter() - start
    ) * 1000

    output = interpreter.get_tensor(
        output_details[0]["index"]
    )[0]

    # --------------------------------------------------------
    # OUTPUT DEQUANTIZATION
    # --------------------------------------------------------

    output_scale, output_zero = \
        output_details[0]["quantization"]

    if output_scale != 0:

        output_float = (
            output.astype(np.float32)
            - output_zero
        ) * output_scale

    else:

        output_float = output.astype(
            np.float32
        )

    return output_float, elapsed_ms, sample_quantized


# ============================================================
# DASHBOARD
# ============================================================

class DiagnosticDashboard:

    def __init__(self, root):

        self.root = root

        root.title(
            "Nova — Diagnostic Real-Time Test"
        )

        root.geometry(
            "650x720"
        )

        root.configure(
            bg="#1e1e2e"
        )

        # ----------------------------------------------------
        # TITLE
        # ----------------------------------------------------

        tk.Label(
            root,
            text="NOVA REAL-TIME DIAGNOSTIC",
            font=("Segoe UI", 18, "bold"),
            bg="#1e1e2e",
            fg="white"
        ).pack(pady=(20, 5))

        self.status = tk.StringVar(
            value="Loading model..."
        )

        tk.Label(
            root,
            textvariable=self.status,
            font=("Segoe UI", 10),
            bg="#1e1e2e",
            fg="#aaaaaa"
        ).pack(pady=(0, 15))

        # ----------------------------------------------------
        # RECORD BUTTON
        # ----------------------------------------------------

        self.record_button = tk.Button(
            root,
            text="RECORD 1 SECOND",
            font=("Segoe UI", 12, "bold"),
            padx=25,
            pady=10,
            command=self.start_recording
        )

        self.record_button.pack(
            pady=10
        )

        # ----------------------------------------------------
        # RESULTS FRAME
        # ----------------------------------------------------

        frame = tk.Frame(
            root,
            bg="#2a2a3c"
        )

        frame.pack(
            fill="both",
            expand=True,
            padx=25,
            pady=15
        )

        self.output_box = tk.Text(
            frame,
            height=30,
            width=75,
            font=("Consolas", 10),
            bg="#151522",
            fg="white",
            relief="flat"
        )

        self.output_box.pack(
            fill="both",
            expand=True,
            padx=10,
            pady=10
        )

        # ----------------------------------------------------
        # LOAD MODEL
        # ----------------------------------------------------

        self.interpreter = None
        self.input_details = None
        self.output_details = None

        threading.Thread(
            target=self.load,
            daemon=True
        ).start()


    # ========================================================
    # LOAD
    # ========================================================

    def load(self):

        try:

            (
                self.interpreter,
                self.input_details,
                self.output_details
            ) = load_model()

            self.root.after(
                0,
                lambda: self.status.set(
                    "Model loaded — ready"
                )
            )

        except Exception as e:

            self.root.after(
                0,
                lambda: self.status.set(
                    f"MODEL ERROR: {e}"
                )
            )


    # ========================================================
    # RECORD
    # ========================================================

    def start_recording(self):

        if self.interpreter is None:

            self.status.set(
                "Model is still loading..."
            )

            return

        self.record_button.config(
            state="disabled"
        )

        threading.Thread(
            target=self.record_and_test,
            daemon=True
        ).start()


    # ========================================================
    # RECORD + TEST
    # ========================================================

    def record_and_test(self):

        try:

            self.root.after(
                0,
                lambda: self.status.set(
                    "RECORDING — SAY 'NOVA' NOW"
                )
            )

            # ------------------------------------------------
            # RECORD
            # ------------------------------------------------

            audio = sd.rec(
                CLIP_LENGTH,
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32"
            )

            sd.wait()

            audio = audio[:, 0]

            # ------------------------------------------------
            # SAVE RAW AUDIO
            # ------------------------------------------------

            sf.write(
                DEBUG_WAV,
                audio,
                SAMPLE_RATE
            )

            # ------------------------------------------------
            # AUDIO STATISTICS
            # ------------------------------------------------

            audio_min = float(
                np.min(audio)
            )

            audio_max = float(
                np.max(audio)
            )

            audio_mean = float(
                np.mean(audio)
            )

            audio_rms = float(
                np.sqrt(
                    np.mean(
                        audio ** 2
                    )
                )
            )

            # ------------------------------------------------
            # MFCC
            # ------------------------------------------------

            mfcc = extract_mfcc(
                audio
            )

            mfcc_min = float(
                np.min(mfcc)
            )

            mfcc_max = float(
                np.max(mfcc)
            )

            mfcc_mean = float(
                np.mean(mfcc)
            )

            # ------------------------------------------------
            # INPUT DETAILS
            # ------------------------------------------------

            input_info = self.input_details[0]

            input_shape = input_info["shape"]

            input_dtype = input_info["dtype"]

            input_scale, input_zero = \
                input_info["quantization"]

            # ------------------------------------------------
            # CLASSIFY
            # ------------------------------------------------

            probs, latency, sample_q = classify(
                self.interpreter,
                self.input_details,
                self.output_details,
                mfcc
            )

            negative = float(
                probs[0]
            )

            nova = float(
                probs[1]
            )

            # ------------------------------------------------
            # QUANTIZED INPUT STATISTICS
            # ------------------------------------------------

            q_min = int(
                np.min(sample_q)
            )

            q_max = int(
                np.max(sample_q)
            )

            q_mean = float(
                np.mean(sample_q)
            )

            # ------------------------------------------------
            # RESULT
            # ------------------------------------------------

            if nova > negative:

                result = "NOVA DETECTED"

            else:

                result = "NOT DETECTED"

            # ------------------------------------------------
            # DISPLAY
            # ------------------------------------------------

            text = f"""
============================================================
                  NOVA DIAGNOSTIC RESULT
============================================================

AUDIO
------------------------------------------------------------
Sample rate        : {SAMPLE_RATE} Hz
Duration            : {DURATION:.1f} sec
Samples             : {len(audio)}
Channels            : 1 (Mono)

Audio minimum      : {audio_min:.6f}
Audio maximum      : {audio_max:.6f}
Audio mean         : {audio_mean:.6f}
Audio RMS          : {audio_rms:.6f}

------------------------------------------------------------
MFCC
------------------------------------------------------------
MFCC shape         : {mfcc.shape}
MFCC count         : {N_MFCC}
MFCC minimum       : {mfcc_min:.4f}
MFCC maximum       : {mfcc_max:.4f}
MFCC mean          : {mfcc_mean:.4f}

------------------------------------------------------------
TFLITE INPUT
------------------------------------------------------------
Input shape        : {input_shape}
Input dtype        : {input_dtype}
Input scale        : {input_scale}
Input zero point   : {input_zero}

Quantized minimum  : {q_min}
Quantized maximum  : {q_max}
Quantized mean     : {q_mean:.2f}

------------------------------------------------------------
PREDICTION
------------------------------------------------------------
Negative           : {negative * 100:.2f}%
NOVA               : {nova * 100:.2f}%

------------------------------------------------------------
PERFORMANCE
------------------------------------------------------------
Inference latency  : {latency:.3f} ms

------------------------------------------------------------
RESULT
------------------------------------------------------------
                    {result}

------------------------------------------------------------
Saved microphone recording:
                    {DEBUG_WAV}
============================================================
"""

            self.root.after(
                0,
                lambda: self.show_result(text)
            )

        except Exception as e:

            self.root.after(
                0,
                lambda: self.show_result(
                    f"\nERROR:\n{e}\n"
                )
            )

        finally:

            self.root.after(
                0,
                lambda: self.record_button.config(
                    state="normal"
                )
            )

            self.root.after(
                0,
                lambda: self.status.set(
                    "Ready"
                )
            )


    # ========================================================
    # SHOW RESULT
    # ========================================================

    def show_result(self, text):

        self.output_box.delete(
            "1.0",
            tk.END
        )

        self.output_box.insert(
            tk.END,
            text
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    root = tk.Tk()

    app = DiagnosticDashboard(
        root
    )

    root.mainloop()