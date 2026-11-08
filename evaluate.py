import os
import random
import numpy as np
import librosa
from sklearn.model_selection import train_test_split


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = "dataset"

# Nova recordings from Edge Impulse
NOVA_TRAIN_DIR = os.path.join(BASE_DIR, "raw", "training")
NOVA_TEST_DIR = os.path.join(BASE_DIR, "raw", "testing")

# Google Speech Commands
GOOGLE_DIR = os.path.join(BASE_DIR, "external")

# Google background noise
BACKGROUND_DIR = os.path.join(GOOGLE_DIR, "_background_noise_")

# Output
OUT_DIR = os.path.join(BASE_DIR, "processed")

# Audio configuration
SAMPLE_RATE = 16000
CLIP_DURATION_SEC = 1.0
CLIP_LENGTH = int(SAMPLE_RATE * CLIP_DURATION_SEC)

# MFCC configuration
N_MFCC = 13

# Validation split for Nova training + Google training data
VAL_SPLIT = 0.15

# Maximum number of negative samples
#
# We have 1018 Nova recordings.
# Around 1800 negative samples gives a reasonable
# positive/negative ratio without making the dataset huge.
MAX_GOOGLE_NEGATIVES = 1500
MAX_BACKGROUND_NEGATIVES = 300

# Number of background-noise clips generated per noise WAV
BACKGROUND_CLIPS_PER_FILE = 5

RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ============================================================
# LOAD AUDIO
# ============================================================

def load_audio(filepath):
    """
    Load audio as:
        16 kHz
        mono
        exactly 1 second
    """

    audio, sr = librosa.load(
        filepath,
        sr=SAMPLE_RATE,
        mono=True
    )

    # Trim if longer than 1 second
    if len(audio) > CLIP_LENGTH:
        audio = audio[:CLIP_LENGTH]

    # Pad if shorter than 1 second
    elif len(audio) < CLIP_LENGTH:
        audio = np.pad(
            audio,
            (0, CLIP_LENGTH - len(audio)),
            mode="constant"
        )

    return audio


def load_background_random_segment(filepath, rng):
    """
    Load a random 1-second segment from a background-noise WAV.

    This is useful because background-noise files are usually
    much longer than 1 second.
    """

    audio, sr = librosa.load(
        filepath,
        sr=SAMPLE_RATE,
        mono=True
    )

    if len(audio) <= CLIP_LENGTH:
        audio = np.pad(
            audio,
            (0, max(0, CLIP_LENGTH - len(audio))),
            mode="constant"
        )

        return audio[:CLIP_LENGTH]

    max_start = len(audio) - CLIP_LENGTH

    start = rng.randint(0, max_start)

    return audio[start:start + CLIP_LENGTH]


# ============================================================
# MFCC
# ============================================================

def extract_mfcc(audio):
    """
    Convert 1-second audio into MFCC features.

    Output:
        (time_frames, 13)
        normally around (32, 13)
    """

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCC
    )

    return mfcc.T


# ============================================================
# FIND WAV FILES
# ============================================================

def find_wav_files(folder):
    """
    Recursively find all WAV files inside a folder.
    """

    wav_files = []

    if not os.path.isdir(folder):
        return wav_files

    for root, dirs, files in os.walk(folder):

        for fname in files:

            if fname.lower().endswith(".wav"):

                wav_files.append(
                    os.path.join(root, fname)
                )

    return wav_files


# ============================================================
# LOAD NOVA FILES
# ============================================================

def process_nova_folder(folder, label=1):

    features = []
    labels = []
    skipped = []

    if not os.path.isdir(folder):

        print(f"WARNING: folder not found:")
        print(f"  {folder}")

        return features, labels, skipped

    files = find_wav_files(folder)

    print(f"Found {len(files)} WAV files in:")
    print(f"  {folder}")

    for filepath in files:

        try:

            audio = load_audio(filepath)

            mfcc = extract_mfcc(audio)

            features.append(mfcc)

            labels.append(label)

        except Exception as e:

            skipped.append(
                (filepath, str(e))
            )

    return features, labels, skipped


# ============================================================
# LOAD GOOGLE SPEECH COMMANDS
# ============================================================

def process_google_commands(max_samples):

    """
    Load negative examples from Google Speech Commands.

    Every command except _background_noise_ is treated as
    NOT NOVA.

    Label:
        0 = negative
    """

    features = []
    labels = []
    skipped = []

    if not os.path.isdir(GOOGLE_DIR):

        print("WARNING: Google Speech Commands folder not found:")
        print(f"  {GOOGLE_DIR}")

        return features, labels, skipped

    all_files = []

    # Go through command folders
    for folder_name in os.listdir(GOOGLE_DIR):

        folder_path = os.path.join(
            GOOGLE_DIR,
            folder_name
        )

        # Ignore files
        if not os.path.isdir(folder_path):
            continue

        # Background noise is processed separately
        if folder_name == "_background_noise_":
            continue

        wavs = find_wav_files(folder_path)

        all_files.extend(wavs)

    print(
        f"Found {len(all_files)} Google Speech Commands WAV files."
    )

    # Shuffle deterministically
    random.shuffle(all_files)

    selected_files = all_files[:max_samples]

    print(
        f"Using {len(selected_files)} Google files as negative samples."
    )

    for filepath in selected_files:

        try:

            audio = load_audio(filepath)

            mfcc = extract_mfcc(audio)

            features.append(mfcc)

            labels.append(0)

        except Exception as e:

            skipped.append(
                (filepath, str(e))
            )

    return features, labels, skipped


# ============================================================
# LOAD BACKGROUND NOISE
# ============================================================

def process_background_noise(max_samples):

    """
    Generate negative examples from the Google
    _background_noise_ directory.
    """

    features = []
    labels = []
    skipped = []

    if not os.path.isdir(BACKGROUND_DIR):

        print("WARNING: background noise folder not found:")
        print(f"  {BACKGROUND_DIR}")

        return features, labels, skipped

    files = find_wav_files(BACKGROUND_DIR)

    print(
        f"Found {len(files)} background-noise WAV files."
    )

    if len(files) == 0:

        return features, labels, skipped

    rng = random.Random(RANDOM_SEED)

    # Generate clips until requested number is reached
    count = 0

    while count < max_samples:

        filepath = rng.choice(files)

        try:

            audio = load_background_random_segment(
                filepath,
                rng
            )

            mfcc = extract_mfcc(audio)

            features.append(mfcc)

            labels.append(0)

            count += 1

        except Exception as e:

            skipped.append(
                (filepath, str(e))
            )

            # Avoid infinite loop
            if len(skipped) > 20:
                break

    print(
        f"Generated {len(features)} background-noise "
        f"negative samples."
    )

    return features, labels, skipped


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("        NOVA KEYWORD SPOTTING DATASET BUILDER")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. LOAD NOVA TRAINING
    # --------------------------------------------------------

    print("\n[1] Loading NOVA training recordings...")

    nova_train_features, nova_train_labels, skipped_nova_train = \
        process_nova_folder(
            NOVA_TRAIN_DIR,
            label=1
        )

    # --------------------------------------------------------
    # 2. LOAD NOVA TEST
    # --------------------------------------------------------

    print("\n[2] Loading NOVA test recordings...")

    nova_test_features, nova_test_labels, skipped_nova_test = \
        process_nova_folder(
            NOVA_TEST_DIR,
            label=1
        )

    # --------------------------------------------------------
    # 3. LOAD GOOGLE SPEECH COMMANDS
    # --------------------------------------------------------

    print("\n[3] Loading Google Speech Commands...")

    google_features, google_labels, skipped_google = \
        process_google_commands(
            MAX_GOOGLE_NEGATIVES
        )

    # --------------------------------------------------------
    # 4. LOAD BACKGROUND NOISE
    # --------------------------------------------------------

    print("\n[4] Loading background noise...")

    background_features, background_labels, skipped_background = \
        process_background_noise(
            MAX_BACKGROUND_NEGATIVES
        )

    # --------------------------------------------------------
    # DATA SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("RAW DATA SUMMARY")
    print("=" * 70)

    print(
        f"Nova training        : {len(nova_train_features)}"
    )

    print(
        f"Nova test            : {len(nova_test_features)}"
    )

    print(
        f"Google negatives     : {len(google_features)}"
    )

    print(
        f"Background negatives : {len(background_features)}"
    )

    total_nova = (
        len(nova_train_features)
        + len(nova_test_features)
    )

    total_negative = (
        len(google_features)
        + len(background_features)
    )

    print(
        f"\nTotal Nova           : {total_nova}"
    )

    print(
        f"Total negative       : {total_negative}"
    )

    # --------------------------------------------------------
    # CHECK DATA
    # --------------------------------------------------------

    if total_nova == 0:

        print("\nERROR: No Nova recordings found.")

        return

    if total_negative == 0:

        print("\nERROR: No negative recordings found.")

        return

    # --------------------------------------------------------
    # CONVERT TO NUMPY
    # --------------------------------------------------------

    X_nova_train = np.array(
        nova_train_features,
        dtype=np.float32
    )

    y_nova_train = np.array(
        nova_train_labels,
        dtype=np.int64
    )

    X_nova_test = np.array(
        nova_test_features,
        dtype=np.float32
    )

    y_nova_test = np.array(
        nova_test_labels,
        dtype=np.int64
    )

    X_google = np.array(
        google_features,
        dtype=np.float32
    )

    y_google = np.array(
        google_labels,
        dtype=np.int64
    )

    X_background = np.array(
        background_features,
        dtype=np.float32
    )

    y_background = np.array(
        background_labels,
        dtype=np.int64
    )

    # --------------------------------------------------------
    # COMBINE ALL NEGATIVES
    # --------------------------------------------------------

    X_negative = np.concatenate(
        [
            X_google,
            X_background
        ],
        axis=0
    )

    y_negative = np.concatenate(
        [
            y_google,
            y_background
        ],
        axis=0
    )

    # --------------------------------------------------------
    # SPLIT NOVA TRAINING INTO TRAIN / VALIDATION
    # --------------------------------------------------------

    print("\n[5] Creating training / validation split...")

    X_nova_train_final, X_nova_val, \
    y_nova_train_final, y_nova_val = train_test_split(
        X_nova_train,
        y_nova_train,
        test_size=VAL_SPLIT,
        random_state=RANDOM_SEED,
        shuffle=True
    )

    # --------------------------------------------------------
    # SPLIT NEGATIVES
    #
    # 70% train
    # 15% validation
    # 15% test
    # --------------------------------------------------------

    print("[6] Splitting negative samples...")

    X_neg_temp, X_neg_test, \
    y_neg_temp, y_neg_test = train_test_split(
        X_negative,
        y_negative,
        test_size=0.15,
        random_state=RANDOM_SEED,
        shuffle=True
    )

    X_neg_train, X_neg_val, \
    y_neg_train, y_neg_val = train_test_split(
        X_neg_temp,
        y_neg_temp,
        test_size=0.15 / 0.85,
        random_state=RANDOM_SEED,
        shuffle=True
    )

    # --------------------------------------------------------
    # BUILD FINAL TRAIN SET
    # --------------------------------------------------------

    X_train = np.concatenate(
        [
            X_nova_train_final,
            X_neg_train
        ],
        axis=0
    )

    y_train = np.concatenate(
        [
            y_nova_train_final,
            y_neg_train
        ],
        axis=0
    )

    # --------------------------------------------------------
    # BUILD FINAL VALIDATION SET
    # --------------------------------------------------------

    X_val = np.concatenate(
        [
            X_nova_val,
            X_neg_val
        ],
        axis=0
    )

    y_val = np.concatenate(
        [
            y_nova_val,
            y_neg_val
        ],
        axis=0
    )

    # --------------------------------------------------------
    # BUILD FINAL TEST SET
    #
    # Keep ALL Edge Impulse Nova test recordings untouched.
    # Add Google/background negative test samples.
    # --------------------------------------------------------

    X_test = np.concatenate(
        [
            X_nova_test,
            X_neg_test
        ],
        axis=0
    )

    y_test = np.concatenate(
        [
            y_nova_test,
            y_neg_test
        ],
        axis=0
    )

    # --------------------------------------------------------
    # SHUFFLE EACH FINAL SET
    # --------------------------------------------------------

    def shuffle_dataset(X, y):

        indices = np.arange(len(X))

        rng = np.random.default_rng(
            RANDOM_SEED
        )

        rng.shuffle(indices)

        return X[indices], y[indices]

    X_train, y_train = shuffle_dataset(
        X_train,
        y_train
    )

    X_val, y_val = shuffle_dataset(
        X_val,
        y_val
    )

    X_test, y_test = shuffle_dataset(
        X_test,
        y_test
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    os.makedirs(
        OUT_DIR,
        exist_ok=True
    )

    np.save(
        os.path.join(
            OUT_DIR,
            "X_train.npy"
        ),
        X_train
    )

    np.save(
        os.path.join(
            OUT_DIR,
            "y_train.npy"
        ),
        y_train
    )

    np.save(
        os.path.join(
            OUT_DIR,
            "X_val.npy"
        ),
        X_val
    )

    np.save(
        os.path.join(
            OUT_DIR,
            "y_val.npy"
        ),
        y_val
    )

    np.save(
        os.path.join(
            OUT_DIR,
            "X_test.npy"
        ),
        X_test
    )

    np.save(
        os.path.join(
            OUT_DIR,
            "y_test.npy"
        ),
        y_test
    )

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL DATASET")
    print("=" * 70)

    print("\nTRAIN")
    print(
        f"  Shape    : {X_train.shape}"
    )
    print(
        f"  Nova     : {np.sum(y_train == 1)}"
    )
    print(
        f"  Negative : {np.sum(y_train == 0)}"
    )

    print("\nVALIDATION")
    print(
        f"  Shape    : {X_val.shape}"
    )
    print(
        f"  Nova     : {np.sum(y_val == 1)}"
    )
    print(
        f"  Negative : {np.sum(y_val == 0)}"
    )

    print("\nTEST")
    print(
        f"  Shape    : {X_test.shape}"
    )
    print(
        f"  Nova     : {np.sum(y_test == 1)}"
    )
    print(
        f"  Negative : {np.sum(y_test == 0)}"
    )

    print("\n" + "=" * 70)
    print("FEATURE INFORMATION")
    print("=" * 70)

    print(
        f"Sample rate : {SAMPLE_RATE} Hz"
    )

    print(
        f"Audio       : {CLIP_DURATION_SEC} second"
    )

    print(
        "Channels    : Mono"
    )

    print(
        f"MFCC        : {N_MFCC}"
    )

    print(
        f"Feature shape: {X_train.shape[1:]}"
    )

    print("\nSaved files:")

    print(
        "  dataset/processed/X_train.npy"
    )

    print(
        "  dataset/processed/y_train.npy"
    )

    print(
        "  dataset/processed/X_val.npy"
    )

    print(
        "  dataset/processed/y_val.npy"
    )

    print(
        "  dataset/processed/X_test.npy"
    )

    print(
        "  dataset/processed/y_test.npy"
    )

    # --------------------------------------------------------
    # SKIPPED FILES
    # --------------------------------------------------------

    all_skipped = (
        skipped_nova_train
        + skipped_nova_test
        + skipped_google
        + skipped_background
    )

    if all_skipped:

        print("\n" + "=" * 70)
        print("SKIPPED FILES")
        print("=" * 70)

        print(
            f"Total skipped: {len(all_skipped)}"
        )

        for filepath, error in all_skipped[:10]:

            print(f"\n{filepath}")
            print(f"  {error}")

        if len(all_skipped) > 10:

            print(
                f"\n... and "
                f"{len(all_skipped) - 10} more."
            )

    print("\n" + "=" * 70)
    print("DATASET BUILD COMPLETE")
    print("=" * 70)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()