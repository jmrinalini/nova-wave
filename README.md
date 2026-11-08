\# Nova Wave – TinyML Keyword Spotting System



Nova Wave is an AI-powered keyword spotting system designed to recognize the wake word \*\*"Nova"\*\* using a lightweight Deep Learning model.



The project uses a Depthwise Separable Convolutional Neural Network (DS-CNN) to perform keyword detection and TensorFlow Lite INT8 quantization for lightweight model deployment.



\## Features



\- Keyword spotting using a DS-CNN architecture.

\- Audio feature extraction using Mel-Frequency Cepstral Coefficients (MFCC).

\- Lightweight TensorFlow Lite INT8 model.

\- Real-time microphone-based audio testing.

\- Model evaluation and diagnostic scripts.

\- Desktop dashboard for interaction.



\## System Architecture



1\. \*\*Audio Input:\*\* Captures audio through a microphone.

2\. \*\*Preprocessing:\*\* Converts audio into suitable input features.

3\. \*\*Feature Extraction:\*\* Extracts MFCC features.

4\. \*\*Deep Learning:\*\* DS-CNN classifies the input audio.

5\. \*\*Quantization:\*\* Converts the trained model into TensorFlow Lite INT8 format.

6\. \*\*Inference:\*\* Predicts whether the keyword "Nova" is detected.



\## Technologies Used



\- Python

\- TensorFlow

\- Keras

\- NumPy

\- Librosa

\- Scikit-learn

\- TensorFlow Lite

\- SoundDevice

\- SoundFile

\- Tkinter



\## Project Structure



```text

nova-wave/

├── models/

│   ├── nova\_ds\_cnn.keras

│   └── nova\_ds\_cnn\_int8.tflite

├── diagnostic\_nova.py

├── evaluate.py

├── nova\_dashboard.py

├── quantize\_model.py

├── realtime\_test.py

├── test\_nova\_tflite.py

├── tflite\_accuracy.py

├── Train ds cnn small.py

├── requirements.txt

├── .gitignore

└── README.md

```



The dataset is excluded from this repository and is maintained separately.



\## Installation



Clone the repository:



```bash

git clone YOUR\_GITHUB\_REPOSITORY\_URL

cd nova-wave

```



Create a virtual environment:



```bash

python -m venv venv

```



Activate it on Windows:



```bash

venv\\Scripts\\activate

```



Install dependencies:



```bash

pip install -r requirements.txt

```



\## Running the Project



The repository includes scripts for:



\- Model training

\- Model evaluation

\- INT8 quantization

\- TensorFlow Lite testing

\- Real-time microphone testing

\- Diagnostic analysis

\- Desktop dashboard



Run the corresponding Python script for the required operation.



\## Models



| Model | Description |

|---|---|

| `nova\_ds\_cnn.keras` | Trained DS-CNN model |

| `nova\_ds\_cnn\_int8.tflite` | Quantized TensorFlow Lite model |



\## Model Performance



The DS-CNN model achieved \*\*98.97% test accuracy\*\* in the reported evaluation.



\*Note: This is the previously measured test accuracy. Actual real-time performance may vary depending on microphone input and environmental noise.\*



\## Future Improvements



\- Improve real-time keyword detection reliability.

\- Optimize inference for low-power edge devices.

\- Integrate with embedded hardware.

\- Improve robustness under background noise.

\- Explore always-on TinyML deployment.



\## License



This project is available for educational and research purposes.



