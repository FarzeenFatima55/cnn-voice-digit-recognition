# CNN Voice Digit Recognition

A Convolutional Neural Network (CNN) that recognizes spoken digits (0–9) from audio. Audio is converted into log-mel spectrograms and classified by a small CNN built with TensorFlow/Keras.

The model is trained on the **Google Speech Commands v0.02** dataset and also tested on the **Free Spoken Digit Dataset (FSDD)** to measure cross-dataset generalization.

## Features

- Speaker-independent train/validation/test split (official `validation_list.txt` and `testing_list.txt`)
- Log-mel spectrogram features (40 mel bands, 1 second at 16 kHz, input shape `40 x 101 x 1`)
- Lightweight CNN (3 Conv blocks + BatchNorm + Global Average Pooling + Dropout)
- Trained with 3 different random seeds (0, 1, 2) for a mean ± std result
- Cross-dataset evaluation on FSDD (8 kHz audio resampled to 16 kHz)
- Live microphone demo and audio-file upload demo (Google Colab)

## Datasets

| Dataset | Use | Link |
|---|---|---|
| Speech Commands v0.02 | Training, validation, testing | http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz |
| Free Spoken Digit Dataset (FSDD) | Cross-dataset evaluation | https://github.com/Jakobovski/free-spoken-digit-dataset |

The datasets are **not** included in this repository. The notebook downloads them automatically.

## Model Architecture

```
Input (40, 101, 1)
Conv2D(16, 3x3) -> BatchNorm -> MaxPool
Conv2D(32, 3x3) -> BatchNorm -> MaxPool
Conv2D(64, 3x3) -> BatchNorm
GlobalAveragePooling2D -> Dropout(0.3)
Dense(10, softmax)
```

- Optimizer: Adam
- Loss: sparse categorical crossentropy
- Epochs: up to 30 with EarlyStopping (patience 5, best weights restored)
- Batch size: 64

## Results

Run the notebook to generate your own numbers, then fill in the table below.

| Model | Params | Train time | Speech Commands test | FSDD |
|---|---|---|---|---|
| CNN | _fill in_ | _fill in_ | _fill in_ % | _fill in_ % |

## How to Run

### Option 1: Google Colab (recommended)

1. Open `CNN_Voice_Digit_Recognition.ipynb` in [Google Colab](https://colab.research.google.com).
2. Go to **Runtime → Change runtime type → T4 GPU**.
3. Run the cells from top to bottom.

The data download takes about 5–10 minutes and feature extraction another 5–10 minutes.

### Option 2: Locally

```bash
git clone https://github.com/YOUR_USERNAME/cnn-voice-recognition.git
cd cnn-voice-recognition
pip install -r requirements.txt
jupyter notebook
```

Notes for local use:

- Cells starting with `!` (wget, tar, git clone, ffmpeg) are shell commands. Run them in a terminal if your environment does not support them.
- The microphone demo and the file upload demo use `google.colab`, so they only work in Colab.
- **ffmpeg** must be installed for the microphone demo (`sudo apt install ffmpeg`, or `brew install ffmpeg` on macOS).

## Project Structure

```
.
├── CNN_Voice_Digit_Recognition.ipynb   # Full pipeline: data, features, training, evaluation, demo
├── requirements.txt                    # Python dependencies
├── README.md
├── spectrograms.png                    # Sample spectrograms (generated)
├── curve.png                           # Training curve (generated)
├── cm_sc.png                           # Confusion matrix, Speech Commands (generated)
└── cm_fsdd.png                         # Confusion matrix, FSDD (generated)
```

## Pipeline Overview

1. **Download data** — Speech Commands and FSDD.
2. **Split** — speaker-independent split using the official lists.
3. **Feature extraction** — waveform to log-mel spectrogram, normalized with the training-set mean and std.
4. **Train** — CNN trained with 3 seeds.
5. **Evaluate** — test accuracy, confusion matrices, cross-dataset accuracy on FSDD.
6. **Demo** — predict a digit from live microphone input or an uploaded audio file.

## Tech Stack

Python, TensorFlow/Keras, Librosa, NumPy, scikit-learn, Matplotlib, Seaborn

## Acknowledgements

- Pete Warden, *Speech Commands: A Dataset for Limited-Vocabulary Speech Recognition* (2018)
- [Free Spoken Digit Dataset](https://github.com/Jakobovski/free-spoken-digit-dataset) by Zohar Jackson et al.


