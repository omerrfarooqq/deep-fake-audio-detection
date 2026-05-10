Deep Fake Audio Detection


Overview

This repository contains a minimal training pipeline for detecting synthetic (fake) speech at the 2 second clip level. The pipeline extracts MFCC features from audio, trains a hybrid CNN-LSTM binary classifier, and produces evaluation metrics and visualization artifacts.

Dataset

The code expects a dataset root with the following structure:

- training/real/*.wav
- training/fake/*.wav

I generated a manifest named `manifest.csv` that lists `filepath,label` for each WAV file. The manifest used for the completed run contained 13,956 entries. Audio is processed at 16 kHz and clipped or padded to 2 seconds.

Pipeline summary

1. prepare_manifest.py
   - Scans `training/real` and `training/fake` and writes `filepath,label` rows to the CSV manifest.

2. data_utils.py
   - `AudioDataset` reads the manifest, loads audio using `soundfile`, resamples to 16 kHz, pads/trims to 2 seconds, and computes MFCC features via `torchaudio.transforms.MFCC`.

3. model.py
   - `CNNLSTM` model. Convolutional blocks process the spectrogram-like input, then an LSTM summarizes temporal information. The final linear layer produces a single logit for binary classification.

4. train.py
   - Trains the model with a train/validation/test split. The script saves the best model by validation AUC. It also writes `artifacts/training_history.csv` and `artifacts/test_predictions.csv`.

5. visualize.py and plot_metrics.py
   - `visualize.py` creates waveform, spectrogram, mel spectrogram, and MFCC images for representative real and fake samples and a label distribution plot.
   - `plot_metrics.py` reads the saved CSV outputs and generates training curves, confusion matrix, ROC and precision recall curves, and probability distribution plots.

How to reproduce

1. Create a virtual environment and install dependencies

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

2. Create the manifest from your dataset root

```powershell
python prepare_manifest.py "C:/path/to/dataset/root" manifest.csv
```

3. Train the model

```powershell
python train.py --manifest manifest.csv --workdir . --epochs 10 --batch-size 16 --save-model best.pth --num-workers 0 --artifacts-dir artifacts
```

4. Generate visualizations (example)

```powershell
python visualize.py --manifest manifest.csv --outdir artifacts
python plot_metrics.py --history artifacts/training_history.csv --predictions artifacts/test_predictions.csv --outdir artifacts
```

Notes on the run included here

- The audio was resampled to 16000 Hz and fixed at 2 second length. MFCCs used 40 coefficients.
- Training used Adam with learning rate 1e-3.
- The run in this folder used 10 training epochs and batch size 16.

Reported final test metrics from the completed run

- Test loss: 0.0084
- Test accuracy: 0.9971
- Test F1: 0.9971
- Test AUC: 0.9999

Interpretation and caveats

High numbers like these can be real but they are often a sign that the model learned dataset specific artifacts or that there is leakage between train and test. Consider these points before relying on the results in production:

- Files in a dataset might include many near duplicates. Remove duplicates or apply a source level split to ensure the same speaker or original file does not appear in multiple splits.
- Evaluate on out of distribution data produced by other synthesis pipelines or generators to estimate generalization.
- Use augmentations during training such as added noise, room reverb, and codec simulation to reduce reliance on trivial artifacts.

Files in this repository

- `prepare_manifest.py` : generate the CSV manifest
- `data_utils.py` : dataset class and MFCC extraction
- `model.py` : CNN-LSTM classifier
- `train.py` : training loop, metrics, checkpointing
- `visualize.py` : per-sample visualizations
- `plot_metrics.py` : training and evaluation plots
- `requirements.txt` : Python dependencies
- `artifacts/` : generated plots and CSVs (ignored from git)

License

This project is provided without warranty. Use it as a starting point for experiments.

Author: Omer Farooq Khan

