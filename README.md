# Fake vs Real Voice Detection - Training Pipeline

Minimal pipeline to train a hybrid CNN-LSTM binary classifier on 2s audio clips.

Quick start

1. Create manifest CSV from your dataset root (which should contain `training/fake` and `training/real`):

```bash
python prepare_manifest.py "C:/Users/omerf/Downloads/for-2sec/for-2seconds" manifest.csv
```

2. Install dependencies (prefer a venv):

```bash
pip install -r requirements.txt
```

3. Run training:

```bash
python train.py --manifest manifest.csv --workdir ./runs --epochs 10 --batch-size 32 --save-model best.pth
```

Files

- `prepare_manifest.py`: generate `filepath,label` CSV from `training/fake` and `training/real`.
- `data_utils.py`: `AudioDataset` using torchaudio MFCCs.
- `model.py`: `CNNLSTM` model.
- `train.py`: training loop and simple validation.

Visualizations and Artifacts

The training run generates an `artifacts/` folder that contains visualizations and CSV outputs. Below are the key visualizations produced in this project:

## Audio Sample Visualizations

### Real Audio

**Waveform:**
![Real Waveform](artifacts/real_waveform.png)

**Spectrogram:**
![Real Spectrogram](artifacts/real_spectrogram.png)

**Mel-Spectrogram:**
![Real Mel-Spectrogram](artifacts/real_mel.png)

**MFCC:**
![Real MFCC](artifacts/real_mfcc.png)

### Fake Audio

**Waveform:**
![Fake Waveform](artifacts/fake_waveform.png)

**Spectrogram:**
![Fake Spectrogram](artifacts/fake_spectrogram.png)

**Mel-Spectrogram:**
![Fake Mel-Spectrogram](artifacts/fake_mel.png)

**MFCC:**
![Fake MFCC](artifacts/fake_mfcc.png)

## Dataset Statistics

![Label Distribution](artifacts/label_distribution.png)

## Training Curves

**Training Loss:**
![Training Loss](artifacts/training_loss.png)

**Training Accuracy:**
![Training Accuracy](artifacts/training_accuracy.png)

**Validation AUC:**
![Validation AUC](artifacts/val_auc.png)

## Evaluation Metrics

**Confusion Matrix:**
![Confusion Matrix](artifacts/confusion_matrix.png)

**ROC Curve:**
![ROC Curve](artifacts/roc_curve.png)

**Precision-Recall Curve:**
![PR Curve](artifacts/pr_curve.png)

**Prediction Probability Distribution:**
![Probability Distribution](artifacts/probability_distribution.png)

Final metrics from the most recent run included in this folder

- Test loss: 0.0084
- Test accuracy: 0.9971
- Test F1: 0.9971
- Test AUC: 0.9999

Notes and caveats

These results are reported on the dataset split used during training. High metrics can indicate good separation but may also signal dataset artifacts or leakage. Before using the model in a real setting, evaluate on out of distribution data and use source-level splits.

Next steps

- Add argument to tune MFCC and model hyperparameters.
- Add checkpointing, TensorBoard logging, and better data augmentation.
