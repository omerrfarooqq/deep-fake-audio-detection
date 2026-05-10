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

Next steps

- Add argument to tune MFCC and model hyperparameters.
- Add checkpointing, TensorBoard logging, and better data augmentation.
