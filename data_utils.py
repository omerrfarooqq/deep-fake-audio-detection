import os
import torch
import torchaudio
import soundfile as sf
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

TARGET_SR = 16000
TARGET_SECONDS = 2
TARGET_LENGTH = TARGET_SR * TARGET_SECONDS


class AudioDataset(Dataset):
    def __init__(self, manifest_csv, sr=TARGET_SR, n_mfcc=40, return_path=False):
        self.df = pd.read_csv(manifest_csv)
        self.sr = sr
        self.n_mfcc = n_mfcc
        self.return_path = return_path
        self.mfcc_transform = torchaudio.transforms.MFCC(sample_rate=sr, n_mfcc=n_mfcc)

    def __len__(self):
        return len(self.df)

    def _load_wave(self, path):
        data, orig_sr = sf.read(path, dtype="float32")
        if data.ndim == 1:
            wav = torch.from_numpy(data).unsqueeze(0)
        else:
            wav = torch.from_numpy(data.T)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if orig_sr != self.sr:
            wav = torchaudio.functional.resample(wav, orig_sr, self.sr)
        # pad/trim
        if wav.shape[1] < TARGET_LENGTH:
            pad = TARGET_LENGTH - wav.shape[1]
            wav = torch.nn.functional.pad(wav, (0, pad))
        elif wav.shape[1] > TARGET_LENGTH:
            wav = wav[:, :TARGET_LENGTH]
        return wav

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = row["filepath"]
        label = int(row["label"]) if "label" in row.index else int(row[2])
        wav = self._load_wave(path)
        mfcc = self.mfcc_transform(wav)  # (channel, n_mfcc, time)
        # For model we'll drop channel dim and return (n_mfcc, time)
        mfcc = mfcc.squeeze(0)
        if self.return_path:
            return mfcc, torch.tensor(label, dtype=torch.float32), path
        return mfcc, torch.tensor(label, dtype=torch.float32)
