import argparse
import os
import pandas as pd
import numpy as np
import torch
import torchaudio
import soundfile as sf
import matplotlib.pyplot as plt

TARGET_SR = 16000
TARGET_SECONDS = 2
TARGET_LENGTH = TARGET_SR * TARGET_SECONDS


def load_wave(path, sr=TARGET_SR):
    data, orig_sr = sf.read(path, dtype="float32")
    if data.ndim == 1:
        wav = torch.from_numpy(data).unsqueeze(0)
    else:
        wav = torch.from_numpy(data.T)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if orig_sr != sr:
        wav = torchaudio.functional.resample(wav, orig_sr, sr)
    if wav.shape[1] < TARGET_LENGTH:
        pad = TARGET_LENGTH - wav.shape[1]
        wav = torch.nn.functional.pad(wav, (0, pad))
    elif wav.shape[1] > TARGET_LENGTH:
        wav = wav[:, :TARGET_LENGTH]
    return wav


def plot_waveform(wav, sr, title, out_path):
    t = np.linspace(0, wav.shape[1] / sr, num=wav.shape[1])
    plt.figure(figsize=(10, 3))
    plt.plot(t, wav.squeeze(0).numpy())
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_spectrogram(wav, sr, title, out_path, mel=False):
    if mel:
        transform = torchaudio.transforms.MelSpectrogram(sample_rate=sr, n_mels=64)
    else:
        transform = torchaudio.transforms.Spectrogram(n_fft=400, hop_length=160)
    spec = transform(wav)
    spec_db = torchaudio.functional.amplitude_to_DB(spec, multiplier=10.0, amin=1e-10, db_multiplier=0.0)
    plt.figure(figsize=(10, 4))
    plt.imshow(spec_db.squeeze(0).numpy(), origin="lower", aspect="auto")
    plt.title(title)
    plt.xlabel("Frames")
    plt.ylabel("Mel bins" if mel else "Freq bins")
    plt.colorbar(format="%+2.0f dB")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_mfcc(wav, sr, title, out_path):
    transform = torchaudio.transforms.MFCC(sample_rate=sr, n_mfcc=40)
    mfcc = transform(wav)
    plt.figure(figsize=(10, 4))
    plt.imshow(mfcc.squeeze(0).numpy(), origin="lower", aspect="auto")
    plt.title(title)
    plt.xlabel("Frames")
    plt.ylabel("MFCC")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_artifacts(manifest_csv, outdir):
    df = pd.read_csv(manifest_csv)
    real_path = df[df["label"] == 0].iloc[0]["filepath"]
    fake_path = df[df["label"] == 1].iloc[0]["filepath"]

    samples = [
        ("real", real_path),
        ("fake", fake_path),
    ]

    for label, path in samples:
        wav = load_wave(path)
        plot_waveform(wav, TARGET_SR, f"{label} waveform", os.path.join(outdir, f"{label}_waveform.png"))
        plot_spectrogram(wav, TARGET_SR, f"{label} spectrogram", os.path.join(outdir, f"{label}_spectrogram.png"))
        plot_spectrogram(wav, TARGET_SR, f"{label} mel-spectrogram", os.path.join(outdir, f"{label}_mel.png"), mel=True)
        plot_mfcc(wav, TARGET_SR, f"{label} MFCC", os.path.join(outdir, f"{label}_mfcc.png"))

    # Dataset label distribution
    plt.figure(figsize=(4, 3))
    counts = df["label"].value_counts().sort_index()
    plt.bar(["real", "fake"], [counts.get(0, 0), counts.get(1, 0)])
    plt.title("Label distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "label_distribution.png"), dpi=150)
    plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--outdir", required=True)
    args = p.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    save_artifacts(args.manifest, args.outdir)


if __name__ == "__main__":
    main()
