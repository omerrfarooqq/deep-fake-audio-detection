import os
import io
import torch
import torchaudio
import soundfile as sf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import gradio as gr

from model import CNNLSTM

TARGET_SR = 16000
TARGET_SECONDS = 2
TARGET_LENGTH = TARGET_SR * TARGET_SECONDS


def _load_and_prepare_from_numpy(arr, sr):
	data = arr.astype("float32")
	if data.ndim == 1:
		wav = torch.from_numpy(data).unsqueeze(0)
	else:
		wav = torch.from_numpy(data.T)
	if wav.shape[0] > 1:
		wav = wav.mean(dim=0, keepdim=True)
	if sr != TARGET_SR:
		wav = torchaudio.functional.resample(wav, sr, TARGET_SR)
	if wav.shape[1] < TARGET_LENGTH:
		pad = TARGET_LENGTH - wav.shape[1]
		wav = torch.nn.functional.pad(wav, (0, pad))
	elif wav.shape[1] > TARGET_LENGTH:
		wav = wav[:, :TARGET_LENGTH]
	mfcc = torchaudio.transforms.MFCC(sample_rate=TARGET_SR, n_mfcc=40)(wav)
	return wav, mfcc


def _make_mel_image(wav_tensor):
	with torch.no_grad():
		spec = torchaudio.transforms.MelSpectrogram(sample_rate=TARGET_SR, n_mels=64)(wav_tensor)
		spec_db = torchaudio.functional.amplitude_to_DB(spec, multiplier=10.0, amin=1e-10, db_multiplier=0.0)
		arr = spec_db.squeeze(0).cpu().numpy()
	fig, ax = plt.subplots(figsize=(6, 3))
	ax.imshow(arr, origin="lower", aspect="auto")
	ax.set_axis_off()
	buf = io.BytesIO()
	plt.tight_layout()
	fig.savefig(buf, format="png", dpi=100)
	plt.close(fig)
	buf.seek(0)
	return Image.open(buf)


def predict(audio):
	# audio can be (sr, np_array) when type='numpy' or a filepath string
	if audio is None:
		return "No audio", 0.0, None

	if isinstance(audio, tuple) and len(audio) == 2:
		sr, arr = audio
		wav, mfcc = _load_and_prepare_from_numpy(arr, sr)
	elif isinstance(audio, np.ndarray):
		# assume TARGET_SR
		wav, mfcc = _load_and_prepare_from_numpy(audio, TARGET_SR)
	elif isinstance(audio, str) and os.path.exists(audio):
		arr, sr = sf.read(audio, dtype="float32")
		wav, mfcc = _load_and_prepare_from_numpy(arr, sr)
	else:
		return "Unsupported input", 0.0, None

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model = CNNLSTM()
	model_path = "best.pth"
	if not os.path.exists(model_path):
		return "Model not found (best.pth)", 0.0, _make_mel_image(wav)
	model.load_state_dict(torch.load(model_path, map_location=device))
	model.to(device)
	model.eval()
	with torch.no_grad():
		x = mfcc.to(device)
		# ensure batch dim
		if x.dim() == 3:
			x = x.unsqueeze(0)
		logits = model(x)
		prob = float(torch.sigmoid(logits).cpu().item())
		label = "fake" if prob > 0.5 else "real"

	img = _make_mel_image(wav)
	return label, prob, img


def serve():
	title = "Deep Fake Audio Detector"
	desc = "Upload a short audio clip (<= 2s). Returns label, fake probability, and mel spectrogram image."
	iface = gr.Interface(
		fn=predict,
		inputs=gr.Audio(source="upload", type="numpy", label="Audio"),
		outputs=[gr.Textbox(label="Label"), gr.Number(label="Fake probability"), gr.Image(type="pil", label="Mel spectrogram")],
		title=title,
		description=desc,
		allow_flagging=False,
	)
	iface.launch()


if __name__ == "__main__":
	serve()

