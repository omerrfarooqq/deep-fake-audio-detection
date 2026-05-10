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
	elif isinstance(audio, np.ndarray):
		# assume TARGET_SR
		sr, arr = TARGET_SR, audio
	elif isinstance(audio, str) and os.path.exists(audio):
		arr, sr = sf.read(audio, dtype="float32")
	else:
		return "Unsupported input", 0.0, None

	# Convert to float32 if needed
	arr = arr.astype("float32")
	
	# Handle stereo -> mono
	if arr.ndim > 1:
		arr = arr.mean(axis=1)
	
	# Resample if needed
	if sr != TARGET_SR:
		wav_tensor = torch.from_numpy(arr).unsqueeze(0)
		wav_tensor = torchaudio.functional.resample(wav_tensor, sr, TARGET_SR)
		arr = wav_tensor.squeeze(0).numpy()
	
	# Split into 2-second chunks if longer than 2 seconds
	total_samples = len(arr)
	chunk_size = TARGET_LENGTH
	
	if total_samples <= chunk_size:
		chunks = [arr]
	else:
		# Split into overlapping chunks (50% overlap)
		hop_size = chunk_size // 2
		chunks = []
		start = 0
		while start < total_samples:
			end = min(start + chunk_size, total_samples)
			chunk = arr[start:end]
			# Pad if necessary
			if len(chunk) < chunk_size:
				chunk = np.pad(chunk, (0, chunk_size - len(chunk)), mode='constant')
			chunks.append(chunk)
			start += hop_size
			if end == total_samples:
				break
	
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model = CNNLSTM()
	model_path = os.path.join(os.path.dirname(__file__), "best.pth")
	if not os.path.exists(model_path):
		return "Model not found (best.pth)", 0.0, _make_mel_image(torch.from_numpy(arr).unsqueeze(0))
	
	model.load_state_dict(torch.load(model_path, map_location=device))
	model.to(device)
	model.eval()
	
	probs = []
	with torch.no_grad():
		for chunk in chunks:
			# chunk is 1D numpy array of shape (32000,)
			wav_chunk = torch.from_numpy(chunk).unsqueeze(0)  # (1, 32000)
			mfcc = torchaudio.transforms.MFCC(sample_rate=TARGET_SR, n_mfcc=40)(wav_chunk)  # (1, 40, time)
			x = mfcc.unsqueeze(0).to(device)  # (1, 1, 40, time) - add batch dim
			logits = model(x)
			prob = float(torch.sigmoid(logits).cpu().item())
			probs.append(prob)
	
	# Average probability across chunks
	avg_prob = np.mean(probs)
	label = "fake" if avg_prob > 0.5 else "real"
	
	# Use first chunk for visualization
	img = _make_mel_image(torch.from_numpy(chunks[0]).unsqueeze(0))
	return label, avg_prob, img


def serve():
	title = "Deep Fake Audio Detector"
	desc = "Upload an audio clip of any length. The model processes it in 2-second chunks with 50% overlap and returns an averaged prediction."
	iface = gr.Interface(
		fn=predict,
		inputs=gr.Audio(type="numpy", label="Audio"),
		outputs=[gr.Textbox(label="Label"), gr.Number(label="Fake probability"), gr.Image(type="pil", label="Mel spectrogram (first chunk)")],
		title=title,
		description=desc,
	)
	iface.launch()


if __name__ == "__main__":
	serve()

