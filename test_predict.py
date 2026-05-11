from gradio_app import predict
import glob

real_files = glob.glob(r"C:/Users/omerf/Downloads/for-2sec/for-2seconds/training/real/*.wav")[:5]
fake_files = glob.glob(r"C:/Users/omerf/Downloads/for-2sec/for-2seconds/training/fake/*.wav")[:5]

print('Testing real samples:')
for p in real_files:
	label, prob, _ = predict(p)
	print(p, '->', label, prob)

print('\nTesting fake samples:')
for p in fake_files:
	label, prob, _ = predict(p)
	print(p, '->', label, prob)
