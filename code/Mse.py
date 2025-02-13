import soundfile as sf
import numpy as np
from pystoi import stoi
from pesq import pesq
from scipy.signal import resample_poly
ref, fs_ref = sf.read( 'D:\Ben\FinalProject\Github\Train-test-dataset\Train\Actions - South Of The Water+-.stem_vocals.wav')    # input - reffrence mixture path
pred, fs_pred = sf.read('D:\Ben\FinalProject\TestSongs_Train\Actions - South Of The Water.stem.mp4_vocals.wav')   # input - prediction from U-Net
if ref.ndim > 1:
    ref = np.mean(ref, axis=-1)

if pred.ndim > 1:
    pred= np.mean(pred, axis=-1)
if fs_ref > 16000:
    ref = resample_poly(ref[:4410000], 16000, fs_ref)
if fs_pred > 16000:
    pred = resample_poly(pred[:4410000], 16000, fs_pred)

print("pred sample:{}".format(fs_pred))
d = stoi(ref, pred, fs_ref, extended=False)
print("STOI:{}".format(d))
print(pesq(16000, ref, pred, 'wb'))
mse = np.sum((ref - pred)**2, axis=0) / len(pred) # calculation of MSE

noise = pred - ref
std_noise = np.std(noise)
std_ref = np.std(ref)
snr = std_ref / (std_noise + np.finfo(np.float32).eps)
snr_db = 20 * np.log10(snr + np.finfo(np.float32).eps)
print("SNR:{}".format(snr_db))
print("MSE = {}".format(mse))


