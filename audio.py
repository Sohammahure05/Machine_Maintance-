from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from config import SR


def resample(y, rate):
    if rate != SR:
        y = librosa.resample(y, orig_sr=rate, target_sr=SR)
    return y.astype(np.float32)


def read_audio(source, channels=(0,)):
    try:
        data, rate = sf.read(source, always_2d=True, dtype="float32")
    except Exception:
        if hasattr(source, "seek"):
            source.seek(0)
        data, rate = librosa.load(source, sr=None, mono=False)
        data = data.reshape(-1, 1) if data.ndim == 1 else data.T
    picked = [c for c in channels if c < data.shape[1]] or [0]
    return [resample(data[:, c], rate) for c in picked]


def read_table(source, rate):
    raw = source.read() if hasattr(source, "read") else Path(source).read_bytes()
    tokens = raw.decode("utf-8", errors="ignore").replace(",", " ").replace(";", " ").split()
    start = 0
    while start < len(tokens):
        try:
            float(tokens[start])
            break
        except ValueError:
            start += 1
    return resample(np.array(tokens[start:], dtype=np.float32), rate)


def add_noise(y, snr_db, rng):
    power = np.mean((y - y.mean()) ** 2) + 1e-12
    noise_power = power / (10 ** (snr_db / 10))
    return (y + rng.normal(0, np.sqrt(noise_power), len(y))).astype(np.float32)
