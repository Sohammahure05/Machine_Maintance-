import librosa
import numpy as np
from scipy.stats import kurtosis

from config import N_MELS, N_MFCC, SEG_SEC, SR

SEG_LEN = SR * SEG_SEC
GROUPS = 16
CONTRAST_BANDS = 7

BLOCKS = [
    ("Log-mel mean", N_MELS),
    ("Log-mel spread", N_MELS),
    ("MFCC mean", N_MFCC),
    ("MFCC spread", N_MFCC),
    ("Band low level", GROUPS),
    ("Band high level", GROUPS),
    ("Band peak", GROUPS),
    ("MFCC change rate", N_MFCC),
    ("Contrast mean", CONTRAST_BANDS),
    ("Contrast spread", CONTRAST_BANDS),
    ("Spectral shape and spikiness", 12),
]
N_FEATURES = sum(n for _, n in BLOCKS)
FEATURE_NAMES = [f"{name} {i}" for name, n in BLOCKS for i in range(n)]


def describe(seg):
    seg = seg - seg.mean()
    seg = seg / (np.sqrt(np.mean(seg ** 2)) + 1e-9)
    mel = librosa.feature.melspectrogram(y=seg, sr=SR, n_fft=1024, hop_length=512, n_mels=N_MELS)
    logmel = librosa.power_to_db(mel + 1e-10)
    mfcc = librosa.feature.mfcc(S=logmel, n_mfcc=N_MFCC)
    delta = librosa.feature.delta(mfcc)
    contrast = librosa.feature.spectral_contrast(y=seg, sr=SR, n_fft=1024, hop_length=512)
    grouped = np.stack([g.mean(axis=0) for g in np.array_split(logmel, GROUPS, axis=0)])
    spectral = [
        librosa.feature.spectral_centroid(y=seg, sr=SR),
        librosa.feature.spectral_bandwidth(y=seg, sr=SR),
        librosa.feature.spectral_rolloff(y=seg, sr=SR),
        librosa.feature.spectral_flatness(y=seg),
        librosa.feature.zero_crossing_rate(seg),
    ]
    scalars = [v for f in spectral for v in (f.mean(), f.std())]
    scalars += [np.max(np.abs(seg)), kurtosis(seg)]
    row = np.concatenate([
        logmel.mean(axis=1), logmel.std(axis=1), mfcc.mean(axis=1), mfcc.std(axis=1),
        np.percentile(grouped, 10, axis=1), np.percentile(grouped, 90, axis=1), grouped.max(axis=1),
        delta.std(axis=1), contrast.mean(axis=1), contrast.std(axis=1), scalars,
    ])
    return np.nan_to_num(row).astype(np.float32)


def segment_features(y):
    if len(y) < SEG_LEN:
        y = np.pad(y, (0, SEG_LEN - len(y)))
    count = len(y) // SEG_LEN
    return np.vstack([describe(y[i * SEG_LEN:(i + 1) * SEG_LEN]) for i in range(count)])
