import json

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from audio import add_noise, read_audio, read_table
from config import (AUG_SNRS, CACHE_PATH, CHANNELS, FEATURE_VERSION, MIMII_ROOT, MODEL_DIR, N_MELS,
                    N_MFCC, SEED, SEG_SEC, SR, SUBF_ROOT, SUBF_SR, TRAIN_SNR)
from dataset import scan_mimii, scan_subf
from features import N_FEATURES, segment_features

SIGNATURE = json.dumps({
    "version": FEATURE_VERSION, "channels": CHANNELS, "aug": AUG_SNRS, "sr": SR, "seg": SEG_SEC,
    "mels": N_MELS, "mfcc": N_MFCC, "train_snr": TRAIN_SNR, "subf_sr": SUBF_SR,
})
TEXT = ["source", "tag", "machine", "mid", "variant"]
NUMBERS = ["snr", "label", "channel", "seg", "file", "unit"]


def variants_for(source, snr):
    variants = [("clean", None)]
    if source == "mimii" and snr == TRAIN_SNR:
        variants += [(f"aug{s}", s) for s in AUG_SNRS]
    return variants


def extract(path, source, variants, seed):
    signals = [read_table(path, SUBF_SR)] if source == "subf" else read_audio(path, CHANNELS)
    rng = np.random.default_rng(seed)
    blocks = []
    for name, snr in variants:
        for channel, y in enumerate(signals):
            blocks.append((name, channel, segment_features(y if snr is None else add_noise(y, snr, rng))))
    return blocks


def load():
    if not CACHE_PATH.exists():
        return None
    data = np.load(CACHE_PATH, allow_pickle=False)
    if str(data["signature"]) != SIGNATURE:
        return None
    meta = pd.DataFrame({k: data["m_" + k] for k in TEXT + NUMBERS})
    return data["X"], meta, list(data["paths"])


def save(X, meta, paths):
    MODEL_DIR.mkdir(exist_ok=True)
    arrays = {"m_" + k: meta[k].astype(str).to_numpy().astype("U32") for k in TEXT}
    arrays.update({"m_" + k: meta[k].to_numpy().astype(np.int64) for k in NUMBERS})
    np.savez(CACHE_PATH, X=X, paths=np.array(paths).astype(str), signature=np.array(SIGNATURE), **arrays)


def update(files, source):
    cached = load()
    if cached:
        X_old, meta_old, paths = cached
    else:
        X_old, meta_old, paths = np.empty((0, N_FEATURES), np.float32), None, []
    todo = files[~files["path"].isin(set(paths))].reset_index(drop=True)
    if todo.empty:
        return
    print(f"Extracting features for {len(todo)} new {source} files")
    results = Parallel(n_jobs=4, verbose=5)(
        delayed(extract)(r.path, source, variants_for(source, r.snr), SEED + i)
        for i, r in enumerate(todo.itertuples())
    )
    records, blocks = [], []
    for row, result in zip(todo.itertuples(), results):
        file_index = len(paths)
        paths.append(row.path)
        codes = {name: k for k, (name, _) in enumerate(variants_for(source, row.snr))}
        for variant, channel, X in result:
            blocks.append(X)
            records.append((source, row.tag, row.machine, row.mid, variant, row.snr, row.label,
                            channel, file_index, file_index * 16 + codes[variant], len(X)))
    table = np.array(records, dtype=object)
    counts = table[:, -1].astype(int)
    columns = {name: np.repeat(table[:, j], counts) for j, name in enumerate(TEXT + NUMBERS[:3] + NUMBERS[4:])}
    meta_new = pd.DataFrame(columns)
    meta_new["seg"] = np.concatenate([np.arange(c) for c in counts])
    meta_new = meta_new[TEXT + NUMBERS]
    for k in NUMBERS:
        meta_new[k] = meta_new[k].astype(np.int64)
    meta = meta_new if meta_old is None else pd.concat([meta_old, meta_new], ignore_index=True)
    save(np.vstack([X_old] + blocks).astype(np.float32), meta, paths)


def ensure():
    if MIMII_ROOT.exists():
        files = scan_mimii(MIMII_ROOT)
        if not files.empty:
            update(files, "mimii")
    if SUBF_ROOT.exists():
        files = scan_subf(SUBF_ROOT)
        if not files.empty:
            update(files, "subf")
    return load()


if __name__ == "__main__":
    ensure()
