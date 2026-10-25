import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)

from config import CRITICAL_FLAG_RATIO, N_MELS, SEG_SEC, SR, WATCH_FLAG_RATIO
from features import segment_features
from models import pool, predict_parts


def ood_scores(X, mu, sd):
    return np.clip(np.abs((X - mu) / sd), 0, 10).mean(axis=1)


def safe_auc(y, score, **kwargs):
    y = np.asarray(y)
    if len(set(y.tolist())) < 2:
        return None
    return float(roc_auc_score(y, score, **kwargs))


def summarize(y, score, threshold):
    y = np.asarray(y)
    score = np.asarray(score)
    pred = (score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    return {
        "files": int(len(y)),
        "abnormal_files": int(y.sum()),
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall),
        "specificity": float(specificity),
        "balanced_accuracy": float((recall + specificity) / 2),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "auc": safe_auc(y, score),
        "pauc": safe_auc(y, score, max_fpr=0.1),
        "confusion": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def longest_run(flags):
    best = current = 0
    for flag in flags:
        current = current + 1 if flag else 0
        best = max(best, current)
    return best


def health_index(score, threshold):
    if score < threshold:
        return 100 - 50 * score / threshold
    return 50 - 50 * (score - threshold) / max(1 - threshold, 1e-9)


def maintenance_level(abnormal, flag_ratio, novel):
    if abnormal and flag_ratio >= CRITICAL_FLAG_RATIO:
        return "Critical", (
            "Abnormal sound is present in nearly the whole recording. If it is safe to do so, "
            "stop the machine and inspect it before further use."
        )
    if abnormal:
        return "Maintenance required", (
            "The recording is classified as abnormal. Schedule an inspection soon and check "
            "bearings, rotating parts and mountings for this machine type."
        )
    if flag_ratio >= WATCH_FLAG_RATIO:
        return "Watch", (
            "The recording as a whole is classified as normal, but some segments show abnormal "
            "behaviour. Keep the machine in service, record again soon, and inspect it if the "
            "same pattern returns."
        )
    if novel:
        return "Watch", (
            "The recording is classified as normal, but the sound is unusual compared with normal "
            "machine sound. Keep the machine in service, record again soon, and inspect it if the "
            "pattern returns."
        )
    return "Healthy", (
        "No abnormal behaviour was detected. Continue routine maintenance and keep recording "
        "periodically to build a baseline."
    )


def analyse(signals, entry):
    per_channel = [segment_features(s) for s in signals]
    count = min(len(f) for f in per_channel)
    features = np.stack([f[:count] for f in per_channel])
    flat = features.reshape(-1, features.shape[-1])
    parts = predict_parts(entry["models"], flat)
    probabilities = parts["ensemble"].reshape(len(signals), count).mean(axis=0)

    threshold = entry["threshold"]
    score = pool(probabilities, entry["pooling"])
    abnormal = score >= threshold
    flags = probabilities >= threshold
    flag_ratio = float(flags.mean())
    peak_index = int(probabilities.argmax())
    novelty = float(parts["novelty"].mean())
    novel = novelty > entry["novelty_limit"]

    signal = signals[0]
    duration = len(signal) / SR
    distance = float(ood_scores(flat, entry["mu"], entry["sd"]).mean())
    peak = float(np.max(np.abs(signal)))
    clipped = float(np.mean(np.abs(signal) >= 0.999 * peak)) if peak > 0 else 0.0
    rms = float(np.sqrt(np.mean(signal ** 2)))
    checks = [
        ("Recording length", duration >= SEG_SEC, f"{duration:.1f} s"),
        ("Signal level", rms > 1e-4, f"RMS {rms:.4f}"),
        ("Clipping", clipped < 0.01, f"{clipped:.2%} of samples at peak"),
        ("Similarity to training data", distance <= entry["ood_limit"],
         f"distance {distance:.2f}, limit {entry['ood_limit']:.2f}"),
    ]
    reliable = all(ok for _, ok, _ in checks)

    level, advice = maintenance_level(abnormal, flag_ratio, novel)
    if not reliable:
        advice += " Some input checks failed, so confirm this result by inspection before acting on it."

    return {
        "status": "Abnormal" if abnormal else "Normal",
        "level": level,
        "advice": advice,
        "score": float(score),
        "mean_score": float(probabilities.mean()),
        "pooling": entry["pooling"],
        "threshold": float(threshold),
        "health": float(health_index(score, threshold)),
        "confidence": float(min(abs(score - threshold) / max(threshold, 1 - threshold), 1.0)),
        "segments": int(len(flags)),
        "flagged": int(flags.sum()),
        "flag_ratio": flag_ratio,
        "peak_score": float(probabilities[peak_index]),
        "peak_time": float(peak_index * SEG_SEC),
        "longest_run": float(longest_run(flags) * SEG_SEC),
        "duration": float(duration),
        "novelty": novelty,
        "novelty_limit": float(entry["novelty_limit"]),
        "novel": bool(novel),
        "distance": distance,
        "reliable": bool(reliable),
        "checks": checks,
        "profile": [round(float(v), 2) for v in features[:, :, :N_MELS].mean(axis=(0, 1))],
        "segment_scores": probabilities,
        "segment_flags": flags,
        "segment_times": np.arange(len(flags)) * SEG_SEC,
        "channels": len(signals),
    }
