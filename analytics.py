import librosa
import numpy as np

from config import N_MELS, SR, TREND_SLOPE, TREND_WINDOW

EDGES = librosa.mel_frequencies(n_mels=N_MELS + 2, fmin=0.0, fmax=SR / 2)
CENTERS = EDGES[1:-1]


def trend(health):
    values = np.asarray(health[-TREND_WINDOW:], dtype=float)
    if len(values) < 3:
        return {"direction": "Not enough data", "slope": 0.0, "change": 0.0, "points": int(len(values))}
    slope = float(np.polyfit(np.arange(len(values)), values, 1)[0])
    direction = "Declining" if slope <= -TREND_SLOPE else "Improving" if slope >= TREND_SLOPE else "Stable"
    return {"direction": direction, "slope": slope, "change": float(values[-1] - values[0]),
            "points": int(len(values))}


def compare(result, prev):
    if prev.empty:
        return None
    normal = prev[prev["status"] == "Normal"]
    base = normal if len(normal) >= 2 else prev
    baseline = np.median(np.stack(base["profile"].to_list()), axis=0)
    delta = np.asarray(result["profile"]) - baseline
    changes = []
    for band in np.array_split(np.arange(N_MELS), 8):
        low = EDGES[band[0]]
        high = EDGES[min(band[-1] + 2, len(EDGES) - 1)]
        changes.append((float(delta[band].mean()), float(low), float(high)))
    changes.sort(key=lambda c: -abs(c[0]))
    baseline_health = float(base["health"].median())
    return {
        "baseline_profile": baseline,
        "baseline_health": baseline_health,
        "health_change": float(result["health"] - baseline_health),
        "readings_used": int(len(base)),
        "top_changes": changes[:3],
    }


def alerts(prev, trend_info, comparison):
    found = []
    recent = prev.tail(3)
    abnormal = int((recent["status"] == "Abnormal").sum()) if len(recent) else 0
    if len(recent) >= 2 and abnormal >= 2:
        found.append(f"{abnormal} of the last {len(recent)} saved readings were abnormal.")
    if trend_info["direction"] == "Declining":
        found.append(
            f"Health index is declining: {trend_info['change']:+.0f} points over the last "
            f"{trend_info['points']} readings."
        )
    if comparison and comparison["health_change"] <= -10:
        found.append(
            f"This reading is {abs(comparison['health_change']):.0f} points below this machine's own "
            f"baseline ({comparison['baseline_health']:.0f})."
        )
    return found


def describe_changes(comparison):
    lines = []
    for change, low, high in comparison["top_changes"]:
        direction = "louder" if change > 0 else "quieter"
        lines.append(f"{abs(change):.1f} dB {direction} between {low:.0f} and {high:.0f} Hz")
    return lines
