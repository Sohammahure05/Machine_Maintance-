import io
import json
import textwrap

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import analytics
from config import EVAL_PATH, SEG_SEC, SR

PAGE = (8.27, 11.69)
COLORS = {
    "Healthy": "#2e7d32",
    "Watch": "#ef6c00",
    "Maintenance required": "#c62828",
    "Critical": "#7f0000",
}

NOTES = [
    "Health index: 50 is the decision boundary learned on unseen machines. Values above 50 "
    "indicate normal operation and values below 50 indicate abnormal operation.",
    "Abnormality timeline: each bar is a 2 second segment. Red bars are segments flagged as "
    "abnormal and the dashed line is the decision threshold.",
    "Novelty score: how unusual the sound is compared with normal machine sound, learned from "
    "normal recordings only. It can warn about a sound the classifier has never seen.",
    "Input checks: the recording is compared with the training recordings. A failed check "
    "means the microphone, machine or noise level may differ from the training data, so the "
    "verdict should be confirmed by inspection.",
    "Baseline comparison: spectral shape is compared with this machine's own earlier readings. "
    "Segments are loudness-normalised, so differences show a change in sound character, not volume.",
    "This report supports maintenance planning. It does not replace a physical inspection.",
]


def plot_timeline(ax, result):
    colors = ["#c62828" if f else "#2e7d32" for f in result["segment_flags"]]
    ax.bar(result["segment_times"], result["segment_scores"], width=SEG_SEC * 0.9,
           align="edge", color=colors)
    ax.axhline(result["threshold"], color="black", linestyle="--", linewidth=1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Abnormal probability")
    ax.set_title("Abnormality timeline")


def plot_waveform(ax, signal):
    step = max(1, len(signal) // 6000)
    ax.plot(np.arange(0, len(signal), step) / SR, signal[::step], linewidth=0.6, color="#1565c0")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Waveform")


def plot_spectrogram(ax, signal):
    ax.specgram(signal, NFFT=1024, Fs=SR, noverlap=512, cmap="magma")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title("Spectrogram")


def plot_trend(ax, prev, result):
    count = len(prev)
    if count:
        ax.plot(range(1, count + 1), prev["health"], marker="o", color="#1565c0", label="saved readings")
    ax.scatter([count + 1], [result["health"]], color="#c62828", zorder=3, label="this reading")
    ax.axhline(50, color="black", linestyle="--", linewidth=1)
    ax.set_ylim(0, 105)
    ax.set_xticks(range(1, count + 2))
    ax.set_xlabel("Reading number")
    ax.set_ylabel("Health index")
    ax.set_title("Health index over time (50 = decision boundary)")
    ax.legend(loc="lower left")


def plot_profile(ax, result, comparison):
    ax.plot(analytics.CENTERS, comparison["baseline_profile"], color="#1565c0", label="machine baseline")
    ax.plot(analytics.CENTERS, result["profile"], color="#c62828", label="this reading")
    ax.set_xscale("symlog", linthresh=500)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Level (dB, normalised)")
    ax.set_title("Spectral profile: this reading vs baseline")
    ax.legend()


def load_reference(machine):
    if not EVAL_PATH.exists():
        return []
    data = json.loads(EVAL_PATH.read_text())
    rows = [r for r in data.get("results", []) if r["machine"] == machine]
    for r in rows:
        r["independent"] = data.get("setup", {}).get("independent", False)
    return rows


def summary_rows(result):
    novelty = f"{result['novelty']:.3f} (limit {result['novelty_limit']:.3f})"
    return [
        ["Health index", f"{result['health']:.0f} / 100"],
        ["File score", f"{result['score']:.3f} ({result['pooling']} of segments)"],
        ["Decision threshold", f"{result['threshold']:.3f}"],
        ["Decision confidence", f"{result['confidence']:.0%}"],
        ["Segments flagged", f"{result['flagged']} of {result['segments']} ({result['flag_ratio']:.0%})"],
        ["Peak segment score", f"{result['peak_score']:.3f} at {result['peak_time']:.0f} s"],
        ["Longest abnormal stretch", f"{result['longest_run']:.0f} s"],
        ["Novelty score", novelty + (" unusual" if result["novel"] else "")],
        ["Recording length", f"{result['duration']:.1f} s, {result['channels']} channel(s) used"],
        ["Input reliability", "Good" if result["reliable"] else "Check inputs"],
    ]


def summary_page(result, meta):
    fig = plt.figure(figsize=PAGE)
    color = COLORS[result["level"]]
    fig.text(0.08, 0.95, "Machine Condition Report", fontsize=20, weight="bold")
    fig.text(0.08, 0.925,
             f"Machine {meta['code']}: {meta['name']} ({meta['type']}), {meta['location'] or 'no location'}",
             fontsize=10)
    fig.text(0.08, 0.907, f"File: {meta['file']}   |   Generated: {meta['time']}", fontsize=9, color="#444444")
    fig.text(0.08, 0.845, result["status"].upper(), fontsize=26, weight="bold", color="white",
             bbox=dict(facecolor=color, edgecolor="none", boxstyle="round,pad=0.4"))
    fig.text(0.40, 0.855, result["level"], fontsize=16, weight="bold", color=color)
    fig.text(0.40, 0.832, f"Health index {result['health']:.0f} / 100", fontsize=11)

    ax = fig.add_axes([0.08, 0.46, 0.84, 0.32])
    ax.axis("off")
    table = ax.table(cellText=summary_rows(result), colLabels=["Indicator", "Value"],
                     loc="upper left", cellLoc="left", colWidths=[0.40, 0.60])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.6)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor("#cccccc")
        if row == 0:
            cell.set_facecolor("#eeeeee")
            cell.set_text_props(weight="bold")

    fig.text(0.08, 0.425, "Maintenance recommendation", fontsize=12, weight="bold")
    fig.text(0.08, 0.405, textwrap.fill(result["advice"], 95), fontsize=10, va="top")
    plot_timeline(fig.add_axes([0.10, 0.08, 0.82, 0.22]), result)
    return fig


def detail_page(result, signal, reference):
    fig = plt.figure(figsize=PAGE)
    plot_waveform(fig.add_axes([0.10, 0.80, 0.82, 0.13]), signal)
    plot_spectrogram(fig.add_axes([0.10, 0.60, 0.82, 0.14]), signal)

    fig.text(0.08, 0.54, "Input checks", fontsize=12, weight="bold")
    lines = [f"[{'OK' if ok else 'FAIL'}]  {name}: {detail}" for name, ok, detail in result["checks"]]
    fig.text(0.08, 0.52, "\n".join(lines), fontsize=9.5, va="top", family="monospace", linespacing=1.6)

    fig.text(0.08, 0.42, "Reference performance of this model", fontsize=12, weight="bold")
    if reference:
        rows = []
        for r in reference:
            auc = "n/a" if r["auc"] is None else f"{r['auc']:.3f}"
            kind = "unseen machine" if r["independent"] else "all machines (not independent)"
            rows.append(f"{r['snr']} dB, {kind}: accuracy {r['accuracy']:.1%}, recall {r['recall']:.1%}, "
                        f"AUC {auc} ({r['files']} files)")
        text = "\n".join(rows)
    else:
        text = "Run evaluate.py to add test results here."
    fig.text(0.08, 0.40, text, fontsize=9.5, va="top", family="monospace", linespacing=1.6)

    fig.text(0.08, 0.31, "How to read this report", fontsize=12, weight="bold")
    body = "\n\n".join(textwrap.fill(n, 100) for n in NOTES)
    fig.text(0.08, 0.29, body, fontsize=8.5, va="top", linespacing=1.35)
    return fig


def history_page(result, history):
    fig = plt.figure(figsize=PAGE)
    prev, comparison, trend_info = history["prev"], history["comparison"], history["trend"]
    fig.text(0.08, 0.95, "History and comparison", fontsize=18, weight="bold")
    fig.text(0.08, 0.925, f"{len(prev)} earlier reading(s) with the current model. "
                          f"Trend: {trend_info['direction']}.", fontsize=10, color="#444444")
    plot_trend(fig.add_axes([0.10, 0.68, 0.82, 0.19]), prev, result)
    if comparison is not None:
        plot_profile(fig.add_axes([0.10, 0.42, 0.82, 0.19]), result, comparison)
        changes = analytics.describe_changes(comparison)
        text = [f"Compared with {comparison['readings_used']} earlier reading(s): health index "
                f"{comparison['health_change']:+.0f} points against the baseline ({comparison['baseline_health']:.0f})."]
        text += ["Largest spectral changes: " + "; ".join(changes) + "."] if changes else []
        fig.text(0.08, 0.37, textwrap.fill(" ".join(text), 100), fontsize=9.5, va="top")
    fig.text(0.08, 0.28, "Alerts from history", fontsize=12, weight="bold")
    alerts = history["alerts"] or ["No history alerts."]
    fig.text(0.08, 0.26, "\n".join(textwrap.fill("- " + a, 100) for a in alerts), fontsize=9.5, va="top")
    recent = prev.tail(8)
    if len(recent):
        fig.text(0.08, 0.17, "Latest saved readings", fontsize=12, weight="bold")
        rows = [f"{r.recorded_at:%Y-%m-%d %H:%M}  {r.status:9s} {r.level:21s} health {r.health:5.0f}"
                for r in recent.itertuples()]
        fig.text(0.08, 0.15, "\n".join(rows), fontsize=8.5, va="top", family="monospace", linespacing=1.4)
    return fig


def build_pdf(result, signal, meta, reference, history=None):
    buffer = io.BytesIO()
    pages = [summary_page(result, meta), detail_page(result, signal, reference)]
    if history is not None:
        pages.append(history_page(result, history))
    with PdfPages(buffer) as pdf:
        for fig in pages:
            pdf.savefig(fig)
            plt.close(fig)
    return buffer.getvalue()
