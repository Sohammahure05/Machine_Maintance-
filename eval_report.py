import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import precision_recall_curve, roc_curve

from config import EVAL_PDF_PATH, REPORT_DIR
from features import BLOCKS

PAGE = (8.27, 11.69)
COLORS = {6: "#2e7d32", 0: "#ef6c00", -6: "#c62828"}


def findings(rows):
    ordered = sorted(rows, key=lambda r: -r["snr"])
    lines = [
        "Abnormal recall: " + ", ".join(f"{r['snr']} dB {r['recall']:.0%}" for r in ordered),
        "Normal recognised correctly: " + ", ".join(f"{r['snr']} dB {r['specificity']:.0%}" for r in ordered),
        "AUC: " + ", ".join(f"{r['snr']} dB {'n/a' if r['auc'] is None else format(r['auc'], '.2f')}" for r in ordered),
    ]
    for r in ordered:
        if r["auc"] is None:
            continue
        if r["auc"] >= 0.9 and r["recall"] < 0.6:
            lines.append(f"At {r['snr']} dB the ranking is good (AUC {r['auc']:.2f}) but recall is low, "
                         "which points to a threshold that is too strict for this noise level.")
        elif r["auc"] < 0.7:
            lines.append(f"At {r['snr']} dB the model barely separates normal from abnormal "
                         f"(AUC {r['auc']:.2f}), which points to weak features or model at this noise level.")
    if len(ordered) > 1 and ordered[0]["recall"] - ordered[-1]["recall"] >= 0.2:
        lines.append(f"Recall falls by {ordered[0]['recall'] - ordered[-1]['recall']:.0%} from "
                     f"{ordered[0]['snr']} dB to {ordered[-1]['snr']} dB: noise is hiding faults.")
    return lines


def wrap(text, width=105):
    import textwrap
    return "\n".join(textwrap.fill(t, width) for t in text)


def table_rows(rows):
    out = []
    for r in sorted(rows, key=lambda r: -r["snr"]):
        auc = "n/a" if r["auc"] is None else f"{r['auc']:.3f}"
        out.append([f"{r['snr']} dB", r["files"], r["abnormal_files"], f"{r['accuracy']:.3f}",
                    f"{r['recall']:.3f}", f"{r['specificity']:.3f}", f"{r['precision']:.3f}",
                    f"{r['f1']:.3f}", auc])
    return out


def summary_page(machine, setup, rows):
    fig = plt.figure(figsize=PAGE)
    fig.text(0.08, 0.95, f"Model evaluation: {machine}", fontsize=20, weight="bold")
    note = ("Test machines were never used in training or calibration."
            if setup["independent"] else
            "WARNING: the model was trained on all machine IDs, so these results are not independent.")
    fig.text(0.08, 0.92, f"Trained {setup['trained_at']}  |  training IDs {', '.join(setup['train_ids'])}  |  "
                         f"test IDs {', '.join(setup['test_ids'])}", fontsize=9, color="#444444")
    fig.text(0.08, 0.90, note, fontsize=9, color="#c62828" if not setup["independent"] else "#2e7d32")
    fig.text(0.08, 0.875, f"Decision threshold {setup['threshold']:.3f}  |  segment pooling: {setup['pooling']}  |  "
                          f"calibration AUC on unseen machines: {setup['calibration_auc']}", fontsize=9)
    ax = fig.add_axes([0.06, 0.70, 0.88, 0.15])
    ax.axis("off")
    table = ax.table(cellText=table_rows(rows),
                     colLabels=["Noise", "Files", "Abnormal", "Accuracy", "Recall", "Specificity", "Precision", "F1", "AUC"],
                     loc="upper left", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.8)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor("#cccccc")
        if row == 0:
            cell.set_facecolor("#eeeeee")
            cell.set_text_props(weight="bold")
    fig.text(0.08, 0.64, "Findings (read from the numbers above)", fontsize=12, weight="bold")
    fig.text(0.08, 0.62, wrap(findings(rows)), fontsize=9.5, va="top", linespacing=1.5)
    fig.text(0.08, 0.38, wrap([
        "Recall is the share of abnormal files that were caught. Specificity is the share of normal files "
        "that were correctly left alone. Accuracy alone can hide poor recall because normal files outnumber abnormal ones.",
        "AUC measures how well abnormal files are ranked above normal ones, independent of the threshold.",
    ]), fontsize=8.5, va="top", color="#444444", linespacing=1.4)
    return fig


def curves_page(machine, rows, plots):
    fig = plt.figure(figsize=PAGE)
    fig.suptitle(f"{machine}: ranking quality and score separation", fontsize=14, weight="bold", y=0.97)
    roc = fig.add_axes([0.09, 0.62, 0.38, 0.28])
    pr = fig.add_axes([0.58, 0.62, 0.38, 0.28])
    for r in sorted(rows, key=lambda r: -r["snr"]):
        data = plots[r["snr"]]
        if len(set(data["y"].tolist())) < 2:
            continue
        fpr, tpr, _ = roc_curve(data["y"], data["score"])
        precision, recall, _ = precision_recall_curve(data["y"], data["score"])
        roc.plot(fpr, tpr, color=COLORS.get(r["snr"]), label=f"{r['snr']} dB")
        pr.plot(recall, precision, color=COLORS.get(r["snr"]), label=f"{r['snr']} dB")
    roc.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    roc.set(xlabel="False alarm rate", ylabel="Abnormal caught", title="ROC curve")
    pr.set(xlabel="Recall", ylabel="Precision", title="Precision-recall curve")
    roc.legend(); pr.legend()
    snrs = sorted(plots, reverse=True)
    for i, snr in enumerate(snrs):
        ax = fig.add_axes([0.09, 0.38 - i * 0.15, 0.86, 0.08])
        data = plots[snr]
        ax.hist(data["score"][data["y"] == 0], bins=30, range=(0, 1), alpha=0.7, color="#2e7d32", label="normal")
        ax.hist(data["score"][data["y"] == 1], bins=30, range=(0, 1), alpha=0.7, color="#c62828", label="abnormal")
        ax.axvline(data["threshold"], color="black", linestyle="--", linewidth=1)
        ax.set(title=f"Score distribution at {snr} dB (dashed line = threshold)", xlabel="File score")
        ax.legend()
    return fig


def detail_page(machine, rows, plots):
    fig = plt.figure(figsize=PAGE)
    fig.suptitle(f"{machine}: errors, threshold and model parts", fontsize=14, weight="bold", y=0.97)
    snrs = sorted(plots, reverse=True)
    for i, snr in enumerate(snrs):
        ax = fig.add_axes([0.16 + i * 0.30, 0.72, 0.22, 0.16])
        cm = np.array(next(r for r in rows if r["snr"] == snr)["confusion"])
        ax.imshow(cm, cmap="Blues")
        for a in range(2):
            for b in range(2):
                ax.text(b, a, cm[a, b], ha="center", va="center")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["normal", "abnormal"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["normal", "abnormal"])
        ax.set(title=f"{snr} dB", xlabel="Predicted")
        if i == 0:
            ax.set_ylabel("True")
    hardest = plots[snrs[-1]]
    sweep = fig.add_axes([0.09, 0.42, 0.86, 0.20])
    grid = np.linspace(0.02, 0.98, 49)
    y, score = hardest["y"], hardest["score"]
    recall = [((score >= t) & (y == 1)).sum() / max((y == 1).sum(), 1) for t in grid]
    precision = [((score >= t) & (y == 1)).sum() / max((score >= t).sum(), 1) for t in grid]
    f1 = [2 * p * r / max(p + r, 1e-9) for p, r in zip(precision, recall)]
    sweep.plot(grid, recall, label="recall"); sweep.plot(grid, precision, label="precision"); sweep.plot(grid, f1, label="F1")
    sweep.axvline(hardest["threshold"], color="black", linestyle="--", linewidth=1)
    sweep.set(title=f"Threshold sweep at {snrs[-1]} dB (dashed line = chosen threshold)", xlabel="Threshold")
    sweep.legend()
    bars = fig.add_axes([0.09, 0.08, 0.86, 0.26])
    parts = list(next(iter(rows))["auc_by_part"])
    width = 0.8 / len(snrs)
    for i, snr in enumerate(snrs):
        values = [next(r for r in rows if r["snr"] == snr)["auc_by_part"][p] or 0 for p in parts]
        bars.bar(np.arange(len(parts)) + i * width, values, width, color=COLORS.get(snr), label=f"{snr} dB")
    bars.set_xticks(np.arange(len(parts)) + width * (len(snrs) - 1) / 2)
    bars.set_xticklabels(parts)
    bars.set(ylim=(0, 1.05), ylabel="AUC", title="AUC by model part (does the ensemble help?)")
    bars.axhline(0.5, color="gray", linestyle=":", linewidth=1)
    bars.legend()
    return fig


def extras_page(machine, importances, subf_rows):
    fig = plt.figure(figsize=PAGE)
    fig.suptitle(f"{machine}: what the model uses, and the bearing (subf) test", fontsize=14, weight="bold", y=0.97)
    ax = fig.add_axes([0.30, 0.64, 0.64, 0.26])
    sums, start = [], 0
    for name, size in BLOCKS:
        sums.append((name, float(importances[start:start + size].sum())))
        start += size
    sums.sort(key=lambda t: t[1])
    ax.barh([n for n, _ in sums], [v for _, v in sums], color="#1565c0")
    ax.set(title="Random forest importance by feature group", xlabel="Share of total importance")
    fig.text(0.08, 0.58, "Bearing (subf) test: normal vs inner/outer race fault", fontsize=12, weight="bold")
    if not subf_rows:
        fig.text(0.08, 0.55, "No subf files were tested.", fontsize=10)
        return fig
    r = subf_rows
    auc = "n/a" if r["auc"] is None else f"{r['auc']:.3f}"
    lines = [
        f"Files tested: {r['files']} ({r['abnormal_files']} faulty)   accuracy {r['accuracy']:.3f}   "
        f"recall {r['recall']:.3f}   specificity {r['specificity']:.3f}   AUC {auc}",
        "Share flagged abnormal by class: " + ", ".join(f"{k} {v:.0%}" for k, v in r["flagged_rate"].items()),
        f"Share outside the training range: {r['outside_training_range']:.0%}",
        "",
        "This data is single-sensor vibration at 10 kHz, resampled to 16 kHz, while the model learned "
        "factory machine sound. A result near chance, with most files outside the training range, "
        "shows a data mismatch rather than a model fault.",
    ]
    fig.text(0.08, 0.55, wrap(lines, 100), fontsize=9.5, va="top", linespacing=1.5)
    return fig


def build(sections):
    REPORT_DIR.mkdir(exist_ok=True)
    with PdfPages(EVAL_PDF_PATH) as pdf:
        for s in sections:
            pages = [summary_page(s["machine"], s["setup"], s["rows"])]
            if s["plots"]:
                pages += [curves_page(s["machine"], s["rows"], s["plots"]),
                          detail_page(s["machine"], s["rows"], s["plots"])]
            pages.append(extras_page(s["machine"], s["importances"], s["subf"]))
            for fig in pages:
                pdf.savefig(fig)
                plt.close(fig)
