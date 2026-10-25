import json

import joblib
import numpy as np
import pandas as pd

import cache
import eval_report
from config import EVAL_PATH, MODEL_PATH, POOLING_MODES, REPORT_DIR, TEST_SNRS, TRAIN_SNR
from inference import ood_scores, safe_auc, summarize
from models import predict_parts, segment_table, unit_info, unit_mean, unit_scores


def evaluate_set(entry, X, meta):
    parts = predict_parts(entry["models"], X)
    info = unit_info(meta)
    tables = {k: segment_table(meta, parts[k]) for k in ("rf", "hgb", "ensemble")}
    scores = unit_scores(tables["ensemble"], entry["pooling"])
    y = info.loc[scores.index, "label"].to_numpy()
    row = summarize(y, scores.to_numpy(), entry["threshold"])
    novelty = unit_mean(meta, parts["novelty"]).loc[scores.index]
    row["auc_by_part"] = {
        "random forest": safe_auc(y, unit_scores(tables["rf"], entry["pooling"]).loc[scores.index]),
        "boosting": safe_auc(y, unit_scores(tables["hgb"], entry["pooling"]).loc[scores.index]),
        "ensemble": row["auc"],
        "novelty only": safe_auc(y, novelty.to_numpy()),
    }
    row["auc_by_pooling"] = {
        mode: safe_auc(y, unit_scores(tables["ensemble"], mode).loc[scores.index]) for mode in POOLING_MODES
    }
    flagged = scores.to_numpy() >= entry["threshold"]
    return row, {"y": y, "score": scores.to_numpy(), "threshold": entry["threshold"]}, flagged, info.loc[scores.index]


def main():
    loaded = cache.ensure()
    if loaded is None:
        raise SystemExit("No features available. Check data paths in config.py")
    X_all, meta_all, _ = loaded
    bundle = joblib.load(MODEL_PATH)
    holdout = bundle["info"]["holdout_ids"]
    independent = bool(holdout)
    sections, results, subf_results = [], [], []

    for machine, entry in bundle["machines"].items():
        mimii = (meta_all["source"] == "mimii") & (meta_all["machine"] == machine) & (meta_all["variant"] == "clean")
        test_ids = holdout if independent else sorted(meta_all.loc[mimii, "mid"].unique())
        rows, plots = [], {}
        for snr in [TRAIN_SNR] + TEST_SNRS:
            mask = (mimii & (meta_all["snr"] == snr) & meta_all["mid"].isin(test_ids)).to_numpy()
            if not mask.any():
                continue
            row, plot, _, _ = evaluate_set(entry, X_all[mask], meta_all[mask].reset_index(drop=True))
            row.update(machine=machine, snr=snr)
            rows.append(row)
            plots[snr] = plot
        results += rows

        subf_row = None
        mask = (meta_all["source"] == "subf").to_numpy()
        if mask.any():
            sub_meta = meta_all[mask].reset_index(drop=True)
            row, _, flagged, info = evaluate_set(entry, X_all[mask], sub_meta)
            distance = unit_mean(sub_meta, ood_scores(X_all[mask], entry["mu"], entry["sd"])).loc[info.index]
            row.update(
                machine=machine,
                flagged_rate={k: float(flagged[(info["tag"] == k).to_numpy()].mean()) for k in info["tag"].unique()},
                outside_training_range=float(np.mean(distance.to_numpy() > entry["ood_limit"])),
            )
            subf_row = row
            subf_results.append(row)

        if rows:
            sections.append({
                "machine": machine, "rows": rows, "plots": plots, "subf": subf_row,
                "importances": entry["models"]["rf"].feature_importances_,
                "setup": {"trained_at": bundle["info"]["trained_at"], "train_ids": entry["train_ids"],
                          "test_ids": test_ids, "independent": independent, "threshold": entry["threshold"],
                          "pooling": entry["pooling"], "calibration_auc": entry["calibration"]["auc"]},
            })
        else:
            print(f"\nNo test recordings found for {machine}")

    REPORT_DIR.mkdir(exist_ok=True)
    EVAL_PATH.write_text(json.dumps({
        "setup": {"holdout_ids": holdout, "independent": independent, "trained_at": bundle["info"]["trained_at"]},
        "results": results, "subf": subf_results,
    }, indent=2))
    if sections:
        eval_report.build(sections)

    columns = ["machine", "snr", "files", "accuracy", "recall", "specificity", "precision", "f1", "auc", "pauc"]
    if results:
        print("\nHeld-out machine results" if independent else "\nResults (NOT independent: trained on all IDs)")
        print(pd.DataFrame(results)[columns].round(3).to_string(index=False))
        print("\nAUC by model part")
        print(pd.DataFrame([{"machine": r["machine"], "snr": r["snr"], **r["auc_by_part"]} for r in results]).round(3).to_string(index=False))
        print("\nAUC by pooling")
        print(pd.DataFrame([{"machine": r["machine"], "snr": r["snr"], **r["auc_by_pooling"]} for r in results]).round(3).to_string(index=False))
    if subf_results:
        print("\nsubf bearing test")
        print(pd.DataFrame(subf_results)[["machine", "files", "accuracy", "recall", "specificity", "auc", "outside_training_range"]].round(3).to_string(index=False))
    print("\nSaved", EVAL_PATH, "and", eval_report.EVAL_PDF_PATH)


if __name__ == "__main__":
    main()
