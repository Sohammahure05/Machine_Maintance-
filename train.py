import argparse
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

import cache
from config import (HOLDOUT_IDS, MODEL_DIR, MODEL_PATH, NOVELTY_PERCENTILE, OOD_MARGIN,
                    OOD_PERCENTILE, POOLING_MODES, SEED, TRAIN_SNR)
from inference import ood_scores, safe_auc, summarize
from models import fit_models, predict_parts, segment_table, unit_info, unit_mean, unit_scores


def best_threshold(y, score):
    precision, recall, thresholds = precision_recall_curve(y, score)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-9)
    return float(thresholds[np.argmax(f1)])


def fold_groups(meta):
    ids = meta["mid"].to_numpy()
    if len(np.unique(ids)) >= 2:
        return ids
    rng = np.random.default_rng(SEED)
    files = np.unique(meta["file"].to_numpy())
    assign = dict(zip(files, rng.integers(0, 3, len(files))))
    return meta["file"].map(assign).to_numpy()


def out_of_group(X, meta, y, mu, sd):
    groups = fold_groups(meta)
    folds = []
    for group in np.unique(groups):
        train, held = groups != group, groups == group
        print(f"  calibration fold: leaving out {group}")
        models = fit_models(X[train], y[train])
        parts = predict_parts(models, X[held])
        part_meta = meta[held]
        folds.append((
            segment_table(part_meta, parts["ensemble"]),
            unit_info(part_meta),
            unit_mean(part_meta, parts["novelty"]),
            unit_mean(part_meta, ood_scores(X[held], mu, sd)),
        ))
    return (
        pd.concat([f[0] for f in folds]),
        pd.concat([f[1] for f in folds]),
        pd.concat([f[2] for f in folds]),
        pd.concat([f[3] for f in folds]),
    )


def main(all_ids):
    loaded = cache.ensure()
    if loaded is None:
        raise SystemExit("No MIMII wav files found. Check MIMII_ROOT in config.py")
    X_all, meta_all, _ = loaded
    keep = (meta_all["source"] == "mimii") & (meta_all["snr"] == TRAIN_SNR)
    if not all_ids:
        keep &= ~meta_all["mid"].isin(HOLDOUT_IDS)
    if not keep.any():
        raise SystemExit(f"No {TRAIN_SNR} dB recordings available for training")

    machines, holdout = {}, [] if all_ids else list(HOLDOUT_IDS)
    for machine in sorted(meta_all.loc[keep, "machine"].unique()):
        mask = (keep & (meta_all["machine"] == machine)).to_numpy()
        X, meta = X_all[mask], meta_all[mask].reset_index(drop=True)
        y = meta["label"].to_numpy()
        train_ids = sorted(meta["mid"].unique())
        print(f"\n{machine}: {meta['unit'].nunique()} training units from {train_ids}")

        mu, sd = X.mean(axis=0), X.std(axis=0) + 1e-6
        table, info, novelty, distance = out_of_group(X, meta, y, mu, sd)

        auc_by_mode = {}
        for mode in POOLING_MODES:
            scores = unit_scores(table, mode)
            auc_by_mode[mode] = safe_auc(info.loc[scores.index, "label"].to_numpy(), scores.to_numpy())
        pooling = max(auc_by_mode, key=lambda m: auc_by_mode[m] if auc_by_mode[m] is not None else -1)
        scores = unit_scores(table, pooling)
        y_unit = info.loc[scores.index, "label"].to_numpy()
        threshold = best_threshold(y_unit, scores.to_numpy())

        normal_units = info.index[info["label"] == 0]
        novelty_limit = float(np.percentile(novelty.loc[normal_units], NOVELTY_PERCENTILE))
        ood_limit = float(np.percentile(distance, OOD_PERCENTILE) * OOD_MARGIN)
        calibration = summarize(y_unit, scores.to_numpy(), threshold)
        calibration["auc_by_pooling"] = auc_by_mode
        print(f"  pooling={pooling} threshold={threshold:.3f} "
              f"unseen-machine AUC={calibration['auc']} recall={calibration['recall']:.3f}")

        print("  fitting final models")
        machines[machine] = {
            "models": fit_models(X, y), "pooling": pooling, "threshold": threshold,
            "novelty_limit": novelty_limit, "mu": mu, "sd": sd, "ood_limit": ood_limit,
            "calibration": calibration, "train_ids": train_ids,
        }

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump({
        "machines": machines,
        "info": {"holdout_ids": holdout, "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                 "train_snr": TRAIN_SNR},
    }, MODEL_PATH)
    print("\nSaved", MODEL_PATH)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--all-ids", action="store_true")
    main(parser.parse_args().all_ids)
