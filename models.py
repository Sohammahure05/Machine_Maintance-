import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest, RandomForestClassifier
from sklearn.utils.class_weight import compute_sample_weight

from config import BOOST_ITER, N_TREES, SEED


def fit_models(X, y):
    forest = RandomForestClassifier(
        n_estimators=N_TREES, min_samples_leaf=5, max_samples=0.5,
        class_weight="balanced_subsample", n_jobs=-1, random_state=SEED,
    ).fit(X, y)
    boosting = HistGradientBoostingClassifier(
        max_iter=BOOST_ITER, learning_rate=0.08, early_stopping=False, random_state=SEED,
    ).fit(X, y, sample_weight=compute_sample_weight("balanced", y))
    isolation = IsolationForest(n_estimators=200, random_state=SEED, n_jobs=-1).fit(X[y == 0])
    return {"rf": forest, "hgb": boosting, "iso": isolation}


def predict_parts(models, X):
    forest = models["rf"].predict_proba(X)[:, 1]
    boosting = models["hgb"].predict_proba(X)[:, 1]
    return {
        "rf": forest,
        "hgb": boosting,
        "ensemble": (forest + boosting) / 2,
        "novelty": -models["iso"].score_samples(X),
    }


def pool(values, mode):
    ordered = np.sort(np.asarray(values))[::-1]
    if mode == "max":
        return float(ordered[0])
    if mode == "top2":
        return float(ordered[:2].mean())
    return float(ordered.mean())


def segment_table(meta, values):
    frame = pd.DataFrame({"unit": meta["unit"].to_numpy(), "seg": meta["seg"].to_numpy(), "v": values})
    return frame.groupby(["unit", "seg"], sort=False)["v"].mean().reset_index()


def unit_scores(table, mode):
    return table.groupby("unit", sort=False)["v"].agg(lambda v: pool(v.to_numpy(), mode))


def unit_mean(meta, values):
    return pd.Series(values, index=meta["unit"].to_numpy()).groupby(level=0, sort=False).mean()


def unit_info(meta):
    return meta.drop_duplicates("unit").set_index("unit")[["label", "variant", "snr", "mid", "tag"]]
