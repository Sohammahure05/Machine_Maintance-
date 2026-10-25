import re

import numpy as np
import pandas as pd

from config import MACHINES, SEED, SUBF_CLASSES, SUBF_MAX_PER_CLASS

SNR_PATTERN = re.compile(r"(-?\d+)_dB")


def scan_mimii(root):
    rows = []
    for path in sorted(root.rglob("*.wav")):
        parts = path.parts
        snr = next((int(m.group(1)) for p in parts if (m := SNR_PATTERN.search(p))), None)
        machine = next((p for p in parts if p in MACHINES), None)
        ident = next((p for p in parts if p.startswith("id_")), None)
        label = next((int(p == "abnormal") for p in parts if p in ("normal", "abnormal")), None)
        if snr is None or machine is None or ident is None or label is None:
            continue
        rows.append({"path": str(path), "snr": snr, "machine": machine, "mid": ident,
                     "label": label, "tag": ""})
    return pd.DataFrame(rows)


def scan_subf(root):
    rng = np.random.default_rng(SEED)
    rows = []
    for name, label in SUBF_CLASSES.items():
        files = sorted((root / name).rglob("*.csv"))
        if len(files) > SUBF_MAX_PER_CLASS:
            files = [files[i] for i in rng.choice(len(files), SUBF_MAX_PER_CLASS, replace=False)]
        rows += [{"path": str(f), "snr": -99, "machine": "bearing", "mid": "",
                  "label": label, "tag": name} for f in files]
    return pd.DataFrame(rows)
