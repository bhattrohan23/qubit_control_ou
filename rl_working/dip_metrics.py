import glob
import os
import pickle
import numpy as np

BASE = "episodic_data/qubit_control"


def dip_metrics(mean_F):
    return {
        'worst_dip': mean_F.min(),
        'updates_below_0.7': (mean_F < 0.7).sum(),
        'final_F': mean_F[-100:].mean(),
    }


for path in sorted(glob.glob(os.path.join(BASE, "*.pkl"))):
    with open(path, "rb") as f:
        data = pickle.load(f)

    mean_F = np.array([d['mean_fidelity'] for d in data], dtype=float)
    metrics = dip_metrics(mean_F)
    name = os.path.basename(path)
    print(
        f"for {name}, "
        f"worst_dip={metrics['worst_dip']:.4f}, "
        f"updates_below_0.7={metrics['updates_below_0.7']}, "
        f"final_F={metrics['final_F']:.4f}"
    )
