import argparse
import os
import pickle
import re
from datetime import datetime

import numpy as np

VALID_PATTERN     = re.compile(r"^qubit_control_(memoryless|context(\d+))_seed(\d+)\.pkl$")
VALID_PATTERN_TAU = re.compile(r"^qubit_control_(memoryless|context(\d+))_tau([\d.]+)_seed(\d+)\.pkl$")


def parse_filename(fname):
    m = VALID_PATTERN_TAU.match(fname)
    if m:
        condition_raw, k_val, tau_str, seed_str = m.group(1), m.group(2), m.group(3), m.group(4)
        condition = "memoryless" if condition_raw == "memoryless" else f"k={k_val}"
        return condition, int(seed_str), float(tau_str)
    m = VALID_PATTERN.match(fname)
    if m:
        condition_raw, k_val, seed_str = m.group(1), m.group(2), m.group(3)
        condition = "memoryless" if condition_raw == "memoryless" else f"k={k_val}"
        return condition, int(seed_str), None
    return None, None, None


def compute_metrics(data, tau, warmup=50):
    F = np.array([d["mean_fidelity"] for d in data])
    omega = np.array([d["mean_omega_x"] for d in data])
    n = len(F)

    # Asymptotic performance
    final_F_mean = float(np.mean(F[-100:]))
    final_F_std = float(np.std(F[-100:]))
    pct25_start = int(0.75 * n)
    final_F_25pct_mean = float(np.mean(F[pct25_start:]))

    # Training stability
    running_max = np.maximum.accumulate(F)
    drops = running_max[warmup:] - F[warmup:]
    worst_drop_idx_rel = int(np.argmax(drops))
    worst_drop_at = warmup + worst_drop_idx_rel
    worst_drop = float(drops[worst_drop_idx_rel])
    peak_before_drop = float(running_max[worst_drop_at])
    low_at_drop = float(F[worst_drop_at])
    updates_below_07 = int(np.sum(F[warmup:] < 0.7))
    recovery_indices = np.where(F[worst_drop_at + 1:] > 0.9)[0]
    recovery_time_to_F09 = int(recovery_indices[0] + 1) if len(recovery_indices) > 0 else -1

    # Action character
    final_omega_mean = float(np.mean(omega[-100:]))
    final_omega_std = float(np.std(omega[-100:]))
    mean_action_smoothness = float(np.mean(np.abs(np.diff(omega[-100:]))))

    return {
        "n": n,
        "final_F_mean": final_F_mean,
        "final_F_std": final_F_std,
        "final_F_25pct_mean": final_F_25pct_mean,
        "worst_drop": worst_drop,
        "worst_drop_at": worst_drop_at,
        "peak_before_drop": peak_before_drop,
        "low_at_drop": low_at_drop,
        "updates_below_07": updates_below_07,
        "recovery_time_to_F09": recovery_time_to_F09,
        "final_omega_mean": final_omega_mean,
        "final_omega_std": final_omega_std,
        "mean_action_smoothness": mean_action_smoothness,
    }


def write_report(pkl_path, fname, condition, seed, tau, metrics, out_path):
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        f"Run: {fname}",
        f"Generated: {now}",
        "",
        "=== Run metadata ===",
        f"condition: {condition}",
        f"seed: {seed}",
        f"tau: {tau}",
        f"num_updates: {metrics['n']}",
        f"pkl_path: {pkl_path}",
        "",
        "=== Asymptotic performance (last 100 updates) ===",
        f"final_F_mean: {metrics['final_F_mean']:.4f}",
        f"final_F_std: {metrics['final_F_std']:.4f}",
        f"final_F_25pct_mean: {metrics['final_F_25pct_mean']:.4f}",
        "",
        "=== Training stability ===",
        f"worst_drop: {metrics['worst_drop']:.4f}",
        f"worst_drop_at: {metrics['worst_drop_at']}",
        f"peak_before_drop: {metrics['peak_before_drop']:.4f}",
        f"low_at_drop: {metrics['low_at_drop']:.4f}",
        f"updates_below_0.7: {metrics['updates_below_07']}",
        f"recovery_time_to_F09: {metrics['recovery_time_to_F09']}",
        "",
        "=== Action character (last 100 updates) ===",
        f"final_omega_mean: {metrics['final_omega_mean']:.4f}",
        f"final_omega_std: {metrics['final_omega_std']:.4f}",
        f"mean_action_smoothness: {metrics['mean_action_smoothness']:.4f}",
    ]
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Analyze qubit control training pkl files.")
    parser.add_argument("directory", nargs="?", default="episodic_data/qubit_control/",
                        help="Directory containing pkl files")
    parser.add_argument("--tau", type=float, default=1.0, help="Tau value (default: 1.0)")
    args = parser.parse_args()

    directory = args.directory
    tau = args.tau

    fnames = sorted(f for f in os.listdir(directory) if f.endswith(".pkl"))
    for fname in fnames:
        condition, seed, tau_from_fname = parse_filename(fname)
        if condition is None:
            continue

        effective_tau = tau_from_fname if tau_from_fname is not None else tau

        pkl_path = os.path.join(directory, fname)
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        metrics = compute_metrics(data, effective_tau)

        out_fname = fname.replace(".pkl", "_metrics.txt")
        out_path = os.path.join(directory, out_fname)
        write_report(pkl_path, fname, condition, seed, effective_tau, metrics, out_path)
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
