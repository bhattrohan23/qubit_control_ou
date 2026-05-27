"""
Extract proxy evaluation metrics from existing episodic training pkl files.

These are NOT frozen-policy eval — they come from stochastic-policy training
episodes (64 envs per update). But the tail of a converged run gives a good
approximation of E[F], std(F), and rough range.

What you get (approximations):
    mean_F       ~ E[F] under stochastic policy at convergence
    std_F        ~ typical episode-to-episode spread
    max_F        ~ near-ceiling fidelity seen in training
    min_F        ~ near-floor fidelity seen in training
    frac_solved  ~ P(F > 0.5) during training tail

What you cannot get without re-running:
    P(F > 0.90 / 0.95 / 0.99)  — need individual fidelities
    median / p10 / p25          — need individual fidelities
    deterministic-policy bias   — pi.mode() typically ~1-3% better than pi.sample()

Usage:
    python analyze_training_tail.py
    python analyze_training_tail.py --tail 200 --out eval_results/training_tail.csv
"""

import os
import re
import pickle
import argparse
import csv
import numpy as np


def parse_condition(stem: str) -> dict:
    """Parse condition dict from pkl filename stem.

    Examples:
        qubit_control_memoryless_tau0.1_seed42  → k=0,  tau=0.1, seed=42
        qubit_control_context50_tau5_seed42     → k=50, tau=5.0, seed=42
        qubit_control_memoryless_seed42         → k=0,  tau=1.0, seed=42
        qubit_control_context5_seed43           → k=5,  tau=1.0, seed=43
    """
    # strip prefix
    name = stem.removeprefix("qubit_control_")

    k = 0
    if name.startswith("memoryless"):
        k = 0
        name = name.removeprefix("memoryless").lstrip("_")
    elif m := re.match(r"context(\d+)_?(.*)", name):
        k = int(m.group(1))
        name = m.group(2)

    tau = 1.0
    if m := re.search(r"tau([\d.]+)", name):
        tau = float(m.group(1))

    seed = None
    if m := re.search(r"seed(\d+)", name):
        seed = int(m.group(1))

    return {"k": k, "tau": tau, "seed": seed}


def tail_metrics(data: list, tail: int) -> dict:
    """Compute proxy metrics from the final `tail` update steps."""
    chunk = data[-tail:]
    mean_F  = np.mean([d["mean_fidelity"]    for d in chunk])
    std_F   = np.mean([d["std_fidelity"]     for d in chunk])
    max_F   = np.mean([d["max_fidelity"]     for d in chunk])
    min_F   = np.mean([d["min_fidelity"]     for d in chunk])
    frac    = np.mean([d["fraction_solved"]  for d in chunk])
    omegax  = np.mean([d["mean_omega_x"]     for d in chunk])
    n_steps = len(data)
    last_ts = int(data[-1]["timestep"])
    return {
        "n_updates":     n_steps,
        "last_timestep": last_ts,
        "tail_steps":    len(chunk),
        "mean_F":        float(mean_F),
        "std_F":         float(std_F),
        "max_F":         float(max_F),
        "min_F":         float(min_F),
        "frac_F_gt_0.5": float(frac),
        "mean_omega_x":  float(omegax),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="episodic_data/qubit_control")
    parser.add_argument("--tail", type=int, default=100,
                        help="Number of final update steps to average over")
    parser.add_argument("--out", default="eval_results/training_tail.csv")
    args = parser.parse_args()

    pkls = sorted(f for f in os.listdir(args.data_dir) if f.endswith(".pkl"))

    skip = {"qubit_control_memoryless_test.pkl",
            "qubit_control_local_save.pkl",
            "qubit_control_diagnostic_tau1.pkl"}
    pkls = [p for p in pkls if p not in skip]

    rows = []
    for fname in pkls:
        stem = fname.removesuffix(".pkl")
        try:
            cond = parse_condition(stem)
        except Exception as e:
            print(f"  skip {fname}: {e}")
            continue

        path = os.path.join(args.data_dir, fname)
        with open(path, "rb") as f:
            data = pickle.load(f)

        if not data:
            print(f"  skip {fname}: empty")
            continue

        m = tail_metrics(data, args.tail)
        row = {"file": fname, **cond, **m}
        rows.append(row)

    rows.sort(key=lambda r: (r["k"], r["tau"], r.get("seed") or 0))

    # --- print table ---
    W = 80
    print(f"\n{'─'*W}")
    print(f"  Training-tail proxy metrics  (tail={args.tail} update steps)")
    print(f"  Source: episodic_data — stochastic policy, ~64 eps/step")
    print(f"  NOTE: E[F] is ~1-3% lower than frozen deterministic eval")
    print(f"{'─'*W}")
    hdr = f"{'condition':<38} {'k':>4} {'τ':>6} {'seed':>5}  {'E[F]':>7} {'std':>6} {'max':>7} {'min':>7} {'P>0.5':>6}"
    print(hdr)
    print(f"{'─'*W}")
    for r in rows:
        cname = r["file"].removeprefix("qubit_control_").removesuffix(".pkl")
        print(
            f"  {cname:<36} {r['k']:>4}  {r['tau']:>5.1f}  {r['seed'] or '?':>5}"
            f"  {r['mean_F']:>6.4f} {r['std_F']:>6.4f} {r['max_F']:>6.4f}"
            f" {r['min_F']:>6.4f} {r['frac_F_gt_0.5']:>6.4f}"
        )
    print(f"{'─'*W}\n")

    # --- save CSV ---
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fieldnames = ["file", "k", "tau", "seed",
                  "n_updates", "last_timestep", "tail_steps",
                  "mean_F", "std_F", "max_F", "min_F",
                  "frac_F_gt_0.5", "mean_omega_x"]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"Saved → {args.out}")


if __name__ == "__main__":
    main()
