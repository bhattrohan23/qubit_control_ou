#!/usr/bin/env python3
"""
Postprocess final-checkpoint frozen-policy evaluation results.

Parses .txt result files only. PKL parsing is not implemented in this script.

CPU-only. No training, no GPU, no policy re-evaluation. Reads result text
files produced by the qubit-control PPO eval pipeline and emits CSVs, plots,
and a markdown report focused on the FINAL checkpoint of each run.

Usage
-----
    python analyze_final_eval.py --input_dir eval_results --output_dir analysis_outputs
    python analyze_final_eval.py --files a.txt b.txt --output_dir analysis_outputs

Duplicate handling
------------------
If the same (tau, method, seed) appears in more than one input file, the script
errors by default (data-integrity protection: e.g. older `retrain_..._tau10.0_seed42`
and newer `..._tau10_seedXX` files should not be silently mixed). Pass
`--allow_duplicates keep_last` to override and keep the later occurrence.
"""
from __future__ import annotations

import argparse
import os
import re
import sys

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from scipy.stats import trim_mean as _scipy_trim_mean
    from scipy.stats import binomtest as _scipy_binomtest
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

METRICS = [
    "mean_F", "median_F", "p25_F", "p10_F", "min_F",
    "P_gt_90", "P_gt_95", "P_gt_99", "worst5_mean",
]

PROFILE_METRICS = ["mean_F", "median_F", "p10_F", "P_gt_90", "P_gt_95"]

BOOT_N = 10_000
BOOT_SEED = 12345


# ---------------------------------------------------------------------------
# Regexes
# ---------------------------------------------------------------------------

RUN_NAME_RE = re.compile(
    r"^(?:retrain_)?(?P<method>memoryless|context50)"
    r"_tau(?P<tau>\d+(?:\.\d+)?)"
    r"_seed(?P<seed>\d+)\s*$"
)

# Snapshot rows: timestep (with commas), n_eval, then 7 floats.
SNAPSHOT_RE = re.compile(
    r"^\s*([\d,]+)\s+(\d+)"
    r"\s+([\-\d.eE+]+)\s+([\-\d.eE+]+)"      # mean, std
    r"\s+([\-\d.eE+]+)\s+([\-\d.eE+]+)"      # median, p10
    r"\s+([\-\d.eE+]+)\s+([\-\d.eE+]+)\s+([\-\d.eE+]+)\s*$"  # P>0.90, P>0.95, P>0.99
)

FINAL_BLOCK_RE = re.compile(
    r"---\s*FINAL FROZEN-POLICY EVAL.*?---\s*\n(.*?)(?=\n---|\Z)",
    re.DOTALL,
)
SNAPSHOTS_BLOCK_RE = re.compile(
    r"---\s*FROZEN-POLICY EVAL SNAPSHOTS\s*---\s*\n(.*?)(?=\n---|\Z)",
    re.DOTALL,
)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def split_runs(text: str):
    """Yield (run_name, body) per `RUN:` block in `text`."""
    parts = re.split(r"(?m)^RUN:\s*", text)
    for p in parts[1:]:
        nl = p.find("\n")
        if nl < 0:
            yield p.strip(), ""
        else:
            yield p[:nl].strip(), p[nl + 1:]


def _grab_float(block: str, pattern: str) -> float:
    m = re.search(pattern, block)
    return float(m.group(1)) if m else float("nan")


def parse_final_block(text: str) -> dict | None:
    m = FINAL_BLOCK_RE.search(text)
    if not m:
        return None
    block = m.group(1)

    n_eval_m = re.search(r"n_eval:\s*(\d+)", block)
    ef_m     = re.search(r"E\[F\]:\s*([\-\d.eE+]+)\s*[±+/-]+\s*([\-\d.eE+]+)", block)
    medp25_m = re.search(r"Median\s*/\s*p25:\s*([\-\d.eE+]+)\s*/\s*([\-\d.eE+]+)", block)
    p10min_m = re.search(r"p10\s*/\s*min:\s*([\-\d.eE+]+)\s*/\s*([\-\d.eE+]+)", block)

    return {
        "n_eval":      int(n_eval_m.group(1)) if n_eval_m else None,
        "mean_F":      float(ef_m.group(1))     if ef_m     else float("nan"),
        "std_F":       float(ef_m.group(2))     if ef_m     else float("nan"),
        "median_F":    float(medp25_m.group(1)) if medp25_m else float("nan"),
        "p25_F":       float(medp25_m.group(2)) if medp25_m else float("nan"),
        "p10_F":       float(p10min_m.group(1)) if p10min_m else float("nan"),
        "min_F":       float(p10min_m.group(2)) if p10min_m else float("nan"),
        "P_gt_90":     _grab_float(block, r"P\(F\s*>\s*0\.90\):\s*([\-\d.eE+]+)"),
        "P_gt_95":     _grab_float(block, r"P\(F\s*>\s*0\.95\):\s*([\-\d.eE+]+)"),
        "P_gt_99":     _grab_float(block, r"P\(F\s*>\s*0\.99\):\s*([\-\d.eE+]+)"),
        "worst5_mean": _grab_float(block, r"worst[\-\s]*5%\s*mean:\s*([\-\d.eE+]+)"),
    }


def parse_last_snapshot(text: str) -> dict | None:
    m = SNAPSHOTS_BLOCK_RE.search(text)
    if not m:
        return None
    rows = []
    for line in m.group(1).splitlines():
        sm = SNAPSHOT_RE.match(line)
        if not sm:
            continue
        rows.append({
            "timestep": int(sm.group(1).replace(",", "")),
            "n_eval":   int(sm.group(2)),
            "mean_F":   float(sm.group(3)),
            "std_F":    float(sm.group(4)),
            "median_F": float(sm.group(5)),
            "p10_F":    float(sm.group(6)),
            "P_gt_90":  float(sm.group(7)),
            "P_gt_95":  float(sm.group(8)),
            "P_gt_99":  float(sm.group(9)),
        })
    if not rows:
        return None
    rows.sort(key=lambda r: (r["timestep"], r["n_eval"]))
    return rows[-1]


def extract_record(run_name: str, body: str, source_file: str) -> dict | None:
    name_m = RUN_NAME_RE.match(run_name)
    if not name_m:
        print(f"  warning: could not parse RUN name '{run_name}' in {source_file}; skipping")
        return None
    method = name_m.group("method")
    tau    = float(name_m.group("tau"))
    seed   = int(name_m.group("seed"))
    k      = 0 if method == "memoryless" else 50

    final = parse_final_block(body)
    snap  = parse_last_snapshot(body)

    if final is None and snap is None:
        print(f"  warning: no FINAL block or snapshots in '{run_name}' ({source_file}); skipping")
        return None

    if final is None:
        print(f"  warning: FINAL block missing for '{run_name}' ({source_file}); "
              f"falling back to last snapshot (p25, min, worst5_mean will be NaN)")
        rec = {
            "tau": tau, "k": k, "method": method, "seed": seed,
            "timestep": snap["timestep"], "n_eval": snap["n_eval"],
            "mean_F":  snap["mean_F"],  "std_F":  snap["std_F"],
            "median_F": snap["median_F"], "p25_F": float("nan"),
            "p10_F":   snap["p10_F"],   "min_F":  float("nan"),
            "P_gt_90": snap["P_gt_90"], "P_gt_95": snap["P_gt_95"], "P_gt_99": snap["P_gt_99"],
            "worst5_mean": float("nan"),
            "source_file": source_file,
        }
    else:
        timestep = snap["timestep"] if snap else None
        n_eval   = final["n_eval"] if final["n_eval"] is not None else (snap["n_eval"] if snap else None)
        rec = {
            "tau": tau, "k": k, "method": method, "seed": seed,
            "timestep": timestep, "n_eval": n_eval,
            "mean_F":  final["mean_F"],   "std_F":  final["std_F"],
            "median_F": final["median_F"], "p25_F":  final["p25_F"],
            "p10_F":   final["p10_F"],    "min_F":  final["min_F"],
            "P_gt_90": final["P_gt_90"],  "P_gt_95": final["P_gt_95"], "P_gt_99": final["P_gt_99"],
            "worst5_mean": final["worst5_mean"],
            "source_file": source_file,
        }
        for col in ("p25_F", "min_F", "worst5_mean"):
            if np.isnan(rec[col]):
                print(f"  warning: '{col}' missing in FINAL block for '{run_name}' ({source_file}); stored as NaN")
    return rec


# ---------------------------------------------------------------------------
# Stats: IQM + bootstrap CIs (over seeds, never over eval episodes)
# ---------------------------------------------------------------------------

def iqm(values) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return float("nan")
    if _HAS_SCIPY:
        return float(_scipy_trim_mean(arr, proportiontocut=0.25))
    arr_sorted = np.sort(arr)
    n = arr_sorted.size
    k = int(np.floor(n * 0.25))
    middle = arr_sorted[k:n - k] if (n - 2 * k) > 0 else arr_sorted
    return float(middle.mean())


def _boot_indices(n: int, n_boot: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, n, size=(n_boot, n))


def bootstrap_ci_mean(values, n_boot: int = BOOT_N, seed: int = BOOT_SEED) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return (float("nan"), float("nan"))
    idx = _boot_indices(arr.size, n_boot, seed)
    boots = arr[idx].mean(axis=1)
    return (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))


def bootstrap_ci_iqm(values, n_boot: int = BOOT_N, seed: int = BOOT_SEED) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    n = arr.size
    if n == 0:
        return (float("nan"), float("nan"))
    idx = _boot_indices(n, n_boot, seed)
    samples = arr[idx]  # (n_boot, n)
    if _HAS_SCIPY:
        boots = _scipy_trim_mean(samples, proportiontocut=0.25, axis=1)
    else:
        s = np.sort(samples, axis=1)
        k = int(np.floor(n * 0.25))
        boots = s[:, k:n - k].mean(axis=1) if (n - 2 * k) > 0 else s.mean(axis=1)
    return (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------

def summarize_by_tau_method(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (tau, method), g in df.groupby(["tau", "method"]):
        for metric in METRICS:
            vals = g[metric].dropna().to_numpy()
            n = vals.size
            if n == 0:
                rows.append({
                    "tau": tau, "method": method, "metric": metric, "n": 0,
                    "mean": np.nan, "median": np.nan, "IQM": np.nan,
                    "std": np.nan, "min": np.nan, "max": np.nan,
                    "mean_ci_lo": np.nan, "mean_ci_hi": np.nan,
                    "iqm_ci_lo":  np.nan, "iqm_ci_hi":  np.nan,
                })
                continue
            mean_lo, mean_hi = bootstrap_ci_mean(vals)
            iqm_lo,  iqm_hi  = bootstrap_ci_iqm(vals)
            rows.append({
                "tau": tau, "method": method, "metric": metric, "n": n,
                "mean":   float(np.mean(vals)),
                "median": float(np.median(vals)),
                "IQM":    iqm(vals),
                "std":    float(np.std(vals, ddof=1)) if n > 1 else 0.0,
                "min":    float(np.min(vals)),
                "max":    float(np.max(vals)),
                "mean_ci_lo": mean_lo, "mean_ci_hi": mean_hi,
                "iqm_ci_lo":  iqm_lo,  "iqm_ci_hi":  iqm_hi,
            })
    return pd.DataFrame(rows)


def paired_diffs(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for tau, g in df.groupby("tau"):
        mem = g[g["method"] == "memoryless"].set_index("seed")
        ctx = g[g["method"] == "context50"].set_index("seed")
        common = sorted(set(mem.index) & set(ctx.index))
        only_mem = sorted(set(mem.index) - set(ctx.index))
        only_ctx = sorted(set(ctx.index) - set(mem.index))
        if only_mem or only_ctx:
            print(f"  warning: tau={tau:g} seed mismatch — "
                  f"memoryless-only seeds {only_mem}, context50-only seeds {only_ctx}; "
                  f"paired analysis uses intersection {common}.")
        for seed in common:
            row = {"tau": tau, "seed": seed}
            for metric in METRICS:
                row[f"delta_{metric}"] = float(ctx.loc[seed, metric] - mem.loc[seed, metric])
            rows.append(row)
    return pd.DataFrame(rows)


def summarize_paired(paired_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if paired_df.empty:
        return pd.DataFrame(rows)
    for tau, g in paired_df.groupby("tau"):
        for metric in METRICS:
            col = f"delta_{metric}"
            vals = g[col].dropna().to_numpy()
            n = vals.size
            if n == 0:
                continue
            wins = int(np.sum(vals > 0))
            mean_lo, mean_hi = bootstrap_ci_mean(vals)
            iqm_lo,  iqm_hi  = bootstrap_ci_iqm(vals)
            row = {
                "tau": tau, "metric": metric, "n_pairs": n,
                "mean_delta":   float(np.mean(vals)),
                "median_delta": float(np.median(vals)),
                "IQM_delta":    iqm(vals),
                "std_delta":    float(np.std(vals, ddof=1)) if n > 1 else 0.0,
                "min_delta":    float(np.min(vals)),
                "max_delta":    float(np.max(vals)),
                "ca50_wins":    wins,
                "ca50_win_fraction": wins / n,
                "mean_delta_ci_lo": mean_lo, "mean_delta_ci_hi": mean_hi,
                "iqm_delta_ci_lo":  iqm_lo,  "iqm_delta_ci_hi":  iqm_hi,
            }
            if _HAS_SCIPY:
                # Two-sided sign test, excluding exact zeros (standard convention).
                non_zero = vals[vals != 0]
                if non_zero.size > 0:
                    successes = int(np.sum(non_zero > 0))
                    res = _scipy_binomtest(successes, non_zero.size, p=0.5, alternative="two-sided")
                    row["sign_test_p"] = float(res.pvalue)
                else:
                    row["sign_test_p"] = float("nan")
            rows.append(row)
    return pd.DataFrame(rows)


def performance_profiles(df: pd.DataFrame, metrics=PROFILE_METRICS) -> pd.DataFrame:
    thetas = np.round(np.arange(0.0, 1.0 + 1e-9, 0.01), 4)
    rows = []
    for (tau, method), g in df.groupby(["tau", "method"]):
        for metric in metrics:
            vals = g[metric].dropna().to_numpy()
            n = vals.size
            for theta in thetas:
                frac = float(np.mean(vals > theta)) if n > 0 else float("nan")
                rows.append({
                    "tau": tau, "method": method, "metric": metric,
                    "theta": float(theta), "frac_above": frac, "n_seeds": n,
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _fmt_tau_for_file(tau: float) -> str:
    return f"{int(tau)}" if float(tau).is_integer() else f"{tau:g}".replace(".", "p")


def plot_paired_diff_bars(paired_df: pd.DataFrame, metric: str, output_dir: str) -> None:
    if paired_df.empty:
        return
    col = f"delta_{metric}"
    for tau, g in paired_df.groupby("tau"):
        g = g.sort_values("seed")
        deltas = g[col].to_numpy()
        seeds  = g["seed"].astype(str).to_list()
        colors = ["#2a7fbf" if d >= 0 else "#c0392b" for d in deltas]
        fig, ax = plt.subplots(figsize=(max(4.5, 0.5 * len(seeds) + 2), 4))
        ax.bar(seeds, deltas, color=colors, edgecolor="black", linewidth=0.6)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xlabel("seed")
        ax.set_ylabel(f"Δ {metric}  (CA50 − memoryless)")
        ax.set_title(f"Final checkpoint paired differences, τ = {tau:g}")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        out = os.path.join(output_dir, f"paired_diff_final_{metric}_tau{_fmt_tau_for_file(tau)}.png")
        fig.savefig(out, dpi=130)
        plt.close(fig)


def plot_performance_profile(profile_df: pd.DataFrame, metric: str, output_dir: str) -> None:
    sub = profile_df[profile_df["metric"] == metric]
    if sub.empty:
        return
    for tau, g in sub.groupby("tau"):
        fig, ax = plt.subplots(figsize=(6, 4))
        for method, h in g.groupby("method"):
            h_sorted = h.sort_values("theta")
            n_seeds = int(h_sorted["n_seeds"].iloc[0])
            ax.plot(h_sorted["theta"], h_sorted["frac_above"],
                    label=f"{method} (n={n_seeds})", lw=1.8)
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("θ")
        ax.set_ylabel(f"P({metric} > θ) across seeds")
        ax.set_title(f"Performance profile, {metric}, τ = {tau:g}")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        out = os.path.join(output_dir, f"performance_profile_final_{metric}_tau{_fmt_tau_for_file(tau)}.png")
        fig.savefig(out, dpi=130)
        plt.close(fig)


def plot_scatter_memvsctx(df: pd.DataFrame, output_dir: str) -> None:
    for tau, g in df.groupby("tau"):
        mem = g[g["method"] == "memoryless"].set_index("seed")["mean_F"]
        ctx = g[g["method"] == "context50"].set_index("seed")["mean_F"]
        common = sorted(set(mem.index) & set(ctx.index))
        if not common:
            continue
        xs = np.array([mem.loc[s] for s in common])
        ys = np.array([ctx.loc[s] for s in common])
        fig, ax = plt.subplots(figsize=(5.5, 5.5))
        ax.scatter(xs, ys, s=55, color="#2a7fbf", edgecolor="black", linewidth=0.5, zorder=3)
        for s, x, y in zip(common, xs, ys):
            ax.annotate(str(s), (x, y), xytext=(4, 4), textcoords="offset points", fontsize=8)
        lo = float(min(xs.min(), ys.min(), 0.0))
        hi = float(max(xs.max(), ys.max(), 1.0))
        pad = 0.02
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=0.8, alpha=0.7, label="y = x")
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_xlabel("memoryless  final mean_F")
        ax.set_ylabel("context50  final mean_F")
        ax.set_title(f"Final mean_F, CA50 vs memoryless (τ = {tau:g})")
        ax.set_aspect("equal", "box")
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right")
        fig.tight_layout()
        out = os.path.join(output_dir, f"scatter_context_vs_memoryless_final_meanF_tau{_fmt_tau_for_file(tau)}.png")
        fig.savefig(out, dpi=130)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def _fmt(v) -> str:
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if np.isnan(f):
        return "NaN"
    return f"{f:.4f}"


def make_report(final_df: pd.DataFrame, summary_df: pd.DataFrame,
                paired_df: pd.DataFrame, paired_summary_df: pd.DataFrame,
                output_dir: str) -> str:
    L: list[str] = []
    L.append("# Final-checkpoint frozen-policy analysis report")
    L.append("")
    L.append(f"- Final-eval rows parsed: **{len(final_df)}**")
    L.append(f"- τ values: {sorted(final_df['tau'].unique().tolist())}")
    L.append(f"- methods: {sorted(final_df['method'].unique().tolist())}")
    L.append(f"- bootstrap: {BOOT_N:,} resamples, RNG seed {BOOT_SEED}, "
             f"resampling over **seeds** (not eval episodes)")
    if not _HAS_SCIPY:
        L.append("- scipy not available: IQM uses manual trimming, sign-test p-value omitted")
    L.append("")

    # Coverage
    L.append("## Coverage: seeds per (τ, method)")
    L.append("")
    L.append("| τ | method | n_seeds | seeds |")
    L.append("|---|---|---|---|")
    for (tau, method), g in final_df.groupby(["tau", "method"]):
        seeds = sorted(g["seed"].unique().tolist())
        L.append(f"| {tau:g} | {method} | {len(seeds)} | {seeds} |")
    L.append("")

    # Mismatch warnings
    mismatched = False
    for tau, g in final_df.groupby("tau"):
        mem_s = set(g[g["method"] == "memoryless"]["seed"])
        ctx_s = set(g[g["method"] == "context50"]["seed"])
        if mem_s != ctx_s:
            if not mismatched:
                L.append("### Seed-set mismatch warnings")
                L.append("")
                mismatched = True
            only_m = sorted(mem_s - ctx_s)
            only_c = sorted(ctx_s - mem_s)
            inter  = sorted(mem_s & ctx_s)
            L.append(f"- **τ = {tau:g}**: memoryless-only seeds {only_m}, "
                     f"context50-only seeds {only_c}. "
                     f"Paired comparisons below use the intersection only: {inter}.")
    if mismatched:
        L.append("")

    # Per-tau summaries (each tau analyzed separately — never combine taus)
    for tau, g in summary_df.groupby("tau"):
        L.append(f"## Summary by method — τ = {tau:g}")
        L.append("")
        L.append("| method | metric | n | mean | median | IQM | std | min | max | mean 95% CI | IQM 95% CI |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for _, r in g.iterrows():
            L.append(
                f"| {r['method']} | {r['metric']} | {int(r['n'])} | "
                f"{_fmt(r['mean'])} | {_fmt(r['median'])} | {_fmt(r['IQM'])} | "
                f"{_fmt(r['std'])} | {_fmt(r['min'])} | {_fmt(r['max'])} | "
                f"[{_fmt(r['mean_ci_lo'])}, {_fmt(r['mean_ci_hi'])}] | "
                f"[{_fmt(r['iqm_ci_lo'])}, {_fmt(r['iqm_ci_hi'])}] |"
            )
        L.append("")

    # Per-tau paired differences
    if not paired_summary_df.empty:
        for tau, g in paired_summary_df.groupby("tau"):
            L.append(f"## Paired differences (CA50 − memoryless) — τ = {tau:g}")
            L.append("")
            header = ("| metric | n_pairs | mean Δ | median Δ | IQM Δ | std Δ | min Δ | max Δ | "
                      "CA50 wins | mean Δ 95% CI | IQM Δ 95% CI |")
            sep    = "|---|---|---|---|---|---|---|---|---|---|---|"
            if _HAS_SCIPY:
                header = header + " sign-test p |"
                sep    = sep + "---|"
            L.append(header)
            L.append(sep)
            for _, r in g.iterrows():
                cells = [
                    r["metric"], int(r["n_pairs"]),
                    _fmt(r["mean_delta"]), _fmt(r["median_delta"]), _fmt(r["IQM_delta"]),
                    _fmt(r["std_delta"]), _fmt(r["min_delta"]), _fmt(r["max_delta"]),
                    f"{int(r['ca50_wins'])}/{int(r['n_pairs'])} ({r['ca50_win_fraction']:.0%})",
                    f"[{_fmt(r['mean_delta_ci_lo'])}, {_fmt(r['mean_delta_ci_hi'])}]",
                    f"[{_fmt(r['iqm_delta_ci_lo'])}, {_fmt(r['iqm_delta_ci_hi'])}]",
                ]
                if _HAS_SCIPY:
                    cells.append(_fmt(r.get("sign_test_p", float("nan"))))
                L.append("| " + " | ".join(str(c) for c in cells) + " |")
            L.append("")

    # Interpretation (per tau, conservative)
    L.append("## Interpretation (per τ, descriptive only)")
    L.append("")
    L.append("> Each τ is analyzed independently — no cross-τ comparisons are made here. "
             "n is small in every condition, so bootstrap CIs are wide. "
             "\"CI excludes 0\" is described as a direction in the available seeds, **not** "
             "as a statistical significance claim.")
    L.append("")
    if paired_summary_df.empty:
        L.append("- No paired data available — nothing to interpret.")
    else:
        for tau, g in paired_summary_df.groupby("tau"):
            r = g[g["metric"] == "mean_F"]
            if r.empty:
                continue
            r = r.iloc[0]
            n_pairs = int(r["n_pairs"])
            md, lo, hi = r["mean_delta"], r["mean_delta_ci_lo"], r["mean_delta_ci_hi"]
            wins, frac = int(r["ca50_wins"]), r["ca50_win_fraction"]

            if np.isnan(md):
                verdict = "no paired mean_F data available."
            elif np.isnan(lo) or np.isnan(hi):
                verdict = "CI could not be computed."
            elif lo > 0:
                verdict = ("CA50 mean_F was higher than memoryless across the matched seeds "
                           "(the bootstrap CI for the per-seed Δ lies entirely above 0). "
                           "n is small, so this is a directional observation, not a significance claim.")
            elif hi < 0:
                verdict = ("Memoryless mean_F was higher than CA50 across the matched seeds "
                           "(the bootstrap CI for the per-seed Δ lies entirely below 0). "
                           "n is small, so this is a directional observation, not a significance claim.")
            else:
                verdict = ("The bootstrap CI for the per-seed Δ straddles 0 — the sign of the "
                           "effect is **seed-dependent** at this n. The data here do not support "
                           "a directional claim either way.")
            L.append(
                f"- **τ = {tau:g}** — n_pairs = {n_pairs}; mean Δ mean_F = {md:+.4f} "
                f"(95% CI [{lo:+.4f}, {hi:+.4f}]); CA50 wins {wins}/{n_pairs} ({frac:.0%}). {verdict}"
            )
    L.append("")
    L.append("## Files in this directory")
    L.append("")
    L.append("- `final_checkpoint_raw.csv` — one row per final eval (all parsed rows)")
    L.append("- `final_checkpoint_summary_by_tau_method.csv` — per (τ, method, metric) stats + bootstrap CIs")
    L.append("- `final_checkpoint_paired_seed_differences.csv` — per-seed CA50 − memoryless deltas (intersection only)")
    L.append("- `final_checkpoint_paired_difference_summary.csv` — per-(τ, metric) paired-Δ stats + bootstrap CIs")
    L.append("- `performance_profiles_final_long.csv` — long-form P(metric > θ) for θ ∈ [0, 1] step 0.01")
    L.append("- `paired_diff_final_{mean_F,p10_F}_tau*.png` — per-seed Δ bar plots")
    L.append("- `performance_profile_final_{mean_F,p10_F}_tau*.png` — survival-style profiles across seeds")
    L.append("- `scatter_context_vs_memoryless_final_meanF_tau*.png` — per-seed scatter with y=x")
    L.append("")

    out = os.path.join(output_dir, "final_checkpoint_analysis_report.md")
    with open(out, "w") as f:
        f.write("\n".join(L))
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def gather_files(input_dir: str | None, files: list[str] | None) -> list[str]:
    paths: list[str] = []
    if files:
        paths.extend(files)
    if input_dir:
        for root, _, fnames in os.walk(input_dir):
            for fn in fnames:
                if fn.endswith(".txt"):
                    paths.append(os.path.join(root, fn))
    return sorted(set(paths))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input_dir", type=str, default=None,
                   help="Directory to recursively search for .txt result files (PKL files are not parsed)")
    p.add_argument("--files", nargs="+", default=None,
                   help="Explicit list of result .txt files to parse (PKL files are not supported)")
    p.add_argument("--output_dir", type=str, default="analysis_outputs",
                   help="Directory to write CSVs, plots, and report")
    p.add_argument("--allow_duplicates", choices=["error", "keep_last"], default="error",
                   help="Behavior on duplicate (tau, method, seed) across input files. "
                        "Default 'error' aborts so stale + current results are never silently mixed. "
                        "'keep_last' warns and uses the later occurrence.")
    args = p.parse_args()

    if not args.input_dir and not args.files:
        p.error("Provide --input_dir or --files")

    paths = gather_files(args.input_dir, args.files)
    if not paths:
        print("No .txt files found.")
        return 1

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Parsing {len(paths)} file(s)…")
    records: list[dict] = []
    by_key: dict[tuple, dict] = {}
    duplicates: list[tuple] = []  # list of (key, prev_source, new_source)
    for path in paths:
        try:
            with open(path) as f:
                text = f.read()
        except OSError as e:
            print(f"  could not read {path}: {e}")
            continue
        for name, body in split_runs(text):
            rec = extract_record(name, body, path)
            if rec is None:
                continue
            key = (rec["tau"], rec["method"], rec["seed"])
            if key in by_key:
                duplicates.append((key, by_key[key]["source_file"], path))
            by_key[key] = rec  # provisional: later occurrence wins for keep_last mode
            records.append(rec)

    if duplicates:
        if args.allow_duplicates == "error":
            msg_lines = [
                "Duplicate (tau, method, seed) records detected across input files.",
                "This script refuses to silently mix possibly-stale results "
                "(e.g. older retrain_*_tau10.0_seed42 vs. newer *_tau10_seed42 files).",
                "",
                "Conflicts:",
            ]
            for (tau, method, seed), prev, new in duplicates:
                msg_lines.append(f"  (tau={tau:g}, {method}, seed={seed}): {prev}  AND  {new}")
            msg_lines.append("")
            msg_lines.append(
                "Resolve by removing/filtering files, or rerun with "
                "`--allow_duplicates keep_last` to keep the later occurrence."
            )
            raise ValueError("\n".join(msg_lines))
        else:
            for (tau, method, seed), prev, new in duplicates:
                print(f"  duplicate (tau={tau:g}, {method}, seed={seed}): "
                      f"already from {prev}, now in {new}. Keeping later occurrence.")

    if not by_key:
        print("No parsable records found.")
        return 1

    final_df = pd.DataFrame(list(by_key.values()))
    final_df = final_df.sort_values(["tau", "method", "seed"]).reset_index(drop=True)

    # CSVs
    raw_path = os.path.join(args.output_dir, "final_checkpoint_raw.csv")
    final_df.to_csv(raw_path, index=False)

    summary_df = summarize_by_tau_method(final_df)
    summary_path = os.path.join(args.output_dir, "final_checkpoint_summary_by_tau_method.csv")
    summary_df.to_csv(summary_path, index=False)

    paired_df = paired_diffs(final_df)
    paired_path = os.path.join(args.output_dir, "final_checkpoint_paired_seed_differences.csv")
    paired_df.to_csv(paired_path, index=False)

    paired_summary_df = summarize_paired(paired_df)
    paired_summary_path = os.path.join(args.output_dir, "final_checkpoint_paired_difference_summary.csv")
    paired_summary_df.to_csv(paired_summary_path, index=False)

    profile_df = performance_profiles(final_df)
    profile_path = os.path.join(args.output_dir, "performance_profiles_final_long.csv")
    profile_df.to_csv(profile_path, index=False)

    # Plots
    if not paired_df.empty:
        plot_paired_diff_bars(paired_df, "mean_F", args.output_dir)
        plot_paired_diff_bars(paired_df, "p10_F",  args.output_dir)
        plot_scatter_memvsctx(final_df, args.output_dir)
    plot_performance_profile(profile_df, "mean_F", args.output_dir)
    plot_performance_profile(profile_df, "p10_F",  args.output_dir)

    # Report
    report_path = make_report(final_df, summary_df, paired_df, paired_summary_df, args.output_dir)

    # Console summary
    print()
    print("=" * 72)
    print(f"Outputs written to: {os.path.abspath(args.output_dir)}")
    print(f"Final-eval rows parsed: {len(final_df)}")
    print()
    print("Coverage (τ / method → seeds):")
    cov = (final_df.groupby(["tau", "method"])["seed"]
           .agg(lambda s: sorted(s.unique().tolist()))
           .reset_index(name="seeds"))
    cov["n_seeds"] = cov["seeds"].apply(len)
    print(cov[["tau", "method", "n_seeds", "seeds"]].to_string(index=False))
    print()
    print("Paired coverage (matched seeds per τ):")
    if paired_df.empty:
        print("  (no paired data)")
    else:
        pc = (paired_df.groupby("tau")["seed"]
              .agg(lambda s: sorted(s.unique().tolist()))
              .reset_index(name="paired_seeds"))
        pc["n_pairs"] = pc["paired_seeds"].apply(len)
        print(pc[["tau", "n_pairs", "paired_seeds"]].to_string(index=False))
    print()
    print(f"Markdown report: {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
