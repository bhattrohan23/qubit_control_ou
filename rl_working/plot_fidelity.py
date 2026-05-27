import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

# --- inputs ---
base_dir   = "episodic_data/qubit_control"
condition  = "context5"
seeds      = [42, 43, 44]
out_path   = "plots/exp2/context_aware_curve_5000_updates_tau_1.0_multiseed.png"
title      = "5000 updates of Context Aware PPO, τ=1.0 (seeds 42–44, mean ± std)"
omega_max  = 2.0
# --------------

def load_seed(condition, seed):
    path = f"{base_dir}/qubit_control_{condition}_seed{seed}.pkl"
    with open(path, "rb") as f:
        data = pickle.load(f)
    return {
        'timesteps':       np.array([d['timestep']        for d in data], dtype=float),
        'max_fidelity':    np.array([d['max_fidelity']    for d in data], dtype=float),
        'mean_fidelity':   np.array([d['mean_fidelity']   for d in data], dtype=float),
        'fraction_solved': np.array([d['fraction_solved'] for d in data], dtype=float),
        'mean_omega_x':    np.array([d['mean_omega_x']    for d in data], dtype=float),
        'std_omega_x':     np.array([d['std_omega_x']     for d in data], dtype=float),
    }

runs = [load_seed(condition, s) for s in seeds]
timesteps = runs[0]['timesteps']

def agg(key):
    stacked = np.stack([r[key] for r in runs], axis=0)  # (n_seeds, T)
    return stacked.mean(axis=0), stacked.std(axis=0)

max_fid_mean,    max_fid_std    = agg('max_fidelity')
mean_fid_mean,   mean_fid_std   = agg('mean_fidelity')
frac_mean,       frac_std       = agg('fraction_solved')
mean_omega_mean, mean_omega_std = agg('mean_omega_x')

last_n = 100
print(f"Across {len(seeds)} seeds, mean F (last {last_n} updates) = "
      f"{mean_fid_mean[-last_n:].mean():.4f} ± {mean_fid_std[-last_n:].mean():.4f}")

os.makedirs(os.path.dirname(out_path), exist_ok=True)

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle(title, fontsize=13)

def plot_band(ax, x, mean, std, color, label=None):
    ax.plot(x, mean, color=color, label=label)
    ax.fill_between(x, mean - std, mean + std, alpha=0.25, color=color)

# Panel 1: max fidelity
ax = axes[0, 0]
plot_band(ax, timesteps, max_fid_mean, max_fid_std, "steelblue")
ax.set_ylim(0.5, 1.01)
ax.set_xlabel("Timestep")
ax.set_ylabel("Max Fidelity")
ax.set_title("Best episode fidelity")
ax.grid(True, alpha=0.3)

# Panel 2: mean fidelity
ax = axes[0, 1]
plot_band(ax, timesteps, mean_fid_mean, mean_fid_std, "darkorange")
ax.set_ylim(0, 1)
ax.set_xlabel("Timestep")
ax.set_ylabel("Mean Fidelity")
ax.set_title("Mean episode fidelity")
ax.grid(True, alpha=0.3)

# Panel 3: fraction solved
ax = axes[1, 0]
plot_band(ax, timesteps, frac_mean, frac_std, "seagreen")
ax.set_ylim(-0.05, 1.05)
ax.set_xlabel("Timestep")
ax.set_ylabel("Fraction F > 0.5")
ax.set_title("Fraction of episodes solved")
ax.grid(True, alpha=0.3)

# Panel 4: action magnitude
ax = axes[1, 1]
plot_band(ax, timesteps, mean_omega_mean, mean_omega_std, "purple", label="mean |ωx| ± std across seeds")
ax.axhline(omega_max, color="gray", linestyle="--", linewidth=0.8, label="ωmax")
ax.set_xlabel("Timestep")
ax.set_ylabel("|ωx| (rad/s)")
ax.set_title("Action magnitude distribution")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(out_path, dpi=150)
plt.show()
