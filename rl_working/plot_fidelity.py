import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

# --- inputs ---
pkl_path   = "episodic_data/qubit_control/qubit_control_memoryless_test.pkl"
out_path   = "plots/exp2/memoryless_curve_tau_1.0.png"
title      = "Memoryless PPO, τ=1.0"
# Set to the episode length if pkl was generated before the terminal-fid fix
# (mean_fidelity and fraction_solved were averaged over all NUM_STEPS, not
# just the terminal step). Set to 1 for runs generated after the fix.
dilution_correction = 1000
# --------------

with open(pkl_path, "rb") as f:
    data = pickle.load(f)

steps          = np.array([d["timestep"]      for d in data], dtype=float)
max_fid        = np.array([d["max_fidelity"]  for d in data], dtype=float)
mean_fid       = np.array([d["mean_fidelity"] for d in data], dtype=float) * dilution_correction
frac_solved    = np.array([d["fraction_solved"] for d in data], dtype=float) * dilution_correction
mean_omega     = np.array([d["mean_omega_x"]  for d in data], dtype=float)
std_omega      = np.array([d["std_omega_x"]   for d in data], dtype=float)

os.makedirs(os.path.dirname(out_path), exist_ok=True)

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle(title, fontsize=13)

# Panel 1: max fidelity (log scale) — "can the agent solve the task?"
ax = axes[0, 0]
ax.plot(steps, max_fid, color="steelblue")
ax.set_yscale("log")
ax.set_ylim(1e-3, 1.0)
ax.set_xlabel("Timestep")
ax.set_ylabel("Max Fidelity")
ax.set_title("Best episode fidelity")
ax.grid(True, which="both", alpha=0.3)

# Panel 2: mean fidelity (linear) — "has the policy converged?"
ax = axes[0, 1]
ax.plot(steps, mean_fid, color="darkorange")
ax.set_ylim(0, 1)
ax.set_xlabel("Timestep")
ax.set_ylabel("Mean Fidelity")
ax.set_title("Mean episode fidelity")
ax.grid(True, alpha=0.3)

# Panel 3: fraction of episodes with F > 0.5 — "is the policy generalising?"
ax = axes[1, 0]
ax.plot(steps, frac_solved, color="seagreen")
ax.set_ylim(0, 1)
ax.set_xlabel("Timestep")
ax.set_ylabel("Fraction F > 0.5")
ax.set_title("Fraction of episodes solved")
ax.grid(True, alpha=0.3)

# Panel 4: action statistics — "is the policy still exploring?"
ax = axes[1, 1]
ax.plot(steps, mean_omega, color="purple", label="mean |ωx|")
ax.fill_between(steps,
                np.clip(mean_omega - std_omega, 0, None),
                mean_omega + std_omega,
                alpha=0.25, color="purple", label="±1 std")
ax.axhline(2.0, color="gray", linestyle="--", linewidth=0.8, label="ωmax")
ax.set_xlabel("Timestep")
ax.set_ylabel("|ωx| (rad/s)")
ax.set_title("Action magnitude distribution")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(out_path, dpi=150)
plt.show()
