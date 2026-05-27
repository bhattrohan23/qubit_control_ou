import os
import pickle
import matplotlib.pyplot as plt
import numpy as np

BASE = 'episodic_data/qubit_control'

def load(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def unpack(data):
    return {
        'timesteps':       np.array([d['timestep']        for d in data], dtype=float),
        'mean_fidelity':   np.array([d['mean_fidelity']   for d in data], dtype=float),
        'fraction_solved': np.array([d['fraction_solved'] for d in data], dtype=float),
        'mean_omega_x':    np.array([d['mean_omega_x']    for d in data], dtype=float),
        'max_fidelity':    np.array([d['max_fidelity']    for d in data], dtype=float),
    }

def summarize(name, mean_fidelity, mean_omega_x, last_n=100):
    print(f"  {name:<35} | mean F (last {last_n}) = {mean_fidelity[-last_n:].mean():.4f} "
          f"| peak F = {mean_fidelity.max():.4f} | mean |ωx| = {mean_omega_x.mean():.4f}")

os.makedirs('plots', exist_ok=True)

memoryless_seed43 = unpack(load(f'{BASE}/qubit_control_memoryless_seed43.pkl'))
context50         = unpack(load(f'{BASE}/qubit_control_context50_seed43.pkl'))

print("\n=== seed 43 ===")
summarize("Memoryless seed=43",         memoryless_seed43['mean_fidelity'], memoryless_seed43['mean_omega_x'])
summarize("Context-aware k=50 seed=43", context50['mean_fidelity'],         context50['mean_omega_x'])

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].plot(memoryless_seed43['timesteps'], memoryless_seed43['mean_fidelity'], label='Memoryless (k=0)',      color='C0')
axes[0, 0].plot(context50['timesteps'],         context50['mean_fidelity'],         label='Context-aware (k=50)', color='C1')
axes[0, 0].set_ylabel('Mean Fidelity')
axes[0, 0].set_xlabel('Timestep')
axes[0, 0].set_ylim(0, 1)
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)
axes[0, 0].set_title('Mean episode fidelity')

axes[0, 1].plot(memoryless_seed43['timesteps'], memoryless_seed43['fraction_solved'], label='Memoryless',     color='C0')
axes[0, 1].plot(context50['timesteps'],         context50['fraction_solved'],         label='Context-aware', color='C1')
axes[0, 1].set_ylabel('Fraction F > 0.5')
axes[0, 1].set_xlabel('Timestep')
axes[0, 1].set_ylim(-0.05, 1.05)
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)
axes[0, 1].set_title('Fraction of episodes solved')

axes[1, 0].plot(memoryless_seed43['timesteps'], memoryless_seed43['mean_omega_x'], label='Memoryless',     color='C0')
axes[1, 0].plot(context50['timesteps'],         context50['mean_omega_x'],         label='Context-aware', color='C1')
axes[1, 0].axhline(np.pi / 10, linestyle='--', color='gray', label='π/T_gate (analytic π-pulse)')
axes[1, 0].set_ylabel('|Ω_x| (rad/s)')
axes[1, 0].set_xlabel('Timestep')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)
axes[1, 0].set_title('Action magnitude')

axes[1, 1].plot(memoryless_seed43['timesteps'], memoryless_seed43['max_fidelity'], label='Memoryless',     color='C0')
axes[1, 1].plot(context50['timesteps'],         context50['max_fidelity'],         label='Context-aware', color='C1')
axes[1, 1].set_ylabel('Max Fidelity')
axes[1, 1].set_xlabel('Timestep')
axes[1, 1].set_ylim(0.5, 1.01)
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)
axes[1, 1].set_title('Best episode fidelity')

fig.suptitle('Context-aware (k=50) vs Memoryless, τ=1.0, seed=43, 5000 updates', fontsize=13)
plt.tight_layout()
out = 'plots/comparison_context50_vs_memoryless_seed43.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {out}")
