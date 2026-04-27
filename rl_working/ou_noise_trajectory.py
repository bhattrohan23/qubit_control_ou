# ou_noise_trajectory.py
# OU noise: d(δΔ) = -(1/τ) δΔ dt + σ dW
# Illustrates time evolution of H(t) under OU noise on a qubit Bloch sphere

import numpy as np
import matplotlib.pyplot as plt
from qutip import basis, sigmax, sigmay, sigmaz, sesolve, expect

# ============================================================
# STEP 1: OU noise parameters and generator function
# ============================================================
sigma = 1.0    # noise strength
dt = 0.01      # timestep
N = 1000       # number of steps  →  N+1 = 1001 time points
T_gate = N * dt  # total gate time = 10.0

rng = np.random.default_rng(seed=42)

def generate_ou_trajectory(tau, sigma, dt, N_steps, rng=None):
    """Euler-Maruyama discretisation of OU process.
    Returns array of length N_steps."""
    if rng is None:
        rng = np.random.default_rng()
    delta = np.zeros(N_steps)
    for i in range(N_steps - 1):
        eta = rng.standard_normal()
        delta[i + 1] = delta[i] - (1 / tau) * delta[i] * dt + sigma * np.sqrt(dt) * eta
    return delta

# ============================================================
# Original plot: OU noise trajectories for three τ values
# ============================================================
tau_values = [0.1, 1.0, 10.0]
tlist = np.arange(N + 1) * dt   # length N+1 = 1001

fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
for ax, tau in zip(axes, tau_values):
    delta = generate_ou_trajectory(tau, sigma, dt, N + 1, rng)
    ax.plot(tlist, delta, lw=0.8)
    ax.set_ylabel("δΔ(t)")
    ax.set_title(f"τ = {tau}")
    ax.axhline(0, color="gray", lw=0.5, linestyle="--")

axes[-1].set_xlabel("time")
fig.suptitle("Ornstein-Uhlenbeck Noise Trajectories", fontsize=13)
plt.tight_layout()
plt.savefig("ou_noise_trajectory.png", dpi=150)
plt.show()

# ============================================================
# STEP 2: Pauli operators and qubit observables
# ============================================================
sx = sigmax()
sy = sigmay()
sz = sigmaz()

# ============================================================
# STEP 3 & 4: Hamiltonian and initial state helpers
# ============================================================
# H(t) = 0.5 * Δ(t) σz + 0.5 * Ωx(t) σx + 0.5 * Ωy(t) σy  (ℏ = 1)
#
# Pure-noise run: Ωx = Ωy = 0  →  only σz term driven by OU noise
# |+⟩ sits on equator so z-noise causes visible phase drift

omega_x_array = np.zeros(N + 1)   # no drive yet
omega_y_array = np.zeros(N + 1)

psi0 = (basis(2, 0) + basis(2, 1)).unit()   # |+⟩ = (|0⟩+|1⟩)/√2

# ============================================================
# STEP 5: Single trajectory + ensemble
# ============================================================
tau_demo = 1.0    # τ used for plots A, B, C

# --- single trajectory ---
delta_single = generate_ou_trajectory(tau_demo, sigma, dt, N + 1, rng)
H_single = [[sz / 2, delta_single],
            [sx / 2, omega_x_array],
            [sy / 2, omega_y_array]]

result_single = sesolve(H_single, psi0, tlist)
x_single = expect(sx, result_single.states)
y_single = expect(sy, result_single.states)
z_single = expect(sz, result_single.states)
r_single = np.sqrt(x_single**2 + y_single**2 + z_single**2)

print(f"Single trajectory: |r| range [{r_single.min():.4f}, {r_single.max():.4f}]  "
      f"(should be ≈ 1 everywhere)")

# --- ensemble ---
N_ensemble = 100
x_all = np.zeros((N_ensemble, N + 1))
y_all = np.zeros((N_ensemble, N + 1))
z_all = np.zeros((N_ensemble, N + 1))

for i in range(N_ensemble):
    delta_i = generate_ou_trajectory(tau_demo, sigma, dt, N + 1, rng)
    H_i = [[sz / 2, delta_i],
            [sx / 2, omega_x_array],
            [sy / 2, omega_y_array]]
    result_i = sesolve(H_i, psi0, tlist)
    x_all[i] = expect(sx, result_i.states)
    y_all[i] = expect(sy, result_i.states)
    z_all[i] = expect(sz, result_i.states)

x_avg = np.mean(x_all, axis=0)
y_avg = np.mean(y_all, axis=0)
z_avg = np.mean(z_all, axis=0)
r_avg = np.sqrt(x_avg**2 + y_avg**2 + z_avg**2)

# ============================================================
# PLOT A: Single-trajectory Bloch coordinates + |r|
# ============================================================
fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
for ax, data, label, color in zip(
        axes[:3],
        [x_single, y_single, z_single],
        ["⟨σx⟩", "⟨σy⟩", "⟨σz⟩"],
        ["tab:blue", "tab:orange", "tab:green"]):
    ax.plot(tlist, data, lw=0.8, color=color)
    ax.set_ylabel(label)
    ax.axhline(0, color="gray", lw=0.5, linestyle="--")
    ax.set_ylim(-1.1, 1.1)

axes[3].plot(tlist, r_single, lw=0.8, color="black")
axes[3].set_ylabel("|r|(t)")
axes[3].set_ylim(0, 1.1)
axes[3].axhline(1, color="gray", lw=0.5, linestyle="--")
axes[-1].set_xlabel("time")
fig.suptitle(f"Plot A – Single trajectory, τ = {tau_demo},  pure z-noise, |+⟩ initial state",
             fontsize=12)
plt.tight_layout()
plt.savefig("plot_A_single_trajectory.png", dpi=150)
plt.show()

# ============================================================
# PLOT B: Ensemble-averaged Bloch coordinates
# ============================================================
fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
for ax, avg, all_runs, label, color in zip(
        axes,
        [x_avg, y_avg, z_avg],
        [x_all, y_all, z_all],
        ["⟨σx⟩", "⟨σy⟩", "⟨σz⟩"],
        ["tab:blue", "tab:orange", "tab:green"]):
    # light individual trajectories
    for run in all_runs[::10]:
        ax.plot(tlist, run, lw=0.3, alpha=0.3, color=color)
    ax.plot(tlist, avg, lw=1.5, color=color, label="ensemble avg")
    ax.set_ylabel(label)
    ax.axhline(0, color="gray", lw=0.5, linestyle="--")
    ax.set_ylim(-1.1, 1.1)
    ax.legend(loc="upper right", fontsize=8)

axes[-1].set_xlabel("time")
fig.suptitle(f"Plot B – Ensemble average ({N_ensemble} runs), τ = {tau_demo}", fontsize=12)
plt.tight_layout()
plt.savefig("plot_B_ensemble_bloch.png", dpi=150)
plt.show()

# ============================================================
# PLOT C: r_avg(t) — the decoherence envelope
# ============================================================
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(tlist, r_avg, lw=1.5, color="tab:purple", label=f"τ = {tau_demo}")
ax.set_xlabel("time")
ax.set_ylabel("|r̄|(t)")
ax.set_ylim(0, 1.05)
ax.axhline(1, color="gray", lw=0.5, linestyle="--", label="|r| = 1 (pure state)")
ax.legend()
ax.set_title(f"Plot C – Bloch-vector length decay (decoherence), τ = {tau_demo}, "
             f"N = {N_ensemble} trajectories")
plt.tight_layout()
plt.savefig("plot_C_r_decay.png", dpi=150)
plt.show()

# ============================================================
# PLOT D: r_avg(t) for different τ values
# ============================================================
fig, ax = plt.subplots(figsize=(9, 5))

s = 0.5   # target standard deviation (fixed across all τ)

for tau in tau_values:
    sigma_calc = s * np.sqrt(2 / tau)   # normalise variance: Var(OU ss) = σ²τ/2 → set = s²
    x_t = np.zeros((N_ensemble, N + 1))
    y_t = np.zeros((N_ensemble, N + 1))
    z_t = np.zeros((N_ensemble, N + 1))
    for i in range(N_ensemble):
        delta_i = generate_ou_trajectory(tau, sigma_calc, dt, N + 1, rng)
        H_i = [[sz / 2, delta_i],
                [sx / 2, omega_x_array],
                [sy / 2, omega_y_array]]
        result_i = sesolve(H_i, psi0, tlist)
        x_t[i] = expect(sx, result_i.states)
        y_t[i] = expect(sy, result_i.states)
        z_t[i] = expect(sz, result_i.states)
    r_t = np.sqrt(np.mean(x_t, axis=0)**2 +
                  np.mean(y_t, axis=0)**2 +
                  np.mean(z_t, axis=0)**2)
    ax.plot(tlist, r_t, lw=1.5, label=f"τ = {tau}")

ax.set_xlabel("time")
ax.set_ylabel("|r̄|(t)")
ax.set_ylim(0, 1.05)
ax.axhline(1, color="gray", lw=0.5, linestyle="--")
ax.legend()
ax.set_title(f"Plot D – Decoherence for different τ  (s = {s}, σ = s√(2/τ), N = {N_ensemble} runs)")
plt.tight_layout()
plt.savefig("plot_D_tau_comparison.png", dpi=150)
plt.show()

# ============================================================
# PLOT E: Fidelity decay F_avg(t) for different τ values
#   F = |⟨+|ψ(t)⟩|²  averaged over N_ensemble trajectories
#   No control pulses; |+⟩ initial state; target |+⟩
#   Normalized noise: σ = s√(2/τ) so steady-state variance = s²
# ============================================================
fig, ax = plt.subplots(figsize=(9, 5))

proj_plus = psi0 * psi0.dag()   # projector |+⟩⟨+| — F = ⟨ψ|proj|ψ⟩

for tau in tau_values:
    sigma_e = s * np.sqrt(2 / tau)   # normalised: Var(OU ss) = σ²τ/2 = s²
    F_all = np.zeros((N_ensemble, N + 1))
    for i in range(N_ensemble):
        delta_i = generate_ou_trajectory(tau, sigma_e, dt, N + 1, rng)
        H_i = [[sz / 2, delta_i],
                [sx / 2, omega_x_array],
                [sy / 2, omega_y_array]]
        result_i = sesolve(H_i, psi0, tlist)
        F_all[i] = expect(proj_plus, result_i.states)
    F_avg = np.mean(F_all, axis=0)
    ax.plot(tlist, F_avg, lw=1.5, label=f"τ = {tau}")

ax.set_xlabel("time")
ax.set_ylabel("F_avg(t) = |⟨+|ψ(t)⟩|²")
ax.set_ylim(0.4, 1.05)
ax.axhline(1, color="gray", lw=0.5, linestyle="--")
ax.legend()
ax.set_title("Fidelity decay under uncontrolled OU detuning drift")
plt.tight_layout()
plt.savefig("plot_E_fidelity_decay.png", dpi=150)
plt.show()
