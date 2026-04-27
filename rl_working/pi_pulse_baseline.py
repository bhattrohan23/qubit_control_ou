import numpy as np
import matplotlib.pyplot as plt
from qutip import basis, sigmax, sigmay, sigmaz, sesolve, expect
import os
from ou_noise_trajectory import generate_ou_trajectory

#time-step 
dt = 0.01
#number of time-steps
N = 1000
#total time
T_gate = N * dt
tlist = np.arange(N + 1) * dt
#correlation times -- goal is to determine which is the ideal range of tau
tau_values = [0.1, 0.3, 1.0, 3.0, 10.0]
s = 0.5
N_ensemble = 100
rng = np.random.default_rng(seed=42)

os.makedirs("plots/exp1", exist_ok=True)

delta_trajectory = np.zeros(N+1)

#|0> to |1> only
def run_pi_pulse(delta_trajectory, tlist, omega_x_array):
    sx = sigmax()
    sz = sigmaz()
    psi0 = basis(2, 0)
    psi_target = basis(2, 1)
    H = [[sz / 2, delta_trajectory], [sx / 2, omega_x_array]]
    result = sesolve(H, psi0, tlist)
    fidelity = expect(psi_target * psi_target.dag(), result.states[-1])
    x_traj = expect(sigmax(), result.states)
    y_traj = expect(sigmay(), result.states)
    z_traj = expect(sigmaz(), result.states)
    return fidelity, x_traj, y_traj, z_traj


omega_x_val = np.pi / T_gate
omega_x_array = np.ones(N + 1) * omega_x_val

sigma_calc = s * np.sqrt(2 / 1.0)
delta_single = generate_ou_trajectory(1.0, sigma_calc, dt, N + 1, rng)
fidelity_single, x_traj, y_traj, z_traj = run_pi_pulse(delta_single, tlist, omega_x_array)

fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
for ax, data, label, color in zip(
        axes[:3],
        [x_traj, y_traj, z_traj],
        ["⟨σx⟩", "⟨σy⟩", "⟨σz⟩"],
        ["tab:blue", "tab:orange", "tab:green"]):
    ax.plot(tlist, data, lw=0.8, color=color)
    ax.set_ylabel(label)
    ax.axhline(0, color="gray", lw=0.5, linestyle="--")
    ax.set_ylim(-1.1, 1.1)

axes[3].plot(tlist, delta_single, lw=0.8, color="tab:red")
axes[3].set_ylabel("δΔ(t)")
axes[3].axhline(0, color="gray", lw=0.5, linestyle="--")
axes[-1].set_xlabel("time")
fig.suptitle(f"π-Pulse Single Trajectory (τ = 1.0) — Final Fidelity: {fidelity_single:.4f}", fontsize=12)
plt.tight_layout()
plt.savefig("plots/exp1/plot1_single_trajectory_tau1.png", dpi=150)
plt.close()

F_means = np.zeros(len(tau_values))
F_stds = np.zeros(len(tau_values))
F_all_by_tau = {}

for idx, tau in enumerate(tau_values):
    sigma_calc = s * np.sqrt(2 / tau)
    fidelities = np.zeros(N_ensemble)
    for i in range(N_ensemble):
        delta_i = generate_ou_trajectory(tau, sigma_calc, dt, N + 1, rng)
        fidelity_i, _, _, _ = run_pi_pulse(delta_i, tlist, omega_x_array)
        fidelities[i] = fidelity_i
    F_means[idx] = np.mean(fidelities)
    F_stds[idx] = np.std(fidelities)
    F_all_by_tau[tau] = fidelities

print(f"{'τ':<8} | F_mean ± F_std")
print(f"{'--------':<8}-+---------------")
for idx, tau in enumerate(tau_values):
    print(f"{tau:<8} | {F_means[idx]:.4f} ± {F_stds[idx]:.4f}")

fig, ax = plt.subplots(figsize=(8, 5))
ax.errorbar(tau_values, F_means, yerr=F_stds, fmt="o-", capsize=5, lw=1.5)
ax.set_xscale("log")
ax.set_xlabel("τ (correlation time)")
ax.set_ylabel("Mean Fidelity F")
ax.set_title("π-Pulse Fidelity vs OU Correlation Time τ")
plt.tight_layout()
plt.savefig("plots/exp1/plot2_fidelity_vs_tau.png", dpi=150)
plt.close()

fidelities_tau1 = F_all_by_tau[1.0]
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(fidelities_tau1, bins=20, edgecolor="black", alpha=0.75)
ax.axvline(np.mean(fidelities_tau1), color="red", linestyle="--", lw=1.5, label=f"Mean = {np.mean(fidelities_tau1):.4f}")
ax.set_xlabel("Fidelity")
ax.set_ylabel("Count")
ax.set_title(f"Fidelity Distribution: π-pulse at τ = 1.0 (N = {N_ensemble})")
ax.annotate(
    f"Mean = {np.mean(fidelities_tau1):.4f}\nStd = {np.std(fidelities_tau1):.4f}",
    xy=(0.05, 0.95), xycoords="axes fraction",
    va="top", ha="left", fontsize=10,
    bbox=dict(boxstyle="round", fc="white", alpha=0.7)
)
ax.legend()
plt.tight_layout()
plt.savefig("plots/exp1/plot3_fidelity_histogram_tau1.png", dpi=150)
plt.close()


# Six cardinal Bloch states with all transitions under OU noise
# Transitions: 0 -> 1, 1 -> 0, +i -> -i, -i -> +i, + -> -, - -> +
CARDINAL_STATES = {
    "0->1":   (basis(2, 0),                              basis(2, 1)),
    "1->0":   (basis(2, 1),                              basis(2, 0)),
    "+i->-i": ((basis(2, 0) + 1j * basis(2, 1)).unit(),  (basis(2, 0) - 1j * basis(2, 1)).unit()),
    "-i->+i": ((basis(2, 0) - 1j * basis(2, 1)).unit(),  (basis(2, 0) + 1j * basis(2, 1)).unit()),
    "-->-":   ((basis(2, 0) - basis(2, 1)).unit(),        (basis(2, 0) - basis(2, 1)).unit()),
    "+->+":   ((basis(2, 0) + basis(2, 1)).unit(),        (basis(2, 0) + basis(2, 1)).unit())
}


def run_pi_pulse_cardinal(delta_trajectory, tlist, omega_x_array):
    """
    Simulate a pi pulse under OU detuning noise for all 6 cardinal Bloch states.

    Returns a dict keyed by transition label, each value is:
        {
            "fidelity": float,
            "x":        array of <σx> over tlist,
            "y":        array of <σy> over tlist,
            "z":        array of <σz> over tlist,
        }
    """
    sx = sigmax()
    sy = sigmay()
    sz = sigmaz()
    H = [[sz / 2, delta_trajectory], [sx / 2, omega_x_array]]

    results = {}
    for label, (psi_init, psi_target) in CARDINAL_STATES.items():
        res = sesolve(H, psi_init, tlist)
        proj = psi_target * psi_target.dag()
        fidelity = float(expect(proj, res.states[-1]))
        results[label] = {
            "fidelity": fidelity,
            "x": expect(sx, res.states),
            "y": expect(sy, res.states),
            "z": expect(sz, res.states),
        }
    return results


def run_cardinal_ensemble(tau, s, dt, N, tlist, omega_x_array, N_ensemble, rng):
    """
    Run N_ensemble OU noise trajectories for all 6 cardinal transitions.

    Returns a dict keyed by transition label, each containing an array of
    per-trajectory fidelities.
    """
    sigma_calc = s * np.sqrt(2 / tau)
    fidelities = {label: np.zeros(N_ensemble) for label in CARDINAL_STATES}

    for i in range(N_ensemble):
        delta_i = generate_ou_trajectory(tau, sigma_calc, dt, N + 1, rng)
        res = run_pi_pulse_cardinal(delta_i, tlist, omega_x_array)
        for label in CARDINAL_STATES:
            fidelities[label][i] = res[label]["fidelity"]

    return fidelities


def plot_cardinal_single(tau=1.0):
    """Plot single-trajectory Bloch components + detuning for all 6 transitions."""
    sigma_calc = s * np.sqrt(2 / tau)
    delta = generate_ou_trajectory(tau, sigma_calc, dt, N + 1, rng)
    res = run_pi_pulse_cardinal(delta, tlist, omega_x_array)

    os.makedirs("plots/exp1", exist_ok=True)
    n_transitions = len(CARDINAL_STATES)
    fig, axes = plt.subplots(n_transitions, 4, figsize=(16, 3 * n_transitions), sharex=True)

    for row, (label, data) in enumerate(res.items()):
        for col, (component, color) in enumerate(
            zip(["x", "y", "z"], ["tab:blue", "tab:orange", "tab:green"])
        ):
            ax = axes[row, col]
            ax.plot(tlist, data[component], lw=0.8, color=color)
            ax.set_ylim(-1.1, 1.1)
            ax.axhline(0, color="gray", lw=0.5, linestyle="--")
            if row == 0:
                ax.set_title(f"⟨σ{'xyz'[col]}⟩")
            if col == 0:
                ax.set_ylabel(label, fontsize=8)

        ax_delta = axes[row, 3]
        ax_delta.plot(tlist, delta, lw=0.8, color="tab:red")
        ax_delta.axhline(0, color="gray", lw=0.5, linestyle="--")
        if row == 0:
            ax_delta.set_title("δΔ(t)")
        f = data["fidelity"]
        ax_delta.text(0.02, 0.95, f"F={f:.4f}", transform=ax_delta.transAxes,
                      va="top", fontsize=8, bbox=dict(boxstyle="round", fc="white", alpha=0.7))

    axes[-1, 0].set_xlabel("time")
    fig.suptitle(f"π-Pulse All Cardinal States (τ={tau}) — OU Noise", fontsize=13)
    plt.tight_layout()
    out = f"plots/exp1/plot_cardinal_single_tau{tau}.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")


def plot_cardinal_fidelity_vs_tau(tau_values, s, dt, N, tlist, omega_x_array,
                                   N_ensemble, rng):
    """Plot mean fidelity ± std vs tau for all 6 cardinal transitions."""
    os.makedirs("plots/exp1", exist_ok=True)
    # shape: {label: (len(tau_values),)}
    means = {label: [] for label in CARDINAL_STATES}
    stds  = {label: [] for label in CARDINAL_STATES}

    for tau in tau_values:
        fids = run_cardinal_ensemble(tau, s, dt, N, tlist, omega_x_array, N_ensemble, rng)
        for label in CARDINAL_STATES:
            means[label].append(np.mean(fids[label]))
            stds[label].append(np.std(fids[label]))

    fig, ax = plt.subplots(figsize=(9, 5))
    for label in CARDINAL_STATES:
        ax.errorbar(tau_values, means[label], yerr=stds[label],
                    fmt="o-", capsize=4, lw=1.5, label=label)
    ax.set_xscale("log")
    ax.set_xlabel("τ (correlation time)")
    ax.set_ylabel("Mean Fidelity F")
    ax.set_title("π-Pulse Fidelity vs OU τ — All Cardinal States")
    ax.legend(fontsize=8)
    plt.tight_layout()
    out = "plots/exp1/plot_cardinal_fidelity_vs_tau.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")


def plot_avg_cardinal_fidelity(tau_values, s, dt, N, tlist, omega_x_array,
                               N_ensemble, rng):
    """Plot mean cardinal fidelity (averaged over all 6 states) vs tau,
    overlaid with the 0->1 fidelity for comparison."""
    os.makedirs("plots/exp1", exist_ok=True)

    avg_means, avg_stds = [], []
    zero_one_means, zero_one_stds = [], []

    for tau in tau_values:
        fids = run_cardinal_ensemble(tau, s, dt, N, tlist, omega_x_array, N_ensemble, rng)
        # stack (6, N_ensemble) and average across transitions per trajectory
        all_fids = np.stack([fids[label] for label in CARDINAL_STATES], axis=0)
        avg_per_traj = all_fids.mean(axis=0)
        avg_means.append(np.mean(avg_per_traj))
        avg_stds.append(np.std(avg_per_traj))

        zero_one_means.append(np.mean(fids["0->1"]))
        zero_one_stds.append(np.std(fids["0->1"]))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(tau_values, avg_means, yerr=avg_stds,
                fmt="s-", capsize=4, lw=1.5, label="Avg Cardinal Fidelity (6 states)")
    ax.errorbar(tau_values, zero_one_means, yerr=zero_one_stds,
                fmt="o--", capsize=4, lw=1.5, label="0→1 Fidelity")
    ax.set_xscale("log")
    ax.set_xlabel("τ (correlation time)")
    ax.set_ylabel("Mean Fidelity F")
    ax.set_title("π-Pulse: Avg Cardinal Fidelity vs 0→1 under OU Noise")
    ax.legend(fontsize=10)
    plt.tight_layout()
    out = "plots/exp1/plot_avg_cardinal_vs_01.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")


if __name__ == "__main__":
    plot_cardinal_single(tau=1.0)
    plot_cardinal_fidelity_vs_tau(tau_values, s, dt, N, tlist, omega_x_array,
                                   N_ensemble, rng)
    plot_avg_cardinal_fidelity(tau_values, s, dt, N, tlist, omega_x_array,
                               N_ensemble, rng)
