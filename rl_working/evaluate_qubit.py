"""
Post-training evaluation and plotting for Experiment 2: Context-Aware PPO.

Designed to be called inline immediately after a training run returns:

    # at the bottom of a training session (or in a notebook)
    outs = jax.block_until_ready(single_train(rng))
    from rl_working.evaluate_qubit import run_evaluation
    run_evaluation(outs, config)

What run_evaluation() produces from a single training run:
  - plots/exp2/learning_curves_{save_name}.png
  - plots/exp2/fidelity_histogram_tau{tau}.png
  - plots/exp2/pulse_visualization_tau{tau}_k{k}.png

For cross-run plots (fidelity vs tau, context window sweep), collect results
from multiple training runs and call the standalone plot functions:
  - plot_fidelity_vs_tau(results_dict)
  - plot_context_window_sweep(k_values, means, stds)
"""

import os
import pickle

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from rl_working.envs.qubit_control_env import QubitControlEnv
from rl_working.envs.utils.qubit_stepper import fidelity_target_1


# ---------------------------------------------------------------------------
# Policy rollout
# ---------------------------------------------------------------------------

def rollout_policy(network, network_params, env: QubitControlEnv,
                   key: jax.Array) -> dict:
    """Roll out the trained policy for one episode, deterministically.

    Uses pi.mode() (= actor_mean) as the action — no sampling.

    Returns dict with keys:
        fidelity: float
        actions:  np.ndarray (N,)  — scaled omega_x values
        deltas:   np.ndarray (N,)  — raw OU noise values seen by env
        bloch:    np.ndarray (N, 3) — Bloch vector at each step
    """
    params = env.default_params
    obs, state = env.reset_env(key, params)

    actions, deltas, bloch_traj = [], [], []

    for _ in range(params.N):
        pi, _ = network.apply(network_params, obs)
        action = jnp.clip(pi.mode(), -1.0, 1.0)

        delta_t = state.delta_traj_padded[state.timestep + env.noise_window]
        obs, state, _, _, _ = env.step_env(
            jax.random.PRNGKey(0), state, action, params)

        actions.append(float(action[0]) * params.omega_max)
        deltas.append(float(delta_t))
        bloch_traj.append([
            float(state.psi_real[0]),  # x approximation; see note below
            float(state.psi_imag[0]),
            float(state.psi_real[0] ** 2 - state.psi_real[1] ** 2 +
                  state.psi_imag[0] ** 2 - state.psi_imag[1] ** 2),
        ])

    psi = state.psi_real + 1j * state.psi_imag
    fidelity = float(fidelity_target_1(psi))

    return {
        "fidelity": fidelity,
        "actions": np.array(actions),
        "deltas": np.array(deltas),
        "bloch": np.array(bloch_traj),
    }


def evaluate_policy(network, network_params, env: QubitControlEnv,
                    n_episodes: int, rng: jax.Array) -> np.ndarray:
    """Run trained policy for n_episodes. Returns fidelity array (n_episodes,)."""
    keys = jax.random.split(rng, n_episodes)
    fidelities = []
    for key in keys:
        result = rollout_policy(network, network_params, env, key)
        fidelities.append(result["fidelity"])
    return np.array(fidelities)


def evaluate_pi_pulse(env: QubitControlEnv, n_episodes: int,
                      rng: jax.Array) -> np.ndarray:
    """Run constant pi-pulse baseline for n_episodes.

    omega_x = pi / (N * dt), scaled to [-1,1] by omega_max before passing
    to the env. Returns fidelity array (n_episodes,).
    """
    params = env.default_params
    T_gate = params.N * params.dt
    omega_pi = np.pi / T_gate
    # env internally scales action by omega_max; send normalised action
    action = jnp.array([float(np.clip(omega_pi / params.omega_max, -1.0, 1.0))])

    keys = jax.random.split(rng, n_episodes)
    fidelities = []
    for key in keys:
        obs, state = env.reset_env(key, params)
        for _ in range(params.N):
            obs, state, _, _, _ = env.step_env(
                jax.random.PRNGKey(0), state, action, params)
        psi = state.psi_real + 1j * state.psi_imag
        fidelities.append(float(fidelity_target_1(psi)))
    return np.array(fidelities)


# ---------------------------------------------------------------------------
# Individual plot functions (reusable across multiple runs)
# ---------------------------------------------------------------------------

def _ensure_plot_dir():
    os.makedirs("plots/exp2", exist_ok=True)


def plot_learning_curves(pkl_paths: dict, save_name: str = "curves",
                         save: bool = True):
    """Plot episode fidelity vs training update from episodic_data pkl files.

    Args:
        pkl_paths: {label: path} e.g.
            {"k=0": "episodic_data/qubit_control/qubit_control_k0.pkl",
             "k=5": "episodic_data/qubit_control/qubit_control_k5.pkl"}
    """
    _ensure_plot_dir()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    colors = ["steelblue", "firebrick", "green", "orange", "purple"]

    for (label, path), color in zip(pkl_paths.items(), colors):
        if not os.path.exists(path):
            print(f"  Warning: {path} not found, skipping.")
            continue
        with open(path, "rb") as f:
            data = pickle.load(f)
        steps = [d["timestep"] for d in data]
        means = [float(d["mean_fidelity"]) for d in data]
        maxs  = [float(d["max_fidelity"])  for d in data]

        axes[0].plot(steps, means, label=label, color=color)
        axes[1].plot(steps, maxs,  label=label, color=color)

    for ax, title in zip(axes, ["Mean fidelity", "Max fidelity"]):
        ax.set_xlabel("Training update")
        ax.set_ylabel("Fidelity $F$")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save:
        fpath = f"plots/exp2/learning_curves_{save_name}.png"
        plt.savefig(fpath, dpi=150)
        print(f"  Saved {fpath}")
    return fig


def plot_pulse_visualization(traj: dict, tau: float, k: int,
                             save: bool = True):
    """Single-episode: detuning noise, drive pulse, and fidelity annotation.

    Args:
        traj: dict returned by rollout_policy()
        tau, k: used for filename and title only
    """
    _ensure_plot_dir()
    N = len(traj["actions"])
    dt = 0.01
    t = np.linspace(0, N * dt, N)

    fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    axes[0].plot(t, traj["deltas"], color="gray", alpha=0.8, linewidth=0.8)
    axes[0].set_ylabel("δΔ(t)  (detuning noise)")
    axes[0].set_title(f"Episode trace — τ={tau}, k={k}  |  F={traj['fidelity']:.4f}")

    axes[1].plot(t, traj["actions"], color="steelblue", linewidth=0.8)
    axes[1].set_ylabel("Ωx(t)  (drive)")
    axes[1].set_xlabel("Time (units of T_gate/10)")

    plt.tight_layout()
    if save:
        fpath = f"plots/exp2/pulse_visualization_tau{tau}_k{k}.png"
        plt.savefig(fpath, dpi=150)
        print(f"  Saved {fpath}")
    return fig


def plot_fidelity_histogram(fidelities: dict, tau: float,
                            save: bool = True):
    """Fidelity distributions at a fixed tau.

    Args:
        fidelities: {label: np.ndarray} e.g.
            {"pi-pulse": ..., "k=0": ..., "k=5": ...}
    """
    _ensure_plot_dir()
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = {"pi-pulse": "gray", "k=0": "steelblue", "k=5": "firebrick"}

    for label, vals in fidelities.items():
        ax.hist(vals, bins=30, alpha=0.55, density=True,
                label=f"{label}  (μ={vals.mean():.3f})",
                color=colors.get(label, "black"))

    ax.set_xlabel("Fidelity $F$")
    ax.set_ylabel("Density")
    ax.set_title(f"Fidelity distribution at τ={tau}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save:
        fpath = f"plots/exp2/fidelity_histogram_tau{tau}.png"
        plt.savefig(fpath, dpi=150)
        print(f"  Saved {fpath}")
    return fig


def plot_fidelity_vs_tau(results: dict, save: bool = True):
    """Mean fidelity ± std vs tau. Intended for cross-run comparisons.

    Args:
        results: {
            "tau_values": [0.1, 0.3, 1.0, 3.0, 10.0],
            "pi_pulse": {"mean": [...], "std": [...]},
            "k0":       {"mean": [...], "std": [...]},
            "k5":       {"mean": [...], "std": [...]},
        }
    """
    _ensure_plot_dir()
    taus = results["tau_values"]
    fig, ax = plt.subplots(figsize=(6, 4))

    styles = [
        ("pi_pulse", "pi-pulse baseline", "gray",      "s"),
        ("k0",       "PPO k=0",           "steelblue", "o"),
        ("k5",       "PPO k=5",           "firebrick", "^"),
    ]
    for key, label, color, marker in styles:
        if key not in results:
            continue
        m = np.array(results[key]["mean"])
        s = np.array(results[key]["std"])
        ax.errorbar(taus, m, yerr=s, label=label, marker=marker,
                    color=color, capsize=3)

    ax.set_xscale("log")
    ax.set_xlabel(r"$\tau$  (OU correlation time)")
    ax.set_ylabel("Mean fidelity $F$")
    ax.set_title("Fidelity vs OU correlation time")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save:
        fpath = "plots/exp2/fidelity_vs_tau.png"
        plt.savefig(fpath, dpi=150)
        print(f"  Saved {fpath}")
    return fig


def plot_context_window_sweep(k_values: list, means: list, stds: list,
                              tau: float = 1.0, save: bool = True):
    """Mean fidelity vs noise window k at a fixed tau.

    Collect one trained model per k value and pass their evaluation results.
    """
    _ensure_plot_dir()
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.errorbar(k_values, means, yerr=stds, marker="o",
                color="steelblue", capsize=3)
    ax.set_xlabel("Noise window $k$")
    ax.set_ylabel("Mean fidelity $F$")
    ax.set_title(f"Context window sweep at τ={tau}")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save:
        fpath = f"plots/exp2/context_sweep_tau{tau}.png"
        plt.savefig(fpath, dpi=150)
        print(f"  Saved {fpath}")
    return fig


# ---------------------------------------------------------------------------
# Main entry point: call inline after training
# ---------------------------------------------------------------------------

def run_evaluation(outs: dict, config: dict, n_episodes: int = 256,
                   rng_seed: int = 999):
    """Evaluate a completed training run and generate per-run plots.

    Typical usage (inline after ppo.py training):

        outs = jax.block_until_ready(single_train(rng))
        from rl_working.evaluate_qubit import run_evaluation
        run_evaluation(outs, config)

    Generates:
        plots/exp2/learning_curves_{save_name}.png
        plots/exp2/fidelity_histogram_tau{tau}.png
        plots/exp2/pulse_visualization_tau{tau}_k{k}.png

    Args:
        outs:       dict returned by PPO_make_train(config)(rng)
        config:     the config dict passed to PPO_make_train
        n_episodes: episodes for evaluation (default 256)
        rng_seed:   seed for evaluation randomness
    """
    # Lazy import to avoid executing ppo.py module-level code twice
    from rl_working.ppo import CombinedActorCritic

    # --- Extract trained params ---
    train_state = outs["runner_state"][0]
    network_params = train_state.params

    # --- Rebuild env from config (drop ppo-internal keys) ---
    env_kwargs = {k: v for k, v in config["ENV_PARAMS"].items()
                  if k != "ou_noise_params"}
    env = QubitControlEnv(**env_kwargs)
    env_params = env.default_params

    # --- Rebuild network (must match training architecture) ---
    network = CombinedActorCritic(
        action_dim=env.action_space(env_params).shape[0],
        activation=config.get("ACTIVATION", "relu6"),
        layer_size=config.get("LAYER_SIZE", 256),
    )

    tau = env_kwargs.get("tau", 1.0)
    k   = env_kwargs.get("noise_window", 0)
    rng = jax.random.PRNGKey(rng_seed)

    print(f"\n=== evaluate_qubit: tau={tau}, k={k}, n_episodes={n_episodes} ===")

    # --- 1. Learning curves from episodic_data pkl ---
    save_name = config.get("LOCAL_SAVE_NAME", "local_save")
    pkl_path  = f"episodic_data/qubit_control/qubit_control_{save_name}.pkl"
    plot_learning_curves({f"k={k}": pkl_path}, save_name=save_name)

    # --- 2. Policy rollouts ---
    print(f"  Running policy evaluation ({n_episodes} episodes)...")
    rng, rng_pol, rng_pi = jax.random.split(rng, 3)
    fids_policy = evaluate_policy(network, network_params, env,
                                  n_episodes, rng_pol)
    fids_pi     = evaluate_pi_pulse(env, n_episodes, rng_pi)

    print(f"  pi-pulse: mean={fids_pi.mean():.4f}  std={fids_pi.std():.4f}")
    print(f"  policy  : mean={fids_policy.mean():.4f}  std={fids_policy.std():.4f}")

    # --- 3. Fidelity histogram ---
    plot_fidelity_histogram(
        {"pi-pulse": fids_pi, f"k={k}": fids_policy},
        tau=tau,
    )

    # --- 4. Single-episode pulse visualization ---
    rng, rng_vis = jax.random.split(rng)
    traj = rollout_policy(network, network_params, env, rng_vis)
    plot_pulse_visualization(traj, tau=tau, k=k)

    print("  Done.\n")

    # Return raw fidelities so the caller can aggregate across runs
    return {
        "fids_policy": fids_policy,
        "fids_pi":     fids_pi,
        "tau":         tau,
        "k":           k,
    }
