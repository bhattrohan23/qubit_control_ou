"""
Mechanistic comparison: Context-50 vs Memoryless on identical OU noise.
τ=10, seed=42, final checkpoint.

Runs N independent episodes (default 5). Each episode uses the same JAX key
for both agents, so they receive the exact same OU noise trajectory — the only
difference is that Context-50 can observe the last 50 noise values while
Memoryless sees only the Bloch vector.

Plots four panels per episode:
  1. δ(t)       — shared OU noise detuning
  2. ω_x(t)     — pulse sequence from each agent
  3. Bloch z(t) — qubit z-component (z=−1 is target |1⟩)
  4. F(t)       — running fidelity |<1|ψ(t)>|² at every step

Usage:
    python pulse_viz.py                  # 5 episodes (keys 0–4)
    python pulse_viz.py --n_episodes 10  # 10 episodes (keys 0–9)
    python pulse_viz.py --keys 0 3 7     # specific keys
"""

import sys
import os
import argparse
import pickle

parent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "envs"))

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import flax.linen as nn
import distrax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence

from rl_working.envs.qubit_control_env import QubitControlEnv
from rl_working.envs.utils.qubit_stepper import bloch_vector, fidelity_target_1

# --- Config ---
TAU   = 10.0
SEED  = 42

CKPT_DIR = "checkpoints/qubit_control"


# Verbatim copy from ppo.py — if/else chain preserved so param pytree keys
# match the saved checkpoints exactly.
class CombinedActorCritic(nn.Module):
    action_dim: Sequence[int]
    activation: str = "tanh"
    layer_size: int = 128

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        if self.activation == "elu":
            activation = nn.elu
        if self.activation == "leaky_relu":
            activation = nn.leaky_relu
        if self.activation == "relu6":
            activation = nn.relu6
        if self.activation == "selu":
            activation = nn.selu
        else:
            activation = nn.tanh
        actor_mean = nn.Dense(self.layer_size,
                              kernel_init=orthogonal(np.sqrt(2)),
                              bias_init=constant(0.0))(x)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(self.layer_size,
                              kernel_init=orthogonal(np.sqrt(2)),
                              bias_init=constant(0.0))(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean_val = nn.Dense(self.action_dim,
                                  kernel_init=orthogonal(0.01),
                                  bias_init=constant(0.0))(actor_mean)
        actor_logtstd = self.param("log_std", nn.initializers.zeros,
                                   (self.action_dim,))
        pi = distrax.MultivariateNormalDiag(actor_mean_val,
                                            jnp.exp(actor_logtstd))
        critic = nn.Dense(1,
                          kernel_init=orthogonal(1.0),
                          bias_init=constant(0.0))(actor_mean)
        return pi, jnp.squeeze(critic, axis=-1)


def load_checkpoint(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def rollout(ckpt, noise_window, episode_key):
    """Single deterministic episode; returns per-timestep arrays."""
    nc      = ckpt["network_cfg"]
    network = CombinedActorCritic(
        action_dim=nc["action_dim"],
        activation=nc["activation"],
        layer_size=nc["layer_size"],
    )
    params = ckpt["params"]

    env = QubitControlEnv(
        tau=TAU, s=0.5, dt=0.01, N=1000, omega_max=2.0,
        lambda_amp=0.01, lambda_smooth=0.01, w_F=1.0,
        noise_window=noise_window,
    )
    env_params = env.default_params
    obs, state = env.reset_env(episode_key, env_params)

    deltas, omegas, bloch_z, fidelities = [], [], [], []

    for _ in range(env._N):
        delta_t = float(state.delta_traj_padded[state.timestep + env.noise_window])

        pi, _ = network.apply(params, obs)
        action = jnp.clip(pi.mode(), -1.0, 1.0)
        omega_t = float(action[0]) * float(env_params.omega_max)

        obs, state, _, _, _ = env.step_env(
            jax.random.PRNGKey(0), state, action, env_params
        )

        psi = state.psi_real + 1j * state.psi_imag
        bv  = bloch_vector(psi)

        deltas.append(delta_t)
        omegas.append(omega_t)
        bloch_z.append(float(bv[2]))
        fidelities.append(float(fidelity_target_1(psi)))

    return {
        "delta":    np.array(deltas),
        "omega_x":  np.array(omegas),
        "bloch_z":  np.array(bloch_z),
        "fidelity": np.array(fidelities),
    }


def make_plot(key_idx, episode_key, r_ml, r_ca):
    F_ml = r_ml["fidelity"][-1]
    F_ca = r_ca["fidelity"][-1]

    dt       = 0.01
    t        = np.arange(1000) * dt
    omega_pi = np.pi / 10.0

    fig, axes = plt.subplots(4, 1, figsize=(12, 14), sharex=True)

    axes[0].plot(t, r_ml["delta"], color="gray", lw=1.0)
    axes[0].axhline(0, color="k", lw=0.5, ls="--")
    axes[0].set_ylabel("δ(t)  [rad/s]")
    axes[0].set_title("Shared OU noise detuning δ(t)  (τ=10, identical for both agents)")
    axes[0].grid(alpha=0.3)

    axes[1].plot(t, r_ml["omega_x"],
                 label=f"Memoryless  F={F_ml:.3f}", color="C0", lw=1.2)
    axes[1].plot(t, r_ca["omega_x"],
                 label=f"Context-50  F={F_ca:.3f}", color="C1", lw=1.2)
    axes[1].axhline( omega_pi, color="gray", ls="--", lw=0.8, label="±π-pulse ref")
    axes[1].axhline(-omega_pi, color="gray", ls="--", lw=0.8)
    axes[1].set_ylabel("ω_x(t)  [rad/s]")
    axes[1].set_title("Pulse sequence output by each agent")
    axes[1].legend(loc="upper right")
    axes[1].grid(alpha=0.3)

    axes[2].plot(t, r_ml["bloch_z"], label="Memoryless", color="C0", lw=1.2)
    axes[2].plot(t, r_ca["bloch_z"], label="Context-50", color="C1", lw=1.2)
    axes[2].axhline(-1, color="green", ls="--", lw=0.8, label="Target |1⟩  (z=−1)")
    axes[2].axhline( 1, color="gray",  ls="--", lw=0.8, label="Initial |0⟩ (z=+1)")
    axes[2].set_ylabel("Bloch z")
    axes[2].set_ylim(-1.1, 1.1)
    axes[2].set_title("Bloch z-component  (z=−1 means qubit fully in |1⟩)")
    axes[2].legend(loc="upper right")
    axes[2].grid(alpha=0.3)

    axes[3].plot(t, r_ml["fidelity"],
                 label=f"Memoryless  F={F_ml:.3f}", color="C0", lw=1.2)
    axes[3].plot(t, r_ca["fidelity"],
                 label=f"Context-50  F={F_ca:.3f}", color="C1", lw=1.2)
    axes[3].axhline(0.99, color="gold",   ls=":", lw=1.0, label="F=0.99")
    axes[3].axhline(0.90, color="orange", ls=":", lw=1.0, label="F=0.90")
    axes[3].set_ylabel("F(t) = |⟨1|ψ(t)⟩|²")
    axes[3].set_xlabel("Time  (s,  T_gate = 10)")
    axes[3].set_ylim(-0.05, 1.05)
    axes[3].set_title("Running fidelity with target |1⟩")
    axes[3].legend(loc="lower right")
    axes[3].grid(alpha=0.3)

    fig.suptitle(
        f"Pulse comparison: Context-50 vs Memoryless\n"
        f"τ={TAU}, training seed={SEED}, final checkpoint  |  "
        f"episode key={key_idx}  (ΔF={F_ca - F_ml:+.3f})",
        fontsize=13,
    )
    plt.tight_layout()

    os.makedirs("plots", exist_ok=True)
    out = f"plots/pulse_viz_tau{TAU}_seed{SEED}_key{key_idx}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_episodes", type=int, default=5,
                        help="Number of episodes (keys 0..n-1)")
    parser.add_argument("--keys", type=int, nargs="+",
                        help="Specific key indices to run (overrides --n_episodes)")
    args = parser.parse_args()

    key_indices = args.keys if args.keys else list(range(args.n_episodes))

    ckpt_ml = load_checkpoint(
        f"{CKPT_DIR}/qubit_control_retrain_memoryless_tau{TAU}_seed{SEED}.pkl"
    )
    ckpt_ca = load_checkpoint(
        f"{CKPT_DIR}/qubit_control_retrain_context50_tau{TAU}_seed{SEED}.pkl"
    )

    print(f"\nRunning {len(key_indices)} episodes: keys {key_indices}")
    print(f"  {'key':>4}  {'ML F':>7}  {'CA F':>7}  {'ΔF':>7}")

    for key_idx in key_indices:
        episode_key = jax.random.PRNGKey(key_idx)
        r_ml = rollout(ckpt_ml, noise_window=0,  episode_key=episode_key)
        r_ca = rollout(ckpt_ca, noise_window=50, episode_key=episode_key)
        F_ml = r_ml["fidelity"][-1]
        F_ca = r_ca["fidelity"][-1]
        print(f"  {key_idx:>4}  {F_ml:>7.4f}  {F_ca:>7.4f}  {F_ca - F_ml:>+7.4f}")
        make_plot(key_idx, episode_key, r_ml, r_ca)


if __name__ == "__main__":
    main()
