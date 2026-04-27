# validate_env_pi_pulse.py
# Run from rl_working/

import os
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
import numpy as np

import sys
sys.path.insert(0, "envs")
from qubit_control_env import QubitControlEnv
from utils.qubit_stepper import fidelity_target_1

tau_values = [0.1, 0.3, 1.0, 3.0, 10.0]
N_episodes = 100
N_steps = 1000
dt = 0.01
T_gate = N_steps * dt
omega_pi = jnp.pi / T_gate  # constant pi-pulse amplitude

for tau in tau_values:
    env = QubitControlEnv(tau=tau, noise_window=0)
    params = env.default_params

    fidelities = []
    rng = jax.random.PRNGKey(42)

    for ep in range(N_episodes):
        rng, key_reset, key_step = jax.random.split(rng, 3)
        obs, state = env.reset_env(key_reset, params)

        for t in range(N_steps):
            # Constant pi-pulse scaled to [-1, 1] for the env
            action = jnp.array([omega_pi / params.omega_max])
            rng, key_step = jax.random.split(rng)
            obs, state, reward, done, info = env.step_env(
                key_step, state, action, params
            )

        F = fidelity_target_1(state.psi_real + 1j * state.psi_imag)
        fidelities.append(float(F))

    fids = np.array(fidelities)
    print(f"tau={tau:4.1f}  F={fids.mean():.4f} +/- {fids.std():.4f}")
