# rl_working

JAX-based reinforcement learning for quantum control tasks. Trains RL agents (PPO, TD3, DDPG) to find optimal pulse sequences for quantum systems including Rydberg atoms, STIRAP, transmon reset, and single-qubit control.

## Project goals and constraints

<!-- TODO (Rohan): Fill this in. Describe:
  - The high-level research goal (what are you trying to show or achieve?)
  - Physical/experimental constraints the agent must respect (e.g. pulse amplitude limits, smoothness requirements, decoherence timescales)
  - What "success" looks like (fidelity threshold, robustness to noise, etc.)
  - Any scope boundaries — what this project is NOT trying to do
  - Any known tradeoffs or design decisions that should be preserved
-->

## Stack

- **JAX / jit / vmap** — all environments and training loops are JIT-compiled
- **Flax (linen)** — neural network definitions
- **Optax** — optimizers
- **Distrax** — probability distributions for PPO
- **gymnax** — environment base class conventions
- **WandB** — experiment tracking and hyperparameter sweeps

GPU is used when available; falls back to CPU with float64.

## Project structure

```
ppo.py                        PPO training entry point
td3.py                        TD3 training entry point
ppo_vmap_hyp.py               PPO with vmapped hyperparameter sweeps
ddpg_buffer.py                DDPG replay buffer
env_configs/configs.py        Per-environment hyperparameter configs and argparse defaults
envs/
  environment_template.py     Abstract base: SingleStepEnvironment
  qubit_control_env.py        Single-qubit control (new, OU noise, Bloch obs)
  single_rydberg_env.py       Single Rydberg atom CZ gate
  single_rydberg_two_photon_env.py
  single_stirap_env.py
  multistep_stirap_env.py
  single_transmon_reset_env.py
  hamiltonian/                Hamiltonian definitions per system
  utils/
    noise_functions.py        OU process and other noise generators
    qubit_stepper.py          Schrödinger equation stepper, Bloch vector, fidelity
    pulse_waveforms.py        Waveform utilities
    shared_quantum_functions.py
    signal_functions.py
    wrappers.py               VecEnv, LogWrapper
wandb_sweeps/                 WandB sweep YAML configs
```

## Running training

```bash
# PPO
python ppo.py --env_name qubit_control --num_envs 64 --num_updates 500 \
  --tau 1.0 --s 0.5 --omega_max 2.0 --lambda_amp 0.01 --lambda_smooth 0.01

# TD3
python td3.py --env_name qubit_control ...

# WandB sweep
wandb sweep wandb_sweeps/sweep_config_ss_mxsteps_vmap.yaml
wandb agent <sweep_id>
```

## Environments

All environments inherit from `SingleStepEnvironment` and follow gymnax conventions (jit-compatible `reset`/`step`, `EnvState`/`EnvParams` as Flax structs).

| env_name | System | Key params |
|---|---|---|
| `qubit_control` | Single qubit, OU noise | tau, s, omega_max, lambda_amp, lambda_smooth, noise_window |
| `rydberg` | Single Rydberg CZ | gamma, omega_0, blockade_strength |
| `rydberg_two` | Two-photon Rydberg | omega_e, omega_r, delta_r, delta_e |
| `simple_stirap` | STIRAP | gamma, omega_0, delta_s, delta_p |
| `multi_stirap` | Multi-step STIRAP | same as simple_stirap |
| `transmon_reset` | Transmon qubit reset | kappa, chi, delta, anharm, g_coupling |

## qubit_control env specifics

- Physics: `H(t) = 0.5 * delta(t) * sigma_z + 0.5 * omega_x(t) * sigma_x`
- `delta(t)` is OU noise pre-generated at episode reset
- Agent action: `omega_x(t)` at each timestep
- Observation: Bloch vector `[x, y, z]` + (optionally) last `noise_window` delta values
- `noise_window = 0`: memoryless; `noise_window = k`: context-aware
- Reward combines fidelity (`w_F`), amplitude penalty (`lambda_amp`), smoothness penalty (`lambda_smooth`)

## Import paths

Scripts use `rl_working` as the package root (added to `sys.path` at startup):
```python
from rl_working.envs.qubit_control_env import QubitControlEnv
from rl_working.envs.utils.noise_functions import ou_process
```
