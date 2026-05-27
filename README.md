My experiments:
Project: RL control of single-qubit state transfer under OU detuning noise

Overview:
This repository contains code for training and evaluating PPO agents for single-qubit |0> to |1> state transfer under Ornstein-Uhlenbeck detuning noise.

Main components:
- qubit_control_env.py: JAX environment for noisy qubit state transfer.
- qubit_stepper.py: Hamiltonian time evolution and fidelity calculation.
- ppo.py: PPO training loop.
- evaluate_qubit.py / eval_frozen_policy.py: frozen-policy evaluation on fresh OU noise trajectories.
- analysis scripts: parse evaluation outputs and compute final-checkpoint, best-snapshot, paired-seed, IQM, and performance-profile metrics.

Experiments:
- k-sweep: memoryless vs context-aware agents with different noise-history windows.
- tau-sweep: OU correlation-time sweep across fast, intermediate, and quasi-static noise regimes.
- final-checkpoint evaluation across seeds 42–49 for tau=10.
- preliminary best-snapshot/debiased analysis for validation-checkpoint selection.

Status:
Final-checkpoint evaluation does not show a consistent CA50 advantage, but best-snapshot analysis suggests CA50 may more often discover strong intermediate policies. The next proposed experiment is validation-based checkpoint selection with fresh re-evaluation.

# RL4qcWpc

[![Generic badge](https://img.shields.io/badge/arXiv-2305.04899-<COLOR>.svg)](https://arxiv.org/abs/2501.14372)

**Reinforcement Learning for Quantum Control with Physical Constraints**

## Installation

To set up the environment and install dependencies, follow these steps:

### Create and Activate a Virtual Environment

Using Conda:

```sh
export CONPREFIX=qiskit
conda create --prefix $CONPREFIX python=3.10 -y
conda activate $CONPREFIX
```

### Install Dependencies

Install JAX with CUDA support:

```sh
conda install -c nvidia cuda
pip install --upgrade "jax[cuda12]"
```

Install additional required packages:

```sh
pip install qiskit-dynamics gymnax evosax distrax optax flax numpy brax wandb flashbax diffrax
```

## Overview

The implementation is contained in the `rl_working` directory. Our PPO algorithm implementation is based on the JAX-based framework [PureJAX-RL](https://github.com/luchris429/purejax-rl). The other implementations follow the structure of [CleanRL](https://github.com/vwxyzjn/cleanrl). We provide the following RL implementations:

- **Proximal Policy Optimization (PPO):**
  - `ppo_vmap_hyp.py`: PPO with hyperparameter vectorization
  - `ppo.py`: Standard PPO implementation
- **Twin Delayed Deep Deterministic Policy Gradient (TD3):** `td3.py`
- **Deep Deterministic Policy Gradient (DDPG):** `ddpg_buffer.py`

### Environments

Our quantum control environments are located in the `envs` directory, with support for:

- **Lambda system**
- **Rydberg atom**
- **Transmon reset**

### Reproducing Experiments & Notebooks

All experiments in our paper can be reproduced by following the structure of the example sweep provided in `rl_working/wand_sweeps`.

For quick reproducibility, we provide example Jupyter notebooks in the `notebooks` directory. These notebooks allow users to generate key results from our paper and automatically detect GPU or CPU resources for execution.

## Logging

We use **Weights & Biases (W&B)** for experiment tracking. To enable logging, configure your W&B project and entity IDs. Basic local logging is also available within the notebooks for convenience.

---

## Final Notes

Thank you for your interest in **RL4qcWpc**!  
We welcome all contributions — feel free to submit issues, feature requests, or pull requests.  
If you use this codebase or build upon it, please cite our paper:

### Citation

```bibtex
@misc{ernst2025reinforcementlearningquantumcontrol,
      title={Reinforcement Learning for Quantum Control under Physical Constraints}, 
      author={Jan Ole Ernst and Aniket Chatterjee and Tim Franzmeyer and Axel Kuhn},
      year={2025},
      eprint={2501.14372},
      archivePrefix={arXiv},
      primaryClass={quant-ph},
      url={https://arxiv.org/abs/2501.14372}, 
}
