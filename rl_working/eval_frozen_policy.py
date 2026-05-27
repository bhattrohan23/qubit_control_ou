"""
Frozen-policy evaluation on fresh OU noise episodes.

Loads checkpoints saved by ppo.py (LOCAL_LOGGING=True), runs N_eval
deterministic episodes (pi.mode(), no PPO update), and reports:

  mean/std/median/p10/p25/min fidelity
  P(F>0.90), P(F>0.95), P(F>0.99)
  mean cumulative reward, mean |omega_x|, mean action smoothness

Usage:
    # single checkpoint
    python eval_frozen_policy.py \
        --checkpoint checkpoints/qubit_control/qubit_control_memoryless_tau1_seed42.pkl

    # glob over all conditions
    python eval_frozen_policy.py \
        --checkpoint_glob "checkpoints/qubit_control/*.pkl" \
        --n_eval 5000 --seed 77777

    # force CPU
    python eval_frozen_policy.py --cpu --checkpoint ...
"""

import sys
import os

if "--cpu" in sys.argv:
    os.environ["JAX_PLATFORMS"] = "cpu"
    sys.argv.remove("--cpu")

import argparse
import glob
import pickle
import time

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import distrax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# qubit_control_env.py uses sibling-relative imports (`from environment_template import ...`),
# so the envs/ dir must also be on sys.path — mirrors what ppo.py does.
sys.path.append(os.path.join(os.path.dirname(__file__), "envs"))

from rl_working.envs.qubit_control_env import QubitControlEnv
from rl_working.evaluate_qubit import build_fast_eval_fn, compute_eval_metrics


# Replicated verbatim from ppo.py so param pytree keys match exactly.
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


def _print_table(name: str, m: dict, n_eval: int):
    # Keys match what compute_eval_metrics returns (rl_working/evaluate_qubit.py).
    W = 58
    bar = "─" * W
    print(f"\n┌{bar}┐")
    print(f"│ {name:<{W-1}}│")
    print(f"│ n_eval = {n_eval:<{W-11}}│")
    print(f"├{bar}┤")
    print(f"│ {'FIDELITY STATISTICS':<{W-1}}│")
    print(f"│   E[F]          {m['eval_F_mean']:>8.5f}   ±  {m['eval_F_std']:.5f}{'':>{W-42}}│")
    print(f"│   Median        {m['eval_F_median']:>8.5f}{'':>{W-25}}│")
    print(f"│   p10           {m['eval_F_10']:>8.5f}{'':>{W-25}}│")
    print(f"│   p25           {m['eval_F_25']:>8.5f}{'':>{W-25}}│")
    print(f"│   Min           {m['eval_F_min']:>8.5f}{'':>{W-25}}│")
    print(f"├{bar}┤")
    print(f"│ {'ROBUSTNESS':<{W-1}}│")
    print(f"│   P(F > 0.90)   {m['eval_success_09']:>8.5f}{'':>{W-25}}│")
    print(f"│   P(F > 0.95)   {m['eval_success_095']:>8.5f}{'':>{W-25}}│")
    print(f"│   P(F > 0.99)   {m['eval_success_099']:>8.5f}{'':>{W-25}}│")
    print(f"├{bar}┤")
    print(f"│ {'CONTROL SIGNAL':<{W-1}}│")
    print(f"│   Mean |ω_x|    {m['eval_omega_mean']:>8.4f}{'':>{W-25}}│")
    print(f"│   Smoothness    {m['eval_action_smoothness']:>8.4f}{'':>{W-25}}│")
    print(f"│   Mean Σreward  {m['eval_cumulative_reward']:>8.3f}{'':>{W-25}}│")
    print(f"└{bar}┘")


def evaluate_checkpoint(ckpt_path: str, n_eval: int, rng: jax.Array) -> dict:
    with open(ckpt_path, "rb") as f:
        ckpt = pickle.load(f)

    ep = ckpt["env_params"]
    nc = ckpt["network_cfg"]

    env_kwargs = {k: v for k, v in ep.items() if k != "ou_noise_params"}
    env = QubitControlEnv(**env_kwargs)

    network = CombinedActorCritic(
        action_dim=nc["action_dim"],
        activation=nc["activation"],
        layer_size=nc["layer_size"],
    )

    network_params = ckpt["params"]

    print(f"\nBuilding eval fn for {os.path.basename(ckpt_path)} …", flush=True)
    t0 = time.time()
    eval_fn = build_fast_eval_fn(network, env, stochastic=False)

    keys = jax.random.split(rng, n_eval)
    raw = jax.block_until_ready(eval_fn(network_params, keys))
    elapsed = time.time() - t0

    metrics = compute_eval_metrics(raw)
    print(f"  Done in {elapsed:.1f}s")

    name = os.path.splitext(os.path.basename(ckpt_path))[0]
    _print_table(name, metrics, n_eval)

    return {
        "checkpoint": ckpt_path,
        "n_eval": n_eval,
        "env_params": ep,
        "network_cfg": nc,
        "fidelities": np.asarray(raw["fidelity"]),
        "cumulative_rewards": np.asarray(raw["cumulative_reward"]),
        "mean_action_mags": np.asarray(raw["mean_action_mag"]),
        "action_smoothnesses": np.asarray(raw["action_smoothness"]),
        "metrics": metrics,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to a single checkpoint pkl file")
    parser.add_argument("--checkpoint_glob", type=str, default=None,
                        help="Glob pattern matching multiple checkpoint files")
    parser.add_argument("--n_eval", type=int, default=5000,
                        help="Number of fresh evaluation episodes per policy")
    parser.add_argument("--seed", type=int, default=77777,
                        help="RNG seed for evaluation episodes")
    parser.add_argument("--out_dir", type=str, default="eval_results",
                        help="Directory to save evaluation result pkls")
    args = parser.parse_args()

    if args.checkpoint is None and args.checkpoint_glob is None:
        parser.error("Provide --checkpoint or --checkpoint_glob")

    ckpt_paths = []
    if args.checkpoint:
        ckpt_paths.append(args.checkpoint)
    if args.checkpoint_glob:
        ckpt_paths.extend(sorted(glob.glob(args.checkpoint_glob)))

    if not ckpt_paths:
        print("No checkpoint files found.")
        return

    print(f"JAX devices: {jax.devices()}")
    print(f"Evaluating {len(ckpt_paths)} checkpoint(s) with n_eval={args.n_eval}")

    os.makedirs(args.out_dir, exist_ok=True)
    base_rng = jax.random.PRNGKey(args.seed)

    all_results = []
    for i, ckpt_path in enumerate(ckpt_paths):
        rng, sub_rng = jax.random.split(base_rng)
        base_rng = rng

        result = evaluate_checkpoint(ckpt_path, args.n_eval, sub_rng)
        all_results.append(result)

        name = os.path.splitext(os.path.basename(ckpt_path))[0]
        out_path = os.path.join(args.out_dir, f"eval_{name}.pkl")
        with open(out_path, "wb") as f:
            pickle.dump(result, f)
        print(f"  Saved → {out_path}")

    # summary csv
    import csv
    csv_path = os.path.join(args.out_dir, "eval_summary.csv")
    fieldnames = ["checkpoint"] + list(all_results[0]["metrics"].keys())
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in all_results:
            row = {"checkpoint": os.path.basename(r["checkpoint"])}
            row.update(r["metrics"])
            w.writerow(row)
    print(f"\nSummary CSV → {csv_path}")


if __name__ == "__main__":
    main()
