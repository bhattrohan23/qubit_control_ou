#!/bin/bash
# Retrain 18 targeted conditions to generate frozen-policy checkpoints.
# Checkpoints saved to checkpoints/qubit_control/ at end of each run.
# Same hyperparams as original runs: num_envs=16, num_updates=5000.
#
# Estimated time: ~{run_time} per condition — run in tmux/screen or background.
# Usage:
#   bash retrain_for_eval.sh              # sequential
#   bash retrain_for_eval.sh 2>&1 | tee retrain.log

set -e

# ── tau = 0.3 ──────────────────────────────────────────────────────────────
python ppo.py --env qubit_control --tau 0.3 --noise_window 0  --seed 42 --num_envs 16 --num_updates 5000 --local_save_name memoryless_tau0.3_seed42
python ppo.py --env qubit_control --tau 0.3 --noise_window 0  --seed 43 --num_envs 16 --num_updates 5000 --local_save_name memoryless_tau0.3_seed43
python ppo.py --env qubit_control --tau 0.3 --noise_window 0  --seed 44 --num_envs 16 --num_updates 5000 --local_save_name memoryless_tau0.3_seed44

python ppo.py --env qubit_control --tau 0.3 --noise_window 50 --seed 42 --num_envs 16 --num_updates 5000 --local_save_name context50_tau0.3_seed42
python ppo.py --env qubit_control --tau 0.3 --noise_window 50 --seed 43 --num_envs 16 --num_updates 5000 --local_save_name context50_tau0.3_seed43
python ppo.py --env qubit_control --tau 0.3 --noise_window 50 --seed 44 --num_envs 16 --num_updates 5000 --local_save_name context50_tau0.3_seed44

# ── tau = 5 ────────────────────────────────────────────────────────────────
python ppo.py --env qubit_control --tau 5.0 --noise_window 0  --seed 42 --num_envs 16 --num_updates 5000 --local_save_name memoryless_tau5_seed42
python ppo.py --env qubit_control --tau 5.0 --noise_window 0  --seed 43 --num_envs 16 --num_updates 5000 --local_save_name memoryless_tau5_seed43
python ppo.py --env qubit_control --tau 5.0 --noise_window 0  --seed 44 --num_envs 16 --num_updates 5000 --local_save_name memoryless_tau5_seed44

python ppo.py --env qubit_control --tau 5.0 --noise_window 50 --seed 42 --num_envs 16 --num_updates 5000 --local_save_name context50_tau5_seed42
python ppo.py --env qubit_control --tau 5.0 --noise_window 50 --seed 43 --num_envs 16 --num_updates 5000 --local_save_name context50_tau5_seed43
python ppo.py --env qubit_control --tau 5.0 --noise_window 50 --seed 44 --num_envs 16 --num_updates 5000 --local_save_name context50_tau5_seed44

# ── tau = 10 ───────────────────────────────────────────────────────────────
python ppo.py --env qubit_control --tau 10.0 --noise_window 0  --seed 42 --num_envs 16 --num_updates 5000 --local_save_name memoryless_tau10_seed42
python ppo.py --env qubit_control --tau 10.0 --noise_window 0  --seed 43 --num_envs 16 --num_updates 5000 --local_save_name memoryless_tau10_seed43
python ppo.py --env qubit_control --tau 10.0 --noise_window 0  --seed 44 --num_envs 16 --num_updates 5000 --local_save_name memoryless_tau10_seed44

python ppo.py --env qubit_control --tau 10.0 --noise_window 50 --seed 42 --num_envs 16 --num_updates 5000 --local_save_name context50_tau10_seed42
python ppo.py --env qubit_control --tau 10.0 --noise_window 50 --seed 43 --num_envs 16 --num_updates 5000 --local_save_name context50_tau10_seed43
python ppo.py --env qubit_control --tau 10.0 --noise_window 50 --seed 44 --num_envs 16 --num_updates 5000 --local_save_name context50_tau10_seed44

echo "All 18 runs complete. Checkpoints in checkpoints/qubit_control/"
