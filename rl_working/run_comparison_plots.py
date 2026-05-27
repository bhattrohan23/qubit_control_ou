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

def unpack_eval(data):
    return {
        'timesteps':       np.array([d['timestep']                               for d in data], dtype=float),
        'mean_fidelity':   np.array([d['det']['eval_F_mean']                     for d in data], dtype=float),
        'fraction_solved': np.array([np.mean(np.asarray(d['fidelities']) > 0.5)  for d in data], dtype=float),
        'mean_omega_x':    np.array([d['det']['eval_omega_mean']                 for d in data], dtype=float),
        'max_fidelity':    np.array([np.max(np.asarray(d['fidelities']))         for d in data], dtype=float),
    }


def unpack_frozen_eval(d, n_batches=10):
    """Adapter so an eval_frozen_policy.py output pkl (single-timepoint final eval,
    5000 episode samples) can flow through the same 4-panel plotting code as
    `unpack_eval`. The 5000 episodes are split into `n_batches` chunks; each
    batch contributes one (x, y) point per metric, so the curve visualises
    estimator stability rather than a training trajectory.
    """
    F = np.asarray(d['fidelities'])
    A = np.asarray(d['mean_action_mags'])
    n = len(F)
    n_per = n // n_batches
    if n_per == 0:
        raise ValueError(f"Need at least n_batches={n_batches} episodes, got {n}.")
    edges = np.arange(1, n_batches + 1) * n_per

    F_chunks = [F[i*n_per:(i+1)*n_per] for i in range(n_batches)]
    A_chunks = [A[i*n_per:(i+1)*n_per] for i in range(n_batches)]
    return {
        'timesteps':       edges.astype(float),
        'mean_fidelity':   np.array([c.mean()          for c in F_chunks]),
        'fraction_solved': np.array([np.mean(c > 0.5)  for c in F_chunks]),
        'mean_omega_x':    np.array([c.mean()          for c in A_chunks]),
        'max_fidelity':    np.array([c.max()           for c in F_chunks]),
    }

def summarize(name, mean_fidelity, mean_omega_x, last_n=100):
    print(f"  {name:<40} | mean F (last {last_n}) = {mean_fidelity[-last_n:].mean():.4f} "
          f"| peak F = {mean_fidelity.max():.4f} | mean |ωx| = {mean_omega_x.mean():.4f}")

def make_plot(context_k, seed):
    memoryless = unpack(load(f'{BASE}/qubit_control_memoryless_seed{seed}.pkl'))
    context    = unpack(load(f'{BASE}/qubit_control_context{context_k}_seed{seed}.pkl'))

    print(f"\n=== context{context_k} vs memoryless, seed {seed} ===")
    summarize(f"Memoryless seed={seed}",             memoryless['mean_fidelity'], memoryless['mean_omega_x'])
    summarize(f"Context-aware k={context_k} seed={seed}", context['mean_fidelity'],    context['mean_omega_x'])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(memoryless['timesteps'], memoryless['mean_fidelity'], label='Memoryless (k=0)',           color='C0')
    axes[0, 0].plot(context['timesteps'],    context['mean_fidelity'],    label=f'Context-aware (k={context_k})', color='C1')
    axes[0, 0].set_ylabel('Mean Fidelity')
    axes[0, 0].set_xlabel('Timestep')
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    axes[0, 0].set_title('Mean episode fidelity')

    axes[0, 1].plot(memoryless['timesteps'], memoryless['fraction_solved'], label='Memoryless',     color='C0')
    axes[0, 1].plot(context['timesteps'],    context['fraction_solved'],    label='Context-aware', color='C1')
    axes[0, 1].set_ylabel('Fraction F > 0.5')
    axes[0, 1].set_xlabel('Timestep')
    axes[0, 1].set_ylim(-0.05, 1.05)
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    axes[0, 1].set_title('Fraction of episodes solved')

    axes[1, 0].plot(memoryless['timesteps'], memoryless['mean_omega_x'], label='Memoryless',     color='C0')
    axes[1, 0].plot(context['timesteps'],    context['mean_omega_x'],    label='Context-aware', color='C1')
    axes[1, 0].axhline(np.pi / 10, linestyle='--', color='gray', label='π/T_gate (analytic π-pulse)')
    axes[1, 0].set_ylabel('|Ω_x| (rad/s)')
    axes[1, 0].set_xlabel('Timestep')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    axes[1, 0].set_title('Action magnitude')

    axes[1, 1].plot(memoryless['timesteps'], memoryless['max_fidelity'], label='Memoryless',     color='C0')
    axes[1, 1].plot(context['timesteps'],    context['max_fidelity'],    label='Context-aware', color='C1')
    axes[1, 1].set_ylabel('Max Fidelity')
    axes[1, 1].set_xlabel('Timestep')
    axes[1, 1].set_ylim(0.0, 1.01)
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)
    axes[1, 1].set_title('Best episode fidelity')

    fig.suptitle(f'Context-aware (k={context_k}) vs Memoryless, τ=1.0, seed={seed}, 5000 updates', fontsize=13)
    plt.tight_layout()

    os.makedirs('plots', exist_ok=True)
    out = f'plots/comparison_context{context_k}_vs_memoryless_seed{seed}.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")

def make_tau_plot(tau, seed, retrain_eval=False, frozen_eval=False, eval_dir='eval_results'):
    if frozen_eval:
        # New eval_frozen_policy.py outputs (single-timepoint, 5000 eps each).
        # Chunk into 10 batches so make_tau_plot's curve code works unchanged.
        mem_pkl = load(f'{eval_dir}/eval_qubit_control_memoryless_tau{tau}_seed{seed}.pkl')
        ctx_pkl = load(f'{eval_dir}/eval_qubit_control_context50_tau{tau}_seed{seed}.pkl')
        memoryless = unpack_frozen_eval(mem_pkl)
        context    = unpack_frozen_eval(ctx_pkl)
    elif retrain_eval:
        def _resolve_eval(label):
            for cand in (
                f'{BASE}/qubit_control_retrain_{label}_tau{tau}_seed{seed}_eval.pkl',
                f'{BASE}/qubit_control_{label}_tau{tau}_seed{seed}_eval.pkl',
            ):
                if os.path.exists(cand):
                    return cand
            raise FileNotFoundError(
                f"No periodic-eval pkl for {label} tau={tau} seed={seed} "
                f"(tried both retrain_ and non-prefixed names under {BASE})"
            )
        memoryless = unpack_eval(load(_resolve_eval('memoryless')))
        context    = unpack_eval(load(_resolve_eval('context50')))
    else:
        memoryless = unpack(load(f'{BASE}/qubit_control_memoryless_tau{tau}_seed{seed}.pkl'))
        context    = unpack(load(f'{BASE}/qubit_control_context50_tau{tau}_seed{seed}.pkl'))

    mode_tag = '[frozen-eval]' if frozen_eval else ('[retrain-eval]' if retrain_eval else '')
    print(f"\n=== context50 vs memoryless, tau={tau}, seed={seed} {mode_tag} ===")
    summarize(f"Memoryless tau={tau} seed={seed}",         memoryless['mean_fidelity'], memoryless['mean_omega_x'])
    summarize(f"Context-aware k=50 tau={tau} seed={seed}", context['mean_fidelity'],    context['mean_omega_x'])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(memoryless['timesteps'], memoryless['mean_fidelity'], label='Memoryless (k=0)',      color='C0')
    axes[0, 0].plot(context['timesteps'],    context['mean_fidelity'],    label='Context-aware (k=50)', color='C1')
    axes[0, 0].set_ylabel('Mean Fidelity')
    axes[0, 0].set_xlabel('Timestep')
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    axes[0, 0].set_title('Mean episode fidelity')

    axes[0, 1].plot(memoryless['timesteps'], memoryless['fraction_solved'], label='Memoryless',     color='C0')
    axes[0, 1].plot(context['timesteps'],    context['fraction_solved'],    label='Context-aware', color='C1')
    axes[0, 1].set_ylabel('Fraction F > 0.5')
    axes[0, 1].set_xlabel('Timestep')
    axes[0, 1].set_ylim(-0.05, 1.05)
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    axes[0, 1].set_title('Fraction of episodes solved')

    axes[1, 0].plot(memoryless['timesteps'], memoryless['mean_omega_x'], label='Memoryless',     color='C0')
    axes[1, 0].plot(context['timesteps'],    context['mean_omega_x'],    label='Context-aware', color='C1')
    axes[1, 0].axhline(np.pi / 10, linestyle='--', color='gray', label='π/T_gate (analytic π-pulse)')
    axes[1, 0].set_ylabel('|Ω_x| (rad/s)')
    axes[1, 0].set_xlabel('Timestep')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    axes[1, 0].set_title('Action magnitude')

    axes[1, 1].plot(memoryless['timesteps'], memoryless['max_fidelity'], label='Memoryless',     color='C0')
    axes[1, 1].plot(context['timesteps'],    context['max_fidelity'],    label='Context-aware', color='C1')
    axes[1, 1].set_ylabel('Max Fidelity')
    axes[1, 1].set_xlabel('Timestep')
    axes[1, 1].set_ylim(0.0, 1.01)
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)
    axes[1, 1].set_title('Best episode fidelity')

    if frozen_eval:
        suptitle = f'Context-aware (k=50) vs Memoryless, τ={tau}, seed={seed} [frozen-policy eval, 5000 eps in 10 batches]'
        out = f'plots/frozen_eval_tau{tau}_seed{seed}_context50_vs_memoryless.png'
    elif retrain_eval:
        suptitle = f'Context-aware (k=50) vs Memoryless, τ={tau}, seed={seed} [frozen-policy eval]'
        out = f'plots/retrain_eval_tau{tau}_seed{seed}_context50_vs_memoryless.png'
    else:
        suptitle = f'Context-aware (k=50) vs Memoryless, τ={tau}, seed={seed}, 5000 updates'
        out = f'plots/comparison_tau{tau}_seed{seed}_context50_vs_memoryless.png'

    fig.suptitle(suptitle, fontsize=13)
    plt.tight_layout()

    os.makedirs('plots', exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")

def run_legacy_plots():
    """Per-seed comparison plots for the original (seed 42–44) sweep.

    These outputs already exist in plots/; only re-run with --include-legacy
    if you've regenerated the underlying episodic_data.
    """
    for context_k in [20, 30]:
        for seed in [42, 43, 44]:
            make_plot(context_k, seed)
    for tau in [0.1, 0.3, 3, 5, 6, 7, 8, 10]:
        for seed in [42, 43, 44]:
            make_tau_plot(tau, seed)
    for tau in [5.0, 10.0]:
        for seed in [42, 43, 44]:
            make_tau_plot(tau, seed, retrain_eval=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--include-legacy', action='store_true',
                        help='Also regenerate the original per-seed plots for seeds 42–44.')
    parser.add_argument('--tau', type=float, default=10,
                        help='OU correlation time (default 10).')
    parser.add_argument('--seeds', type=int, nargs='+', default=[45, 46, 47, 48, 49],
                        help='Seeds to plot per-seed (default 45 46 47 48 49).')
    args = parser.parse_args()

    if args.include_legacy:
        run_legacy_plots()

    # Per-seed training-curve plots (existing format).
    for s in args.seeds:
        make_tau_plot(args.tau, s)

    # Per-seed periodic frozen-policy eval plots (true learning curves,
    # 10 snapshots × n_eval per snapshot), from the *_eval.pkl files.
    for s in args.seeds:
        make_tau_plot(args.tau, s, retrain_eval=True)
