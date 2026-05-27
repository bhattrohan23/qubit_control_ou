import re
import matplotlib.pyplot as plt

BASE = 'episodic_data/qubit_control'

conditions = [
    ('memoryless', 0),
    ('context20',  20),
    ('context30',  30),
    ('context50',  50),
]
seeds = [42, 43, 44]
colors = {42: 'red', 43: 'green', 44: 'blue'}

def read_worst_drop(condition, seed):
    fname = f'{BASE}/qubit_control_{condition}_seed{seed}_metrics.txt'
    with open(fname) as f:
        for line in f:
            m = re.match(r'worst_drop:\s*([\d.]+)', line)
            if m:
                return float(m.group(1))
    raise ValueError(f'worst_drop not found in {fname}')

fig, ax = plt.subplots(figsize=(8, 5))

for seed in seeds:
    drops = [read_worst_drop(cond, seed) for cond, _ in conditions]
    xs = [k for _, k in conditions]
    ax.plot(xs, drops, marker='o', color=colors[seed], label=f'seed {seed}')

ax.set_xlabel('Context window k')
ax.set_ylabel('Worst drop')
ax.set_xticks([0, 20, 30, 50])
ax.set_title('Worst fidelity drop vs. context window')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
out = 'plots/worst_drop_vs_context.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved: {out}')
