# bloch_animation.py
# Animate qubit state on Bloch sphere under OU detuning noise + Control Pulse


import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from qutip import basis, sigmax, sigmay, sigmaz, sesolve, expect, Bloch
from PIL import Image
import io

# ============================================================
# SETUP
# ============================================================
sigma = 1.0    # noise strength
dt = 0.01      # timestep
N = 1000       # number of steps  ->  N+1 = 1001 time points
T_gate = N * dt  # total gate time = 10.0

rng = np.random.default_rng(seed=42)

def generate_ou_trajectory(tau, sigma, dt, N_steps, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    delta = np.zeros(N_steps)
    for i in range(N_steps - 1):
        eta = rng.standard_normal()
        delta[i + 1] = delta[i] - (1 / tau) * delta[i] * dt + sigma * np.sqrt(dt) * eta
    return delta

tau = 1.0
tlist = np.arange(N + 1) * dt

sx = sigmax()
sy = sigmay()
sz = sigmaz()

# --- FIX 1 & 2: Set initial state to |0> and turn on a pi-pulse ---
psi0 = basis(2, 0)  # Start at North Pole (|0>)

# A constant pulse that integrates to Pi over T_gate to flip from 0 to 1
omega_x_array = np.full(N + 1, np.pi / T_gate) 
omega_y_array = np.zeros(N + 1)

# Generate ONE OU noise trajectory (the Z-axis crosswind)
delta = generate_ou_trajectory(tau, sigma, dt, N + 1, rng)

# Build H and propagate with sesolve
H = [[sz / 2, delta],
     [sx / 2, omega_x_array],
     [sy / 2, omega_y_array]]

result = sesolve(H, psi0, tlist)

# Extract Bloch coordinates
sx_exp = expect(sigmax(), result.states)
sy_exp = expect(sigmay(), result.states)
sz_exp = expect(sigmaz(), result.states)

# ============================================================
# ANIMATION
# ============================================================
skip = 5
frame_indices = np.arange(0, len(sx_exp), skip)
gif_frames = []

fig = plt.figure(figsize=(5, 5))

for i in frame_indices:
    fig.clf()
    ax = Axes3D(fig, azim=20, elev=15)
    sphere = Bloch(axes=ax)
    sphere.point_color = ['b']
    sphere.point_size = [20]
    sphere.vector_color = ['r', 'g']

    # Trail
    sphere.add_points([sx_exp[:i+1], sy_exp[:i+1], sz_exp[:i+1]])
    # Current state vector
    sphere.add_vectors([sx_exp[i], sy_exp[i], sz_exp[i]])
    # |1> target marker (South Pole)
    sphere.add_vectors([0, 0, -1])

    sphere.make_sphere()

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=80)
    buf.seek(0)
    gif_frames.append(Image.open(buf).copy())
    buf.close()

plt.close(fig)

gif_frames[0].save(
    'bloch_ou_flip.gif',
    save_all=True,
    append_images=gif_frames[1:],
    loop=0,
    duration=50,  # ms per frame -> ~20 fps
)
print("Saved bloch_ou_flip.gif")
