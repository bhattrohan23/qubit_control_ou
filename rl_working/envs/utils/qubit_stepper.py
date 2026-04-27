"""
JIT-compiled 2x2 quantum evolution functions for single-qubit control.

All functions are pure JAX — no NumPy, no Python control flow — so they
compose freely with jax.jit, jax.vmap, and jax.lax.scan.

Hamiltonian: H = 0.5 * delta * sigma_z + 0.5 * omega_x * sigma_x
"""

import jax
jax.config.update("jax_platform_name", "cpu")
import jax.numpy as jnp


def analytic_unitary(omega_x, delta, dt):
    """Compute U = exp(-i H dt) analytically for 2x2 qubit Hamiltonian.

    H = 0.5 * delta * sigma_z + 0.5 * omega_x * sigma_x

    Returns a 2x2 complex jnp array.
    """
    omega_eff = jnp.sqrt(omega_x ** 2 + delta ** 2)
    theta = 0.5 * omega_eff * dt
    cos_t = jnp.cos(theta)
    sin_t = jnp.sin(theta)
    # Avoid division by zero when omega_eff ~ 0 (returns identity)
    safe_omega = jnp.where(omega_eff > 1e-10, omega_eff, 1.0)
    factor = jnp.where(omega_eff > 1e-10, sin_t / safe_omega, 0.0)

    U00 = cos_t - 1j * factor * delta
    U01 = -1j * factor * omega_x
    U10 = -1j * factor * omega_x
    U11 = cos_t + 1j * factor * delta
    return jnp.array([[U00, U01], [U10, U11]])


def evolve(psi, omega_x, delta, dt):
    """Apply one timestep of evolution: psi' = U @ psi. psi(t + dt) = U * psi(t)

    Args:
        psi: 2-element complex jnp array (state vector)
        omega_x: drive amplitude
        delta: detuning (noise)
        dt: timestep

    Returns:
        New state vector psi' (2-element complex jnp array).
    """
    U = analytic_unitary(omega_x, delta, dt)
    return U @ psi


def bloch_vector(psi):
    """Extract Bloch vector from state vector psi.

    Returns jnp array [x, y, z] where:
        x = 2 Re(psi[0] * conj(psi[1]))
        y = 2 Im(psi[0] * conj(psi[1]))
        z = |psi[0]|^2 - |psi[1]|^2
    """
    x = 2.0 * jnp.real(psi[0] * jnp.conj(psi[1]))
    y = 2.0 * jnp.imag(psi[0] * jnp.conj(psi[1]))
    z = jnp.abs(psi[0]) ** 2 - jnp.abs(psi[1]) ** 2
    return jnp.array([x, y, z])


#calculates fidelity (0 --> 1 specifically)
def fidelity_target_1(psi):
    """Fidelity with respect to |1> target state.

    F = |<1|psi>|^2 = |psi[1]|^2
    """
    return jnp.abs(psi[1]) ** 2


if __name__ == "__main__":
    import jax
    import numpy as np

    print("=== qubit_stepper validation ===\n")

    # 1. Pi-pulse with no noise -> F ~ 1.0
    dt = 0.01
    N = 1000
    T = N * dt  # total time = 10.0
    omega_pi = jnp.pi / T
    psi = jnp.array([1.0 + 0j, 0.0 + 0j])
    for _ in range(N):
        psi = evolve(psi, omega_pi, 0.0, dt)
    F = fidelity_target_1(psi)
    print(f"Pi-pulse fidelity (omega_x = pi/T, delta=0): F = {F:.6f}  (expected ~1.0)")

    # 2. Compare analytic_unitary to scipy.linalg.expm for random H matrices
    try:
        from scipy.linalg import expm
        rng = np.random.default_rng(42)
        errs = []
        for _ in range(20):
            ox = float(rng.uniform(-5, 5))
            d = float(rng.uniform(-5, 5))
            dt_test = float(rng.uniform(0.001, 0.1))
            # scipy reference
            sx = np.array([[0, 1], [1, 0]], dtype=complex)
            sz = np.array([[1, 0], [0, -1]], dtype=complex)
            H = 0.5 * d * sz + 0.5 * ox * sx
            U_ref = expm(-1j * H * dt_test)
            U_jax = np.array(analytic_unitary(ox, d, dt_test))
            errs.append(np.max(np.abs(U_jax - U_ref)))
        print(f"scipy.expm comparison: max element error over 20 random matrices = {max(errs):.2e}  (expected < 1e-6)")
    except ImportError:
        print("scipy not available, skipping expm comparison")

    # 3. Norm preservation over 1000 steps: |ψ[0]|² + |ψ[1]|² = 1 is a necessity of quantum mechanics (probability conservation)
    psi = jnp.array([1.0 + 0j, 0.0 + 0j])
    rng_key = jax.random.PRNGKey(0)
    omegas = jax.random.uniform(rng_key, (N,), minval=-2.0, maxval=2.0)
    deltas = jax.random.uniform(jax.random.split(rng_key)[0], (N,), minval=-1.0, maxval=1.0)
    norms = []
    for i in range(N):
        psi = evolve(psi, float(omegas[i]), float(deltas[i]), dt)
        norms.append(float(jnp.sum(jnp.abs(psi) ** 2)))
    max_norm_err = max(abs(n - 1.0) for n in norms)
    print(f"Norm preservation over {N} steps: max |norm - 1| = {max_norm_err:.2e}  (expected < 1e-5)")

    # 4. Bloch vector: verify x^2 + y^2 + z^2 = 1 for a known pure state
    psi_test = jnp.array([1.0 / jnp.sqrt(2) + 0j, 1.0 / jnp.sqrt(2) + 0j])
    bv = bloch_vector(psi_test)
    bloch_norm = float(jnp.sum(bv ** 2))
    print(f"Bloch vector |r|^2 for |+> state: {bloch_norm:.6f}  (expected 1.0)")
    print(f"  Bloch vector: x={float(bv[0]):.4f}, y={float(bv[1]):.4f}, z={float(bv[2]):.4f}  (expected x=1, y=0, z=0)")

    print("\nAll validation checks complete.")
