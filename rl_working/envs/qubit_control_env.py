"""
Single-qubit control environment for JAX-based PPO training.

Physics: H(t) = 0.5 * delta(t) * sigma_z + 0.5 * omega_x(t) * sigma_x
where delta(t) is OU noise (pre-generated at episode reset) and omega_x(t)
is the agent's action.

The `noise_window` parameter controls context:
  noise_window = 0  -> memoryless agent (obs = Bloch vector only)
  noise_window = k  -> context-aware agent (obs = Bloch + k most recent delta values)

Both modes use the same env class, same physics, same reward — only
the observation dimension differs.
"""

import jax
import jax.numpy as jnp
from functools import partial
from flax import struct
import chex
from typing import Tuple, Optional, Union
from gymnax.environments import spaces

from environment_template import SingleStepEnvironment
from utils.noise_functions import ou_process
from utils.qubit_stepper import evolve, bloch_vector, fidelity_target_1


@struct.dataclass
class EnvState:
    psi_real: jnp.ndarray          # Re(psi), shape (2,)
    psi_imag: jnp.ndarray          # Im(psi), shape (2,)
    delta_traj_padded: jnp.ndarray # OU noise, shape (N+1+noise_window,)
    timestep: int
    prev_action: float
    reward: float
    fidelity: float


@struct.dataclass
class EnvParams:
    tau: float = 1.0
    s: float = 0.5
    dt: float = 0.01
    N: int = 1000
    omega_max: float = 2.0
    lambda_amp: float = 0.01
    lambda_smooth: float = 0.01
    w_F: float = 1.0
    noise_window: int = 0
    min_action: float = -1.0
    max_action: float = 1.0


class QubitControlEnv(SingleStepEnvironment):
    """1000-step single-qubit control under OU-noise detuning.

    The agent outputs a scalar action in [-1, 1] at each of the N=1000
    timesteps. The action is scaled to [-omega_max, omega_max] internally.

    Episode lifecycle:
      reset() -> generates a fresh OU noise trajectory, initialises psi = |0>
      step()  -> applies one dt of unitary evolution, accumulates reward
      done    -> True when timestep reaches N; terminal reward uses log-fidelity
    """

    def __init__(self, tau=1.0, s=0.5, dt=0.01, N=1000, omega_max=2.0,
                 lambda_amp=0.01, lambda_smooth=0.01, w_F=1.0,
                 noise_window=0, min_action=-1.0, max_action=1.0, **kwargs):
        self._tau = tau
        self._s = s
        self._dt = dt
        self._N = N
        self._omega_max = omega_max
        self._lambda_amp = lambda_amp
        self._lambda_smooth = lambda_smooth
        self._w_F = w_F
        self.noise_window = noise_window
        self.obs_dim = 3 + noise_window

        # OU process parameters
        # alpha^2 = dt/tau gives mean-reversion of ~1 correlation time per tau
        # sigma = s gives stationary std = s (see ou_process formula)
        self._alpha = jnp.sqrt(dt / tau)
        self._sigma = s

        self._default_params = EnvParams(
            tau=tau, s=s, dt=dt, N=N, omega_max=omega_max,
            lambda_amp=lambda_amp, lambda_smooth=lambda_smooth,
            w_F=w_F, noise_window=noise_window,
            min_action=min_action, max_action=max_action,
        )

    @property
    def default_params(self) -> EnvParams:
        return self._default_params

    @partial(jax.jit, static_argnums=(0,))
    def reset_env(self, key: chex.PRNGKey, params: EnvParams
                  ) -> Tuple[chex.Array, EnvState]:
        key_ou, _ = jax.random.split(key)

        # Generate N+1 noise values (one per timestep 0..N)
        delta = ou_process(key_ou, self._N + 1, self._alpha, 0.0, self._sigma)

        # Pad with noise_window zeros at the front so get_obs can always
        # slice [timestep : timestep+noise_window] without out-of-bounds
        delta_padded = jnp.concatenate([
            jnp.zeros(self.noise_window), delta
        ])

        psi_real = jnp.array([1.0, 0.0])
        psi_imag = jnp.array([0.0, 0.0])

        state = EnvState(
            psi_real=psi_real,
            psi_imag=psi_imag,
            delta_traj_padded=delta_padded,
            timestep=0,
            prev_action=0.0,
            reward=0.0,
            fidelity=0.0,
        )
        return self.get_obs(state), state

    @partial(jax.jit, static_argnums=(0,))
    def step_env(self, key: chex.PRNGKey, state: EnvState,
                 action: Union[int, float], params: EnvParams
                 ) -> Tuple[chex.Array, EnvState, float, bool, dict]:

        # Current noise value (offset by noise_window due to front-padding)
        delta_t = state.delta_traj_padded[state.timestep + self.noise_window]

        # Scale action from [-1,1] to [-omega_max, omega_max]
        a_t = action[0] * params.omega_max

        # Reconstruct complex psi
        psi = state.psi_real + 1j * state.psi_imag

        # Evolve one timestep
        psi_new = evolve(psi, a_t, delta_t, params.dt)

        # Per-step shaping reward
        r_shaping = (
            -params.lambda_amp * a_t ** 2
            - params.lambda_smooth * (a_t - state.prev_action) ** 2
        )

        new_t = state.timestep + 1

        # Terminal vs non-terminal via jax.lax.cond (JIT-safe)
        def terminal_fn(_):
            F = fidelity_target_1(psi_new)
            r_term = -params.w_F * jnp.log(1.0 - F + 1e-8)
            return r_shaping + r_term, True, F

        def non_terminal_fn(_):
            return r_shaping, False, 0.0

        reward, done, fidelity = jax.lax.cond(
            new_t >= params.N,
            terminal_fn,
            non_terminal_fn,
            operand=None,
        )

        new_state = EnvState(
            psi_real=jnp.real(psi_new),
            psi_imag=jnp.imag(psi_new),
            delta_traj_padded=state.delta_traj_padded,
            timestep=new_t,
            prev_action=a_t,
            reward=reward,
            fidelity=fidelity,
        )
        info = {
            "fid": fidelity,
            "fidelity": fidelity,
            "mean-omega-x": jnp.abs(a_t),
            "mean-delta": jnp.abs(delta_t),
            "smoothness-penalty": params.lambda_smooth * (a_t - state.prev_action) ** 2,
            "amp-penalty": params.lambda_amp * a_t ** 2,
            "reward": reward,
        }
        return self.get_obs(new_state), new_state, reward, done, info

    def get_obs(self, state: EnvState) -> chex.Array:
        """Build observation: Bloch vector + optional noise history."""
        psi = state.psi_real + 1j * state.psi_imag
        bloch = bloch_vector(psi)

        if self.noise_window > 0:
            # delta_traj_padded is front-padded by noise_window zeros,
            # so timestep t reads history from index t..t+noise_window-1
            noise_hist = jax.lax.dynamic_slice(
                state.delta_traj_padded,
                (state.timestep,),
                (self.noise_window,),
            )
            return jnp.concatenate([bloch, noise_hist])
        else:
            return bloch

    def is_terminal(self, state: EnvState, params: EnvParams) -> bool:
        return state.timestep >= params.N

    def action_space(self, params: Optional[EnvParams] = None):
        return spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=jnp.float32)

    def observation_space(self, params: Optional[EnvParams] = None):
        return spaces.Box(
            low=-jnp.inf, high=jnp.inf,
            shape=(self.obs_dim,), dtype=jnp.float32,
        )

    @property
    def name(self) -> str:
        return "QubitControl"

    @property
    def num_actions(self) -> int:
        return 1

    @property
    def log_vals(self) -> dict:
        return {
            "fidelity": 0.0,
            "mean-omega-x": 0.0,
            "mean-delta": 0.0,
            "smoothness-penalty": 0.0,
            "amp-penalty": 0.0,
        }
