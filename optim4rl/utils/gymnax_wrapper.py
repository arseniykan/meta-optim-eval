
import jax
import jax.numpy as jnp
import numpy as np
from gymnax.environments import spaces, environment
from typing import NamedTuple, Optional, Tuple, Union
from brax import envs
from flax import struct
from functools import partial
from gymnax.wrappers.purerl import GymnaxWrapper
import chex
from jax import vjp, flatten_util
from jax.tree_util import tree_flatten

class GymnaxGymWrapper:
    def __init__(self, env, env_params, config):
        self.env = env
        self.env_params = env_params
        # Handle continuous/discrete action space difference
        # Assuming config has CONTINUOUS flag or inference
        # If config is None, try to infer
        if config and config.get("CONTINUOUS", False):
             self.action_space = env.action_space(self.env_params).shape[0]
        elif hasattr(env.action_space(self.env_params), 'n'):
             self.action_space = env.action_space(self.env_params).n
        else:
             # Fallback for continuous if no 'n'
             self.action_space = env.action_space(self.env_params).shape[0]

        self.observation_space = env.observation_space(self.env_params).shape

    def reset(self, rng, params=None):
        rng, _rng = jax.random.split(rng)
        obs, env_state = self.env.reset(_rng, self.env_params)
        state = (env_state, rng)
        return obs, state

    def step(self, key, state, action, params=None):
        env_state, rng = state
        rng, _rng = jax.random.split(rng)
        obs, env_state, reward, done, info = self.env.step(
            _rng, env_state, action, self.env_params
        )
        state = (env_state, rng)
        return obs, state, reward, done, info

class FlatWrapper:
    def __init__(self, env):
        self.env = env
        self.observation_space = np.prod(env.observation_space)
        self.action_space = env.action_space

    def reset(self, rng, params=None):
        obs, env_state = self.env.reset(rng, params)
        obs = jnp.reshape(obs, (self.observation_space,))
        return obs, env_state

    def step(self, key, state, action, params=None):
        obs, state, reward, done, info = self.env.step(key, state, action, params)
        obs = jnp.reshape(obs, (self.observation_space,))
        return obs, state, reward, done, info

class EpisodeStats(NamedTuple):
    episode_returns: float
    episode_lengths: int
    returned_episode_returns: float
    returned_episode_lengths: int

class GymnaxLogWrapper(GymnaxWrapper):
    def __init__(self, env):
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space

    def reset(self, rng, params=None):
        obs, env_state = self.env.reset(rng, params)
        state = (env_state, EpisodeStats(0, 0, 0, 0))
        return obs, state

    def step(self, key, state, action, params=None):
        env_state, episode_stats = state
        obs, env_state, reward, done, info = self.env.step(
            key, env_state, action, params
        )
        new_episode_return = episode_stats.episode_returns + reward
        new_episode_length = episode_stats.episode_lengths + 1
        new_episode_stats = EpisodeStats(
            episode_returns=new_episode_return * (1 - done),
            episode_lengths=new_episode_length * (1 - done),
            returned_episode_returns=episode_stats.returned_episode_returns * (1 - done)
            + new_episode_return * done,
            returned_episode_lengths=episode_stats.returned_episode_lengths * (1 - done)
            + new_episode_length * done,
        )
        state = (env_state, new_episode_stats)
        info = {}
        # Map to Brax metric names if possible
        info["eval/episode_reward"] = new_episode_stats.returned_episode_returns
        info["returned_episode_returns"] = new_episode_stats.returned_episode_returns
        info["returned_episode_lengths"] = new_episode_stats.returned_episode_lengths
        return obs, state, reward, done, info

class ClipAction(GymnaxWrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, key, state, action, params=None):
        action = jnp.clip(action, -1, 1)
        return self._env.step(key, state, action, params)

class TransformObservation(GymnaxWrapper):
    def __init__(self, env, transform_obs):
        super().__init__(env)
        self.transform_obs = transform_obs

    def reset(self, key, params=None):
        obs, state = self._env.reset(key, params)
        return self.transform_obs(obs), state

    def step(self, key, state, action, params=None):
        obs, state, reward, done, info = self._env.step(key, state, action, params)
        return self.transform_obs(obs), state, reward, done, info

class VecEnv(GymnaxWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.reset = jax.vmap(self._env.reset, in_axes=(0, None))
        self.step = jax.vmap(self._env.step, in_axes=(0, 0, 0, None))
    
    # Add Brax-like attributes used by PPO
    @property
    def action_size(self):
        return self._env.action_space
        
    @property
    def observation_size(self):
        return self._env.observation_space[0] # assuming 1D obs after flatten


# Brax State Compat
@struct.dataclass
class BraxState:
    pipeline_state: Optional[any]
    obs: jnp.ndarray
    reward: jnp.ndarray
    done: jnp.ndarray
    metrics: dict = struct.field(default_factory=dict)
    info: dict = struct.field(default_factory=dict)

class BraxGymnaxWrapper:
    def __init__(self, env, env_params, config=None):
        self._env = env
        self._env_params = env_params
        
        # Action/Obs sizes
        dummy_rng = jax.random.PRNGKey(0)
        dummy_obs, _ = self._env.reset(dummy_rng, self._env_params)
        self.observation_size = dummy_obs.shape[-1]
        
        # Action size logic
        if config and config.get("CONTINUOUS", False):
             self.action_size = env.action_space(self._env_params).shape[0]
        elif hasattr(env.action_space(self._env_params), 'n'):
             self.action_size = int(env.action_space(self._env_params).n)
        else:
             self.action_size = int(env.action_space(self._env_params).shape[0])

    def reset(self, rng):
        obs, state = self._env.reset(rng, self._env_params)
        return BraxState(
            pipeline_state=state,
            obs=obs,
            reward=jnp.zeros(()),
            done=jnp.zeros(()),
            metrics={}
        )

    def step(self, state, action):
        rng, _rng = jax.random.split(state.pipeline_state[1]) if isinstance(state.pipeline_state, tuple) else jax.random.split(jax.random.PRNGKey(0)) 
        # Note: Gymnax state usually includes RNG, but if not we might have issues.
        # Our GymnaxGymWrapper returns (env_state, rng).
        
        # We need to unwrap our specific state structure if we are using GymnaxGymWrapper
        # But here we are wrapping the RAW gymnax env primarily?
        # Let's assume we wrap a GymnaxGymWrapper instance.
        
        next_obs, next_env_state, reward, done, info = self._env.step(
            rng, state.pipeline_state, action, self._env_params
        )
        
        return BraxState(
            pipeline_state=next_env_state,
            obs=next_obs,
            reward=reward,
            done=done,
            metrics=info
        )

def symlog(x):
    return jnp.sign(x) * jnp.log(jnp.abs(x) + 1)
