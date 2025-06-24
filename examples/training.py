import os
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from multiprocessing import Pool
from typing import Any, NamedTuple, Optional, Sequence, Tuple, Union

import chex
import distrax
import flax.linen as nn
import imageio
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from flax import struct
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from gymnasium import envs
from gymnax.environments import environment, spaces
from gymnax.wrappers.purerl import GymnaxWrapper
from tqdm import tqdm

import pointax


class ActorCritic(nn.Module):
    action_dim: Sequence[int]
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        actor_mean = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        actor_logtstd = self.param("log_std", nn.initializers.zeros, (self.action_dim,))
        pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_logtstd))

        critic = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        critic = activation(critic)
        critic = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return pi, jnp.squeeze(critic, axis=-1)


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray


@struct.dataclass
class LogEnvState:
    env_state: environment.EnvState
    episode_returns: float
    episode_lengths: int
    returned_episode_returns: float
    returned_episode_lengths: int
    timestep: int


class LogWrapper(GymnaxWrapper):
    """Wrapper that logs episode returns and lengths."""

    def __init__(self, env: environment.Environment):
        super().__init__(env)

    @partial(jax.jit, static_argnums=(0,))
    def reset(
        self, key: chex.PRNGKey, params: Optional[environment.EnvParams] = None
    ) -> Tuple[chex.Array, LogEnvState]:
        obs, env_state = self._env.reset(key, params)
        state = LogEnvState(
            env_state=env_state,
            episode_returns=0.0,
            episode_lengths=0,
            returned_episode_returns=0.0,
            returned_episode_lengths=0,
            timestep=0,
        )
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: LogEnvState,
        action: Union[int, float, chex.Array],
        params: Optional[environment.EnvParams] = None,
    ) -> Tuple[chex.Array, LogEnvState, jnp.ndarray, jnp.ndarray, dict]:
        obs, env_state, reward, done, info = self._env.step(
            key, state.env_state, action, params
        )
        new_episode_return = state.episode_returns + reward
        new_episode_length = state.episode_lengths + 1
        state = LogEnvState(
            env_state=env_state,
            episode_returns=new_episode_return * (1 - done),
            episode_lengths=new_episode_length * (1 - done),
            returned_episode_returns=state.returned_episode_returns * (1 - done)
            + new_episode_return * done,
            returned_episode_lengths=state.returned_episode_lengths * (1 - done)
            + new_episode_length * done,
            timestep=state.timestep + 1,
        )
        return obs, state, reward, done, info


def save_video_worker(args):
    """Worker function for multiprocessing video rendering."""
    states, env_params, filename, fps = args

    # Create environment for rendering (each process needs its own)
    env = pointax.make_medium(
        reward_type="sparse", reset_target=True, continuing_task=True
    )
    env = LogWrapper(env)

    def render_frame(i):
        # Extract the actual environment state from LogEnvState
        env_state_i = jax.tree.map(lambda x: x[i], states.env_state)
        frame = env._env.render(env_state_i, env_params, mode="rgb_array")
        return np.array(frame)

    frames = [render_frame(i) for i in range(len(states.env_state.position))]
    imageio.mimsave(filename, frames, fps=fps)
    return filename


def save_video(env, states, params, filename="pointmaze.mp4", fps=30):
    """Render trajectory and save as video."""

    def render_frame(i):
        # Extract the actual environment state from LogEnvState
        env_state_i = jax.tree.map(lambda x: x[i], states.env_state)
        frame = env._env.render(env_state_i, params, mode="rgb_array")
        return np.array(frame)

    print(f"Rendering {len(states.env_state.position)} frames...")
    frames = [render_frame(i) for i in range(len(states.env_state.position))]

    print("Saving video...")
    imageio.mimsave(filename, frames, fps=fps)
    print(f"Saved video: {filename}")


def evaluate_agent_vmap(train_state, network, env, env_params, keys):
    """Vmapped evaluation - run all episodes in parallel."""

    def single_episode(key):
        # Reset environment
        key, reset_key = jax.random.split(key)
        obs, env_state = env.reset(reset_key, env_params)

        # Store initial state
        initial_env_state = env_state
        max_steps = 500

        def step_fn(carry, step_idx):
            obs, env_state, key, success, done, actual_length = carry

            # Get action from policy (deterministic)
            pi, _ = network.apply(train_state.params, obs)
            action = pi.mean()

            # Step environment
            key, step_key = jax.random.split(key)
            new_obs, new_env_state, reward, new_done, info = env.step(
                step_key, env_state, action, env_params
            )

            # Update only if not already done
            obs = jnp.where(done, obs, new_obs)
            env_state = jax.tree.map(
                lambda old, new: jnp.where(done, old, new), env_state, new_env_state
            )

            # Update success and done flags
            new_success = info.get("is_success", False)
            success = success | (new_success & ~done)  # Only if we weren't already done
            done = done | new_done
            actual_length = jnp.where(
                done, actual_length, step_idx + 2
            )  # +2: initial state + current step

            return (obs, env_state, key, success, done, actual_length), env_state

        # Initial carry
        init_carry = (
            obs,
            env_state,
            key,
            False,
            False,
            1,
        )  # length starts at 1 for initial state

        # Run episode for up to max_steps
        final_carry, states_trajectory = jax.lax.scan(
            step_fn, init_carry, jnp.arange(max_steps)
        )

        _, _, _, success, _, actual_length = final_carry

        # Prepend initial state to trajectory
        initial_state_batch = jax.tree.map(
            lambda x: jnp.expand_dims(x, 0), initial_env_state
        )
        all_states = jax.tree.map(
            lambda init, traj: jnp.concatenate([init, traj], axis=0),
            initial_state_batch,
            states_trajectory,
        )

        return success, actual_length, all_states

    # Vmap over all episodes
    return jax.vmap(single_episode)(keys)


def make_train(config):
    config["NUM_UPDATES"] = int(
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )

    env = pointax.make_medium(
        reward_type="sparse", reset_target=True, continuing_task=True
    )
    env = LogWrapper(env)
    env_params = env.default_params

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    def train(rng):
        # INIT NETWORK
        network = ActorCritic(
            env.action_space(env_params).shape[0], activation=config["ACTIVATION"]
        )
        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros(env.observation_space(env_params).shape)
        network_params = network.init(_rng, init_x)
        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            train_state, env_state, last_obs, rng = runner_state

            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, rng = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                pi, value = network.apply(train_state.params, last_obs)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0, None)
                )(rng_step, env_state, action, env_params)
                transition = Transition(
                    done, action, value, reward, log_prob, last_obs, info
                )
                runner_state = (train_state, env_state, obsv, rng)
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, rng = runner_state
            _, last_val = network.apply(train_state.params, last_obs)

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    )
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, traj_batch, gae, targets):
                        # RERUN NETWORK
                        pi, value = network.apply(params, traj_batch.obs)
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )

                        # CALCULATE ACTOR LOSS
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"],
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                        )
                        return total_loss, (value_loss, loss_actor, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, traj_batch, advantages, targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                train_state, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)

                total_batch_size = config["NUM_STEPS"] * config["NUM_ENVS"]
                assert (
                    config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
                    == total_batch_size
                ), "minibatch size * num minibatches must equal total batch size"

                permutation = jax.random.permutation(_rng, total_batch_size)
                batch = (traj_batch, advantages, targets)
                batch = jax.tree_util.tree_map(
                    lambda x: x.reshape((total_batch_size,) + x.shape[2:]), batch
                )
                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(
                        x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                    ),
                    shuffled_batch,
                )
                train_state, total_loss = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                update_state = (train_state, traj_batch, advantages, targets, rng)
                return update_state, total_loss

            update_state = (train_state, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]
            metric = traj_batch.info
            rng = runner_state[3]

            runner_state = (train_state, env_state, last_obs, rng)
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, _rng)
        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metrics": metrics}

    return train


if __name__ == "__main__":
    config = {
        "LR": 2.5e-4,
        "NUM_ENVS": 64,
        "NUM_STEPS": 128,
        "TOTAL_TIMESTEPS": 5e6,
        "UPDATE_EPOCHS": 4,
        "NUM_MINIBATCHES": 32,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.2,
        "ENT_COEF": 0.01,
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 0.5,
        "ACTIVATION": "tanh",
        "ANNEAL_LR": True,
    }

    print("Training PPO on UMaze...")
    print(f"Total timesteps: {config['TOTAL_TIMESTEPS']:,}")

    # Train
    rng = jax.random.PRNGKey(0)
    train_jit = jax.jit(make_train(config))
    results = train_jit(rng)

    print("Training completed! Running evaluation...")

    # Setup for evaluation
    env = pointax.make_medium(
        reward_type="sparse", reset_target=True, continuing_task=True
    )
    env = LogWrapper(env)
    env_params = env.default_params
    network = ActorCritic(env.action_space(env_params).shape[0])
    train_state = results["runner_state"][0]

    # Run evaluation episodes using vmap for parallel execution
    print("Running evaluation episodes...")
    eval_key = jax.random.PRNGKey(999)

    # Generate keys for all episodes
    eval_keys = jax.random.split(eval_key, 48)

    # Run all 48 episodes in parallel!
    print("Evaluating 48 episodes in parallel...")
    successes, lengths, all_states = evaluate_agent_vmap(
        train_state, network, env, env_params, eval_keys
    )

    # Convert to Python lists and prepare video data
    success_rates = [float(s) for s in successes]
    episode_lengths = [int(l) for l in lengths]
    episode_data = []

    os.makedirs("videos", exist_ok=True)

    # Prepare data for video rendering
    for i in range(48):
        # Extract states for this episode
        episode_states = jax.tree.map(lambda x: x[i], all_states)
        filename = f"videos/episode_{i + 1}.mp4"
        episode_data.append((episode_states, env_params, filename, 30))

        print(
            f"Episode {i + 1}: Success={success_rates[i]}, Length={episode_lengths[i]:.0f}"
        )

    # Render all videos using multiprocessing
    print("\nRendering videos using multiprocessing...")

    num_processes = min(os.cpu_count(), len(episode_data))
    with Pool(processes=num_processes) as pool:
        # Use tqdm to monitor progress
        results_iter = pool.imap(save_video_worker, episode_data)
        completed_videos = list(
            tqdm(results_iter, total=len(episode_data), desc="Rendering videos")
        )

    # Print summary
    print(f"\n{'=' * 50}")
    print("EVALUATION RESULTS")
    print(f"{'=' * 50}")
    print(f"Success Rate: {np.mean(success_rates):.1%}")
    print(f"Mean Episode Length: {np.mean(episode_lengths):.1f}")
    print(f"Videos saved in 'videos/' directory")
    print(f"Total videos rendered: {len(completed_videos)}")
    print(f"{'=' * 50}")
