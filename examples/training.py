import argparse

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
import distrax
import pointax
from flax.linen.initializers import constant, orthogonal
from typing import NamedTuple
from flax.training.train_state import TrainState
from functools import partial


class ActorCritic(nn.Module):
    action_dim: int
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        # Actor network
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

        # Log std parameter
        actor_logtstd = self.param("log_std", nn.initializers.zeros, (self.action_dim,))
        pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_logtstd))

        # Critic network
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


def make_train(config):
    config["NUM_UPDATES"] = int(
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
            config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )

    # Create environment
    env = pointax.make(config["MAZE_LAYOUT"], reward_type="sparse")
    env_params = env.default_params

    def linear_schedule(count):
        frac = (
                1.0
                - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
                / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    def train(rng):
        # Initialize network
        network = ActorCritic(
            action_dim=2, activation=config["ACTIVATION"]
        )
        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros((6,))  # observation space
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

        # Initialize environment
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)

        # Training loop
        def _update_step(runner_state, unused):
            train_state, env_state, last_obs, rng = runner_state

            # Collect trajectories
            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, rng = runner_state

                # Select action
                rng, _rng = jax.random.split(rng)
                pi, value = network.apply(train_state.params, last_obs)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

                # Step environment
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0, None)
                )(rng_step, env_state, action, env_params)

                transition = Transition(
                    done, action, value, reward, log_prob, last_obs
                )
                runner_state = (train_state, env_state, obsv, rng)
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # Calculate GAE
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

            # Update network
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, traj_batch, gae, targets):
                        # Rerun network
                        pi, value = network.apply(params, traj_batch.obs)
                        log_prob = pi.log_prob(traj_batch.action)

                        # Value loss
                        value_pred_clipped = traj_batch.value + (
                                value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                                0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )

                        # Actor loss
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

                # Reshape and shuffle batch
                total_batch_size = config["NUM_STEPS"] * config["NUM_ENVS"]
                permutation = jax.random.permutation(_rng, total_batch_size)
                batch = (traj_batch, advantages, targets)
                batch = jax.tree.map(
                    lambda x: x.reshape((total_batch_size,) + x.shape[2:]), batch
                )
                shuffled_batch = jax.tree.map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )

                # Create minibatches
                minibatches = jax.tree.map(
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
            rng = update_state[-1]
            runner_state = (train_state, env_state, last_obs, rng)

            return runner_state, loss_info

        # Main training loop
        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, _rng)
        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )

        return {"runner_state": runner_state, "metrics": metrics}

    return train


def evaluate_agent(train_state, env_params, eval_key, network_apply, env, num_episodes=10, max_steps=1000):
    """JIT-compiled evaluation function using lax.scan."""

    def single_episode(key):
        """Single episode using lax.scan."""
        obs, env_state = env.reset(key, env_params)

        def step_fn(carry, step_idx):
            obs, env_state, key, total_reward, done, success, length = carry

            # Only step if not done
            key, step_key = jax.random.split(key)

            # Get deterministic action (mean of policy)
            pi, _ = network_apply(train_state.params, obs)
            action = pi.mean()

            # Step environment
            new_obs, new_env_state, reward, new_done, info = env.step(
                step_key, env_state, action, env_params
            )

            # Update only if not already done
            obs = jnp.where(done, obs, new_obs)
            env_state = jax.tree.map(
                lambda old, new: jnp.where(done, old, new), env_state, new_env_state
            )
            total_reward = jnp.where(done, total_reward, total_reward + reward)
            length = jnp.where(done, length, length + 1)

            # Update success and done flags
            new_success = info.get("is_success", False)
            success = success | (new_success & ~done)  # Only if we weren't already done
            done = done | new_done

            return (obs, env_state, key, total_reward, done, success, length), None

        # Initial state
        init_carry = (obs, env_state, key, 0.0, False, False, 0)

        # Run episode
        (final_obs, final_env_state, final_key, total_reward, done, success, length), _ = jax.lax.scan(
            step_fn, init_carry, jnp.arange(max_steps)
        )

        return total_reward, success, length

    # Vectorize over multiple episodes
    episode_fn = jax.vmap(single_episode)

    # Generate keys for all episodes
    eval_keys = jax.random.split(eval_key, num_episodes)

    # Run all episodes in parallel
    rewards, successes, lengths = episode_fn(eval_keys)

    return {
        "mean_reward": jnp.mean(rewards),
        "success_rate": jnp.mean(successes),
        "mean_length": jnp.mean(lengths),
        "rewards": rewards,
        "successes": successes,
        "lengths": lengths
    }


def parse_args():
    parser = argparse.ArgumentParser(description="PPO training on PointMaze")

    # Environment
    parser.add_argument("--maze-layout", type=str, default="Large",
                        help="Maze layout (default: Large)")

    # Training hyperparameters
    parser.add_argument("--lr", type=float, default=2.5e-4,
                        help="Learning rate (default: 2.5e-4)")
    parser.add_argument("--num-envs", type=int, default=256,
                        help="Number of parallel environments (default: 256)")
    parser.add_argument("--num-steps", type=int, default=128,
                        help="Number of steps per environment per update (default: 128)")
    parser.add_argument("--total-timesteps", type=int, default=5000000,
                        help="Total timesteps to train (default: 5000000)")
    parser.add_argument("--update-epochs", type=int, default=4,
                        help="Number of policy update epochs (default: 4)")
    parser.add_argument("--num-minibatches", type=int, default=32,
                        help="Number of minibatches (default: 32)")

    # PPO hyperparameters
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor (default: 0.99)")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
                        help="GAE lambda (default: 0.95)")
    parser.add_argument("--clip-eps", type=float, default=0.2,
                        help="PPO clipping parameter (default: 0.2)")
    parser.add_argument("--ent-coef", type=float, default=0.01,
                        help="Entropy coefficient (default: 0.01)")
    parser.add_argument("--vf-coef", type=float, default=0.5,
                        help="Value function coefficient (default: 0.5)")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
                        help="Maximum gradient norm (default: 0.5)")

    # Network
    parser.add_argument("--activation", type=str, default="tanh",
                        choices=["tanh", "relu"], help="Activation function (default: tanh)")

    # Learning rate schedule
    parser.add_argument("--no-anneal-lr", action="store_true",
                        help="Disable learning rate annealing")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    config = {
        "MAZE_LAYOUT": args.maze_layout,
        "LR": args.lr,
        "NUM_ENVS": args.num_envs,
        "NUM_STEPS": args.num_steps,
        "TOTAL_TIMESTEPS": args.total_timesteps,
        "UPDATE_EPOCHS": args.update_epochs,
        "NUM_MINIBATCHES": args.num_minibatches,
        "GAMMA": args.gamma,
        "GAE_LAMBDA": args.gae_lambda,
        "CLIP_EPS": args.clip_eps,
        "ENT_COEF": args.ent_coef,
        "VF_COEF": args.vf_coef,
        "MAX_GRAD_NORM": args.max_grad_norm,
        "ACTIVATION": args.activation,
        "ANNEAL_LR": not args.no_anneal_lr,
    }

    print(f"Training PPO on PointMaze {config['MAZE_LAYOUT']}...")
    print(f"Total timesteps: {config['TOTAL_TIMESTEPS']:,}")
    print(f"Updates: {config['TOTAL_TIMESTEPS'] // config['NUM_STEPS'] // config['NUM_ENVS']}")

    # Train
    rng = jax.random.PRNGKey(42)
    train_jit = jax.jit(make_train(config))
    print("Compiling...")
    results = train_jit(rng)
    print("Training completed!")

    # Evaluate
    print("\nEvaluating trained agent...")
    print("Compiling evaluation function...")
    env = pointax.make(config["MAZE_LAYOUT"], reward_type="sparse")
    env_params = env.default_params
    network = ActorCritic(action_dim=2, activation=config["ACTIVATION"])
    train_state = results["runner_state"][0]

    # Create JIT-compiled evaluation function with static arguments
    eval_fn = jax.jit(partial(evaluate_agent,
                              network_apply=network.apply,
                              env=env,
                              num_episodes=100))

    eval_results = eval_fn(train_state, env_params, jax.random.PRNGKey(999))

    print(f"\n{'=' * 50}")
    print("EVALUATION RESULTS (100 episodes)")
    print(f"{'=' * 50}")
    print(f"Success Rate: {float(eval_results['success_rate']):.1%}")
    print(f"Mean Episode Length: {float(eval_results['mean_length']):.1f}")
    print(f"{'=' * 50}")