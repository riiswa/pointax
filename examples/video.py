#!/usr/bin/env python3
"""
Minimal PointMaze demo with JAX scan and video generation.
"""
from cgitb import reset

import jax
import jax.numpy as jnp
import jax.random as jr
from jax import lax
import numpy as np
import imageio
from functools import partial
import pointax


def step_fn(carry, x, env, params):
    """Single step with random policy and auto-reset."""
    state, key = carry
    key, action_key, step_key, reset_key = jr.split(key, 4)

    action = jr.uniform(action_key, (2,), minval=-1.0, maxval=1.0)
    _, new_state, reward, done, _ = env.step_env(step_key, state, action, params)

    # Reset if episode is done
    _, reset_state = env.reset_env(reset_key, params)
    final_state = jax.tree.map(
        lambda new_val, reset_val: lax.select(done, reset_val, new_val),
        new_state, reset_state
    )

    return (final_state, key), (final_state, reward)


def run_episode(env, key, num_steps=200):
    """Run episode with random policy."""
    params = env.default_params
    key, reset_key = jr.split(key)
    _, initial_state = env.reset_env(reset_key, params)

    scan_fn = partial(step_fn, env=env, params=params)
    _, (states, rewards) = lax.scan(scan_fn, (initial_state, key), jnp.arange(num_steps))

    return states, rewards


def save_video(env, states, params, filename="pointmaze.mp4", fps=30, max_workers=8):
    """Render trajectory and save as video with multithreading."""
    from concurrent.futures import ThreadPoolExecutor

    def render_frame(i):
        state_i = jax.tree.map(lambda x: x[i], states)
        frame = env.render(state_i, params, mode="rgb_array")
        return i, np.array(frame)

    print(f"Rendering {len(states.position)} frames with {max_workers} threads...")

    # Parallel rendering
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(render_frame, range(len(states.position))))

    # Sort by index to maintain order
    results.sort(key=lambda x: x[0])
    frames = [frame for _, frame in results]

    print("Saving video...")
    imageio.mimsave(filename, frames, fps=fps)
    print(f"Saved video: {filename}")


def main():
    """Main demo."""
    env = pointax.make_large_diverse_gr(reward_type="sparse", reset_target=True, continuing_task=True)
    params = env.default_params
    key = jr.PRNGKey(42)

    print("Running random policy...")
    states, rewards = run_episode(env, key, num_steps=1000)

    print(f"Total reward: {float(jnp.sum(rewards)):.2f}")

    save_video(env, states, params, "pointmaze.mp4")

    return states, rewards


if __name__ == "__main__":
    states, rewards = main()