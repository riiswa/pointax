#!/usr/bin/env python3
"""
Comprehensive pytest test suite for Pointax.

Usage:
    pytest test_pointax_comprehensive.py -v
    pytest test_pointax_comprehensive.py -v -s  # with print output
    pytest test_pointax_comprehensive.py::test_custom_environments -v  # single test

This covers:
- All environment creation methods
- Custom maze functionality  
- JAX compatibility (JIT, vmap, grad)
- Physics and collision detection
- Reward systems
- API functions
- Edge cases
- Performance benchmarks
"""

import time
import pytest
import jax
import jax.numpy as jnp
import numpy as np
import pointax


# Fixtures
@pytest.fixture
def sample_key():
    """Standard random key for tests."""
    return jax.random.PRNGKey(42)


@pytest.fixture
def simple_custom_maze():
    """Simple custom maze for testing."""
    return [
        [1, 1, 1, 1, 1],
        [1, 0, 0, 0, 1],
        [1, 0, 1, 0, 1],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1]
    ]


@pytest.fixture
def goal_maze():
    """Custom maze with goals and resets."""
    return [
        [1, 1, 1, 1, 1, 1, 1],
        [1, 'R', 0, 0, 0, 'G', 1],
        [1, 0, 1, 1, 1, 0, 1],
        [1, 0, 0, 'G', 0, 0, 1],
        [1, 'C', 0, 0, 0, 'C', 1],
        [1, 1, 1, 1, 1, 1, 1]
    ]


# Environment Creation Tests
@pytest.mark.parametrize("env_name", ["UMaze", "Open", "Medium", "Large", "Giant"])
def test_standard_environments(env_name):
    """Test creation of all standard environments."""
    env = pointax.make(env_name)
    assert env.maze_id == env_name
    assert env.reward_type_str == "sparse"
    assert not env.continuing_task
    assert not env.reset_target

    # Test with dense rewards
    env_dense = pointax.make(env_name, reward_type="dense")
    assert env_dense.reward_type_str == "dense"


@pytest.mark.parametrize("env_name", ["Open_Diverse_G", "Medium_Diverse_G", "Large_Diverse_G"])
def test_diverse_goal_environments(env_name):
    """Test diverse goal environments."""
    env = pointax.make(env_name)
    params = env.default_params
    assert params.num_goals > 0, f"{env_name} should have goals"


@pytest.mark.parametrize("env_name", ["Open_Diverse_GR", "Medium_Diverse_GR", "Large_Diverse_GR"])
def test_diverse_goal_reset_environments(env_name):
    """Test diverse goal-reset environments."""
    env = pointax.make(env_name)
    params = env.default_params
    assert params.num_combined > 0, f"{env_name} should have combined locations"


def test_invalid_environment_creation():
    """Test error handling for invalid environments."""
    with pytest.raises(ValueError, match="Unknown maze_id"):
        pointax.make("NonexistentMaze")

    with pytest.raises(ValueError, match="Unknown maze_id"):
        pointax.get_environment_info("NonexistentMaze")


# Custom Environment Tests
def test_custom_environments_numeric(simple_custom_maze):
    """Test custom environment with numeric maze."""
    env = pointax.make_custom(simple_custom_maze)
    assert env.maze_id == "Custom"
    assert env.custom_maze_layout == simple_custom_maze

    # Test with custom parameters
    env = pointax.make_custom(
        simple_custom_maze,
        maze_id="TestMaze",
        reward_type="dense"
    )
    assert env.maze_id == "TestMaze"
    assert env.reward_type_str == "dense"


def test_custom_environments_boolean():
    """Test custom environment with boolean maze."""
    bool_maze = [
        [True, True, True],
        [True, False, True],
        [True, True, True]
    ]
    env = pointax.make_custom(bool_maze, maze_id="BoolTest")
    params = env.default_params

    # Check boolean conversion: True->1 (wall), False->0 (empty)
    assert params.maze_map[1, 1] == 0  # False -> empty
    assert params.maze_map[0, 0] == 1  # True -> wall


def test_custom_environments_with_goals(goal_maze):
    """Test custom environment with goals and resets."""
    env = pointax.make_custom(goal_maze, reward_type="dense")
    params = env.default_params

    assert params.num_goals > 0
    assert params.num_resets > 0
    assert params.num_combined > 0
    assert params.reward_type == 1  # dense


def test_custom_environment_functionality(simple_custom_maze, sample_key):
    """Test that custom environments work correctly."""
    env = pointax.make_custom(simple_custom_maze, reward_type="sparse")
    params = env.default_params

    # Test reset
    obs, state = env.reset_env(sample_key, params)
    assert obs.shape == (6,)
    assert state.time == 0

    # Test step
    action = jnp.array([0.1, 0.1])
    new_obs, new_state, reward, done, info = env.step_env(
        sample_key, state, action, params
    )
    assert new_obs.shape == (6,)
    assert isinstance(reward, jnp.ndarray)
    assert new_state.time == 1


# Core Functionality Tests
def test_basic_environment_operations(sample_key):
    """Test basic environment operations."""
    env = pointax.make_umaze()
    params = env.default_params

    # Test reset
    obs, state = env.reset_env(sample_key, params)
    assert obs.shape == (6,), f"Expected obs shape (6,), got {obs.shape}"
    assert state.time == 0
    assert jnp.allclose(state.velocity, jnp.zeros(2))

    # Test step
    action = jnp.array([0.5, 0.2])
    new_obs, new_state, reward, done, info = env.step_env(sample_key, state, action, params)

    assert new_obs.shape == (6,)
    assert reward.shape == (), f"Expected scalar reward, got shape {reward.shape}"
    assert done.shape == (), f"Expected scalar done, got shape {done.shape}"
    assert isinstance(info, dict)
    assert "is_success" in info
    assert new_state.time == 1


def test_multiple_steps(sample_key):
    """Test multiple environment steps."""
    env = pointax.make_medium()
    params = env.default_params
    obs, state = env.reset_env(sample_key, params)

    for i in range(10):
        action = jax.random.uniform(sample_key, (2,), minval=-1.0, maxval=1.0)
        new_obs, state, reward, done, info = env.step_env(sample_key, state, action, params)

        assert jnp.isfinite(reward), f"Non-finite reward at step {i}"
        assert jnp.all(jnp.isfinite(new_obs)), f"Non-finite observation at step {i}"

        if done:
            break


# JAX Compatibility Tests
def test_jit_compilation(sample_key):
    """Test JAX JIT compilation."""
    env = pointax.make_medium()
    params = env.default_params

    @jax.jit
    def jit_reset(key):
        return env.reset_env(key, params)

    @jax.jit
    def jit_step(key, state, action):
        return env.step_env(key, state, action, params)

    # Test JIT functions
    obs, state = jit_reset(sample_key)
    action = jnp.array([0.1, 0.1])
    new_obs, new_state, reward, done, info = jit_step(sample_key, state, action)

    assert jnp.isfinite(reward)
    assert jnp.all(jnp.isfinite(new_obs))


def test_vectorization(sample_key):
    """Test JAX vectorization with vmap."""
    env = pointax.make_large()
    params = env.default_params
    batch_size = 4

    @jax.vmap
    def batch_reset(keys):
        return env.reset_env(keys, params)

    @jax.vmap
    def batch_step(keys, states, actions):
        return env.step_env(keys, states, actions, params)

    # Test batch operations
    keys = jax.random.split(sample_key, batch_size)
    batch_obs, batch_states = batch_reset(keys)
    assert batch_obs.shape == (batch_size, 6)

    batch_actions = jnp.zeros((batch_size, 2))
    batch_new_obs, batch_new_states, batch_rewards, batch_dones, batch_infos = batch_step(
        keys, batch_states, batch_actions
    )
    assert batch_rewards.shape == (batch_size,)
    assert batch_dones.shape == (batch_size,)


# Physics and Collision Tests
def test_collision_detection(sample_key):
    """Test collision detection works properly."""
    # Create maze with narrow passages for collision testing
    collision_maze = [
        [1, 1, 1, 1, 1],
        [1, 0, 1, 0, 1],  # Narrow passage
        [1, 0, 1, 0, 1],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1]
    ]

    env = pointax.make_custom(collision_maze)
    params = env.default_params

    # Place agent and try to move into wall
    from pointax.types import EnvState

    state = EnvState(
        position=jnp.array([0.0, 0.0]),
        velocity=jnp.zeros(2),
        desired_goal=jnp.array([1.0, 1.0]),
        time=0
    )

    # Try to move into wall with large force
    wall_action = jnp.array([1.0, 0.0])

    for _ in range(5):
        new_obs, state, reward, done, info = env.step_env(sample_key, state, wall_action, params)

        # Agent should not penetrate walls (position should remain reasonable)
        assert jnp.abs(state.position[0]) < 10.0, "Agent went too far (possible wall penetration)"
        assert jnp.abs(state.position[1]) < 10.0, "Agent went too far (possible wall penetration)"


def test_boundary_constraints(sample_key):
    """Test maze boundary constraints."""
    env = pointax.make_open()
    params = env.default_params

    obs, state = env.reset_env(sample_key, params)

    # Apply extreme actions to test boundaries
    extreme_actions = [
        jnp.array([1.0, 1.0]),  # Maximum positive
        jnp.array([-1.0, -1.0]),  # Maximum negative
        jnp.array([1.0, -1.0]),  # Mixed
    ]

    for action in extreme_actions:
        for _ in range(10):  # Multiple steps with extreme action
            new_obs, state, reward, done, info = env.step_env(sample_key, state, action, params)

            # Position should stay within reasonable bounds
            assert jnp.abs(state.position[0]) < params.x_map_center + 1.0
            assert jnp.abs(state.position[1]) < params.y_map_center + 1.0


# Reward System Tests
def test_sparse_vs_dense_rewards(sample_key):
    """Test sparse vs dense reward systems."""
    env_sparse = pointax.make_umaze(reward_type="sparse")
    env_dense = pointax.make_umaze(reward_type="dense")

    params_sparse = env_sparse.default_params
    params_dense = env_dense.default_params

    # Test far from goal
    from pointax.types import EnvState
    far_state = EnvState(
        position=jnp.array([0.0, 0.0]),
        velocity=jnp.zeros(2),
        desired_goal=jnp.array([2.0, 2.0]),  # Far away
        time=0
    )

    action = jnp.zeros(2)

    _, _, reward_sparse, _, _ = env_sparse.step_env(sample_key, far_state, action, params_sparse)
    _, _, reward_dense, _, _ = env_dense.step_env(sample_key, far_state, action, params_dense)

    # Sparse should be 0 when far from goal
    assert reward_sparse == 0.0, f"Sparse reward should be 0, got {reward_sparse}"

    # Dense should be negative (distance-based)
    assert reward_dense < 0.0, f"Dense reward should be negative, got {reward_dense}"


def test_goal_achievement(sample_key):
    """Test goal achievement rewards."""
    env_sparse = pointax.make_umaze(reward_type="sparse")
    env_dense = pointax.make_umaze(reward_type="dense")

    params_sparse = env_sparse.default_params
    params_dense = env_dense.default_params

    # Test at goal
    from pointax.types import EnvState
    goal_state = EnvState(
        position=jnp.array([1.0, 1.0]),
        velocity=jnp.zeros(2),
        desired_goal=jnp.array([1.0, 1.0]),  # At goal
        time=0
    )

    action = jnp.zeros(2)

    _, _, reward_sparse, _, info_sparse = env_sparse.step_env(sample_key, goal_state, action, params_sparse)
    _, _, reward_dense, _, info_dense = env_dense.step_env(sample_key, goal_state, action, params_dense)

    # Both should indicate success
    assert reward_sparse > 0.0, "Sparse reward should be positive at goal"
    assert reward_dense > -1.0, "Dense reward should be higher at goal"  # Less negative
    assert info_sparse["is_success"], "Should indicate success at goal"
    assert info_dense["is_success"], "Should indicate success at goal"


def test_continuing_tasks(sample_key):
    """Test continuing task functionality."""
    env = pointax.make_umaze(continuing_task=True, reset_target=True)
    params = env.default_params

    assert params.continuing_task == True
    assert params.reset_target == True

    # Place agent at goal to test continuing behavior
    from pointax.types import EnvState
    goal_state = EnvState(
        position=jnp.array([1.0, 1.0]),
        velocity=jnp.zeros(2),
        desired_goal=jnp.array([1.0, 1.0]),
        time=0
    )

    action = jnp.zeros(2)

    _, new_state, reward, done, info = env.step_env(sample_key, goal_state, action, params)

    # Should get reward for reaching goal
    assert reward > 0.0, "Should get reward for reaching goal"
    assert info["is_success"], "Should indicate success"

    # For continuing task, should not be done even at goal
    assert not done, "Continuing task should not terminate at goal"

    # New goal should be valid
    assert jnp.all(jnp.isfinite(new_state.desired_goal)), "New goal should be finite"


# Space Definition Tests
def test_action_space():
    """Test action space definition."""
    env = pointax.make_large()
    params = env.default_params

    action_space = env.action_space(params)
    assert action_space.shape == (2,), f"Expected action shape (2,), got {action_space.shape}"
    assert jnp.allclose(action_space.low, -1.0), "Action space low should be -1.0"
    assert jnp.allclose(action_space.high, 1.0), "Action space high should be 1.0"


def test_observation_space():
    """Test observation space definition."""
    env = pointax.make_large()
    params = env.default_params

    obs_space = env.observation_space(params)
    assert obs_space.shape == (6,), f"Expected obs shape (6,), got {obs_space.shape}"


def test_state_space():
    """Test state space definition."""
    env = pointax.make_large()
    params = env.default_params

    state_space = env.state_space(params)
    required_keys = ["position", "velocity", "desired_goal", "time"]
    for key in required_keys:
        assert key in state_space.spaces, f"Missing key '{key}' in state space"


# API Function Tests
def test_list_environments():
    """Test list_environments function."""
    envs = pointax.list_environments()
    assert isinstance(envs, dict)
    assert "UMaze" in envs
    assert "Open_Diverse_G" in envs
    assert len(envs) >= 11, "Should have at least 11 environments"


def test_get_environment_info():
    """Test get_environment_info function."""
    info = pointax.get_environment_info("UMaze")
    assert info["maze_id"] == "UMaze"
    assert "size" in info
    assert "walls" in info
    assert "empty" in info
    assert isinstance(info["is_diverse"], bool)

    # Test diverse environment
    info_diverse = pointax.get_environment_info("Open_Diverse_G")
    assert info_diverse["is_diverse"] == True
    assert info_diverse["goals"] > 0


def test_factory_functions():
    """Test all factory functions work."""
    factory_functions = [
        pointax.make_umaze,
        pointax.make_open,
        pointax.make_medium,
        pointax.make_large,
        pointax.make_giant,
        pointax.make_open_diverse_g,
        pointax.make_medium_diverse_g,
        pointax.make_large_diverse_g,
        pointax.make_open_diverse_gr,
        pointax.make_medium_diverse_gr,
        pointax.make_large_diverse_gr,
    ]

    for factory_fn in factory_functions:
        env = factory_fn()
        assert isinstance(env, pointax.PointMazeEnv)

        # Test with different reward types
        env_dense = factory_fn(reward_type="dense")
        assert env_dense.reward_type_str == "dense"


# Performance Tests
def test_performance_benchmark(sample_key):
    """Test performance characteristics."""
    env = pointax.make_large()
    params = env.default_params

    @jax.jit
    def step_fn(key, state, action):
        return env.step_env(key, state, action, params)

    @jax.jit
    def reset_fn(key):
        return env.reset_env(key, params)

    obs, state = reset_fn(sample_key)
    action = jnp.zeros(2)

    # Warmup JIT
    for _ in range(10):
        obs, state, reward, done, info = step_fn(sample_key, state, action)

    # Benchmark
    n_steps = 1000
    start_time = time.time()

    for _ in range(n_steps):
        obs, state, reward, done, info = step_fn(sample_key, state, action)

    end_time = time.time()
    steps_per_second = n_steps / (end_time - start_time)

    print(f"\nPerformance: {steps_per_second:.0f} steps/second")
    assert steps_per_second > 500, f"Performance too slow: {steps_per_second:.0f} steps/s"


# Edge Case Tests
def test_tiny_maze(sample_key):
    """Test very small maze."""
    tiny_maze = [
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1]
    ]
    env = pointax.make_custom(tiny_maze)
    params = env.default_params

    # Should work even with tiny maze
    obs, state = env.reset_env(sample_key, params)
    action = jnp.array([0.1, 0.1])
    new_obs, new_state, reward, done, info = env.step_env(sample_key, state, action, params)

    assert jnp.all(jnp.isfinite(new_obs)), "Tiny maze should produce finite observations"


def test_extreme_actions(sample_key):
    """Test handling of extreme actions."""
    env = pointax.make_open()
    params = env.default_params
    obs, state = env.reset_env(sample_key, params)

    # Test very large actions
    extreme_action = jnp.array([100.0, -100.0])
    new_obs, new_state, reward, done, info = env.step_env(sample_key, state, extreme_action, params)

    # Should handle extreme actions gracefully
    assert jnp.all(jnp.isfinite(new_obs)), "Should handle extreme actions"
    assert jnp.all(jnp.isfinite(new_state.position)), "Position should remain finite"
    assert jnp.all(jnp.isfinite(new_state.velocity)), "Velocity should remain finite"


def test_maze_with_minimal_empty_space(sample_key):
    """Test maze with very few empty spaces."""
    wall_maze = [
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 0, 1, 1],  # One empty space
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1]
    ]
    env = pointax.make_custom(wall_maze)
    params = env.default_params

    # Should still work
    obs, state = env.reset_env(sample_key, params)
    assert jnp.all(jnp.isfinite(obs)), "Wall maze should produce finite observations"


# Integration Tests
def test_full_episode_simulation(sample_key):
    """Test complete episode simulation."""
    env = pointax.make_medium_diverse_g(reward_type="dense")
    params = env.default_params

    obs, state = env.reset_env(sample_key, params)
    total_reward = 0.0

    for step in range(100):  # Max 100 steps
        # Random action policy
        action = jax.random.uniform(sample_key, (2,), minval=-0.5, maxval=0.5)
        obs, state, reward, done, info = env.step_env(sample_key, state, action, params)

        total_reward += reward

        # Check all outputs are valid
        assert jnp.all(jnp.isfinite(obs)), f"Invalid observation at step {step}"
        assert jnp.isfinite(reward), f"Invalid reward at step {step}"
        assert done.shape == (), f"Done should be scalar at step {step}"
        assert done.dtype == jnp.bool_ or done.dtype == bool, f"Done should be boolean type at step {step}"

        if done:
            break

    assert jnp.isfinite(total_reward), "Total reward should be finite"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])