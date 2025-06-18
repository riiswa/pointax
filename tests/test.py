"""Comprehensive tests for PointMaze environment library."""

import pytest
import jax
import jax.numpy as jnp
import pointax


class TestEnvironmentCreation:
    """Test environment creation and initialization."""

    def test_standard_environments(self):
        """Test creation of all standard environments."""
        standard_envs = [
            "UMaze", "Open", "Medium", "Large", "Giant"
        ]

        for maze_id in standard_envs:
            env = pointax.make(maze_id)
            assert env.maze_id == maze_id
            assert env.reward_type_str == "sparse"
            assert not env.continuing_task
            assert not env.reset_target

    def test_diverse_goal_environments(self):
        """Test creation of diverse goal environments."""
        diverse_g_envs = [
            "Open_Diverse_G", "Medium_Diverse_G", "Large_Diverse_G"
        ]

        for maze_id in diverse_g_envs:
            env = pointax.make(maze_id)
            assert env.maze_id == maze_id
            params = env.default_params
            # Should have specific goal locations
            assert params.num_goals > 0

    def test_diverse_goal_reset_environments(self):
        """Test creation of diverse goal-reset environments."""
        diverse_gr_envs = [
            "Open_Diverse_GR", "Medium_Diverse_GR", "Large_Diverse_GR"
        ]

        for maze_id in diverse_gr_envs:
            env = pointax.make(maze_id)
            assert env.maze_id == maze_id
            params = env.default_params
            # Should have combined locations
            assert params.num_combined > 0

    def test_reward_types(self):
        """Test different reward types."""
        env_sparse = pointax.make_umaze(reward_type="sparse")
        env_dense = pointax.make_umaze(reward_type="dense")

        assert env_sparse.default_params.reward_type == 0
        assert env_dense.default_params.reward_type == 1

    def test_continuing_task_options(self):
        """Test continuing task configuration."""
        env = pointax.make_umaze(
            continuing_task=True,
            reset_target=True
        )
        params = env.default_params
        assert params.continuing_task
        assert params.reset_target

    def test_invalid_maze_id(self):
        """Test error handling for invalid maze IDs."""
        with pytest.raises(ValueError, match="Unknown maze_id"):
            pointax.make("NonexistentMaze")


class TestEnvironmentDynamics:
    """Test environment step and reset functionality."""

    @pytest.fixture
    def env_and_key(self):
        """Fixture providing environment and random key."""
        env = pointax.make_umaze()
        key = jax.random.PRNGKey(42)
        return env, key

    def test_reset_functionality(self, env_and_key):
        """Test environment reset."""
        env, key = env_and_key
        params = env.default_params

        obs, state = env.reset_env(key, params)

        # Check observation shape and contents
        assert obs.shape == (6,)  # [pos_x, pos_y, vel_x, vel_y, goal_x, goal_y]
        assert state.time == 0
        assert jnp.allclose(state.velocity, jnp.zeros(2))

        # Position and goal should be different
        position = obs[:2]
        goal = obs[4:6]
        assert not jnp.allclose(position, goal)

    def test_step_functionality(self, env_and_key):
        """Test environment step."""
        env, key = env_and_key
        params = env.default_params

        # Reset and take a step
        obs, state = env.reset_env(key, params)
        action = jnp.array([0.5, 0.0])  # Move right

        new_obs, new_state, reward, done, info = env.step_env(
            key, state, action, params
        )

        # Check output shapes
        assert new_obs.shape == (6,)
        assert reward.shape == ()
        assert done.shape == ()
        assert isinstance(info, dict)

        # Time should advance
        assert new_state.time == state.time + 1

        # Position should change (unless hitting wall immediately)
        # Velocity should be non-zero after applying force
        assert not jnp.allclose(new_state.velocity, jnp.zeros(2))

    def test_wall_collisions(self, env_and_key):
        """Test wall collision handling."""
        env, key = env_and_key
        params = env.default_params

        obs, state = env.reset_env(key, params)

        # Try to move into a wall (large force towards boundary)
        large_action = jnp.array([1.0, 1.0])

        # Take multiple steps to try to hit a wall
        for _ in range(10):
            new_obs, new_state, reward, done, info = env.step_env(
                key, state, large_action, params
            )
            state = new_state

            # Position should remain within maze bounds
            position = new_state.position
            # Basic sanity check - position shouldn't be extreme
            assert jnp.abs(position[0]) < 100.0
            assert jnp.abs(position[1]) < 100.0

    def test_goal_achievement(self):
        """Test goal achievement and reward."""
        env = pointax.make_umaze(reward_type="sparse")
        params = env.default_params
        key = jax.random.PRNGKey(42)

        # Create state where agent is at goal
        goal_pos = jnp.array([1.0, 1.0])
        state = pointax.EnvState(
            position=goal_pos,
            velocity=jnp.zeros(2),
            desired_goal=goal_pos,
            time=0
        )

        action = jnp.zeros(2)  # No action
        new_obs, new_state, reward, done, info = env.step_env(
            key, state, action, params
        )

        # Should receive reward for being at goal
        assert reward > 0.0
        assert info["is_success"]

        # For non-continuing task, should be done
        if not params.continuing_task:
            assert done

    def test_dense_vs_sparse_rewards(self):
        """Test difference between dense and sparse rewards."""
        key = jax.random.PRNGKey(42)

        env_sparse = pointax.make_umaze(reward_type="sparse")
        env_dense = pointax.make_umaze(reward_type="dense")

        params_sparse = env_sparse.default_params
        params_dense = env_dense.default_params

        # Create state far from goal
        far_pos = jnp.array([0.0, 0.0])
        goal_pos = jnp.array([2.0, 2.0])
        state = pointax.EnvState(
            position=far_pos,
            velocity=jnp.zeros(2),
            desired_goal=goal_pos,
            time=0
        )

        action = jnp.zeros(2)

        _, _, reward_sparse, _, _ = env_sparse.step_env(
            key, state, action, params_sparse
        )
        _, _, reward_dense, _, _ = env_dense.step_env(
            key, state, action, params_dense
        )

        # Sparse should be 0 when far from goal
        assert reward_sparse == 0.0
        # Dense should be positive but small
        assert 0.0 < reward_dense < 1.0


class TestMazeUtilities:
    """Test maze utility functions."""

    def test_list_environments(self):
        """Test environment listing."""
        envs = pointax.list_environments()
        assert isinstance(envs, dict)
        assert "UMaze" in envs
        assert "Open_Diverse_G" in envs
        assert "Large_Diverse_GR" in envs

    def test_get_environment_info(self):
        """Test environment info retrieval."""
        info = pointax.get_environment_info("UMaze")
        assert info["maze_id"] == "UMaze"
        assert info["size"] == "5x5"
        assert not info["is_diverse"]
        assert info["walls"] > 0
        assert info["empty"] > 0

        # Test diverse environment
        info_diverse = pointax.get_environment_info("Open_Diverse_G")
        assert info_diverse["is_diverse"]
        assert info_diverse["goals"] > 0

    def test_invalid_environment_info(self):
        """Test error handling for invalid environment info."""
        with pytest.raises(ValueError, match="Unknown maze_id"):
            pointax.get_environment_info("NonexistentMaze")


class TestJAXCompatibility:
    """Test JAX-specific functionality."""

    def test_jit_compilation(self):
        """Test that environment functions can be JIT compiled."""
        env = pointax.make_umaze()
        params = env.default_params

        # JIT compile step function
        @jax.jit
        def jit_step(key, state, action):
            return env.step_env(key, state, action, params)

        # JIT compile reset function
        @jax.jit
        def jit_reset(key):
            return env.reset_env(key, params)

        key = jax.random.PRNGKey(42)

        # Test JIT compiled functions
        obs, state = jit_reset(key)
        action = jnp.array([0.1, 0.1])
        new_obs, new_state, reward, done, info = jit_step(key, state, action)

        # Should execute without error
        assert new_obs.shape == (6,)
        assert isinstance(reward, jnp.ndarray)

    def test_vectorization(self):
        """Test environment vectorization."""
        env = pointax.make_umaze()
        params = env.default_params

        # Create vectorized version
        batch_size = 8

        @jax.vmap
        def batch_reset(keys):
            return env.reset_env(keys, params)

        @jax.vmap
        def batch_step(keys, states, actions):
            return env.step_env(keys, states, actions, params)

        # Test batch operations
        keys = jax.random.split(jax.random.PRNGKey(42), batch_size)

        batch_obs, batch_states = batch_reset(keys)
        assert batch_obs.shape == (batch_size, 6)

        batch_actions = jnp.zeros((batch_size, 2))
        batch_new_obs, batch_new_states, batch_rewards, batch_dones, batch_infos = batch_step(
            keys, batch_states, batch_actions
        )

        assert batch_new_obs.shape == (batch_size, 6)
        assert batch_rewards.shape == (batch_size,)
        assert batch_dones.shape == (batch_size,)


class TestSpaceDefinitions:
    """Test action and observation space definitions."""

    def test_action_space(self):
        """Test action space definition."""
        env = pointax.make_umaze()
        params = env.default_params
        action_space = env.action_space(params)

        assert action_space.shape == (2,)
        assert jnp.allclose(action_space.low, -1.0)
        assert jnp.allclose(action_space.high, 1.0)

    def test_observation_space(self):
        """Test observation space definition."""
        env = pointax.make_umaze()
        params = env.default_params
        obs_space = env.observation_space(params)

        assert obs_space.shape == (6,)  # [pos, vel, goal]

    def test_state_space(self):
        """Test state space definition."""
        env = pointax.make_umaze()
        params = env.default_params
        state_space = env.state_space(params)

        assert "position" in state_space.spaces
        assert "velocity" in state_space.spaces
        assert "desired_goal" in state_space.spaces
        assert "time" in state_space.spaces


class TestFactoryFunctions:
    """Test individual factory functions."""

    def test_all_factory_functions(self):
        """Test that all factory functions work."""
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


if __name__ == "__main__":
    pytest.main([__file__])