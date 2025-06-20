"""Main PointMaze environment implementation."""

from typing import Tuple, Optional, Dict, Any
import jax
import jax.numpy as jnp
from jax import lax
import chex
from gymnax.environments import environment, spaces

from pointax.types import EnvState, EnvParams
from pointax.mazes import get_maze_layout, convert_maze_to_numeric, compute_all_locations


class PointMazeEnv(environment.Environment[EnvState, EnvParams]):
    """
    JAX implementation of Point Maze environment with diverse goal support.

    A 2-DoF point mass navigates through a maze to reach goal locations.
    Supports various maze layouts including diverse goal/reset configurations.

    Key Features:
    - JAX-native implementation for fast vectorization
    - Support for diverse goal mazes with special cell markers
    - Optimized collision detection and physics
    - Configurable reward types (sparse/dense)
    - Continuing task support with goal resetting

    Special cell markers:
    - 'G': Goal locations
    - 'R': Reset locations
    - 'C': Combined (both goal and reset) locations
    - 0: Empty locations
    - 1: Walls
    """

    def __init__(
            self,
            maze_id: str = "UMaze",
            reward_type: str = "sparse",
            continuing_task: bool = False,
            reset_target: bool = False
    ):
        """Initialize PointMaze environment.

        Args:
            maze_id: Maze layout identifier (e.g., 'UMaze', 'Open_Diverse_G')
            reward_type: 'sparse' or 'dense' reward function
            continuing_task: Whether task continues after reaching goal
            reset_target: Whether to reset goal when reached in continuing task
        """
        super().__init__()
        self.maze_id = maze_id
        self.reward_type_str = reward_type
        self.continuing_task = continuing_task
        self.reset_target = reset_target

    @property
    def default_params(self) -> EnvParams:
        """Get default environment parameters."""
        maze_layout = get_maze_layout(self.maze_id)
        maze_map = convert_maze_to_numeric(maze_layout)
        reward_type_int = 0 if self.reward_type_str == "sparse" else 1

        # Compute all location types
        locations_data = compute_all_locations(maze_map)

        # Pre-compute maze boundaries for optimization
        map_length, map_width = maze_map.shape
        x_map_center = map_width / 2 * 1.0
        y_map_center = map_length / 2 * 1.0

        return EnvParams(
            maze_map=maze_map,
            continuing_task=self.continuing_task,
            reset_target=self.reset_target,
            reward_type=reward_type_int,
            **locations_data,
            # Pre-computed values for faster access
            x_map_center=x_map_center,
            y_map_center=y_map_center,
            map_length=map_length,
            map_width=map_width
        )

    def step_env(
            self,
            key: chex.PRNGKey,
            state: EnvState,
            action: chex.Array,
            params: EnvParams
    ) -> Tuple[chex.Array, EnvState, chex.Array, chex.Array, Dict[str, Any]]:
        """Step the environment forward one timestep.

        Args:
            key: Random key for stochastic operations
            state: Current environment state
            action: Action to take [force_x, force_y]
            params: Environment parameters

        Returns:
            Tuple of (observation, new_state, reward, done, info)
        """
        # Clip actions and velocity
        action = jnp.clip(action, -1.0, 1.0)
        clipped_velocity = jnp.clip(state.velocity, -params.max_velocity, params.max_velocity)

        # Update velocity with forces
        new_velocity = jnp.clip(
            clipped_velocity + action * params.dt,
            -params.max_velocity,
            params.max_velocity
        )

        # Update position
        new_position = state.position + new_velocity * params.dt

        # Handle wall collisions
        new_position, new_velocity = self._handle_wall_collisions(
            new_position, new_velocity, state.position, params
        )

        # Compute reward and goal achievement
        distance_to_goal = jnp.linalg.norm(new_position - state.desired_goal)
        goal_reached = distance_to_goal <= params.goal_threshold

        # Reward computation
        reward = lax.select(
            params.reward_type == 0,
            goal_reached.astype(jnp.float32),  # sparse
            jnp.exp(-distance_to_goal)  # dense
        )

        # Goal reset logic for continuing tasks
        should_reset_goal = params.continuing_task & params.reset_target & goal_reached
        new_goal = lax.select(
            should_reset_goal,
            self._generate_goal_position(key, params),
            state.desired_goal
        )

        # Update state
        new_state = state.replace(
            position=new_position,
            velocity=new_velocity,
            desired_goal=new_goal,
            time=state.time + 1
        )

        # Termination logic
        done_steps = new_state.time >= params.max_steps_in_episode
        done_goal = (~params.continuing_task) & goal_reached
        done = done_steps | done_goal

        # Create observation
        obs = jnp.concatenate([new_position, new_velocity, new_goal])

        # Info dict
        info = {
            "is_success": goal_reached,
            "discount": self.discount(new_state, params)
        }

        return (
            lax.stop_gradient(obs),
            lax.stop_gradient(new_state),
            reward,
            done,
            info
        )

    def reset_env(
            self,
            key: chex.PRNGKey,
            params: EnvParams,
            options: Optional[Dict] = None
    ) -> Tuple[chex.Array, EnvState]:
        """Reset environment to initial state.

        Args:
            key: Random key for stochastic operations
            params: Environment parameters
            options: Optional reset configuration

        Returns:
            Tuple of (initial_observation, initial_state)
        """
        key, pos_key, goal_key = jax.random.split(key, 3)

        # Generate goal and reset positions
        goal_position = self._generate_goal_position(goal_key, params)
        goal_position = self._add_position_noise(goal_position, goal_key, params)

        reset_position = self._generate_reset_position(pos_key, params)
        reset_position = self._add_position_noise(reset_position, pos_key, params)

        # Initialize state
        state = EnvState(
            position=reset_position,
            velocity=jnp.zeros(2),
            desired_goal=goal_position,
            time=0
        )

        # Create initial observation
        obs = jnp.concatenate([reset_position, jnp.zeros(2), goal_position])

        return obs, state

    def get_obs(self, state: EnvState, params: EnvParams, key=None) -> chex.Array:
        """Get observation from current state.

        Args:
            state: Current environment state
            params: Environment parameters
            key: Unused, for compatibility

        Returns:
            Observation array [position, velocity, goal]
        """
        return jnp.concatenate([state.position, state.velocity, state.desired_goal])

    def is_terminal(self, state: EnvState, params: EnvParams) -> chex.Array:
        """Check if state is terminal.

        Args:
            state: Current environment state
            params: Environment parameters

        Returns:
            Boolean indicating if episode should terminate
        """
        done_steps = state.time >= params.max_steps_in_episode
        distance_to_goal = jnp.linalg.norm(state.position - state.desired_goal)
        done_goal = distance_to_goal <= params.goal_threshold

        return lax.select(
            params.continuing_task,
            done_steps,  # Continuing task only terminates on time limit
            done_steps | done_goal  # Normal task terminates on goal or time
        )

    def _generate_goal_position(self, key: chex.PRNGKey, params: EnvParams) -> chex.Array:
        """Generate goal position from appropriate location set.

        Priority: specific goals > combined > empty
        """
        has_specific_goals = params.num_goals > 0
        has_combined = params.num_combined > 0

        def use_goal_locations():
            goal_idx = jax.random.randint(key, (), 0, params.num_goals)
            return params.goal_locations[goal_idx]

        def use_combined_locations():
            combined_idx = jax.random.randint(key, (), 0, params.num_combined)
            return params.combined_locations[combined_idx]

        def use_empty_locations():
            empty_idx = jax.random.randint(key, (), 0, params.num_empty)
            return params.empty_locations[empty_idx]

        return lax.cond(
            has_specific_goals,
            use_goal_locations,
            lambda: lax.cond(
                has_combined,
                use_combined_locations,
                use_empty_locations
            )
        )

    def _generate_reset_position(self, key: chex.PRNGKey, params: EnvParams) -> chex.Array:
        """Generate reset position from appropriate location set.

        Priority: specific resets > combined > empty
        """
        has_specific_resets = params.num_resets > 0
        has_combined = params.num_combined > 0

        def use_reset_locations():
            reset_idx = jax.random.randint(key, (), 0, params.num_resets)
            return params.reset_locations[reset_idx]

        def use_combined_locations():
            combined_idx = jax.random.randint(key, (), 0, params.num_combined)
            return params.combined_locations[combined_idx]

        def use_empty_locations():
            empty_idx = jax.random.randint(key, (), 0, params.num_empty)
            return params.empty_locations[empty_idx]

        return lax.cond(
            has_specific_resets,
            use_reset_locations,
            lambda: lax.cond(
                has_combined,
                use_combined_locations,
                use_empty_locations
            )
        )

    def _add_position_noise(
            self,
            position: chex.Array,
            key: chex.PRNGKey,
            params: EnvParams
    ) -> chex.Array:
        """Add uniform noise to position coordinates."""
        noise_range = params.position_noise_range * params.maze_size_scaling
        noise = jax.random.uniform(
            key, (2,),
            minval=-noise_range,
            maxval=noise_range
        )
        return position + noise

    def _handle_wall_collisions(
            self,
            new_pos: chex.Array,
            velocity: chex.Array,
            old_pos: chex.Array,
            params: EnvParams
    ) -> Tuple[chex.Array, chex.Array]:
        """Handle wall collisions using optimized boundary detection."""
        x, y = new_pos

        # Convert to cell indices
        i = jnp.floor((params.y_map_center - y) / params.maze_size_scaling).astype(int)
        j = jnp.floor((x + params.x_map_center) / params.maze_size_scaling).astype(int)

        # Clamp to valid indices
        i = jnp.clip(i, 0, params.map_length - 1)
        j = jnp.clip(j, 0, params.map_width - 1)

        # Check if cell is a wall (only value 1 is a wall)
        is_wall = params.maze_map[i, j] == 1

        # Revert to old position if hitting wall
        final_pos = lax.select(is_wall, old_pos, new_pos)
        final_vel = lax.select(is_wall, jnp.zeros(2), velocity)

        return final_pos, final_vel

    @property
    def name(self) -> str:
        """Environment name."""
        dense_suffix = "Dense" if self.reward_type_str == "dense" else ""
        return f"PointMaze_{self.maze_id}{dense_suffix}-v3"

    @property
    def num_actions(self) -> int:
        """Number of actions."""
        return 2

    def action_space(self, params: EnvParams = None) -> spaces.Box:
        """Action space: continuous forces in x,y directions."""
        return spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=jnp.float32
        )

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Observation space: [position, velocity, goal] = 6 elements."""
        return spaces.Box(
            low=-jnp.inf, high=jnp.inf, shape=(6,), dtype=jnp.float32
        )

    def state_space(self, params: EnvParams) -> spaces.Dict:
        """State space of the environment."""
        return spaces.Dict({
            "position": spaces.Box(
                low=-jnp.inf, high=jnp.inf, shape=(2,), dtype=jnp.float32
            ),
            "velocity": spaces.Box(
                low=-params.max_velocity, high=params.max_velocity,
                shape=(2,), dtype=jnp.float32
            ),
            "desired_goal": spaces.Box(
                low=-jnp.inf, high=jnp.inf, shape=(2,), dtype=jnp.float32
            ),
            "time": spaces.Discrete(params.max_steps_in_episode),
        })


    def render(self, state, params, mode="human"):
        """Simple minimalist rendering: black walls, white space, red goal, green agent."""
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches


        map_length, map_width = params.maze_map.shape

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.set_xlim(-params.x_map_center - 0.5, params.x_map_center + 0.5)
        ax.set_ylim(-params.y_map_center - 0.5, params.y_map_center + 0.5)
        ax.set_aspect('equal')
        ax.axis('off')  # Minimalist - no axes

        # Draw maze: black walls, white empty space
        for i in range(map_length):
            for j in range(map_width):
                x = j - params.x_map_center + 0.5
                y = params.y_map_center - i - 0.5

                color = 'black' if params.maze_map[i, j] == 1 else 'white'
                rect = patches.Rectangle((x - 0.5, y - 0.5), 1, 1,
                                         facecolor=color, edgecolor='none')
                ax.add_patch(rect)

        # Draw goal (red circle)
        goal_circle = patches.Circle(state.desired_goal, 0.2, color='red', alpha=0.8)
        ax.add_patch(goal_circle)

        # Draw agent (green circle)
        agent_circle = patches.Circle(state.position, 0.15, color='green', alpha=0.9)
        ax.add_patch(agent_circle)

        plt.title(f'Step: {state.time}', fontsize=10)

        if mode == "human":
            plt.show()
        elif mode == "rgb_array":
            fig.canvas.draw()

            # Get the RGBA buffer and convert to RGB (compatible with newer matplotlib)
            buf = fig.canvas.buffer_rgba()
            w, h = fig.canvas.get_width_height()

            # Convert RGBA to RGB by dropping alpha channel
            rgba_array = jnp.frombuffer(buf, dtype=jnp.uint8).reshape((h, w, 4))
            rgb_array = rgba_array[:, :, :3]

            plt.close(fig)
            return rgb_array

        plt.close(fig)