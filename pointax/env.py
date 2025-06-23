"""Main PointMaze environment implementation with improved physics."""

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
    JAX implementation of Point Maze environment with diverse goal support and proper collision physics.

    A 2-DoF point mass navigates through a maze to reach goal locations.
    Supports various maze layouts including diverse goal/reset configurations.

    Key Features:
    - JAX-native implementation for fast vectorization
    - Support for diverse goal mazes with special cell markers
    - Realistic collision detection with line intersection math
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
        # Clip actions and scale them appropriately
        action = jnp.clip(action, -1.0, 1.0) / 10.0  # Scale similar to original

        # Get current position and velocity
        old_position = state.position
        old_velocity = jnp.clip(state.velocity, -params.max_velocity, params.max_velocity)

        # Update velocity with action forces
        new_velocity = jnp.clip(
            old_velocity + action,
            -params.max_velocity,
            params.max_velocity
        )

        # Compute intended new position
        intended_position = old_position + new_velocity * params.dt

        # Handle collisions with proper physics
        final_position, final_velocity = self._handle_wall_collisions(
            intended_position, new_velocity, old_position, params
        )

        # Apply boundary constraints (maze edges)
        x_min = -params.x_map_center
        x_max = params.x_map_center
        y_min = -params.y_map_center
        y_max = params.y_map_center

        final_position = jnp.array([
            jnp.clip(final_position[0], x_min, x_max),
            jnp.clip(final_position[1], y_min, y_max)
        ])

        # Compute reward and goal achievement
        distance_to_goal = jnp.linalg.norm(final_position - state.desired_goal)
        goal_reached = distance_to_goal <= params.goal_threshold

        # Reward computation
        reward = lax.select(
            params.reward_type == 0,
            goal_reached.astype(jnp.float32),  # sparse
            -distance_to_goal  # dense (negative distance like original)
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
            position=final_position,
            velocity=final_velocity,
            desired_goal=new_goal,
            time=state.time + 1
        )

        # Termination logic
        done_steps = new_state.time >= params.max_steps_in_episode
        done_goal = (~params.continuing_task) & goal_reached
        done = done_steps | done_goal

        # Create observation
        obs = jnp.concatenate([final_position, final_velocity, new_goal])

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
        """Handle wall collisions accounting for sphere radius.

        Coordinate System:
        - Cell (0,0) is at world position (-x_map_center, y_map_center)
        - Cell (i,j) is at world position (-x_map_center + j*maze_size_scaling, y_map_center - i*maze_size_scaling)
        - Maze cells are 1x1 units in size (maze_size_scaling = 1.0)

        Checks distance to wall boundaries, not just discrete cell occupancy.
        """
        sphere_radius = 0.1  # Match MuJoCo sphere size
        x_new, y_new = new_pos

        # Convert position to cell coordinates (continuous)
        def pos_to_cell_continuous(x, y):
            i = (params.y_map_center - y) / params.maze_size_scaling
            j = (x + params.x_map_center) / params.maze_size_scaling
            return i, j

        # Convert to discrete cell indices
        def pos_to_cell_discrete(x, y):
            i = jnp.floor((params.y_map_center - y) / params.maze_size_scaling).astype(int)
            j = jnp.floor((x + params.x_map_center) / params.maze_size_scaling).astype(int)
            i = jnp.clip(i, 0, params.map_length - 1)
            j = jnp.clip(j, 0, params.map_width - 1)
            return i, j

        # Check if sphere center is too close to any wall
        i_center, j_center = pos_to_cell_continuous(x_new, y_new)

        # Get the cell indices for checking surrounding cells
        i_discrete, j_discrete = pos_to_cell_discrete(x_new, y_new)

        # Check 3x3 grid around current position for nearby walls
        collision_detected = False

        # Use vectorized approach instead of loops with conditionals
        di_range = jnp.arange(-1, 2)
        dj_range = jnp.arange(-1, 2)

        # Create meshgrid for all combinations
        di_grid, dj_grid = jnp.meshgrid(di_range, dj_range, indexing='ij')
        di_flat = di_grid.flatten()
        dj_flat = dj_grid.flatten()

        # Calculate all check positions
        check_i_all = jnp.clip(i_discrete + di_flat, 0, params.map_length - 1)
        check_j_all = jnp.clip(j_discrete + dj_flat, 0, params.map_width - 1)

        # Check which cells are walls
        is_wall_all = params.maze_map[check_i_all, check_j_all] == 1

        # For each wall cell, calculate distance
        def calculate_wall_distance(idx):
            check_i = check_i_all[idx]
            check_j = check_j_all[idx]
            is_wall = is_wall_all[idx]

            # Wall cell boundaries
            wall_left = check_j.astype(float)
            wall_right = check_j.astype(float) + 1
            wall_bottom = check_i.astype(float)
            wall_top = check_i.astype(float) + 1

            # Find closest point on wall cell boundary to sphere center
            closest_j = jnp.clip(j_center, wall_left, wall_right)
            closest_i = jnp.clip(i_center, wall_bottom, wall_top)

            # Distance from sphere center to closest wall boundary point
            dist_to_wall = jnp.sqrt((j_center - closest_j)**2 + (i_center - closest_i)**2)

            # Convert back to world units
            dist_to_wall_world = dist_to_wall * params.maze_size_scaling

            # Check if sphere would penetrate this wall
            penetration = sphere_radius - dist_to_wall_world
            has_penetration = is_wall & (penetration > 0)

            return has_penetration

        # Check all surrounding cells
        penetration_checks = jnp.array([calculate_wall_distance(i) for i in range(len(check_i_all))])
        collision_detected = jnp.any(penetration_checks)

        # If collision detected, resolve it
        final_pos = lax.select(
            collision_detected,
            self._resolve_wall_collision_with_distance(new_pos, old_pos, velocity, params),
            new_pos
        )

        # Update velocity - zero if collision occurred
        final_vel = lax.select(
            collision_detected,
            self._resolve_collision_velocity(new_pos, old_pos, velocity, params),
            velocity
        )

        return final_pos, final_vel

    def _resolve_wall_collision_with_distance(
            self,
            new_pos: chex.Array,
            old_pos: chex.Array,
            velocity: chex.Array,
            params: EnvParams
    ) -> chex.Array:
        """Resolve wall collision by pushing sphere away from wall boundaries."""
        sphere_radius = 0.1
        x_new, y_new = new_pos

        # Convert position to cell coordinates
        def pos_to_cell_continuous(x, y):
            i = (params.y_map_center - y) / params.maze_size_scaling
            j = (x + params.x_map_center) / params.maze_size_scaling
            return i, j

        def pos_to_cell_discrete(x, y):
            i = jnp.floor((params.y_map_center - y) / params.maze_size_scaling).astype(int)
            j = jnp.floor((x + params.x_map_center) / params.maze_size_scaling).astype(int)
            i = jnp.clip(i, 0, params.map_length - 1)
            j = jnp.clip(j, 0, params.map_width - 1)
            return i, j

        i_center, j_center = pos_to_cell_continuous(x_new, y_new)
        i_discrete, j_discrete = pos_to_cell_discrete(x_new, y_new)

        # Vectorized approach for finding closest wall
        di_range = jnp.arange(-1, 2)
        dj_range = jnp.arange(-1, 2)
        di_grid, dj_grid = jnp.meshgrid(di_range, dj_range, indexing='ij')
        di_flat = di_grid.flatten()
        dj_flat = dj_grid.flatten()

        # Calculate all check positions
        check_i_all = jnp.clip(i_discrete + di_flat, 0, params.map_length - 1)
        check_j_all = jnp.clip(j_discrete + dj_flat, 0, params.map_width - 1)

        # Check which cells are walls
        is_wall_all = params.maze_map[check_i_all, check_j_all] == 1

        # Calculate penetration for each potential wall
        def calculate_penetration_data(idx):
            check_i = check_i_all[idx]
            check_j = check_j_all[idx]
            is_wall = is_wall_all[idx]

            # Wall cell boundaries
            wall_left = check_j.astype(float)
            wall_right = check_j.astype(float) + 1
            wall_bottom = check_i.astype(float)
            wall_top = check_i.astype(float) + 1

            # Find closest point on wall boundary
            closest_j = jnp.clip(j_center, wall_left, wall_right)
            closest_i = jnp.clip(i_center, wall_bottom, wall_top)

            # Distance and direction to wall
            wall_to_center_j = j_center - closest_j
            wall_to_center_i = i_center - closest_i
            dist_to_wall = jnp.sqrt(wall_to_center_j**2 + wall_to_center_i**2)

            # How much we need to push away
            required_distance = sphere_radius / params.maze_size_scaling
            penetration = required_distance - dist_to_wall

            # Only consider this wall if it's actually penetrated
            valid_penetration = is_wall & (penetration > 0)

            # Direction vector (normalized)
            direction_magnitude = jnp.maximum(dist_to_wall, 1e-6)
            push_dir_j = wall_to_center_j / direction_magnitude
            push_dir_i = wall_to_center_i / direction_magnitude

            return jnp.where(
                valid_penetration,
                jnp.array([penetration, push_dir_j, push_dir_i]),
                jnp.array([0.0, 0.0, 0.0])  # No penetration
            )

        # Get penetration data for all surrounding cells
        penetration_data = jnp.array([calculate_penetration_data(i) for i in range(len(check_i_all))])

        # Find the maximum penetration
        penetrations = penetration_data[:, 0]
        max_penetration_idx = jnp.argmax(penetrations)
        max_penetration = penetrations[max_penetration_idx]

        # Get the corresponding push direction
        push_direction_j = penetration_data[max_penetration_idx, 1]
        push_direction_i = penetration_data[max_penetration_idx, 2]

        # Apply push in cell coordinates, then convert back to world coordinates
        pushed_j = j_center + push_direction_j * max_penetration
        pushed_i = i_center + push_direction_i * max_penetration

        # Convert back to world coordinates
        corrected_x = pushed_j * params.maze_size_scaling - params.x_map_center
        corrected_y = params.y_map_center - pushed_i * params.maze_size_scaling

        # Ensure corrected position stays within maze bounds (safety check)
        world_width = params.map_width * params.maze_size_scaling
        world_height = params.map_length * params.maze_size_scaling
        corrected_x = jnp.clip(corrected_x, -params.x_map_center + sphere_radius,
                              -params.x_map_center + world_width - sphere_radius)
        corrected_y = jnp.clip(corrected_y, params.y_map_center - world_height + sphere_radius,
                              params.y_map_center - sphere_radius)

        # If no penetration found, return original position
        return jnp.where(
            max_penetration > 0,
            jnp.array([corrected_x, corrected_y]),
            new_pos
        )

    def _resolve_collision_velocity(
            self,
            new_pos: chex.Array,
            old_pos: chex.Array,
            velocity: chex.Array,
            params: EnvParams
    ) -> chex.Array:
        """Resolve velocity after collision.

        For simplicity, we zero the velocity component in the direction of collision.
        A more sophisticated implementation would reflect velocity off the wall.
        """
        x_new, y_new = new_pos
        x_old, y_old = old_pos
        vx, vy = velocity

        # Determine which direction hit the wall
        dx = x_new - x_old
        dy = y_new - y_old

        # Zero velocity components that would cause further wall penetration
        new_vx = lax.select(jnp.abs(dx) > jnp.abs(dy), 0.0, vx)
        new_vy = lax.select(jnp.abs(dy) > jnp.abs(dx), 0.0, vy)

        return jnp.array([new_vx, new_vy])

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
        import numpy as np

        map_length, map_width = params.maze_map.shape

        # Calculate figsize to match maze aspect ratio
        aspect_ratio = map_width / map_length
        base_size = 4
        if aspect_ratio > 1:
            figsize = (base_size, base_size / aspect_ratio)
        else:
            figsize = (base_size * aspect_ratio, base_size)

        fig, ax = plt.subplots(figsize=figsize, dpi=200)
        # ax.set_xlim(-params.x_map_center - 0.5, params.x_map_center + 0.5)
        # ax.set_ylim(-params.y_map_center - 0.5, params.y_map_center + 0.5)
        ax.set_aspect('equal')
        ax.axis('off')

        # Fix: Only walls (value=1) are black, everything else is white
        maze_visual = (params.maze_map != 1).astype(float)  # walls=0 (black), non-walls=1 (white)
        extent = [-params.x_map_center,
                  map_width - params.x_map_center,
                  params.y_map_center - map_length,
                  params.y_map_center]
        ax.imshow(maze_visual, cmap='gray', extent=extent, origin='upper',
                  vmin=0, vmax=1, interpolation='nearest')

        # Draw goal and agent
        ax.add_patch(patches.Circle(state.desired_goal, 0.2, color='red', alpha=0.8))
        ax.add_patch(patches.Circle(state.position, 0.1, color='green', alpha=0.9))

        if mode == "human":
            plt.show()
        elif mode == "rgb_array":
            fig.canvas.draw()
            buf = fig.canvas.buffer_rgba()
            w, h = fig.canvas.get_width_height()
            rgba_array = np.frombuffer(buf, dtype=np.uint8).reshape((h, w, 4))
            rgb_array = rgba_array[:, :, :3]
            plt.close(fig)
            return rgb_array

        plt.close(fig)