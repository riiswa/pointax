"""Fixed PointMaze environment with proper collision handling."""

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
    JAX implementation of Point Maze environment with fixed collision physics.

    Matches MuJoCo reference implementation parameters:
    - Robot radius: 0.1
    - Motor gear: 100
    - Friction: 0.5
    - Goal threshold: 0.45
    """

    def __init__(
            self,
            maze_id: str = "UMaze",
            reward_type: str = "sparse",
            continuing_task: bool = False,
            reset_target: bool = False
    ):
        super().__init__()
        self.maze_id = maze_id
        self.reward_type_str = reward_type
        self.continuing_task = continuing_task
        self.reset_target = reset_target

    @property
    def default_params(self) -> EnvParams:
        """Get default environment parameters matching MuJoCo implementation."""
        maze_layout = get_maze_layout(self.maze_id)
        maze_map = convert_maze_to_numeric(maze_layout)
        reward_type_int = 0 if self.reward_type_str == "sparse" else 1

        locations_data = compute_all_locations(maze_map)

        map_length, map_width = maze_map.shape
        x_map_center = map_width / 2 * 1.0
        y_map_center = map_length / 2 * 1.0

        return EnvParams(
            maze_map=maze_map,
            continuing_task=self.continuing_task,
            reset_target=self.reset_target,
            reward_type=reward_type_int,
            **locations_data,
            # Fixed parameters to match MuJoCo
            robot_radius=0.1,           # From XML: sphere size="0.1"
            goal_threshold=0.45,        # From gymnasium: success threshold
            motor_gear=100.0,           # From XML: gear="100"
            friction_coeff=0.5,         # From XML: friction=".5 .1 .1"
            dt=0.01,                    # From XML: timestep="0.01"
            mass=1.0,                   # Computed from density * volume
            max_velocity=5.0,           # Reasonable velocity limit
            maze_size_scaling=1.0,      # From gymnasium: maze_size_scaling=1
            # Pre-computed values
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
        """Step the environment with fixed collision handling."""

        # Apply motor gear scaling to match MuJoCo
        scaled_action = jnp.clip(action, -1.0, 1.0) * params.motor_gear

        # Physics integration
        old_position = state.position
        old_velocity = jnp.clip(state.velocity, -params.max_velocity, params.max_velocity)

        # Update velocity with forces
        acceleration = scaled_action / params.mass
        new_velocity = old_velocity + acceleration * params.dt
        new_velocity = jnp.clip(new_velocity, -params.max_velocity, params.max_velocity)

        # Compute intended new position
        intended_position = old_position + new_velocity * params.dt

        # Handle collisions with simplified ray-casting approach
        final_position, final_velocity = self._handle_collisions(
            old_position, intended_position, new_velocity, params
        )

        # Apply maze boundary constraints
        maze_bounds_x = params.x_map_center - params.robot_radius
        maze_bounds_y = params.y_map_center - params.robot_radius

        final_position = jnp.array([
            jnp.clip(final_position[0], -maze_bounds_x, maze_bounds_x),
            jnp.clip(final_position[1], -maze_bounds_y, maze_bounds_y)
        ])

        # Compute reward and goal achievement
        distance_to_goal = jnp.linalg.norm(final_position - state.desired_goal)
        goal_reached = distance_to_goal <= params.goal_threshold

        # Reward computation
        reward = lax.select(
            params.reward_type == 0,
            goal_reached.astype(jnp.float32),  # sparse
            -distance_to_goal  # dense
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

    def _handle_collisions(
            self,
            old_pos: chex.Array,
            intended_pos: chex.Array,
            velocity: chex.Array,
            params: EnvParams
    ) -> Tuple[chex.Array, chex.Array]:
        """Simplified collision handling using sphere-AABB intersection with margin."""

        # Add safety margin to prevent sticking
        effective_radius = params.robot_radius + 0.002

        def world_pos_to_cell(pos):
            """Convert world position to grid cell indices."""
            # Fixed coordinate transformation
            i = jnp.floor((params.y_map_center - pos[1]) / params.maze_size_scaling).astype(int)
            j = jnp.floor((pos[0] + params.x_map_center) / params.maze_size_scaling).astype(int)
            i = jnp.clip(i, 0, params.map_length - 1)
            j = jnp.clip(j, 0, params.map_width - 1)
            return i, j

        def cell_to_world_center(i, j):
            """Convert grid cell indices to world coordinates (cell center)."""
            # Fixed coordinate transformation
            x = (j + 0.5) * params.maze_size_scaling - params.x_map_center
            y = params.y_map_center - (i + 0.5) * params.maze_size_scaling
            return jnp.array([x, y])

        def sphere_aabb_collision(sphere_center, sphere_radius, aabb_center, aabb_half_size):
            """Check sphere-AABB collision and return corrected position."""
            # Find closest point on AABB to sphere center
            aabb_min = aabb_center - aabb_half_size
            aabb_max = aabb_center + aabb_half_size

            closest_point = jnp.array([
                jnp.clip(sphere_center[0], aabb_min[0], aabb_max[0]),
                jnp.clip(sphere_center[1], aabb_min[1], aabb_max[1])
            ])

            # Vector from closest point to sphere center
            diff = sphere_center - closest_point
            distance = jnp.linalg.norm(diff)

            # Check for collision
            is_colliding = distance < sphere_radius

            # Compute corrected position if colliding
            def correct_position():
                # Avoid division by zero
                safe_distance = jnp.maximum(distance, 1e-8)
                normal = diff / safe_distance
                # Push sphere out by penetration amount + small extra margin
                penetration = sphere_radius - distance + 1e-6  # Extra tiny margin to prevent edge cases
                return sphere_center + normal * penetration

            corrected_pos = lax.cond(
                is_colliding,
                correct_position,
                lambda: sphere_center
            )

            return corrected_pos, is_colliding

        # Get cells that might contain walls we could collide with
        current_cell = world_pos_to_cell(intended_pos)

        # Collect all collision corrections instead of applying them sequentially
        total_correction = jnp.zeros(2)
        collision_count = 0

        # Check 3x3 neighborhood around current cell
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                check_i = current_cell[0] + di
                check_j = current_cell[1] + dj

                # Skip if cell is out of bounds
                valid_cell = (
                        (check_i >= 0) & (check_i < params.map_length) &
                        (check_j >= 0) & (check_j < params.map_width)
                )

                def check_wall_collision():
                    is_wall = params.maze_map[check_i, check_j] == 1

                    def handle_wall():
                        wall_center = cell_to_world_center(check_i, check_j)
                        wall_half_size = jnp.array([
                            params.maze_size_scaling * 0.5,
                            params.maze_size_scaling * 0.5
                        ])

                        corrected_pos, collided = sphere_aabb_collision(
                            intended_pos,  # Always use intended_pos, not accumulated corrections
                            effective_radius,  # Use effective radius with margin
                            wall_center,
                            wall_half_size
                        )

                        correction = lax.select(
                            collided,
                            corrected_pos - intended_pos,
                            jnp.zeros(2)
                        )

                        return correction, collided.astype(jnp.int32)

                    return lax.cond(
                        is_wall,
                        handle_wall,
                        lambda: (jnp.zeros(2), 0)
                    )

                correction, collision_flag = lax.cond(
                    valid_cell,
                    check_wall_collision,
                    lambda: (jnp.zeros(2), 0)
                )

                total_correction += correction
                collision_count += collision_flag

        # Apply averaged correction to handle multiple simultaneous collisions better
        any_collision = collision_count > 0
        safe_collision_count = jnp.maximum(collision_count, 1)  # Avoid division by zero

        final_correction = lax.select(
            any_collision,
            total_correction / safe_collision_count.astype(jnp.float32),
            jnp.zeros(2)
        )

        corrected_position = intended_pos + final_correction

        # Update velocity based on actual movement (handles stopping/sliding)
        actual_movement = corrected_position - old_pos
        actual_velocity = actual_movement / params.dt

        # Apply friction if there was a collision
        final_velocity = lax.select(
            any_collision,
            actual_velocity * (1.0 - params.friction_coeff * params.dt),
            velocity
        )

        # Ensure velocity limits
        final_velocity = jnp.clip(final_velocity, -params.max_velocity, params.max_velocity)

        return corrected_position, final_velocity

    def reset_env(
            self,
            key: chex.PRNGKey,
            params: EnvParams,
            options: Optional[Dict] = None
    ) -> Tuple[chex.Array, EnvState]:
        """Reset environment to initial state."""
        key, pos_key, goal_key = jax.random.split(key, 3)

        goal_position = self._generate_goal_position(goal_key, params)
        goal_position = self._add_position_noise(goal_position, goal_key, params)

        reset_position = self._generate_reset_position(pos_key, params)
        reset_position = self._add_position_noise(reset_position, pos_key, params)

        state = EnvState(
            position=reset_position,
            velocity=jnp.zeros(2),
            desired_goal=goal_position,
            time=0
        )

        obs = jnp.concatenate([reset_position, jnp.zeros(2), goal_position])
        return obs, state

    def get_obs(self, state: EnvState, params: EnvParams, key=None) -> chex.Array:
        """Get observation from current state."""
        return jnp.concatenate([state.position, state.velocity, state.desired_goal])

    def is_terminal(self, state: EnvState, params: EnvParams) -> chex.Array:
        """Check if state is terminal."""
        done_steps = state.time >= params.max_steps_in_episode
        distance_to_goal = jnp.linalg.norm(state.position - state.desired_goal)
        done_goal = distance_to_goal <= params.goal_threshold

        return lax.select(
            params.continuing_task,
            done_steps,
            done_steps | done_goal
        )

    def _generate_goal_position(self, key: chex.PRNGKey, params: EnvParams) -> chex.Array:
        """Generate goal position from appropriate location set."""
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
        """Generate reset position from appropriate location set."""
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
        """Minimalist rendering: black walls, white space, green ball, red goal."""
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        import numpy as np

        map_length, map_width = params.maze_map.shape

        # Calculate figsize to match true maze proportions (smaller base size)
        aspect_ratio = map_width / map_length
        base_size = 2
        if aspect_ratio > 1:
            figsize = (base_size * aspect_ratio, base_size)
        else:
            figsize = (base_size, base_size / aspect_ratio)

        fig, ax = plt.subplots(figsize=figsize, dpi=150)
        ax.set_aspect('equal')
        ax.axis('off')

        maze_visual = (params.maze_map != 1).astype(float)
        extent = [-params.x_map_center,
                  params.x_map_center,
                  -params.y_map_center,
                  params.y_map_center]
        ax.imshow(maze_visual, cmap='gray', extent=extent, origin='upper',
                  vmin=0, vmax=1, interpolation='nearest')

        # Draw goal area threshold (light red circle)
        ax.add_patch(patches.Circle(state.desired_goal, params.goal_threshold,
                                    color='red', alpha=0.2, zorder=8))

        # Draw goal (red circle with true size)
        goal_size = params.robot_radius * 0.8  # Slightly smaller than robot for clarity
        ax.add_patch(patches.Circle(state.desired_goal, goal_size,
                                    color='red', alpha=1.0, zorder=10))

        # Draw robot (green circle with true size)
        ax.add_patch(patches.Circle(state.position, params.robot_radius,
                                    color='green', alpha=1.0, zorder=11))

        # Draw velocity arrow (better scaling, thinner)
        velocity = state.velocity
        velocity_magnitude = np.linalg.norm(velocity)

        if velocity_magnitude > 0.1:  # Only show arrow for significant movement
            # Better scaling: linear with velocity magnitude
            arrow_length = velocity_magnitude * 0.1  # More responsive scaling

            # Arrow starts from robot edge, not center
            velocity_direction = velocity / velocity_magnitude
            arrow_start = state.position + velocity_direction * params.robot_radius * 1.1
            arrow_end = arrow_start + velocity_direction * arrow_length

            # Thinner arrow with consistent width
            ax.annotate('',
                        xy=arrow_end,
                        xytext=arrow_start,
                        arrowprops=dict(
                            arrowstyle='->',
                            color='green',
                            lw=1.,  # Thinner arrow
                            alpha=0.8,
                            shrinkA=0,
                            shrinkB=0
                        ),
                        zorder=12)

        # # Set axis limits to show full maze with small margin
        # margin = params.robot_radius * 2
        # ax.set_xlim(-params.x_map_center - margin, params.x_map_center + margin)
        # ax.set_ylim(-params.y_map_center - margin, params.y_map_center + margin)

        plt.tight_layout(pad=1)

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
