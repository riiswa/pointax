"""JAX implementation of PointMaze environment with collision physics."""

from typing import Tuple, Optional, Dict, Any, List, Union
import jax
import jax.numpy as jnp
from jax import lax
import chex
from gymnax.environments import environment, spaces

from pointax.types import EnvState, EnvParams
from pointax.mazes import get_maze_layout, convert_maze_to_numeric, compute_all_locations


class PointMazeEnv(environment.Environment[EnvState, EnvParams]):
    """JAX PointMaze environment matching MuJoCo reference implementation.

    Features:
    - Fixed collision physics with sphere-AABB intersection
    - Configurable reward types (sparse/dense)
    - Support for diverse goal/reset configurations
    - Continuing tasks with goal reset capability
    - Full JAX compatibility (JIT, vmap, grad)
    """

    def __init__(
            self,
            maze_id: str = "UMaze",
            reward_type: str = "sparse",
            continuing_task: bool = False,
            reset_target: bool = False,
            custom_maze_layout: Optional[List[List[Union[int, str, bool]]]] = None
    ):
        super().__init__()
        self.maze_id = maze_id
        self.reward_type_str = reward_type
        self.continuing_task = continuing_task
        self.reset_target = reset_target
        self.custom_maze_layout = custom_maze_layout

    @property
    def default_params(self) -> EnvParams:
        """Default environment parameters matching MuJoCo implementation."""
        # Use custom layout if provided, otherwise get predefined layout
        if self.custom_maze_layout is not None:
            maze_layout = self.custom_maze_layout
        else:
            maze_layout = get_maze_layout(self.maze_id)

        maze_map = convert_maze_to_numeric(maze_layout)
        reward_type_int = 0 if self.reward_type_str == "sparse" else 1
        locations_data = compute_all_locations(maze_map)

        map_length, map_width = maze_map.shape
        x_center = map_width / 2 * 1.0
        y_center = map_length / 2 * 1.0

        return EnvParams(
            maze_map=maze_map,
            continuing_task=self.continuing_task,
            reset_target=self.reset_target,
            reward_type=reward_type_int,
            **locations_data,
            # MuJoCo-matched physics constants
            robot_radius=0.1,
            goal_threshold=0.45,
            motor_gear=100.0,
            friction_coeff=0.5,
            dt=0.01,
            mass=1.0,
            max_velocity=5.0,
            maze_size_scaling=1.0,
            x_map_center=x_center,
            y_map_center=y_center,
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
        """Execute one environment step with physics and collision handling."""
        # Apply motor scaling and integrate physics
        scaled_action = jnp.clip(action, -1.0, 1.0) * params.motor_gear
        old_pos, old_vel = state.position, state.velocity

        # Velocity integration with force application
        old_vel = jnp.clip(old_vel, -params.max_velocity, params.max_velocity)
        acceleration = scaled_action / params.mass
        new_vel = old_vel + acceleration * params.dt
        new_vel = jnp.clip(new_vel, -params.max_velocity, params.max_velocity)

        # Position integration with collision handling
        intended_pos = old_pos + new_vel * params.dt
        final_pos, final_vel = self._resolve_collisions(
            old_pos, intended_pos, new_vel, params
        )

        # Enforce maze boundaries
        final_pos = self._apply_boundary_constraints(final_pos, params)

        # Goal achievement and reward computation
        goal_dist = jnp.linalg.norm(final_pos - state.desired_goal)
        goal_reached = goal_dist <= params.goal_threshold

        reward = lax.select(
            params.reward_type == 0,
            goal_reached.astype(jnp.float32),  # sparse
            -goal_dist  # dense
        )

        # Handle goal reset for continuing tasks
        should_reset_goal = (params.continuing_task &
                           params.reset_target &
                           goal_reached)
        new_goal = lax.select(
            should_reset_goal,
            self._sample_goal_position(key, params),
            state.desired_goal
        )

        # Update state
        new_state = state.replace(
            position=final_pos,
            velocity=final_vel,
            desired_goal=new_goal,
            time=state.time + 1
        )

        # Termination conditions
        time_up = new_state.time >= params.max_steps_in_episode
        goal_done = (~params.continuing_task) & goal_reached
        done = time_up | goal_done

        # Construct output
        obs = jnp.concatenate([final_pos, final_vel, new_goal])
        info = {
            "is_success": goal_reached,
            "discount": self.discount(new_state, params)
        }

        return (
            lax.stop_gradient(obs),
            lax.stop_gradient(new_state),
            reward,
            jnp.astype(done, bool),
            info
        )

    def reset_env(
            self,
            key: chex.PRNGKey,
            params: EnvParams,
            options: Optional[Dict] = None
    ) -> Tuple[chex.Array, EnvState]:
        """Reset environment to initial conditions."""
        key1, key2, key3 = jax.random.split(key, 3)

        goal_pos = self._sample_goal_position(key1, params)
        goal_pos = self._add_noise(goal_pos, key2, params)

        reset_pos = self._sample_reset_position(key3, params)
        reset_pos = self._add_noise(reset_pos, key1, params)

        state = EnvState(
            position=reset_pos,
            velocity=jnp.zeros(2),
            desired_goal=goal_pos,
            time=0
        )

        obs = jnp.concatenate([reset_pos, jnp.zeros(2), goal_pos])
        return obs, state

    def _resolve_collisions(
            self,
            old_pos: chex.Array,
            intended_pos: chex.Array,
            velocity: chex.Array,
            params: EnvParams
    ) -> Tuple[chex.Array, chex.Array]:
        """Handle wall collisions using sphere-AABB intersection."""
        effective_radius = params.robot_radius + 0.002  # Anti-sticking margin

        # Get potentially colliding cells in 3x3 neighborhood
        current_cell = self._world_to_cell(intended_pos, params)

        # Accumulate collision corrections
        total_correction = jnp.zeros(2)
        collision_count = 0

        # Check surrounding cells for walls
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                cell_i = current_cell[0] + di
                cell_j = current_cell[1] + dj

                # Bounds check
                valid_cell = (
                    (cell_i >= 0) & (cell_i < params.map_length) &
                    (cell_j >= 0) & (cell_j < params.map_width)
                )

                def check_wall():
                    is_wall = params.maze_map[cell_i, cell_j] == 1

                    def resolve_wall_collision():
                        wall_center = self._cell_to_world(cell_i, cell_j, params)
                        wall_half_size = jnp.full(2, params.maze_size_scaling * 0.5)

                        corrected_pos, collided = self._sphere_aabb_collision(
                            intended_pos, effective_radius, wall_center, wall_half_size
                        )

                        correction = lax.select(
                            collided,
                            corrected_pos - intended_pos,
                            jnp.zeros(2)
                        )
                        return correction, collided.astype(jnp.int32)

                    return lax.cond(
                        is_wall,
                        resolve_wall_collision,
                        lambda: (jnp.zeros(2), 0)
                    )

                correction, collision_flag = lax.cond(
                    valid_cell,
                    check_wall,
                    lambda: (jnp.zeros(2), 0)
                )

                total_correction += correction
                collision_count += collision_flag

        # Apply averaged correction for multiple simultaneous collisions
        any_collision = collision_count > 0
        safe_count = jnp.maximum(collision_count, 1)

        final_correction = lax.select(
            any_collision,
            total_correction / safe_count.astype(jnp.float32),
            jnp.zeros(2)
        )

        corrected_pos = intended_pos + final_correction

        # Update velocity based on actual movement with friction
        actual_movement = corrected_pos - old_pos
        actual_velocity = actual_movement / params.dt

        final_velocity = lax.select(
            any_collision,
            actual_velocity * (1.0 - params.friction_coeff * params.dt),
            velocity
        )

        final_velocity = jnp.clip(final_velocity, -params.max_velocity, params.max_velocity)

        return corrected_pos, final_velocity

    def _sphere_aabb_collision(
            self,
            sphere_center: chex.Array,
            sphere_radius: float,
            aabb_center: chex.Array,
            aabb_half_size: chex.Array
    ) -> Tuple[chex.Array, bool]:
        """Detect and resolve sphere-AABB collision."""
        aabb_min = aabb_center - aabb_half_size
        aabb_max = aabb_center + aabb_half_size

        # Find closest point on AABB to sphere center
        closest_point = jnp.array([
            jnp.clip(sphere_center[0], aabb_min[0], aabb_max[0]),
            jnp.clip(sphere_center[1], aabb_min[1], aabb_max[1])
        ])

        # Check collision and compute correction
        diff = sphere_center - closest_point
        distance = jnp.linalg.norm(diff)
        is_colliding = distance < sphere_radius

        def compute_correction():
            safe_distance = jnp.maximum(distance, 1e-8)
            normal = diff / safe_distance
            penetration = sphere_radius - distance + 1e-6
            return sphere_center + normal * penetration

        corrected_pos = lax.cond(
            is_colliding,
            compute_correction,
            lambda: sphere_center
        )

        return corrected_pos, is_colliding

    def _world_to_cell(self, pos: chex.Array, params: EnvParams) -> chex.Array:
        """Convert world coordinates to grid cell indices."""
        i = jnp.floor((params.y_map_center - pos[1]) / params.maze_size_scaling).astype(int)
        j = jnp.floor((pos[0] + params.x_map_center) / params.maze_size_scaling).astype(int)
        i = jnp.clip(i, 0, params.map_length - 1)
        j = jnp.clip(j, 0, params.map_width - 1)
        return jnp.array([i, j])

    def _cell_to_world(self, i: int, j: int, params: EnvParams) -> chex.Array:
        """Convert grid cell indices to world coordinates (cell center)."""
        x = (j + 0.5) * params.maze_size_scaling - params.x_map_center
        y = params.y_map_center - (i + 0.5) * params.maze_size_scaling
        return jnp.array([x, y])

    def _apply_boundary_constraints(self, position: chex.Array, params: EnvParams) -> chex.Array:
        """Ensure position stays within maze boundaries."""
        bounds_x = params.x_map_center - params.robot_radius
        bounds_y = params.y_map_center - params.robot_radius

        return jnp.array([
            jnp.clip(position[0], -bounds_x, bounds_x),
            jnp.clip(position[1], -bounds_y, bounds_y)
        ])

    def _sample_goal_position(self, key: chex.PRNGKey, params: EnvParams) -> chex.Array:
        """Sample goal position from appropriate location set."""
        has_goals = params.num_goals > 0
        has_combined = params.num_combined > 0

        def use_goals():
            idx = jax.random.randint(key, (), 0, params.num_goals)
            return params.goal_locations[idx]

        def use_combined():
            idx = jax.random.randint(key, (), 0, params.num_combined)
            return params.combined_locations[idx]

        def use_empty():
            idx = jax.random.randint(key, (), 0, params.num_empty)
            return params.empty_locations[idx]

        return lax.cond(
            has_goals,
            use_goals,
            lambda: lax.cond(has_combined, use_combined, use_empty)
        )

    def _sample_reset_position(self, key: chex.PRNGKey, params: EnvParams) -> chex.Array:
        """Sample reset position from appropriate location set."""
        has_resets = params.num_resets > 0
        has_combined = params.num_combined > 0

        def use_resets():
            idx = jax.random.randint(key, (), 0, params.num_resets)
            return params.reset_locations[idx]

        def use_combined():
            idx = jax.random.randint(key, (), 0, params.num_combined)
            return params.combined_locations[idx]

        def use_empty():
            idx = jax.random.randint(key, (), 0, params.num_empty)
            return params.empty_locations[idx]

        return lax.cond(
            has_resets,
            use_resets,
            lambda: lax.cond(has_combined, use_combined, use_empty)
        )

    def _add_noise(self, position: chex.Array, key: chex.PRNGKey, params: EnvParams) -> chex.Array:
        """Add uniform noise to position coordinates."""
        noise_range = params.position_noise_range * params.maze_size_scaling
        noise = jax.random.uniform(key, (2,), minval=-noise_range, maxval=noise_range)
        return position + noise

    def get_obs(self, state: EnvState, params: EnvParams, key=None) -> chex.Array:
        """Get observation from current state."""
        return jnp.concatenate([state.position, state.velocity, state.desired_goal])

    def is_terminal(self, state: EnvState, params: EnvParams) -> chex.Array:
        """Check if state is terminal."""
        time_up = state.time >= params.max_steps_in_episode
        goal_dist = jnp.linalg.norm(state.position - state.desired_goal)
        goal_reached = goal_dist <= params.goal_threshold

        return lax.select(
            params.continuing_task,
            time_up,
            time_up | goal_reached
        )

    # Space and metadata properties
    @property
    def name(self) -> str:
        """Environment name."""
        suffix = "Dense" if self.reward_type_str == "dense" else ""
        return f"pointax/PointMaze_{self.maze_id}{suffix}"

    @property
    def num_actions(self) -> int:
        """Number of actions."""
        return 2

    def action_space(self, params: EnvParams = None) -> spaces.Box:
        """Continuous action space: forces in x,y directions."""
        return spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=jnp.float32)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Observation space: [position, velocity, goal]."""
        return spaces.Box(low=-jnp.inf, high=jnp.inf, shape=(6,), dtype=jnp.float32)

    def state_space(self, params: EnvParams) -> spaces.Dict:
        """Environment state space."""
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
        """Render environment with matplotlib."""
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        import numpy as np

        map_length, map_width = params.maze_map.shape

        # Calculate proportional figure size
        aspect_ratio = map_width / map_length
        base_size = 2
        figsize = (base_size * aspect_ratio, base_size) if aspect_ratio > 1 else (base_size, base_size / aspect_ratio)

        fig, ax = plt.subplots(figsize=figsize, dpi=150)
        ax.set_aspect('equal')
        ax.axis('off')

        # Render maze (black walls, white space)
        maze_visual = (params.maze_map != 1).astype(float)
        extent = [-params.x_map_center, params.x_map_center,
                  -params.y_map_center, params.y_map_center]
        ax.imshow(maze_visual, cmap='gray', extent=extent, origin='upper',
                  vmin=0, vmax=1, interpolation='nearest')

        # Draw goal threshold area (light red)
        ax.add_patch(patches.Circle(
            state.desired_goal, params.goal_threshold,
            color='red', alpha=0.2, zorder=8
        ))

        # Draw goal (red) and robot (green)
        goal_size = params.robot_radius * 0.8
        ax.add_patch(patches.Circle(
            state.desired_goal, goal_size,
            color='red', alpha=1.0, zorder=10
        ))
        ax.add_patch(patches.Circle(
            state.position, params.robot_radius,
            color='green', alpha=1.0, zorder=11
        ))

        # Draw velocity arrow if moving
        velocity_mag = np.linalg.norm(state.velocity)
        if velocity_mag > 0.1:
            velocity_dir = state.velocity / velocity_mag
            arrow_start = state.position + velocity_dir * params.robot_radius * 1.1
            arrow_end = arrow_start + velocity_dir * velocity_mag * 0.1

            ax.annotate('', xy=arrow_end, xytext=arrow_start,
                       arrowprops=dict(arrowstyle='->', color='green',
                                     lw=1.0, alpha=0.8), zorder=12)

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