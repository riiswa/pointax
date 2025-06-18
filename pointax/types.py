"""Data structures for PointMaze environments."""

import chex
from flax import struct
from gymnax.environments import environment


@struct.dataclass
class EnvState(environment.EnvState):
    """State of the PointMaze environment.

    Attributes:
        position: [x, y] position of the point ball
        velocity: [vx, vy] velocity of the point ball
        desired_goal: [x, y] position of the desired goal
        time: Current timestep
    """
    position: chex.Array
    velocity: chex.Array
    desired_goal: chex.Array
    time: int


@struct.dataclass
class EnvParams(environment.EnvParams):
    """Parameters for the PointMaze environment.

    Attributes:
        max_steps_in_episode: Maximum steps before truncation
        dt: Time step for physics simulation (10 Hz = 0.1)
        goal_threshold: Distance threshold for goal achievement
        max_velocity: Maximum velocity clipping value
        maze_size_scaling: Scaling factor for maze coordinates
        maze_height: Height of maze walls (for visualization)
        position_noise_range: Range of uniform noise added to positions
        maze_map: Discrete maze representation as numeric array
        continuing_task: Whether task continues after reaching goal
        reset_target: Whether to reset goal when reached in continuing task
        reward_type: 0 = sparse, 1 = dense (JAX-compatible integer)

        Location arrays for different cell types:
        empty_locations: Coordinates of empty cells (0)
        goal_locations: Coordinates of goal cells ('G' -> 2)
        reset_locations: Coordinates of reset cells ('R' -> 3)
        combined_locations: Coordinates of combined cells ('C' -> 4)

        Counts for each location type:
        num_empty: Number of empty cells
        num_goals: Number of goal cells
        num_resets: Number of reset cells
        num_combined: Number of combined cells

        Pre-computed maze boundaries for optimization:
        x_map_center: X coordinate of maze center
        y_map_center: Y coordinate of maze center
        map_length: Number of rows in maze
        map_width: Number of columns in maze
    """
    max_steps_in_episode: int = 1000
    dt: float = 0.1
    goal_threshold: float = 0.45
    max_velocity: float = 5.0
    maze_size_scaling: float = 1.0
    maze_height: float = 0.4
    position_noise_range: float = 0.25
    maze_map: chex.Array = None
    continuing_task: bool = False
    reset_target: bool = False
    reward_type: int = 0

    # Location arrays for different cell types
    empty_locations: chex.Array = None
    goal_locations: chex.Array = None
    reset_locations: chex.Array = None
    combined_locations: chex.Array = None

    # Counts for each location type
    num_empty: int = 0
    num_goals: int = 0
    num_resets: int = 0
    num_combined: int = 0

    # Pre-computed maze boundaries
    x_map_center: float = 0.0
    y_map_center: float = 0.0
    map_length: int = 0
    map_width: int = 0