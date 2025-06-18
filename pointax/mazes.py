"""Maze definitions and utilities for PointMaze environments."""

from typing import Dict, Union, List
import jax.numpy as jnp
import chex

# Standard maze layouts
MAZE_LAYOUTS = {
    "UMaze": [
        [1, 1, 1, 1, 1],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 0, 1],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1]
    ],

    "Open": [
        [1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1]
    ],

    "Medium": [
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 1, 1, 0, 0, 1],
        [1, 0, 0, 1, 0, 0, 0, 1],
        [1, 1, 0, 0, 0, 1, 1, 1],
        [1, 0, 0, 1, 0, 0, 0, 1],
        [1, 0, 1, 0, 0, 1, 0, 1],
        [1, 0, 0, 0, 1, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1]
    ],

    "Large": [
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
        [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
        [1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
        [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1],
        [1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1],
        [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    ],

    "Giant": [
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1],
        [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1],
        [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
        [1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1],
        [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1],
        [1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1],
        [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
        [1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1],
        [1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1],
        [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    ],

    # Diverse Goal variations
    "Open_Diverse_G": [
        [1, 1, 1, 1, 1, 1, 1],
        [1, 'R', 'G', 'G', 'G', 'G', 1],
        [1, 'G', 'G', 'G', 'G', 'G', 1],
        [1, 'G', 'G', 'G', 'G', 'G', 1],
        [1, 1, 1, 1, 1, 1, 1]
    ],

    "Medium_Diverse_G": [
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, 'R', 0, 1, 1, 0, 0, 1],
        [1, 0, 0, 1, 0, 0, 'G', 1],
        [1, 1, 0, 0, 0, 1, 1, 1],
        [1, 0, 0, 1, 0, 0, 0, 1],
        [1, 'G', 1, 0, 0, 1, 0, 1],
        [1, 0, 0, 0, 1, 'G', 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1]
    ],

    "Large_Diverse_G": [
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 'R', 0, 0, 0, 1, 'G', 0, 0, 0, 0, 1],
        [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        [1, 0, 0, 0, 0, 'G', 0, 1, 0, 0, 'G', 1],
        [1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
        [1, 0, 'G', 1, 0, 1, 0, 0, 0, 0, 0, 1],
        [1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1],
        [1, 0, 0, 1, 'G', 0, 'G', 1, 0, 'G', 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    ],

    # Diverse Goal-Reset variations
    "Open_Diverse_GR": [
        [1, 1, 1, 1, 1, 1, 1],
        [1, 'C', 'C', 'C', 'C', 'C', 1],
        [1, 'C', 'C', 'C', 'C', 'C', 1],
        [1, 'C', 'C', 'C', 'C', 'C', 1],
        [1, 1, 1, 1, 1, 1, 1]
    ],

    "Medium_Diverse_GR": [
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, 'C', 0, 1, 1, 0, 0, 1],
        [1, 0, 0, 1, 0, 0, 'C', 1],
        [1, 1, 0, 0, 0, 1, 1, 1],
        [1, 0, 0, 1, 0, 0, 0, 1],
        [1, 'C', 1, 0, 0, 1, 0, 1],
        [1, 0, 0, 0, 1, 'C', 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1]
    ],

    "Large_Diverse_GR": [
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 'C', 0, 0, 0, 1, 'C', 0, 0, 0, 0, 1],
        [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        [1, 0, 0, 0, 0, 'C', 0, 1, 0, 0, 'C', 1],
        [1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
        [1, 0, 'C', 1, 0, 1, 0, 0, 0, 0, 0, 1],
        [1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1],
        [1, 0, 0, 1, 'C', 0, 'C', 1, 0, 'C', 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    ]
}


def get_maze_layout(maze_id: str) -> List[List[Union[int, str]]]:
    """Get maze layout by ID.

    Args:
        maze_id: Maze identifier (e.g., 'UMaze', 'Open_Diverse_G')

    Returns:
        Maze layout as list of lists with integer walls (1),
        empty cells (0), and string markers ('G', 'R', 'C')

    Raises:
        KeyError: If maze_id is not found
    """
    if maze_id in MAZE_LAYOUTS:
        return MAZE_LAYOUTS[maze_id]
    else:
        # Default to UMaze if not found
        return MAZE_LAYOUTS["UMaze"]


def convert_maze_to_numeric(maze: List[List[Union[int, str]]]) -> chex.Array:
    """Convert maze with string symbols to numeric representation for JAX.

    Converts:
    - 0 -> 0 (empty)
    - 1 -> 1 (wall)
    - 'G' -> 2 (goal)
    - 'R' -> 3 (reset)
    - 'C' -> 4 (combined)

    Args:
        maze: Maze layout with mixed int/str cells

    Returns:
        JAX-compatible numeric array
    """
    symbol_map = {'G': 2, 'R': 3, 'C': 4}

    numeric_maze = []
    for row in maze:
        numeric_row = []
        for cell in row:
            if isinstance(cell, str):
                numeric_row.append(symbol_map[cell])
            else:
                numeric_row.append(cell)
        numeric_maze.append(numeric_row)

    return jnp.array(numeric_maze)


def compute_all_locations(maze_map: chex.Array) -> Dict:
    """Compute coordinates for all location types in the maze.

    Args:
        maze_map: Numeric maze array

    Returns:
        Dictionary containing location arrays and counts for each cell type
    """
    map_length, map_width = maze_map.shape
    x_map_center = map_width / 2 * 1.0
    y_map_center = map_length / 2 * 1.0

    # Create all possible coordinates
    i_coords, j_coords = jnp.meshgrid(
        jnp.arange(map_length), jnp.arange(map_width), indexing='ij'
    )
    x_coords = (j_coords + 0.5) * 1.0 - x_map_center
    y_coords = y_map_center - (i_coords + 0.5) * 1.0
    all_coords = jnp.stack([x_coords.flatten(), y_coords.flatten()], axis=1)

    # Create masks for each cell type
    flat_maze = maze_map.flatten()
    empty_mask = (flat_maze == 0)  # Empty cells
    goal_mask = (flat_maze == 2)  # 'G' cells
    reset_mask = (flat_maze == 3)  # 'R' cells
    combined_mask = (flat_maze == 4)  # 'C' cells

    # Get coordinates for each type
    max_cells = map_length * map_width

    # Empty locations
    empty_indices = jnp.where(empty_mask, size=max_cells, fill_value=0)[0]
    empty_coords = all_coords[empty_indices]
    num_empty = int(jnp.sum(empty_mask))

    # Goal locations
    goal_indices = jnp.where(goal_mask, size=max_cells, fill_value=0)[0]
    goal_coords = all_coords[goal_indices]
    num_goals = int(jnp.sum(goal_mask))

    # Reset locations
    reset_indices = jnp.where(reset_mask, size=max_cells, fill_value=0)[0]
    reset_coords = all_coords[reset_indices]
    num_resets = int(jnp.sum(reset_mask))

    # Combined locations
    combined_indices = jnp.where(combined_mask, size=max_cells, fill_value=0)[0]
    combined_coords = all_coords[combined_indices]
    num_combined = int(jnp.sum(combined_mask))

    # Ensure at least one valid location for each type
    if num_empty == 0:
        empty_coords = empty_coords.at[0].set(jnp.array([0.0, 0.0]))
        num_empty = 1

    return {
        'empty_locations': empty_coords,
        'goal_locations': goal_coords,
        'reset_locations': reset_coords,
        'combined_locations': combined_coords,
        'num_empty': num_empty,
        'num_goals': num_goals,
        'num_resets': num_resets,
        'num_combined': num_combined,
    }


def get_available_mazes() -> List[str]:
    """Get list of all available maze IDs.

    Returns:
        List of maze identifiers
    """
    return list(MAZE_LAYOUTS.keys())


def is_diverse_maze(maze_id: str) -> bool:
    """Check if maze is a diverse variation.

    Args:
        maze_id: Maze identifier

    Returns:
        True if maze contains diverse goals/resets
    """
    return 'Diverse' in maze_id