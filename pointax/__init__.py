"""Pointax: JAX PointMaze Environment Library.

A high-performance JAX implementation of the PointMaze environment from
Gymnasium Robotics, featuring diverse goal configurations and optimized
collision detection.

Example usage:
    >>> import pointax
    >>> env = pointax.make_open_diverse_g(reward_type="dense")
    >>> print(pointax.list_environments())
"""

from pointax.env import PointMazeEnv
from pointax.types import EnvState, EnvParams
from pointax.mazes import get_available_mazes, is_diverse_maze

__version__ = "0.1.0"
__author__ = "riiswa"

# Export main classes
__all__ = [
    "PointMazeEnv",
    "EnvState",
    "EnvParams",
    # Standard environments
    "make_umaze",
    "make_open",
    "make_medium",
    "make_large",
    "make_giant",
    # Diverse goal environments
    "make_open_diverse_g",
    "make_medium_diverse_g",
    "make_large_diverse_g",
    # Diverse goal-reset environments
    "make_open_diverse_gr",
    "make_medium_diverse_gr",
    "make_large_diverse_gr",
    # Utility functions
    "list_environments",
    "get_environment_info",
]


# Standard maze environments
def make_umaze(reward_type: str = "sparse", **kwargs) -> PointMazeEnv:
    """Create PointMaze_UMaze environment.

    Args:
        reward_type: 'sparse' or 'dense' reward function
        **kwargs: Additional environment parameters

    Returns:
        PointMazeEnv instance
    """
    return PointMazeEnv(maze_id="UMaze", reward_type=reward_type, **kwargs)


def make_open(reward_type: str = "sparse", **kwargs) -> PointMazeEnv:
    """Create PointMaze_Open environment.

    Args:
        reward_type: 'sparse' or 'dense' reward function
        **kwargs: Additional environment parameters

    Returns:
        PointMazeEnv instance
    """
    return PointMazeEnv(maze_id="Open", reward_type=reward_type, **kwargs)


def make_medium(reward_type: str = "sparse", **kwargs) -> PointMazeEnv:
    """Create PointMaze_Medium environment.

    Args:
        reward_type: 'sparse' or 'dense' reward function
        **kwargs: Additional environment parameters

    Returns:
        PointMazeEnv instance
    """
    return PointMazeEnv(maze_id="Medium", reward_type=reward_type, **kwargs)


def make_large(reward_type: str = "sparse", **kwargs) -> PointMazeEnv:
    """Create PointMaze_Large environment.

    Args:
        reward_type: 'sparse' or 'dense' reward function
        **kwargs: Additional environment parameters

    Returns:
        PointMazeEnv instance
    """
    return PointMazeEnv(maze_id="Large", reward_type=reward_type, **kwargs)


def make_giant(reward_type: str = "sparse", **kwargs) -> PointMazeEnv:
    """Create PointMaze_Giant environment.

    Args:
        reward_type: 'sparse' or 'dense' reward function
        **kwargs: Additional environment parameters

    Returns:
        PointMazeEnv instance
    """
    return PointMazeEnv(maze_id="Giant", reward_type=reward_type, **kwargs)


# Diverse goal environments
def make_open_diverse_g(reward_type: str = "sparse", **kwargs) -> PointMazeEnv:
    """Create PointMaze_Open_Diverse_G environment.

    Features multiple goal locations with single reset location.

    Args:
        reward_type: 'sparse' or 'dense' reward function
        **kwargs: Additional environment parameters

    Returns:
        PointMazeEnv instance
    """
    return PointMazeEnv(maze_id="Open_Diverse_G", reward_type=reward_type, **kwargs)


def make_medium_diverse_g(reward_type: str = "sparse", **kwargs) -> PointMazeEnv:
    """Create PointMaze_Medium_Diverse_G environment.

    Features multiple goal locations with single reset location.

    Args:
        reward_type: 'sparse' or 'dense' reward function
        **kwargs: Additional environment parameters

    Returns:
        PointMazeEnv instance
    """
    return PointMazeEnv(maze_id="Medium_Diverse_G", reward_type=reward_type, **kwargs)


def make_large_diverse_g(reward_type: str = "sparse", **kwargs) -> PointMazeEnv:
    """Create PointMaze_Large_Diverse_G environment.

    Features multiple goal locations with single reset location.

    Args:
        reward_type: 'sparse' or 'dense' reward function
        **kwargs: Additional environment parameters

    Returns:
        PointMazeEnv instance
    """
    return PointMazeEnv(maze_id="Large_Diverse_G", reward_type=reward_type, **kwargs)


# Diverse goal-reset environments
def make_open_diverse_gr(reward_type: str = "sparse", **kwargs) -> PointMazeEnv:
    """Create PointMaze_Open_Diverse_GR environment.

    Features combined goal/reset locations for maximum diversity.

    Args:
        reward_type: 'sparse' or 'dense' reward function
        **kwargs: Additional environment parameters

    Returns:
        PointMazeEnv instance
    """
    return PointMazeEnv(maze_id="Open_Diverse_GR", reward_type=reward_type, **kwargs)


def make_medium_diverse_gr(reward_type: str = "sparse", **kwargs) -> PointMazeEnv:
    """Create PointMaze_Medium_Diverse_GR environment.

    Features combined goal/reset locations for maximum diversity.

    Args:
        reward_type: 'sparse' or 'dense' reward function
        **kwargs: Additional environment parameters

    Returns:
        PointMazeEnv instance
    """
    return PointMazeEnv(maze_id="Medium_Diverse_GR", reward_type=reward_type, **kwargs)


def make_large_diverse_gr(reward_type: str = "sparse", **kwargs) -> PointMazeEnv:
    """Create PointMaze_Large_Diverse_GR environment.

    Features combined goal/reset locations for maximum diversity.

    Args:
        reward_type: 'sparse' or 'dense' reward function
        **kwargs: Additional environment parameters

    Returns:
        PointMazeEnv instance
    """
    return PointMazeEnv(maze_id="Large_Diverse_GR", reward_type=reward_type, **kwargs)


# Utility functions
def list_environments() -> dict:
    """List all available environments.

    Returns:
        Dictionary mapping environment names to descriptions
    """
    environments = {
        # Standard environments
        "UMaze": "5x5 U-shaped maze with single path",
        "Open": "7x5 open rectangular maze",
        "Medium": "8x8 maze with moderate complexity",
        "Large": "12x9 large complex maze",
        "Giant": "16x12 very large maze",

        # Diverse goal environments
        "Open_Diverse_G": "Open maze with multiple goal locations",
        "Medium_Diverse_G": "Medium maze with diverse goal placement",
        "Large_Diverse_G": "Large maze with many goal options",

        # Diverse goal-reset environments
        "Open_Diverse_GR": "Open maze with combined goal/reset locations",
        "Medium_Diverse_GR": "Medium maze with flexible goal/reset spots",
        "Large_Diverse_GR": "Large maze with maximum location diversity",
    }

    return environments


def get_environment_info(maze_id: str) -> dict:
    """Get detailed information about a specific environment.

    Args:
        maze_id: Environment identifier

    Returns:
        Dictionary with environment details
    """
    available = get_available_mazes()

    if maze_id not in available:
        raise ValueError(f"Unknown maze_id '{maze_id}'. Available: {available}")

    # Get maze layout to analyze
    from .mazes import get_maze_layout
    layout = get_maze_layout(maze_id)

    info = {
        "maze_id": maze_id,
        "size": f"{len(layout)}x{len(layout[0])}",
        "is_diverse": is_diverse_maze(maze_id),
        "total_cells": len(layout) * len(layout[0]),
    }

    # Count cell types
    cell_counts = {"walls": 0, "empty": 0, "goals": 0, "resets": 0, "combined": 0}
    for row in layout:
        for cell in row:
            if cell == 1:
                cell_counts["walls"] += 1
            elif cell == 0:
                cell_counts["empty"] += 1
            elif cell == 'G':
                cell_counts["goals"] += 1
            elif cell == 'R':
                cell_counts["resets"] += 1
            elif cell == 'C':
                cell_counts["combined"] += 1

    info.update(cell_counts)

    return info


# Create environment registry for easy lookup
ENVIRONMENT_REGISTRY = {
    "UMaze": make_umaze,
    "Open": make_open,
    "Medium": make_medium,
    "Large": make_large,
    "Giant": make_giant,
    "Open_Diverse_G": make_open_diverse_g,
    "Medium_Diverse_G": make_medium_diverse_g,
    "Large_Diverse_G": make_large_diverse_g,
    "Open_Diverse_GR": make_open_diverse_gr,
    "Medium_Diverse_GR": make_medium_diverse_gr,
    "Large_Diverse_GR": make_large_diverse_gr,
}


def make(maze_id: str, **kwargs) -> PointMazeEnv:
    """Create environment by maze_id.

    Args:
        maze_id: Environment identifier
        **kwargs: Environment parameters

    Returns:
        PointMazeEnv instance

    Raises:
        ValueError: If maze_id is not recognized
    """
    if maze_id not in ENVIRONMENT_REGISTRY:
        available = list(ENVIRONMENT_REGISTRY.keys())
        raise ValueError(f"Unknown maze_id '{maze_id}'. Available: {available}")

    return ENVIRONMENT_REGISTRY[maze_id](**kwargs)