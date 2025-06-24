import pointax

# Standard environments
env = pointax.make_umaze(reward_type="sparse")
env = pointax.make_large(reward_type="dense")

# Diverse goal environments
env = pointax.make_open_diverse_g()
env = pointax.make_large_diverse_gr(reward_type="dense")

# Using the generic make function
env = pointax.make("Medium_Diverse_G", reward_type="dense")

### Custom Maze Creation
import pointax

# Simple custom maze (1=wall, 0=empty)
custom_maze = [
    [1, 1, 1, 1, 1],
    [1, 0, 0, 0, 1],
    [1, 0, 1, 0, 1],
    [1, 0, 0, 0, 1],
    [1, 1, 1, 1, 1]
]
env = pointax.make_custom(custom_maze)

# Custom maze with specific goal/reset locations
maze_with_goals = [
    [1, 1, 1, 1, 1, 1, 1],
    [1, 'R', 0, 0, 0, 'G', 1],  # 'R'=reset, 'G'=goal
    [1, 0, 1, 1, 1, 0, 1],
    [1, 0, 0, 'G', 0, 0, 1],    # Multiple goals possible
    [1, 1, 1, 1, 1, 1, 1]
]
env = pointax.make_custom(maze_with_goals, reward_type="dense")

# Using boolean values (True=wall, False=empty)
bool_maze = [
    [True,  True,  True ],
    [True,  False, True ],
    [True,  True,  True ]
]
env = pointax.make_custom(bool_maze, maze_id="MyMaze")