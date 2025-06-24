# Pointax: JAX-Native PointMaze Environment

**High-performance JAX implementation of PointMaze environments with MuJoCo-inspired physics.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![JAX](https://img.shields.io/badge/JAX-compatible-green.svg)](https://github.com/google/jax)

Pointax provides a complete JAX implementation of the PointMaze environment from Gymnasium Robotics, featuring full JIT compilation, vectorization support, and a simplified but accurate 2D physics engine inspired by MuJoCo.

## Physics Engine

Pointax implements a simplified but accurate 2D physics engine inspired by MuJoCo while being fully differentiable and JIT-compilable:

- **Collision Detection**: Sphere-AABB intersection with anti-sticking mechanisms
- **Force Integration**: Proper velocity and position updates with motor scaling
- **Boundary Handling**: Smooth wall interactions with friction coefficients
- **Parameter Matching**: Robot radius (0.1m), motor gear (100x), and dynamics match MuJoCo reference

## Quick Start

```python
import jax
import pointax

# Create environment
env = pointax.make_umaze()
params = env.default_params

# Reset and step
key = jax.random.PRNGKey(42)
obs, state = env.reset_env(key, params)

action = jax.numpy.array([0.5, 0.0])  # Move right
obs, state, reward, done, info = env.step_env(key, state, action, params)

print(f"Reward: {reward}, Success: {info['is_success']}")
```

## Installation

From source:
```bash
git clone https://github.com/riiswa/pointax.git
cd pointax
pip install -e .
```

## Environment Catalog

### Standard Mazes
| Environment | Size | Description |
|-------------|------|-------------|
| `UMaze` | 5×5 | U-shaped maze with single path |
| `Open` | 7×5 | Open rectangular arena |
| `Medium` | 8×8 | Moderately complex maze |
| `Large` | 12×9 | Large complex maze |
| `Giant` | 16×12 | Very large maze |

### Diverse Goal Mazes
| Environment | Description |
|-------------|-------------|
| `Open_Diverse_G` | Open maze with multiple goal locations |
| `Medium_Diverse_G` | Medium maze with diverse goal placement |
| `Large_Diverse_G` | Large maze with many goal options |

### Diverse Goal-Reset Mazes
| Environment | Description |
|-------------|-------------|
| `Open_Diverse_GR` | Open maze with flexible goal/reset locations |
| `Medium_Diverse_GR` | Medium maze with combined goal/reset spots |
| `Large_Diverse_GR` | Large maze with maximum location diversity |

## Usage Examples

### Basic Environments

```python
import pointax

# Standard environments
env = pointax.make_umaze(reward_type="sparse")
env = pointax.make_large(reward_type="dense")

# Diverse environments
env = pointax.make_open_diverse_g()
env = pointax.make_large_diverse_gr(reward_type="dense")
```

### Custom Maze Creation

Create environments from simple 2D layouts:

```python
# Simple custom maze (1=wall, 0=empty)
custom_maze = [
    [1, 1, 1, 1, 1],
    [1, 0, 0, 0, 1],
    [1, 0, 1, 0, 1],
    [1, 0, 0, 0, 1],
    [1, 1, 1, 1, 1]
]
env = pointax.make_custom(custom_maze)

# Maze with specific goal/reset locations
maze_with_goals = [
    [1, 1, 1, 1, 1, 1, 1],
    [1, 'R', 0, 0, 0, 'G', 1],  # 'R'=reset, 'G'=goal
    [1, 0, 1, 1, 1, 0, 1],
    [1, 0, 0, 'G', 0, 0, 1],     # Multiple goals
    [1, 'C', 0, 0, 0, 'C', 1],   # 'C'=combined goal/reset
    [1, 1, 1, 1, 1, 1, 1]
]
env = pointax.make_custom(maze_with_goals, reward_type="dense")

# Boolean maze (intuitive for many users)
bool_maze = [
    [True,  True,  True ],   # True = wall
    [True,  False, True ],   # False = empty
    [True,  True,  True ]
]
env = pointax.make_custom(bool_maze, maze_id="MyMaze")
```

### JAX Transformations

Pointax is designed for JAX workflows:

```python
import jax
import jax.numpy as jnp

env = pointax.make_medium()
params = env.default_params

# JIT compilation
@jax.jit
def fast_step(key, state, action):
    return env.step_env(key, state, action, params)

# Vectorization
@jax.vmap
def batch_step(keys, states, actions):
    return env.step_env(keys, states, actions, params)

# Use with batches
batch_size = 64
keys = jax.random.split(jax.random.PRNGKey(42), batch_size)
actions = jnp.zeros((batch_size, 2))
# ... batch operations
```

## Environment Specification

### Observation Space
- **Type**: `Box(-inf, inf, (6,), float32)`
- **Contents**: `[pos_x, pos_y, vel_x, vel_y, goal_x, goal_y]`

### Action Space
- **Type**: `Box(-1.0, 1.0, (2,), float32)`
- **Contents**: Continuous forces in x and y directions

### Rewards
- **Sparse**: 1.0 when goal reached (distance ≤ 0.45m), 0.0 otherwise
- **Dense**: `-distance_to_goal` (negative distance for gradient-based learning)

### Maze Symbols
| Symbol | Value | Description |
|--------|-------|-------------|
| `1` or `True` | 1 | Wall (impassable) |
| `0` or `False` | 0 | Empty space |
| `'G'` | 2 | Goal location |
| `'R'` | 3 | Reset location |
| `'C'` | 4 | Combined goal/reset location |


## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Citation

If you use Pointax in your research, please cite:

```bibtex
@software{pointax2025,
  title={Pointax: JAX-Native PointMaze Environment},
  author={Waris Radji},
  year={2025},
  url={https://github.com/riiswa/pointax}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Original PointMaze environment from [Gymnasium Robotics](https://github.com/Farama-Foundation/Gymnasium-Robotics)
- [Gymnax](https://github.com/RobertTLange/gymnax) for their API
- [MuJoCo](https://github.com/google-deepmind/mujoco) physics engine for inspiration
- JAX team for the excellent framework