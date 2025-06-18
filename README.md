# Pointax: PointMaze Environment for JAX

A high-performance JAX implementation of the PointMaze environment from Gymnasium Robotics.

## Features

- 🚀 **JAX-Native**: Full JAX compatibility with JIT compilation and vectorization support
- 🎯 **Diverse Goals**: Support for multiple goal/reset configurations
- 🔧 **Flexible**: Configurable reward types, continuing tasks, and maze layouts


## Quick Start

```python
import jax
import pointax

# Create environment
env = pointax.make_umaze()
params = env.default_params

# Reset environment
key = jax.random.PRNGKey(42)
obs, state = env.reset_env(key, params)

# Take a step
action = jax.numpy.array([0.5, 0.0])  # Move right
obs, state, reward, done, info = env.step_env(key, state, action, params)

print(f"Reward: {reward}, Done: {done}, Success: {info['is_success']}")
```

## Available Environments

### Standard Mazes

| Environment | Size | Description |
|-------------|------|-------------|
| `UMaze` | 5×5 | U-shaped maze with single path |
| `Open` | 7×5 | Open rectangular arena |
| `Medium` | 8×8 | Moderately complex maze |
| `Large` | 12×9 | Large complex maze |
| `Giant` | 16×12 | Very large maze |

### Diverse Goal Mazes

These environments feature multiple goal locations with designated reset areas:

| Environment | Description |
|-------------|-------------|
| `Open_Diverse_G` | Open maze with multiple goal locations |
| `Medium_Diverse_G` | Medium maze with diverse goal placement |
| `Large_Diverse_G` | Large maze with many goal options |

### Diverse Goal-Reset Mazes

These environments allow both goals and resets at multiple locations:

| Environment | Description |
|-------------|-------------|
| `Open_Diverse_GR` | Open maze with combined goal/reset locations |
| `Medium_Diverse_GR` | Medium maze with flexible goal/reset spots |
| `Large_Diverse_GR` | Large maze with maximum location diversity |

## Usage Examples

### Basic Environment Creation

```python
import pointax

# Standard environments
env = pointax.make_umaze(reward_type="sparse")
env = pointax.make_large(reward_type="dense")

# Diverse goal environments
env = pointax.make_open_diverse_g()
env = pointax.make_large_diverse_gr(reward_type="dense")

# Using the generic make function
env = pointax.make("Medium_Diverse_G", reward_type="dense")
```

### Continuing Tasks

```python
# Create environment that continues after reaching goals
env = pointax.make_umaze(
    continuing_task=True,
    reset_target=True,  # Reset goal when reached
    reward_type="sparse"
)
```

### JAX Vectorization

```python
import jax
import jax.numpy as jnp

env = pointax.make_umaze()
params = env.default_params

# Vectorize for batch processing
batch_size = 64

@jax.vmap
def batch_reset(keys):
    return env.reset_env(keys, params)

@jax.vmap
def batch_step(keys, states, actions):
    return env.step_env(keys, states, actions, params)

# Use with batched operations
keys = jax.random.split(jax.random.PRNGKey(42), batch_size)
batch_obs, batch_states = batch_reset(keys)

actions = jnp.zeros((batch_size, 2))
batch_obs, batch_states, rewards, dones, infos = batch_step(
    keys, batch_states, actions
)
```

## Environment Details

### Action Space

- **Type**: `Box(-1.0, 1.0, (2,), float32)`
- **Description**: Continuous forces applied in x and y directions
- **Range**: [-1.0, 1.0] for each dimension
- **Units**: Force (N)

### Observation Space

- **Type**: `Box(-inf, inf, (6,), float32)`
- **Components**:
  - `obs[0:2]`: Agent position (x, y) in meters
  - `obs[2:4]`: Agent velocity (vx, vy) in m/s
  - `obs[4:6]`: Goal position (x, y) in meters

### Rewards

#### Sparse Reward (default)
- **1.0**: When agent reaches goal (distance ≤ 0.45m)
- **0.0**: Otherwise

#### Dense Reward
- **Formula**: `exp(-distance_to_goal)`
- **Range**: (0, 1], higher values for closer proximity

### Termination

- **Time limit**: 1000 steps (configurable)
- **Goal reached**: Only for non-continuing tasks
- **Continuing tasks**: Never terminate on goal achievement, only on time limit

## Maze Cell Types

| Symbol | Value | Description |
|--------|-------|-------------|
| `1` | 1 | Wall (impassable) |
| `0` | 0 | Empty space |
| `'G'` | 2 | Goal location |
| `'R'` | 3 | Reset location |
| `'C'` | 4 | Combined goal/reset location |


## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Run the test suite (`pytest`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this library in your research, please cite:

```bibtex
@software{pointmaze_jax,
  title={JPointax: PointMaze Environment for JAX},
  author={Waris Radji},
  year={2025},
  url={https://github.com/riiswa/pointax}
}
```

## Acknowledgments

- Original PointMaze environment from [Gymnasium Robotics](https://github.com/Farama-Foundation/Gymnasium-Robotics)
- D4RL paper by Fu et al. for maze environment inspiration
- JAX team for the excellent JAX framework
- Gymnax for the environment interface standards