# Contributing to Pointax

Thank you for your interest in contributing to Pointax! 

## Quick Start

1. **Fork and clone**
   ```bash
   git clone https://github.com/your-username/pointax.git
   cd pointax
   ```

2. **Setup development environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -e ".[dev,examples]"
   ```

3. **Run tests**
   ```bash
   pytest tests/ -v
   ```

## Making Changes

1. **Create a branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Write clear, documented code
   - Add tests for new features
   - Follow existing code style

3. **Test your changes**
   ```bash
   # Run tests
   pytest tests/ -v
   
   # Check code style
   black pointax tests examples
   ```

4. **Submit a pull request**
   - Describe what your changes do
   - Reference any related issues
   - Ensure all tests pass

## Types of Contributions

- 🐛 **Bug fixes** - Fix issues or improve existing functionality
- ✨ **New features** - Add new maze layouts, environments, or capabilities
- 📚 **Documentation** - Improve docs, examples, or tutorials
- 🧪 **Tests** - Add test coverage or performance benchmarks

## Code Style

- Use `black` for formatting: `black pointax tests examples`
- Write clear docstrings for public functions
- Add type hints where helpful
- Keep functions focused and well-named

## Questions?

- Open an issue for bugs or feature requests
- Start a discussion for questions or ideas
- Check existing issues before creating new ones

We appreciate all contributions, big and small! 🚀