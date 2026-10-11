# Contributing to francois-feed

Thank you for your interest in contributing to francois-feed!

## How Can I Contribute?

### Reporting Bugs

- Check for existing issues using the GitHub search.
- Use a clear and descriptive title.
- Describe the exact steps to reproduce the problem.

### Suggesting Enhancements

- Check for existing suggestions.
- Provide a detailed description of the suggested enhancement.
- Explain why this enhancement would be useful.

### Pull Requests

- Create a new branch for your changes.
- Ensure your code follows the project's formatting rules (run `uv run ruff format .`).
- Ensure all linter checks, type checks, and tests pass (`uv run ruff check .`, `uv run pyright`, `uv run pytest tests/ -v`).
- Write clear, concise commit messages.
- Ensure all CI checks pass.

## Development Setup

1. **Clone the repository:**

   ```bash
   git clone https://github.com/phine-apps/francois-feed.git
   cd francois-feed
   ```

2. **Install dependencies:**

   ```bash
   uv sync
   ```

3. **Run the script:**

   ```bash
   uv run main.py
   ```

4. **Quality checks (Linting, Formatting, Types, Tests):**

   ```bash
   # Format code
   uv run ruff format .

   # Run linter
   uv run ruff check .

   # Run static type checker
   uv run pyright

   # Run unit tests
   uv run pytest tests/ -v
   ```

By contributing, you agree that your contributions will be licensed under the MIT License of the project.
