# Contributing

## Development environment

The repository ships a [VS Code Dev Container](.devcontainer/) as the supported development environment. With Docker and the VS Code **Dev Containers** extension installed, open the repository and choose **Reopen in Container**. The container provisions everything on first build: Python 3.12, the project installed editable with its `learning` extra, and the type-checking/linting/testing toolchain — no manual setup and no virtual environment (the container is the isolation boundary).

Personal environment variables (e.g. `TZ=America/Vancouver`) can be set container-wide in `.devcontainer/devcontainer.env` — one `KEY=VALUE` per line. To get started, rename (or copy) [`devcontainer.env.example`](.devcontainer/devcontainer.env.example) to `devcontainer.env`, edit it, and rebuild the container. The file is gitignored and created empty on first container start if absent, so it is entirely optional.

### Type checking and linting

Run from the container terminal, from the repository root:

```bash
pyright        # type check
ruff check .   # lint
```

Both tools are configured in `pyproject.toml` and both should pass clean before a change is submitted.

### Tests

Run from the container terminal, from the repository root:

```bash
pytest             # full suite: package tests (tests/) + example tests (examples/*/tests/)
pytest tests       # package tests only
```

The suite must pass before a change is submitted. See [doc/plan/00000006-add-automated-testing/automated-testing.md](doc/plan/00000006-add-automated-testing/automated-testing.md) for the suite layout, the fixture-game strategy, and testing conventions.

## Python version

This project targets Python 3.12+. Follow up-to-date language standards accordingly.

## Code conventions

### Imports

**Within a package** (e.g. `game_engine_core/`, and any future sibling packages): use relative imports.

```python
# correct — inside game_engine_core
from ..protocols.game_ply import GamePly
from .game_position import GamePosition
```

**In `examples/`**: import the library packages absolutely, exactly as an external consumer of the package would. Sibling modules within a single example may be imported relatively.

```python
# correct — inside examples/
from game_engine_core.protocols.game_ply import GamePly
from game_engine_core.engines.mcts_engine import MCTSEngine
from .tictactoe_ply import TicTacToePly  # sibling within the same example
```

Ruff enforces these conventions only partially: it bans parent-relative imports in `examples/` (TID252), but no ruff rule can *require* relative imports, so the within-package convention relies on code review rather than tooling.
