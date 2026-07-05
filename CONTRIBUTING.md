# Contributing

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
