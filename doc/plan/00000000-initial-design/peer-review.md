# Peer Review — Initial Design

## Summary

This is a general review of the initial codebase covering the core framework (`game_engine_core`) and the TicTacToe example. The design is clean and the protocol-based abstraction boundary is well-drawn. The MCTS implementation is correct. The findings below are mostly consistency and minor design issues — nothing that breaks the system today.

---

## Comments

### Major

| # | Status | Resolution | Location | Comment | Suggested Change | Code Snippet |
|---|--------|------------|----------|---------|-----------------|--------------|
| 1 | Resolved | Changed to relative imports; added import convention to CONTRIBUTING.md. | [game_engine_core/engines/random_engine.py#L4-L5](../../../game_engine_core/engines/random_engine.py#L4-L5) | Uses absolute imports (`game_engine_core.protocols...`) while every other file in the package uses relative imports (`..protocols...`). Inconsistent and fragile — if the package is ever renamed or imported in an unusual way, this breaks while everything else works. | Change to relative imports to match the rest of the package. | `from game_engine_core.protocols.game_ply import GamePly` |
| 2 | Resolved | Replaced `Tuple`/`List` with built-in `tuple`/`list`; removed deprecated `typing` imports. | [game_engine_core/game/standard_game.py#L1](../../../game_engine_core/game/standard_game.py#L1) | Imports `Tuple` and `List` from `typing`, which are deprecated since Python 3.9. The rest of the codebase uses the built-in `tuple[...]` and `list[...]` forms consistently. Also affects line 25. | Replace with built-in types: `list[tuple[str, str]]`. | `from typing import Literal, Tuple, List, Any` |
| 3 | Deferred | Keeping class constant as scaffolding for now; added TODO comment at the declaration with the suggested fix. | [game_engine_core/engines/mcts_engine.py#L58](../../../game_engine_core/engines/mcts_engine.py#L58) | `TEMPERATURE` is a class constant, not an instance parameter. You cannot have two `MCTSEngine` instances with different temperatures without subclassing, and setting `MCTSEngine.TEMPERATURE = x` silently affects all instances. | Add `temperature: float = 0.0` to `__init__` and store as `self._temperature`. | `TEMPERATURE: float = 0.0` |

---

### Minor

| # | Status | Resolution | Location | Comment | Suggested Change | Code Snippet |
|---|--------|------------|----------|---------|-----------------|--------------|
| 4 | Resolved | Removed the guard; formula now runs unconditionally. | [game_engine_core/engines/mcts_engine.py#L47-L48](../../../game_engine_core/engines/mcts_engine.py#L47-L48) | The `parent is None or parent.visits == 0` guard in `puct_value` is unreachable in normal use. `puct_value` is only called on children via `max(current.children, ...)`, so `parent` is always set and already has at least one visit. | Remove the guard and let the formula run naturally, or add an assertion to document the invariant instead. | `if self.parent is None or self.parent.visits == 0:` |
| 5 | Resolved | No policy → uniform prior (1.0 for all moves); policy provided → raises ValueError if a move is missing. | [game_engine_core/engines/mcts_engine.py#L112](../../../game_engine_core/engines/mcts_engine.py#L112) | When a policy is provided but a move is missing from it, the fallback prior is `1.0`. A uniform prior is typically `1 / n_legal_moves`, so `1.0` would dramatically over-favour unlisted moves relative to listed ones. | Use a small fallback (e.g. `1e-6`) or `1.0 / len(node.unexplored_plies + node.children)` to keep scale consistent. | `prior = node.policy.get(str(ply), 1.0)` |
| 6 | Won't Fix | `__eq__` is not part of the `GamePly` protocol and nothing in the framework relies on it. Left to each game's ply class to add if needed. | [examples/tictactoe/tictactoe_ply.py](../../../examples/tictactoe/tictactoe_ply.py) | `TicTacToePly` has no `__eq__` or `__hash__`. Two `TicTacToePly(3)` instances compare unequal by identity. The MCTS policy keying via `str(ply)` works around this, but callers comparing plies directly (e.g. in tests or game logic) will get surprising results. | Add `__eq__` comparing `self._square` and `__hash__` returning `hash(self._square)`. | *(no current implementation)* |
| 7 | Resolved | Added empty `__init__.py` to `game_engine_core/` and `game_engine_core/players/`. | [game_engine_core/](../../../game_engine_core) | `game_engine_core/` has no top-level `__init__.py`, and `game_engine_core/players/` also has no `__init__.py`, unlike the other subpackages. Python will treat these as implicit namespace packages, which works, but it is inconsistent. | Add empty `__init__.py` files to `game_engine_core/` and `game_engine_core/players/` to match the other subpackages. | *(missing file)* |
