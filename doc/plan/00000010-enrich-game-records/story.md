# Story: Enrich Game Records (issue #10)

## Goal

Make the game record produced by `StandardGame.run()` carry two things it currently cannot express: **why** a game ended, and a **game-specific display rendering** of each ply that can be richer than the ply's identity string. Both are driven by a downstream game implementation (Capture the Flag) but are generic needs — chess would want them for the same reasons (e.g. "checkmate" vs "stalemate"; `Nf3` vs `G1F3`).

Two changes, bundled because they reshape the same seams (`GamePosition`/`GameResult`/`StandardGame` and the protocols around game logging):

1. **`GameResult.result_reason`** — record why the game ended, sourced from a new `outcome_reason` property on `GamePosition`.
2. **`GameLogging` protocol with `ply_annotation`** — split logging concerns out of `GameUI` and let games render plies for the game log independently of `str(ply)`.

Both are **breaking protocol changes, made without defaults or compatibility shims** — the library's only consumers are this repo and one downstream repo, and optional protocol members age badly.

## Motivation

**Result reason.** `GameResult` carries only `outcome: Literal[1, 0, -1]`. Nearly every game has multiple ways to reach the same outcome (capture, resignation, timeout, stalemate, no-progress draw, …) and downstream record formats want to state which one occurred; today they can only write "Unknown". The architecture already forces the reason to be position-derived — `StandardGame.run()` terminates only when `position.outcome` becomes non-`None` — so the position is the one place that always knows why.

**Ply annotation.** `GamePly.__str__` is the framework's **identity key** (policy dictionaries, MCTS visit distributions, training targets) and must stay simple and unique. But the string worth *logging* is often richer and is a function of context the ply alone doesn't have: capture markers need the pre-ply board, chess-style disambiguation needs the other legal plies, check/mate suffixes need the post-ply position. That computation can be expensive, so it must not live on `GamePosition` or `GamePly` where engines would pay for it during search — it belongs in a logging seam invoked once per *executed* ply.

**Protocol split.** `GameUI` is currently doing double duty: `render_board`/`get_next_ply` are genuine interactive UI (used by `HumanPlayer`), while `text_board` is a logging function — headless tournament games still require a `GameUI` just to feed the logs. Combining UI and logging should have been a per-game decision, not an engine-level one. Splitting them makes headless play explicit and gives `ply_annotation` its natural home.

## Scope

### 1. `result_reason` on game results

- `GamePosition` gains a required property `outcome_reason: str | None`: `None` while the game is ongoing; a short human-readable, game-specific string (e.g. `"Three in a row"`, `"Stalemate"`) once `outcome` is non-`None`. Free-form `str`, not an enum — the vocabulary is game-specific.
- `GameResult` gains `result_reason: str`, populated by `StandardGame.run()` from the terminal position.
- Tournament reporting includes the reason in each per-game JSON file.

### 2. `GameLogging` protocol and `ply_annotation`

- New protocol `GameLogging[TPly, TPosition]` with:
  - `text_board(position)` — moves here from `GameUI`.
  - `ply_annotation(from_position, ply, to_position) -> str` — the string to log for an executed ply. `(from_position, ply, to_position)` is sufficient for mainstream notations (captures and disambiguation from the pre-ply position and its legal plies; end-of-game markers from the post-ply position). `str(ply)` remains a valid trivial implementation.
- `GameUI` slims to the interactive pair: `render_board`, `get_next_ply`. (`HumanPlayer` is unaffected — it holds its own `GameUI` reference.)
- `StandardGame` takes a required `GameLogging` (feeds `opening_board` and the game log) and an optional `game_ui: GameUI | None` used only for board rendering; headless play passes `None` instead of supplying a UI that is never shown.
- `Tournament` takes a `GameLogging` instead of a `GameUI` and runs games with `game_ui=None`.
- The game log keeps its `(ply annotation, board after ply)` shape; the annotation entry is now `ply_annotation(...)` instead of `str(ply)`. The identity contract on `GamePly.__str__` is untouched.
- A game may implement `GameUI` and `GameLogging` on one class (structural protocols; the method sets don't collide) or on separate ones.

### 3. In-repo implementers and tests

- TicTacToe example and the Nim test fixture implement the new members: real `outcome_reason` strings, and trivial `ply_annotation` returning `str(ply)` (no fancy notation needed for either game). The TicTacToe UI class implements both protocols.
- Tests updated for the new protocol members, `GameResult` field, `StandardGame`/`Tournament` signatures, and reporting output; new coverage that `ply_annotation` (not `str(ply)`) is what lands in the game log, and that `result_reason` reaches the record and report.
- Docstrings that describe the old arrangement (e.g. `Tournament`'s "the GameUI is only consulted for text_board") and the README updated to match.

## Non-goals

- No richer notation for TicTacToe or Nim — trivial `str(ply)` annotations only; the first real consumer of rich annotations is the downstream CtF repo.
- No resignation/timeout mechanics: `StandardGame`'s only exit path remains `position.outcome`, so such endings are expressible only if a game encodes them in its position. Adding other exit paths is out of scope.
- No backwards-compatibility layer for the protocol changes.
