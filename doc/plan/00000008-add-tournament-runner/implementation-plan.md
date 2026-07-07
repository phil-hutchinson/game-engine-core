# Implementation Plan: Add Tournament Runner

Story: [story.md](story.md)

Testing approach: the reusable pieces (result aggregation, the round-robin runner, reporting, checkpoint helpers) are covered by automated tests in `tests/core` and `tests/learning`, reusing the Nim fixture game — its single-take mode makes every game a forced line with a known winner, so tournament outcomes are exactly predictable. The example scripts (`train.py` checkpointing, `tournament.py`) are verified manually by running them, matching how the other examples are treated. Each step also requires `pytest`, `pyright`, and `ruff check .` to pass clean, per CONTRIBUTING.md.

### Step 1 — Result models and aggregation

Implement the tournament data layer in a new `game_engine_core/tournament/` package: a per-game record type (the two players' names, which had side 1, and the `GameResult`), plus aggregation from a list of records to standings (wins/draws/losses and 1/0.5/0 points per player, sorted by points) and a cross-table (each player's aggregate score against each other player). Aggregation is pure — records in, tables out — so it can be tested with fabricated records, including draws and edge cases that real fixture games can't produce (the Nim fixture has no draws).

Depends on: nothing — the record type introduced here is what the runner (Step 2) produces and the reporter (Step 3) consumes.

Verification (automated): New tests in `tests/core` build record lists by hand (wins on either side, draws, multiple games per pairing) and assert standings points/ordering and cross-table cell values. Run `pytest tests/core`; confirm `pyright` and `ruff check .` pass clean.

### Step 2 — Round-robin tournament runner

Implement the `Tournament` runner in `game_engine_core/tournament/`: takes a list of players (validating names are unique, since results are keyed by name), a position factory, a `GameUI`, and `games_per_pairing`. Every pair of players meets `games_per_pairing` times with side 1 alternating between them; each game runs headless through `StandardGame` (no final-board render). The runner maps each game's absolute `GameResult` outcome (side-1-relative) back to the players who held the sides, and returns a completed-tournament result holding the per-game records plus the Step 1 aggregations.

Depends on: Step 1 (produces its record type and returns its aggregations).

Verification (automated): Tests in `tests/core` run tournaments over the Nim fixture in single-take mode with scripted players, asserting: total game count for N players and k games per pairing, side assignment alternating within each pairing, outcome attribution correct regardless of which side a player held (the forced-line fixture makes each game's winner predictable from who moved first), and duplicate names rejected. Run `pytest tests/core`; confirm `pyright` and `ruff check .` pass clean.

### Step 3 — Tournament reporting

Implement reporting for a completed tournament: given an output folder (caller supplies the path; the runner does not invent one), write `standings.md` containing the standings and cross-table, and a `games/` subfolder with one JSON file per game (player names, side assignment, outcome, and the ply-by-ply log from `GameResult.game_log`). Also print the standings and cross-table summary to the console.

Depends on: Step 2 (serialises the completed-tournament result it returns).

Verification (automated): Tests in `tests/core` run a small Nim tournament into a pytest `tmp_path`, then assert `standings.md` exists and names every player, the `games/` folder holds exactly one parseable JSON file per game record, and each file's outcome and plies match its record. Run `pytest tests/core`; confirm `pyright` and `ruff check .` pass clean.

### Step 4 — Checkpoint naming and discovery helpers

Implement checkpoint file conventions in `game_engine_learning`: given a weights directory and an iteration number, produce the checkpoint path (zero-padded so lexical and numeric order agree, e.g. `checkpoint-00005.pt`); given a directory, discover the checkpoints in it and return them in iteration order with their iteration numbers. This is the seam both training (writes) and tournaments (read) share, in any game repo. The helpers deal only in paths and names — actually saving/loading tensors stays with the callers, so the module stays torch-free.

Depends on: nothing in Steps 1–3 (parallel track); placed here so both example scripts (Steps 5–6) can use it.

Verification (automated): Tests in `tests/learning` cover path formatting, round-tripping (paths produced by the naming helper are found by discovery with the right iterations), ordering with mixed iteration widths, and ignoring non-checkpoint files. Run `pytest tests/learning`; confirm `pyright` and `ruff check .` pass clean.

### Step 5 — Checkpointing in the training example

Extend `examples/tictactoe_learning/train.py` with a `--checkpoint-every n` option (default off, preserving current behaviour): every n iterations, save the model weights to a checkpoint file in the existing gitignored weights directory, named via the Step 4 helpers, alongside the final `model.pt`.

Depends on: Step 4 (checkpoint naming).

Verification (manual): Run `python -m examples.tictactoe_learning.train --iterations 4 --games 5 --checkpoint-every 2` and confirm the weights directory gains checkpoint files for iterations 2 and 4 plus the final `model.pt`, and that a run without the flag writes no checkpoints.

### Step 6 — TicTacToe tournament example

Implement `examples/tictactoe_learning/tournament.py`: discover checkpoints via the Step 4 helpers, load each into a `TicTacToeNNEvaluator` + `MCTSEngine` + `AIPlayer` named after its iteration (e.g. `nn-iter-05`), add reference players (a `RandomEngine` player and the heuristic-evaluator MCTS player from `examples/tictactoe/`) to demonstrate mixing in extra players at creation, run the round-robin, and report into a timestamped folder under a new gitignored tournaments directory. Fail with a clear message if no checkpoints exist ("run train.py with --checkpoint-every first"). Expose the knobs that matter as CLI options (games per pairing, MCTS iterations).

Depends on: Steps 2–5 (runs the runner and reporter over players built from Step 5's checkpoints).

Verification (manual): After a training run with checkpoints enabled (Step 5's command, or longer for clearer separation), run `python -m examples.tictactoe_learning.tournament` and confirm: a timestamped folder appears containing `standings.md` and per-game JSONs, the console shows standings and cross-table, random sits at or near the bottom, and later checkpoints generally outscore earlier ones. (Strength ordering is statistical, not guaranteed per run — random at the bottom is the hard expectation; checkpoint ordering trending upward is the goal signal.)

### Step 7 — README check

Verify `README.md` is still accurate given the story's changes, updating it if warranted (the `/update-readme` command automates this). The story adds a `game_engine_core` subpackage, a `game_engine_learning` module, and a new example entry point; anything the README says about package layout or example usage may need to reflect them.

Depends on: Steps 1–6 (the README can only be checked against the story's final shape).

Verification (manual): Read the resulting README diff (or confirmation that no change was needed) and confirm it matches the story's actual changes.

## After completion

Update this plan if any step's substance changed during implementation, and record each step's commit against it in the peer review when the story is reviewed.
