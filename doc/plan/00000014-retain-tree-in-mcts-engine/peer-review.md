# Peer Review — retain-tree-in-mcts-engine

## Summary

The change adds `observe_ply`/`reset` to the `GameEngine`/`Player` protocols and implements
tree retention in `MCTSEngine`: `select_ply` warm-starts from a retained root, `observe_ply`
re-roots onto the matching child (or clears the root on a miss), and `reset` clears the root
for a new game. `StandardGame.run` broadcasts resets and plies to both players, and the default
`iterations` drops to 1,000. Test coverage for re-rooting, the miss fallback, reset, and
cross-game isolation was added, and the README was updated. `pyright` and `ruff check .` are
both clean, and the full `pytest` suite (110 tests) passes.

## Comments

### Minor

| # | Status | Resolution | Location | Comment | Suggested Change | Code Snippet |
|---|--------|------------|----------|---------|-----------------|--------------|
| 1 | Resolved | Removed the two `reset()` calls from `Tournament.run`; `StandardGame.run` is now the sole caller. | [game_engine_core/tournament/tournament.py#L64](../../../game_engine_core/tournament/tournament.py#L64) | `Tournament.run` now calls `first.reset()` / `second.reset()` before every game, but `StandardGame.run` (added in this same change) already resets both players at the start of `run()`. This makes every game's reset happen twice. Neither the story's Scope section nor the implementation plan lists `tournament.py` as a file to change — the story explicitly attributes cross-game isolation to `StandardGame.run` alone ("`StandardGame.run` calls `reset()` on both players at the start of each game ... Required because engine instances are reused across games"). The duplication is harmless today because `reset()` is idempotent, but it's undocumented scope creep and leaves two call sites responsible for the same invariant. | Remove the two `reset()` calls from `Tournament.run` and rely solely on `StandardGame.run`, per the story's stated design — or, if `Tournament` is meant to own this responsibility instead, update the story/plan to say so. | `first.reset()`<br>`second.reset()` |
| 2 | Resolved | Comment updated to reference `_create_root`/`_grow_tree`. | [tests/core/test_mcts_engine.py#L30](../../../tests/core/test_mcts_engine.py#L30) | The comment still refers to `_build_tree`, which this change removed and replaced with `_create_root` + `_grow_tree` (the two lines directly below the comment call the new methods). The comment is now stale. | Update the comment to say `_create_root`/`_grow_tree` instead of `_build_tree`. | `# Inspects the tree via the private _build_tree: the per-node value signs are` |
| 3 | Resolved | Trailing whitespace stripped from the blank lines in `observe_ply`. | [game_engine_core/engines/mcts_engine.py#L79](../../../game_engine_core/engines/mcts_engine.py#L79) | Two blank lines introduced by this change inside `observe_ply` (after the early-return and after the miss-return) contain trailing whitespace. | Strip the trailing whitespace on the blank lines in the new `observe_ply` method. | `        ` (trailing whitespace on otherwise-blank lines) |

## Discrepancy check

- **Story vs. implementation plan:** consistent — the plan explicitly treats the story's design
  as already implemented in Step 1 and adds verification/test/quality/README steps on top. No
  contradictions found between the two documents.
- **Story/plan vs. diff:** the implementation matches the story's design (observation-driven
  re-rooting, `str(ply)` matching, reset-on-miss, split `select_ply`/`select_ply_with_policy`
  behaviour, default `iterations` reduced to 1,000) and the plan's steps (retention code, protocol
  methods, `AIPlayer`/`HumanPlayer`/`RandomEngine` updates, `StandardGame.run` reset/broadcast,
  test coverage, quality gates, README update) with the one exception noted in Minor #1 — the
  `Tournament.run` reset calls are additional code not called for by either document.
- **README step:** the implementation plan's Step 4 (verify `README.md` is current given the new
  protocol methods) was carried out — `README.md` now documents `MCTSEngine`'s retention behaviour
  and its interaction with `StandardGame`/`Tournament` and `select_ply_with_policy`.

## Quality gates

- `pyright`: 0 errors, 0 warnings, 0 informations.
- `ruff check .`: all checks passed.
- `pytest`: 110 passed.
