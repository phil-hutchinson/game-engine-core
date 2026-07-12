# Peer Review — Widen Tournament Position Factory (issue #11)

## Summary

Widens `Tournament.position_factory` from a zero-arg callable to one receiving the two participants in side order `(side_one, side_other)`, invoked once per game by `_play_game`, with the side-order contract and ignore-the-arguments pattern documented in the class docstring. All in-repo callers (both tournament test files and the tictactoe-learning example) were moved to two-parameter factories in lockstep, a new test pins the side-order contract against `GameRecord`, and the README's Tournaments section gained a sentence describing the seam. `pyright` reports 0 errors/0 warnings, `ruff check .` passes, and the full suite (78 tests) is green.

The implementation matches the story's scope and non-goals: `SelfPlayCollector`, `StandardGame`, and the `Player` protocol are untouched, and the change is a clean break with no compatibility shim, as required. The merge of `origin/main` (story #10's `game_ui` → `game_logging` rename) was resolved consistently across all touched files.

## Comments

### Minor

| # | Status | Resolution | Location | Comment | Suggested Change | Code Snippet |
|---|--------|------------|----------|---------|-----------------|--------------|
| 1 | Closed | No Action Required | [implementation-plan.md#L36](implementation-plan.md#L36) | Step 1's verification cites running `mypy` "per CONTRIBUTING.md", but CONTRIBUTING.md specifies only `ruff` and `pytest`, and mypy is not installed in the project environment. The check actually performed was pyright (which is what this review ran, clean). Documentation-only discrepancy between the plan and the repo's toolchain. | Amend the plan's Step 1 verification to name `pyright` (or drop the type-checker mention) so the plan reflects the tooling that exists. | `` `ruff check .` and `mypy` per CONTRIBUTING.md `` |
| 2 | Closed | No Action Required | [README.md#L61](../../../README.md#L61) | The story scoped the README change to "README updated **if it shows a `Tournament` construction**" — it does not, yet a sentence describing the factory seam was added anyway. The sentence is accurate and arguably improves the section, but it is technically beyond the story's stated condition, so flagging for an explicit accept/revert decision. | Accept as-is (the added capability is user-facing and the sentence is correct), or revert the sentence to keep the diff strictly within story scope. | `Each game's starting position comes from a factory called with the two participants in side order…` |

