# Peer Review — Add a Capture-Time Policy Transform to SelfPlayCollector (#18)

## Summary

The change adds an optional `policy_transform` hook to `SelfPlayCollector`,
applied in `_play_game` at the point each step is recorded — while the
`position` (and thus `active_player_id`) is still in scope — and defaulting to
`None` (identity). This is additive and non-breaking: it introduces a
`PolicyTransform` type alias, stores the hook on the instance, documents the
capture-time contract in the class docstring, and adds a test that re-keys the
distribution by `position.active_player_id`. `pyright` reports **0 errors, 0
warnings**; `ruff check .` reports **All checks passed**; `pytest tests/learning/`
is green (25 passed). The implementation faithfully covers the story's scope and
all three plan steps; the findings below are minor process/consistency notes
only.

## Comments

### Minor

| # | Status | Resolution | Location | Comment | Suggested Change | Code Snippet |
|---|--------|------------|----------|---------|-----------------|--------------|
| 1 | Closed | Won't fix — deviation is intentional and accepted; PEP 695 generic form kept as-is. | [../../../game_engine_learning/self_play_collector.py#L15-L17](../../../game_engine_learning/self_play_collector.py#L15-L17) | The plan (Step 1) said to declare the alias "mirroring how `PolicyLossFn` / `ValueLossFn` are declared in `training_loop.py`", where those are plain module-level assignments (`ValueLossFn = Callable[...]`). The implementation instead uses a PEP 695 generic `type` statement. This is a justified deviation — the alias must be generic over `TPosition`, which a plain assignment at module scope cannot express (the story's illustrative `PolicyTransform = Callable[[TPosition, ...]]` would leave `TPosition` unbound) — but it does depart from the stated "mirror" and is worth recording. No change required; consider aligning `training_loop.py`'s aliases to the same PEP 695 form in a future cleanup for consistency. | Keep as-is; optionally note the intentional deviation, or later migrate `PolicyLossFn`/`ValueLossFn` to `type ... = ...`. | `type PolicyTransform[TPosition: GamePosition[Any]] = Callable[[TPosition, dict[str, float]], dict[str, float]]` |
| 2 | Closed | Confirmed by author: `/update-readme` was run and no README change was warranted; only the plan was not annotated. No action needed. | [../../implementation-plan.md#L52-L57](../../implementation-plan.md#L52-L57) | Plan Step 3 requires checking `README.md` and, if no update is warranted, *recording* that no update is needed. `README.md` genuinely needs no change (it does not enumerate the collector's constructor surface — confirmed by grep for `SelfPlayCollector`/`policy_transform`), so the outcome is correct, but there is no committed artifact capturing that the check was performed. | Record the README verification outcome (e.g. a line in the story/plan or commit message) so the manual Step 3 check is evidenced. | n/a |

## Notes (non-blocking, no action required)

- The story's "shape the seam to grow" note is satisfied: the class docstring
  now carries the "capture is the last place the step has its position" guidance
  covering frame-correcting the policy, extracting provenance, and tagging the
  outcome.
- The `PolicyTransform` alias is defined in the same module as
  `SelfPlayCollector` and is therefore importable "alongside the collector",
  consistent with how `PolicyLossFn`/`ValueLossFn` live in `training_loop.py`
  (the package `__init__.py` is empty, so nothing is added to a public `__all__`
  — this matches existing convention).
- The new test correctly exploits the forced pile-3 Nim line (mover alternates
  `1, -1, 1`, verified against `tests/core/nim_fixture.py`) and the last-step-first
  emission order, making the transformed expectation
  `[{"1:1":1.0},{"-1:1":1.0},{"1:1":1.0}]` unambiguous.
- The plan **does** include a README-verification step (Step 3), so no comment is
  raised for a missing README check.
