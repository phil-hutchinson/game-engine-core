# Peer Review — General Cleanup (final)

*This is the review of the changes implemented on `feat/2-general-cleanup`. The general codebase review that spawned this story is [story-peer-review.md](story-peer-review.md); its findings #1, #5–#10 are what this branch implements.*

## Summary

The branch implements the seven in-scope findings from the story peer review: the missing `game_engine_core/game/__init__.py` (packaging, #1), two type-level fixes (`Literal` outcome narrowing in `StandardGame` #5, `Player` protocol bound #6), the `collections.abc` `Callable` import (#10), the `str(ply)` uniqueness docstring on `GamePly` (#9), sample-weighted epoch losses in `TrainingLoop` (#7), and randomised MCTS expansion order (#8). All seven match the implementation plan exactly, land in plan order as one commit each, and the story peer review's Status/Resolution columns were updated as the plan's "After completion" section required. Cross-checks hold up: the `GamePly` docstring's reference to `PositionEvaluation.policy` is accurate, `random` was already imported in `mcts_engine.py`, and the batch-size weighting is mathematically correct for both current loss functions (default `F.mse_loss` and the example's `tictactoe_policy_loss`, both batch means). No Critical or Major issues; two Minor comments below.

## Comments

### Minor

| # | Status | Resolution | Location | Comment | Suggested Change | Code Snippet |
|---|--------|------------|----------|---------|-----------------|--------------|
| 1 | Resolved | Fixed the README Quick-start drift in place: the example now uses the real API (`MCTSEngine(evaluator=..., iterations=...)`, `StandardGame(initial_position=..., players={1: ..., -1: ...}, game_ui=...)`, `game.run()`), cross-checked against `examples/tictactoe/main.py`; also corrected "configurable simulations" to "configurable iterations". Added a "Include a README check" rule to `doc/guidelines/implementation-plan-guide.md` so future plans carry the verification step. | [implementation-plan.md#L17](implementation-plan.md#L17) | The implementation plan contains no step to verify `README.md` is still accurate given the story's changes. None of the seven fixes happens to invalidate the README, but the check matters here precisely because it would have surfaced existing drift: the Quick start example uses `MCTSEngine(num_simulations=200)` (the parameter is `iterations`), passes `position=`/`engine=`/`ui=` keywords and a `players` list to `StandardGame` (actual signature: `initial_position`, a `dict[Literal[1, -1], Player]`, `game_ui`), and calls `game.play()` (the method is `run()`). That drift predates this branch, so it is not a defect of these changes — but a README-verification step in each plan is the mechanism that catches it. | Add a README-verification step to future implementation plans (and to the plan-guide checklist if not already there). Optionally spin the pre-existing Quick-start drift into a small follow-up story or fold it into an upcoming one. | `engine = MCTSEngine(num_simulations=200)` |
| 2 | Resolved | Added the batch-mean requirement to both the `ValueLossFn` and `PolicyLossFn` docstrings, stating that epoch reporting weights each batch's loss by its sample count and is only exact for mean-reduced losses. Module compile-checked; documentation-only change. | [game_engine_learning/training_loop.py#L29-L39](../../../game_engine_learning/training_loop.py#L29-L39) | The finding-#7 fix weights each batch's loss by `len(batch)` and divides by `len(samples)`, which is exact **if** each loss function returns a per-sample mean over its batch. Both current implementations do (`F.mse_loss` defaults to `reduction='mean'`; `tictactoe_policy_loss` ends in `.mean()`), but `ValueLossFn`/`PolicyLossFn` are caller-supplied and their docstrings promise only "scalar loss". A caller plugging in a sum-reduced loss (e.g. `reduction='sum'`, a common choice) would get reported epoch losses inflated by roughly the batch size, with no error. The old mean-of-means code shared this assumption, so this is not a regression — but the new code hard-codes it more explicitly, so it should be stated where implementers look. | Add one line to the `ValueLossFn` and `PolicyLossFn` docstrings: the returned scalar must be the **mean** loss over the batch (not the sum), since epoch reporting weights it by batch size. | `"""(predicted_values, target_values) -> scalar loss. ..."""` |

---

**Findings by severity:** 2 Minor

**Notes verified as correct (no action):**

- **Plan/story/implementation consistency** — every plan step maps to exactly one commit in plan order; nothing was implemented outside the plan; the deferrals (#2, #3, #4, tooling) are consistent with the story's "some changes may require a new story" approach; the story peer review's Status/Resolution columns were updated per the plan's "After completion" section.
- **#1 (packaging)** — `game_engine_core/game/__init__.py` added, empty, consistent with the other package initializers.
- **#5 (`Literal` narrowing)** — the annotation-instead-of-`cast` approach is sound and its trade-off (relies on pyright literal-math evaluation; documented in the resolution) was consciously made; the code comment states the constraint.
- **#6 (`Player` bound)** — `GamePosition[Any]` matches the convention used by `StandardGame`, `MCTSEngine`, and `SelfPlayCollector`; the `Any` import was added.
- **#7 (epoch loss weighting)** — algebra checked: `Σ(mean_i · n_i) / N` is the exact sample-weighted epoch mean; correct for ragged final batches (see comment #2 for the documentation gap).
- **#8 (expansion order)** — single shuffle at `unexplored_plies` initialisation; `pop()` from a shuffled list is uniform without-replacement order; selection and backpropagation untouched as the plan promised.
- **#9 (`GamePly` docstring)** — the stated identity-key uses (policy dicts, visit distributions, training targets) are accurate, and the `PositionEvaluation.policy` cross-reference resolves.
- **#10 (`Callable` import)** — matches `training_loop.py` and CONTRIBUTING.md; `Any` correctly kept in the `typing` import.
