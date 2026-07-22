# Story: Add a Capture-Time Policy Transform to SelfPlayCollector (issue #18)

## Goal

Give `SelfPlayCollector` an optional hook that re-expresses each step's MCTS
visit distribution while the collector still holds the `position` it came from:

```python
PolicyTransform = Callable[[TPosition, dict[str, float]], dict[str, float]]

def __init__(
    self,
    evaluator: NeuralNetworkEvaluator[TPosition],
    engine_factory: Callable[[], MCTSEngine[TPly, TPosition, Any]],
    position_factory: Callable[[], TPosition],
    policy_transform: PolicyTransform[TPosition] | None = None,
):
    ...
```

Default `None` = identity: the stored `target_policy` is the raw
`str(ply) -> probability` distribution exactly as today. When supplied, the
transform is applied at the point of capture in `_play_game` — where the
`position` (and therefore its `active_player_id`, `legal_plies`, etc.) is still
in scope — and its result becomes the `target_policy` stored on the
`TrainingSample`.

Unlike stories #10, #11 and #16, this is **additive and non-breaking**: the new
parameter is optional and defaults to identity, so tic-tac-toe, Nim, and every
other player-independent action space are unaffected with no code change.

## Motivation

`TrainingLoop`'s `PolicyLossFn = Callable[[Tensor, Sequence[Mapping[str, float]]], Tensor]`
must align each `str(ply)` target with its column in the policy logits. For a
**perspective-relative action space** that alignment is not derivable from
`str(ply)` alone.

The driving downstream case (a Capture the Flag engine) has a side-to-move
relative action space: the encoder rotates the board 180° when the mover is
Black, and the policy head's `(movement, row, column)` layout is in that same
rotated frame. The `target_policy` keys, however, are global-frame `str(ply)`
(whatever `MCTSEngine.select_ply_with_policy` returns, stored verbatim). Mapping
a global-frame ply to its logit column therefore requires `active_player_id` —
and the loss is never given it (no position, no player).

That context cannot be recovered on the consumer side:

- **Can't close over the player.** A shuffled training batch mixes
  White-to-move and Black-to-move samples, so there is no single
  `active_player_id` to bind into the loss — it must be per-sample.
- **Can't recover it post-hoc.** `TrainingSample` doesn't carry it, and the
  encoding is deliberately colour-blind (always "our" planes vs "their"
  planes), so nothing in `encoded_position` reveals which side moved.

The value target dodges this because `SelfPlayCollector._play_game` has
*already* resolved the outcome into the mover's perspective (the sign
alternation over the reversed step records). `ValueLossFn` just compares
scalars — no frame to reconcile. Policy is a structured distribution over a
perspective-relative action space, so it still has to be re-expressed in the
mover's frame before it can be compared to the logits. The collector is the last
place that reframing can happen with the position in hand.

This is the same class of gap as story #16 (`pass-position-in-decode-policy`): a
game-specific extension point that was too narrow to interpret state in a
perspective-relative / player-dependent game. `decode_policy` was fixed by
handing it the `position`; this fixes the symmetric seam on the *training-target*
side.

### Why the hook, not a wider contract

The rejected alternative is to extend `TrainingSample` with a
`perspective`/`active_player_id` field **and** widen `PolicyLossFn` to receive
it. That works, but it changes two public contracts and pushes the
game-specific frame math into every consumer instead of resolving it once at
capture. The transform hook is smaller and keeps the loss contract clean:

- the stored `target_policy` is already in the mover's frame,
- `PolicyLossFn(logits, policies)` becomes genuinely player-independent — its
  existing signature is honored exactly as written,
- `TrainingSample` and `PolicyLossFn` need no change.

## Scope

- `SelfPlayCollector.__init__`: new optional `policy_transform` parameter
  (default `None` = identity), stored on the instance.
- `SelfPlayCollector._play_game`: when a transform is set, apply it to
  `(position, policy)` at the `step_records.append(...)` seam, storing the
  transformed distribution; when `None`, store the raw distribution as today.
- A `PolicyTransform` type alias exported alongside the collector, mirroring how
  `PolicyLossFn` / `ValueLossFn` are defined in `training_loop.py`.
- Class docstring updated to document the new parameter and the capture-time
  contract.
- Tests in `tests/learning/test_self_play_collector.py` covering both the
  identity default (unchanged behaviour) and a supplied transform (distribution
  re-keyed using the position).

## Non-goals

- **No change to `TrainingSample` or `PolicyLossFn`** — the whole point is that
  neither contract moves.
- **No change to `NeuralNetworkEvaluator`**, `encode_position`, `decode_policy`,
  or the policy head's logit layout. This story only reframes the *training
  target*, not the network's I/O conventions.
- **The story-12 per-game record is out of scope.** The collector discards the
  starting placement and terminal outcome type today, and the same
  "collector is the only place with the position" argument will eventually
  motivate an optional per-game game-record callback. Its exact fields are still
  soft, so this story does **not** build it — but see the note below.

## Note: shape the seam to grow

`_play_game` already loops per game, and that per-game boundary is the natural
emit point for a future game-record callback (starting placement, terminal
outcome type — story 12's diagnostics). This story should not build that record,
but should leave the capture-time-hook pattern and a short doc note in place so
the same gap isn't rediscovered a third time:

> Game-specific reading of a recorded step (frame-correcting the policy,
> extracting provenance, tagging the outcome) must happen at capture, via a
> hook — the collector is the last place the step still has its position.
