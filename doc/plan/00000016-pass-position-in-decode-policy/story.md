# Story: Pass Position into decode_policy (issue #16)

## Goal

Widen `NeuralNetworkEvaluator.decode_policy` from

```python
decode_policy(self, policy_logits: Tensor, legal_plies: Sequence[TPly]) -> dict[str, float]
```

to

```python
decode_policy(self, policy_logits: Tensor, position: TPosition) -> dict[str, float]
```

Implementations that only need the legal plies keep working unchanged by
reading `position.legal_plies` as their first line; the position itself is now
available for anything else a decoding scheme needs.

This is a **breaking signature change with no compatibility shim** (same
posture as stories #10 and #11): pre-release, so there is no deprecation
period.

## Motivation

The driving case is a downstream project whose `decode_policy` needs
`position.active_player_id` to interpret the policy head's logits (the head is
laid out from the active player's perspective, mirroring the established
`encode_position` convention — see `NeuralNetworkEvaluator`'s docstring — and
decoding needs to know which perspective it's undoing). `legal_plies` alone
has no seam for that.

Passing the whole `GamePosition` rather than adding a second parameter (e.g.
`active_player_id: Literal[1, -1]`) keeps one seam instead of two, and gives
future decoding schemes access to anything else on the position (`outcome`,
`apply_ply`, game-specific fields on a concrete subclass) without another
signature change.

## Scope

- `NeuralNetworkEvaluator.decode_policy`: parameter changed from
  `legal_plies: Sequence[TPly]` to `position: TPosition`; docstring updated to
  document the new parameter and note that implementations needing only the
  legal plies read `position.legal_plies`.
- `NeuralNetworkEvaluator.evaluate_position`: passes `position` to
  `decode_policy` instead of `list(position.legal_plies)`.
- In-repo implementations and their tests updated to the new signature:
  - `examples/tictactoe_learning/tictactoe_nn_evaluator.py` and
    `examples/tictactoe_learning/tests/test_nn_evaluator.py`.
  - `tests/learning/nim_nn.py` (no dedicated direct-call test file for this
    one, per the current test layout — `test_neural_network_evaluator.py`
    exercises it only through `evaluate_position`, which is unaffected in
    behaviour).
- README checked for any `decode_policy` signature shown; updated if so.

## Non-goals

- No change to `encode_position`, which already receives the full position.
- No change to the policy head's logit layout or ordering conventions — this
  story only widens what `decode_policy` receives, not how logits map to
  plies.
- No changes to `PositionEvaluation`, `TrainingLoop`, or `SelfPlayCollector`,
  none of which call `decode_policy` directly.
