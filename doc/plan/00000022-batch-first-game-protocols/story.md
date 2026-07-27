# Story: Batch-first game protocols (issue #22)

Part of the **Fleet Play** epic (#20).

## Goal

Widen the game-facing protocols so the library always calls a **plural**,
whole-batch form, and the game implementation decides whether to service it with
a real tensor path or a sequential loop. A base/mixin provides the looping
default, so a simple game implements the scalar method and gets the plural form
for free, while a performance-critical game overrides the plural method with a
vectorised implementation.

Touchpoints to widen:

- `GamePosition`: `legal_plies` → batch over a sequence of positions;
  `apply_ply` → `apply_plies`; `outcome` → batch `outcomes`.
- `PositionEvaluator`: `evaluate_position` → `evaluate_positions(positions)`
  returning a sequence of `PositionEvaluation`.
- `NeuralNetworkEvaluator`: `encode_position` → `encode_positions`;
  `decode_policy` → `decode_policies`.
- `SelfPlayCollector`: a batch form of the `policy_transform` hook.

`TrainingLoop`'s `policy_loss_fn` / `value_loss_fn` are already batch-shaped
(they score a whole batch of predictions against targets in
`training_loop.py`), so they need no change — noted here only to record that
they are the same kind of seam.

## Motivation

Amdahl's law. The network forward pass is ≈70% of self-play wall-clock on CPU;
moving only that to a batched GPU call caps the achievable speed-up, because the
residual per-game work then dominates. Profiling puts `encode_position` and the
`policy_transform` next after the forward, with legality generation behind them.
Batching those removes most of the residual overhead.

The wave (issue #23) reinforces this: each wave phase already gathers N of
everything (N leaves to test for terminal, N positions to expand, N to
evaluate). With batch-first protocols each phase is a single bulk call instead
of a Python loop of N calls:

- select leaves → one `outcomes(...)` to partition terminal vs. non-terminal
- expand → one `legal_plies(...)` + one `apply_plies(...)`
- one `evaluate_positions(...)`
- capture → one `encode_positions(...)`, one batched `policy_transform`

The design principle is **batch everything at the protocol boundary, let the
game decide what it processes en masse or sequentially.** A position already
carries small CPU tensors, so a game like Capture the Flag can stack them and
compute legality / encodings for the whole fleet at once; tic-tac-toe can ignore
the opportunity and loop. Offering the batch entry point everywhere — even where
a given game will not use it — is what lets the fast games go fast without
taxing the simple ones.

### `decode_policies` and `policy_transform` — the alignment discipline

Both deal with the perspective-relative action space, at opposite ends. During
search, `decode_policy` maps the head's logits (laid out in the mover's frame)
back to global-frame `str(ply)` priors. At capture, `policy_transform`
re-expresses the MCTS visit distribution into the head's column layout so it can
serve as a training target. Both need `position.active_player_id`, so their
batched forms must carry the N positions alongside the N tensors and pair them
element-by-element (`zip(rows, positions)`) — the same positional-identity
alignment the fleet relies on throughout.

## Scope

- Add the plural methods above to the protocols/ABCs.
- Provide a default implementation (base class or mixin) that services each
  plural method by looping the existing scalar method, so existing games and
  evaluators (tic-tac-toe, Nim) keep working unchanged.
- Provide a real batched `evaluate_positions` on `NeuralNetworkEvaluator`: the
  model's `forward` already accepts `(batch, *shape)`, so this stacks encodings,
  runs one forward, splits the rows, and decodes each — replacing the current
  `unsqueeze(0)` / `squeeze(0)` batch-of-one dance.
- Migrate the library's own call sites (engine, collector) to call the plural
  forms.

## Non-goals

- No vectorising of the tree internals — PUCT descent and backpropagation are
  not game touchpoints and stay per-tree CPU work (epic backlog P1).
- No position hash/equality requirement — that is the evaluation cache's
  contract (epic backlog P2), not this story's.
- No obligation on any game to actually vectorise; the looping default must keep
  simple games correct with zero changes.
- The fleet engine and driver that consume these (issues #23, #24) are separate.
