# Story: Fleet tournament play (issue #31)

Part of the **Fleet Play Phase Two** epic (#25).

## Goal

Extend the fleet path beyond self-play so that **non-learning** play — running
many games for tournament / evaluation purposes — can also batch its searches
across games and saturate the GPU. Bring `StandardGame` (or a fleet counterpart)
onto the fleet engine's plural search path.

**Open question raised by #23.** This story was written expecting a plural *play*
form, `select_plies`. #23 shipped only `select_plies_for_training` and explicitly
dropped `select_plies` "for want of a consumer" — an assessment that overlooked this
story, which is exactly that consumer. The training form is not a drop-in: it is named
for training, it builds bare roots every call (no retention, per #30), and it returns a
visit distribution this story has no use for. So this story must first settle whether it
adds a plural play form, generalises the training form, or drives N engines — which
means the "no new engine capability" non-goal below may not survive.

## Motivation

The skateboard aims the fleet engine at self-play only: `SelfPlayCollector` is
the driver (#24), and `StandardGame.run` remains a single-game loop calling
`player.select_ply` one position at a time. But the batching win is not specific
to learning — running a tournament of many games (a new checkpoint against a
baseline, a round-robin, strength sampling) has exactly the same shape and the
same idle-GPU problem when played one game at a time. This story generalises the
fleet driver so evaluation runs benefit too.

## Scope

- A fleet driver for tournament play that advances N games in lockstep through
  the engine's plural search path, analogous to `SelfPlayCollector` but emitting
  game results rather than training samples. See the open question above: that
  path does not exist under a play-appropriate name as of #23.
- Reconcile with `StandardGame`'s concerns (players, logging, UI, result
  reasons) under batched play — likely a headless fleet variant, since per-move
  UI rendering does not fit lockstep advancement.
- Slot alignment and long-tail handling as in self-play (and composable with #29
  refill / #28 backfill if present).

## Non-goals

- No interactive / UI-driven play in the fleet path — fleet tournament play is
  headless by nature.
- No new engine capability — this consumes the phase-1 fleet engine (#23), it
  does not extend it. **Provisional**: see the open question under Goal. #23 shipped
  no plural *play* form, so this non-goal holds only if one of the alternatives
  there works out; settle that before planning this story.
- No change to the self-play driver (#24).
