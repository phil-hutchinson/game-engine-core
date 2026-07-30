# Story: Retention during training decision (issue #30)

Part of the **Fleet Play Phase Two** epic (#25). This is a **decision** story — a
spike plus a call — not a feature.

## Question

When the fleet plays a real ply and moves each game to its next position, should
the search **retain visit statistics** from the previous ply's tree (so the new
search continues the old), or start each ply's search from bare roots?

The skateboard (#23) runs from bare roots and drops retention entirely. This
story decides whether retention-as-strength should come back, specifically **in
learning / self-play mode**.

## Background

Tree retention carries two separable kinds of value:

- **Retention-as-performance** — reusing cached *evaluations*. This is handled
  separately by the position-keyed cache (#27); it is not what this story
  decides.
- **Retention-as-strength** — carrying *visit statistics* forward so the new
  search starts with accumulated knowledge and therefore searches effectively
  deeper. This is the lever in question.

Canonical AlphaZero reuses the played child's subtree, statistics intact, even
during training — the "AlphaZero discards its tree" belief is truer of simplified
reimplementations than the original. But there are real arguments for resetting:
a fresh search per ply is simpler, avoids biasing the new search with stale
statistics gathered under a now-superseded network, and produces cleaner,
more independent visit-distribution training targets.

## What to resolve

- Does retaining visit statistics across plies measurably improve self-play data
  quality / training, enough to justify the complexity in the fleet path?
- If retained, how does it interact with the fleet's per-ply `select_plies_for_training` call
  (which currently rebuilds bare roots) and with tree reuse under a network that
  changes between training iterations?
- Recommendation, with the evidence behind it, feeding back into the #23
  retention stance (and noting any overlap with #27).

## Scope

- A spike: measure bare-roots vs. retained-statistics self-play under otherwise
  equal budgets, on strength and/or training signal.
- A written recommendation and, if adopted, a follow-up implementation story.

## Non-goals

- No production implementation in this story — it decides and, if warranted,
  spawns the work.
- Does not cover retention-as-performance (that is #27).
