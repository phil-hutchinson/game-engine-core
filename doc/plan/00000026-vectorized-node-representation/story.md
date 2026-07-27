# Story: Vectorised node representation (issue #26)

Part of the **Fleet Play Phase Two** epic (#25).

## Goal

Replace the per-child object representation with **parallel arrays on the
parent**. Instead of a parent holding `children: list[MCTSNode]` where each child
is an object carrying its own scalar `prior`, `visits`, and `total_value`, the
parent holds those statistics as arrays indexed by legal-ply slot:

```
priors[slot]        # the policy, dense and positional (replaces the policy dict)
visits[slot]
total_value[slot]
```

PUCT selection over all children becomes one vectorised array computation (score
vector, then `argmax`) instead of a Python loop calling `puct_value()` on each
child object. Child `MCTSNode` objects are materialised **lazily**, only when the
search first descends into a slot.

## Motivation

After the skateboard, the network forward is batched across the fleet, so the
residual cost is the tree-internal CPU work — and PUCT descent is the hot part of
it. Today `_select_leaf` does `max(current.children, key=lambda c: c.puct_value())`,
a Python-level loop over child objects at every node of every descent, in every
tree, in every wave. Expressing the children as arrays turns each node's
selection into a single vectorised op and stops allocating node objects for
children that are never visited.

It also tidies the representation that full expansion (#21) introduced: full
expansion attaches all children at once with priors, which is natural to store as
a dense array and wasteful to store as N eagerly-allocated objects. The
`policy: dict[str, float]` on the node disappears — the priors array *is* the
policy, positional instead of string-keyed.

## Scope

- Store child priors/visits/total_value as parallel arrays on the parent, indexed
  by legal-ply slot.
- Vectorise PUCT selection over those arrays.
- Materialise child nodes lazily on first descent into a slot.
- Remove the `policy` dict from the node in favour of the priors array; reconcile
  the `str(ply)` ↔ slot mapping at the protocol boundary.

## Non-goals

- No change to search *results* — this is a representation/performance refactor,
  not a strength change.
- No change to the fleet loop or the batched protocols (#22, #23); this is
  internal to the tree.
- No move to a DAG / transposition-sharing structure — nodes remain a tree.
