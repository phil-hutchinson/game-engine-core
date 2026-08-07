# Claude project context

## Intended audience

This library targets developers building game engines and AI/game-playing systems. Assume a technically sophisticated audience — contributors and users are comfortable with algorithms, data structures, and software design patterns. Avoid over-explaining fundamentals; focus explanations on non-obvious design decisions and domain-specific constraints. When writing user stories, the "user" is a developer integrating this library — stories should reflect developer goals (e.g. implementing a game, wiring up an AI) rather than end-user interactions.

## Conventions

See [CONTRIBUTING.md](CONTRIBUTING.md) for coding conventions (imports, etc.).

## Story Documentation

The folder `doc/plan/{story-name}/` (where the story name can be derived from the branch) will contain the following, as needed. Pad the story number to 8 digits.

- **`story.md`** — the original story describing what was requested
- **`implementation-plan.md`** — the plan describing what was intended to be implemented
- **`\peer-review.md`** - a peer review that also includes status and resolution of peer review items

The folder `doc/proposed_stories/` holds work that has been thought through but not scheduled, and is not yet an issue — typically a follow-up a story surfaced and deliberately deferred, written up so the reasoning does not have to be rederived. One markdown file per proposal; see that folder's `README.md` for what belongs there and how an entry graduates to an issue. The convention was introduced by story 26.

## Implementation Strategy

The **`implementation-plan.md`** will contain one or more steps, each with a testing strategy. Progress through steps one at a time, pausing after each one to receive confirmation from the developer that the step has been implemented correctly and that there are no issues. In the case that the testing is manual, you may provide the developer with a reminder of what needs to be tested and how this can be done. Always check for files that have not been committed before beginning a new step: if there are files that have not been committed, **stop** and verify whether the developer wants to commit the existing files before continuing.

## Creation of Implementation Plans

Before creating or modifying an `implementation-plan.md`, read `doc/guidelines/implementation-plan-guide.md` and follow it exactly.

## Vocabulary

For this repository, the following terms should be used:

**Ply** — a single action taken by one player in a turn-based game. Preferred over "move" to avoid ambiguity: in common usage "move" can mean one player's action *or* a full round of actions by all players. A ply is always unambiguous — it refers strictly to one player's turn.

**Slot** — an index into one search node's parallel `child_*` arrays, one per legal ply at that node. Slots are *within* a single node: `child_plies[slot]`, `child_priors[slot]`, `child_visits[slot]`. A child exists as a slot before it exists as an object, which is what makes lazy materialisation possible.

**Fleet position** — an index into a fleet of independent games searched in lockstep, one per game. Fleet positions are *across* games: `positions[fleet_position]` is that game's position, and the result at the same index is that game's result.

Never call a fleet position a slot. Both are dense integer indices from zero and they appear together in the same expressions, so confusing them backpropagates a value into the wrong game's tree — a failure that produces plausible-looking output rather than an error. Where a batched call is narrowed to a subset of the fleet (only the games still pending), the index into that *narrowed* list is neither of these; leave it unnamed, or call it a batch position, and keep the mapping back to fleet positions explicit.