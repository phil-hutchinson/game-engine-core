# Implementation Plan: Add a Capture-Time Policy Transform to SelfPlayCollector

Small, additive change. The sequence introduces the hook wired as identity
first (proving the non-breaking claim against the existing suite), then adds a
test that exercises a real transform, then handles docs.

## Step 1 — Add the optional `policy_transform` hook, applied at capture

Introduce a `PolicyTransform` type alias (a callable taking the position and the
raw `str(ply) -> probability` distribution and returning a re-keyed
distribution), defined alongside `SelfPlayCollector` in the same module,
mirroring how `PolicyLossFn` / `ValueLossFn` are declared in `training_loop.py`.
Add an optional `policy_transform` parameter to `SelfPlayCollector.__init__`
defaulting to `None`, stored on the instance. In `_play_game`, at the point
where a step is recorded, apply the transform to `(position, policy)` when one
is set and store its result as the step's policy; when `None`, store the raw
distribution unchanged. Update the class docstring to document the new parameter
and the capture-time contract (the transform runs while the position — and thus
`active_player_id`, `legal_plies` — is still in scope).

Depends on: nothing (entry point).

Verification (automated): Run `pytest tests/learning/test_self_play_collector.py`
and confirm the existing suite still passes unchanged. Because every current
test constructs the collector without a transform, a green run demonstrates the
identity default preserves today's behaviour exactly — i.e. the change is
non-breaking for player-independent action spaces.

## Step 2 — Test a supplied transform

Add a test to `tests/learning/test_self_play_collector.py` that constructs the
collector with a `policy_transform` and asserts the stored `target_policy` is the
transformed distribution rather than the raw one. Use the position to drive the
transform so the test proves the position is genuinely in scope at capture — e.g.
re-key or tag each entry using `position.active_player_id` (which alternates over
the forced Nim line), and assert the resulting keys/values differ per step in the
way the transform dictates. Confirm the raw distribution is `{"1": 1.0}` on the
identity path (already covered) so the transformed expectation is unambiguous.

Depends on: Step 1 (the parameter and capture-time application must exist).

Verification (automated): Run `pytest tests/learning/test_self_play_collector.py`
and confirm the new test passes, showing the transform is applied per step with
the correct position and its output becomes the stored `target_policy`.

## Step 3 — Doc note and README check

Add the short "capture is the last place the step has its position" note near
`SelfPlayCollector` / `TrainingSample` (as a code comment or docstring line) so
the pattern is discoverable and the same gap is not rediscovered a third time —
covering frame-correcting the policy, extracting provenance, and tagging the
outcome as the family of things that must happen at capture via a hook. Then
check `README.md` for anything the change affects: the `SelfPlayCollector`
constructor surface or self-play/training narrative. Update if warranted;
otherwise record that no update is needed. This is additive (new optional
parameter), so a README update is only needed if the README enumerates the
collector's constructor arguments.

Depends on: Steps 1–2 (documents the finished behaviour).

Verification (manual): Confirm the doc note reads correctly in context, and run
`/update-readme` (or review the branch diff against `README.md`) to confirm the
README is either updated to match the new constructor surface or verified as
already accurate. Re-run `pytest tests/learning/` to confirm the full learning
suite is green before closing the story.
