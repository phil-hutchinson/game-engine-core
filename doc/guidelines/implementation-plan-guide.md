# Implementation Plan Guide

An implementation plan breaks a story into an ordered sequence of steps, each of which builds on the previous and can be verified before proceeding.

## Structure

Each plan is a numbered list of steps. For each step, provide:

- **What to implement** — a high-level description of the change (no code)
- **Why it comes here** — how it depends on prior steps and what later steps depend on it
- **How to verify** — how to confirm this step is working before moving on

## Rules

### Build bottom-up — no forward dependencies

Earlier steps must not depend on anything introduced in a later step. Design the sequence so each step is independently compilable and runnable. If two things are mutually dependent, find the minimal stub that breaks the cycle and introduce it first.

### Stay high-level — no code in the plan

Describe intent, not implementation. Name the types, modules, or interfaces involved, but do not write function signatures, method bodies, or data structure definitions. The plan communicates *what* and *why*; the code communicates *how*.

### Every step must be verifiable

Choose the verification method in this priority order:

1. **Manual test** — the developer runs the application or a targeted script and observes correct behavior directly. Prefer this when the feature has visible output or runtime behavior that can be exercised quickly.
2. **Automated test** — a unit or integration test that can be run with a single command. Use when the behavior is purely internal and a manual test would require significant scaffolding.
3. **Improvised verification** — a temporary diagnostic (e.g. a short script, a debug print, or a REPL session) that confirms the code-so-far is sound. Use only when neither of the above applies, and note in the plan that it is improvised.

Each step's verification description should be concrete enough that the developer knows exactly what to run and what a passing result looks like.

### Include a README check

Every plan must include a step (typically the last) that verifies `README.md` is still accurate given the story's changes — updating it if the story alters anything the README describes (APIs, package layout, setup instructions), or confirming no update is needed. The `/update-readme` command automates this: it reviews the current branch diff and updates `README.md` if warranted.

## Example step format

```
### Step N — <short title>

Implement <what>.

Depends on: Step N-1 (<why>).

Verification (manual): Run `<command>` and confirm <expected output or behavior>.
```

## What a good plan avoids

- Bundling too much into one step — if a step has more than one verification point, split it
- Verification that only checks compilation — the step should demonstrate runtime correctness
- Steps that introduce infrastructure and features together — separate scaffolding from behaviour