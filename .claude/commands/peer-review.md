Review the current branch diff and produce a peer review document.

Save the document as `doc/plan/$ARGUMENTS/peer-review.md`. If no story name is provided, ask the user for one before proceeding. note that the folder for the story will be closely related to the branch, so it is usually easy to determine if not provided.

The folder `doc/plan/$ARGUMENTS/` should already contain two reference documents — read both before reviewing the diff:

- **`story.md`** — the original story describing what was requested
- **`implementation-plan.md`** — the plan describing what was intended to be implemented

Before reviewing the diff, run `pyright` and `ruff check .` from the repository root. Note the result of each in the review document's Summary. If either reports findings, file each as a review comment at the appropriate severity (a type error is typically Critical or Major; a lint finding is typically Minor) — do not fix them.

The review should cover not just code quality, but also discrepancies between these documents and the actual changes: requirements from the story that are missing or misimplemented, deviations from the implementation plan without apparent justification, and anything implemented that is not covered by either document. Also identify discrepancies between the story and the implementation plan.

Also check that the implementation plan included a step to verify `README.md` is still up to date given the story's changes. If that step is absent, raise it as a comment.

## Document format

Use this structure for the document:

```markdown
# Peer Review — <story name>

## Summary

<2–3 sentence overview of the changes>

## Comments

### Critical

| # | Status | Resolution | Location | Comment | Suggested Change | Code Snippet |
|---|--------|------------|----------|---------|-----------------|--------------|
| 1 | Open | | [path/to/file.py#L42](relative/path/to/file.py#L42) | ... | ... | `code` |

### Major

(same table)

### Minor

(same table)
```

**Severity definitions:**
- **Critical** — correctness bugs, data loss, security issues, broken contracts
- **Major** — logic errors, missing edge cases, significant design problems
- **Minor** — naming, style, small inefficiencies, missing comments

**Per-comment fields:**
- **#** — issue number, incrementing from 1 across all severity sections (do not restart at 1 for each section)
- **Status** — always `Open` initially
- **Resolution** — leave blank
- **Location** — a single markdown link combining file path and line number(s), e.g. `[game_engine_core/engines/mcts_engine.py#L42](../../game_engine_core/engines/mcts_engine.py#L42)` or `#L42-L50` for a range. The link must be a **relative path from the peer review file's location** (`doc/plan/<story>/peer-review.md`), so paths to project source files will typically start with `../../../`.
- **Comment** — what the problem is
- **Suggested Change** — concrete fix or direction
- **Code Snippet** — the relevant lines as an inline code snippet or fenced block

Omit any severity section that has no comments.

## After completing the review

Stop. Present the path to the saved file and a brief count of findings by severity. Do not attempt to fix any of the comments — leave that for the user to action.
