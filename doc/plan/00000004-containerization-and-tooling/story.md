# Story: Containerization and Tooling (issue #4)

## Goal

Containerize the development environment so that working on the project requires no local software installation beyond a container runtime, and every developer gets the same versions of every tool. Adopt pyright and ruff as project tooling running inside that environment.

Containerization is strictly a within-repo development setup — it facilitates work on the packages, the examples, and whatever follows (tests, tooling). The packages themselves must remain consumable externally with no reference to, or hidden or soft dependency on, the container environment.

## Motivation

The main objective is a consistent developer experience: no reliance on locally installed interpreters, runtimes, or linters, and no drift between the tool versions on different machines.

Containerization also unblocks the tooling deferred from the general-cleanup story (see the [tooling recommendations](../00000002-general-cleanup/story-peer-review.md) in its peer review):

- **pyright** — the codebase is heavily and carefully typed, but no type checker runs outside the author's editor. CLI pyright requires a node runtime, which is unwanted on the host machine; inside a container that objection disappears.
- **ruff** — mechanically enforces the CONTRIBUTING.md import conventions (relative imports within packages, no deprecated `typing` aliases). `.gitignore` already anticipates `.ruff_cache/`.

## Scope

### 1. Containerized development environment

The intended direction is a VS Code Dev Container, so that the IDE (Pylance), the CLI tools, and the project runtime all share one environment. The environment must support the full existing workflow: installing the project with its `learning` extra (torch) and running the examples (`examples.tictactoe`, `examples.tictactoe_learning.*`). Base image choice, dependency caching, and how the `learning` extra is handled are decided in the implementation plan.

The container becomes the environment's isolation boundary, replacing the repo-local `.venv` — a virtual environment inside a dedicated dev container is redundant, so the venv-based workflow is retired for in-repo development. Project setup (installing the package and its extras) happens as part of the container build rather than as manual steps.

### 2. pyright

Add CLI pyright with a `[tool.pyright]` configuration in `pyproject.toml`, runnable inside the container. Note that peer-review findings #5 and #6 were fixed from manual analysis only — the first full pyright run may surface additional findings. Mechanical, low-risk fixes may be made within this story; anything affecting behaviour or design gets recorded for a follow-up story.

### 3. ruff

Add ruff with a `[tool.ruff]` configuration in `pyproject.toml`, including the `TID` and `UP` rule groups that enforce the CONTRIBUTING.md conventions. The same in-story/follow-up split applies to any findings.

### 4. Peer-review process update

With pyright and ruff available as project tooling, the peer-review command (`.claude/commands/peer-review.md`) is updated to run both before reviewing and to file their findings as review comments.

### 5. Setup documentation rework

The README's venv-based Installation snippets describe contributor setup and are superseded by the container workflow. Rework the setup documentation into two audiences:

- **Contributors** — open the repo in the dev container; the environment is built for them, with no manual setup snippets to follow.
- **External consumers** — what is required to use the packages from another project: Python 3.12+, installing the package, and the optional `learning` extra (PyTorch). This documentation must stand entirely on its own, with no mention of containers.

## Out of Scope

- **pytest and automated testing** — lands with the future test-suite story (general-cleanup peer-review finding #4)
- **CI pipeline** — running pyright/ruff automatically on push/PR is a separate concern from making them runnable consistently; a future story
- **Non-trivial fixes to new tool findings** — recorded for follow-up stories rather than fixed here
