# Implementation Plan — Containerization and Tooling

Implements [story.md](story.md): a VS Code Dev Container as the within-repo development environment, adoption of pyright and ruff inside it, and the contributor/consumer split of the setup documentation. The container is strictly dev-side — nothing in the published packages may reference or depend on it.

## Steps

### Step 1 — Dev container scaffolding

Create the `.devcontainer/` setup (a devcontainer definition and its container image definition): a Python 3.12+ base, the node runtime that CLI pyright requires, and the VS Code extensions the environment pre-installs (Python, Pylance, Ruff). Include Claude Code in the environment via Anthropic's official dev container feature, with a persistent named volume for its configuration directory so authentication and settings survive container rebuilds. Add a repository-level `.gitattributes` establishing line-ending normalization, so the same checkout presents a clean `git status` to both the Windows host git and the container's Linux git (without it, the two disagree on CRLF conversion and the container reports every text file as modified). No project installation yet — this step is environment scaffolding only, kept separate from behaviour per the plan guide.

Depends on: nothing (everything else in the story runs inside this environment).

Verification (manual): Reopen the repo in the dev container from VS Code. In the container terminal, confirm the Python interpreter reports 3.12+ and the node runtime is present; confirm the Python, Pylance, and Ruff extensions show as installed in the container. Pylance should type-check an open file (e.g. `game_engine_core/engines/mcts_engine.py`) without import errors for the standard library. Confirm `git status` inside the container reports a clean working tree (modulo this story's own files), and that `claude` launches in the container terminal and, after a one-time sign-in, remains authenticated after a container rebuild.

### Step 2 — Project installation as part of container provisioning

Make the container provisioning install the project editable with its `learning` extra (torch), directly into the container's Python — no `.venv`; the container is the isolation boundary. Whether this happens at image build or in a post-create hook is decided here, weighing torch's size against rebuild frequency.

Depends on: Step 1 (there must be a container to provision).

Verification (manual): From a fresh container build with no manual setup steps, run `python -m examples.tictactoe --p1 random --p2 random` and confirm a game runs to completion, then `python -m examples.tictactoe_learning.selfplay` and confirm the self-play diagnostic prints its summary statistics. Both commands working proves the package, its `game` subpackage, and the `learning` extra are all installed.

### Step 3 — Adopt pyright

Add CLI pyright to the dev environment with a minimal `[tool.pyright]` section in `pyproject.toml`, runnable with a single command inside the container. Triage the first full run: findings #5 and #6 from the general-cleanup review were fixed by manual analysis only, so new findings are expected. Apply mechanical, low-risk fixes within this step; record anything behavioural or design-level in a findings note in this story folder for follow-up stories.

Depends on: Step 2 (pyright must see the project and its dependencies installed to resolve imports; the node runtime comes from Step 1).

Verification (manual): Run pyright in the container and confirm it completes with zero errors, or with only the findings explicitly recorded for follow-up. If any mechanical fixes touched code, re-run the two example commands from Step 2 and confirm behaviour is unchanged.

### Step 4 — Adopt ruff

Add ruff to the dev environment with a `[tool.ruff]` section in `pyproject.toml`, including the `TID` and `UP` rule groups that enforce the CONTRIBUTING.md import conventions; the exact rule selection beyond those two is decided here. Same triage policy as Step 3: mechanical fixes in-step, anything larger recorded.

Depends on: Step 2 (ruff runs inside the provisioned container; independent of Step 3, ordered after to keep the two tool adoptions sequential and separately verifiable).

Verification (manual): Run `ruff check .` in the container and confirm it passes clean, or with only recorded findings. Confirm the `TID`/`UP` groups are active by checking the rule configuration is picked up (ruff reports the violated rule codes if any are found). If fixes touched code, re-run the two example commands from Step 2.

### Step 5 — Setup documentation rework and README check

Rework the setup documentation for the two audiences from the story: the README's Requirements/Installation sections become external-consumer documentation only (Python 3.12+, package installation, optional `learning` extra — no mention of containers or venvs), and contributor setup becomes "open the repo in the dev container", documented where contributors will find it (README pointer and/or CONTRIBUTING.md, including how to run pyright and ruff). This step also serves as the plan's required README accuracy check: verify the rest of the README still matches the repo after this story's changes — `/update-readme` automates this against the branch diff.

Depends on: Steps 1–4 (documentation must describe the finished environment and tooling, not an intermediate state).

Verification (manual): Review the rendered README and confirm the consumer instructions stand alone with no container or venv references, and that a contributor following only the documented flow (clone, reopen in container) lands in a working environment — cross-checking each documented command against what Steps 2–4 verified.

## After completion

If Steps 3 or 4 recorded findings for follow-up, ensure each has enough context (rule/error code, location, why it was deferred) for a future story to pick up without re-running the triage.
