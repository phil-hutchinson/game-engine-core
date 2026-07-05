FROM mcr.microsoft.com/devcontainers/python:3.12-bookworm

# CLI pyright is an npm package, so the image needs a node runtime.
# Debian bookworm ships node 18 (EOL); take the current LTS from nodesource instead.
RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Pre-create the Claude Code config dir owned by the container user, so the
# named volume mounted here initializes writable (a bare mount point would be
# root-owned and logins would fail to persist).
RUN mkdir -p /home/vscode/.claude && chown vscode:vscode /home/vscode/.claude

# Bake the heavy `learning` extra dependencies into an image layer so rebuilds
# hit the Docker cache instead of re-downloading them. The editable project
# install itself happens in postCreateCommand — it needs the workspace mount,
# which doesn't exist at image build time. CPU wheels: the container has no GPU
# passthrough, and the default Linux wheels pull in the multi-GB CUDA stack.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu "torch>=2.0" \
    && pip install --no-cache-dir numpy
