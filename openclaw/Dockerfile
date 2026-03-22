# =============================================================================
# OpenClaw — Dockerfile
# Requires Node.js 22+ (OpenClaw hard requirement).
# The entrypoint.sh handles first-run configuration at container start.
# =============================================================================

FROM node:22-slim

LABEL maintainer="local"
LABEL description="OpenClaw AI agent gateway — Ollama-only, no cloud models"

# Install runtime dependencies
# chromium: headless browser for OpenClaw's web/browser tools
# fonts-liberation: prevents "couldn't find a font" crashes in headless Chromium
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      curl \
      ca-certificates \
      git \
      chromium \
      fonts-liberation && \
    rm -rf /var/lib/apt/lists/*

# Wrapper that forces headless + no-sandbox so OpenClaw can launch Chromium
# without a display server. CHROME_PATH tells OpenClaw which binary to use.
RUN printf '#!/bin/sh\nexec /usr/bin/chromium --headless --no-sandbox --disable-dev-shm-usage "$@"\n' \
    > /usr/local/bin/chromium-headless && \
    chmod +x /usr/local/bin/chromium-headless
ENV CHROME_PATH=/usr/local/bin/chromium-headless
# Point Playwright at the already-installed system Chromium to avoid a redundant download
ENV PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH=/usr/bin/chromium
ENV PLAYWRIGHT_BROWSERS_PATH=/usr/bin

# Install OpenClaw and MCP servers globally
RUN npm install -g openclaw@latest @playwright/mcp@latest

# Create persistent directories (will be overridden by volume mounts)
RUN mkdir -p /root/.openclaw /root/openclaw/workspace

# Copy and wire up the entrypoint
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

WORKDIR /root

# OpenClaw gateway default port
EXPOSE 18789

ENTRYPOINT ["/entrypoint.sh"]
