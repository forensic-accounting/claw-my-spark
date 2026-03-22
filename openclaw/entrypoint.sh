#!/usr/bin/env bash
# =============================================================================
# OpenClaw entrypoint — DGX Spark / Ollama-only configuration
# Waits for Ollama, writes config (if absent), then starts the gateway.
# =============================================================================
set -euo pipefail

OLLAMA_URL="${OLLAMA_BASE_URL:-http://ollama:11434}"
DEFAULT_MODEL="${DEFAULT_MODEL:-qwen3:14b}"
WORKSPACE_DIR="/root/openclaw/workspace"

# ── 1. Wait for Ollama ────────────────────────────────────────────────────────
echo "[entrypoint] Waiting for Ollama at ${OLLAMA_URL} ..."
MAX_WAIT=120
WAITED=0
until curl -sf "${OLLAMA_URL}/api/tags" > /dev/null 2>&1; do
  if [ "${WAITED}" -ge "${MAX_WAIT}" ]; then
    echo "[entrypoint] ERROR: Ollama did not become ready within ${MAX_WAIT}s. Exiting."
    exit 1
  fi
  echo "[entrypoint]   ...not ready yet, retrying in 5s (${WAITED}s elapsed)"
  sleep 5
  WAITED=$((WAITED + 5))
done
echo "[entrypoint] Ollama is ready."

# ── 2. Apply config from project source and ensure workspace exists ───────────
mkdir -p /root/.openclaw "${WORKSPACE_DIR}"
echo "[entrypoint] Applying config from /config/openclaw.json ..."
cp /config/openclaw.json /root/.openclaw/openclaw.json

# ── 3. Optionally pull the default model if not already present ───────────────
if [ "${PULL_DEFAULT_MODEL:-true}" = "true" ]; then
  echo "[entrypoint] Checking whether model '${DEFAULT_MODEL}' is already pulled..."
  PRESENT=$(curl -sf "${OLLAMA_URL}/api/tags" | grep -c "\"${DEFAULT_MODEL}\"" || true)
  if [ "${PRESENT}" -eq 0 ]; then
    echo "[entrypoint] Pulling model '${DEFAULT_MODEL}' from Ollama (this may take a while)..."
    curl -sf -X POST "${OLLAMA_URL}/api/pull" \
      -H "Content-Type: application/json" \
      -d "{\"name\": \"${DEFAULT_MODEL}\"}" \
      --no-buffer | while IFS= read -r line; do
        echo "[pull] ${line}"
      done
    echo "[entrypoint] Model pull complete."
  else
    echo "[entrypoint] Model '${DEFAULT_MODEL}' already present, skipping pull."
  fi
fi

# ── 4. Start OpenClaw gateway ─────────────────────────────────────────────────
echo "[entrypoint] Starting OpenClaw gateway..."
exec openclaw --log-level debug gateway
