#!/usr/bin/env bash
# scripts/start_local_model.sh
# ──────────────────────────────────────────────────────────────────────────────
# Start Ollama and pull the Qwen3-30B-A3B UD-Q2_K_XL model if not present.
# Run once before `eval run` or `pytest`.
#
# Usage:
#   chmod +x scripts/start_local_model.sh
#   ./scripts/start_local_model.sh
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

MODEL="${LOCAL_MODEL_NAME:-qwen3:30b-a3b-ud-q2_k_xl}"
OLLAMA_HOST="${LOCAL_MODEL_BASE_URL:-http://localhost:11434}"
# Strip /v1 suffix for Ollama CLI commands
OLLAMA_BASE="${OLLAMA_HOST%/v1}"

echo "▶  Checking Ollama server at ${OLLAMA_BASE}…"

# Start Ollama serve in background if not running
if ! curl -sf "${OLLAMA_BASE}/api/tags" > /dev/null 2>&1; then
    echo "   Ollama not running — starting in background…"
    ollama serve &
    OLLAMA_PID=$!
    echo "   PID: $OLLAMA_PID"
    # Wait for server to be ready
    for i in $(seq 1 30); do
        curl -sf "${OLLAMA_BASE}/api/tags" > /dev/null 2>&1 && break
        sleep 2
    done
    echo "   Ollama server ready."
else
    echo "   Ollama already running."
fi

# Pull model if not cached
echo "▶  Checking if model '${MODEL}' is available…"
if curl -sf "${OLLAMA_BASE}/api/tags" | grep -q "${MODEL%%:*}"; then
    echo "   Model already pulled."
else
    echo "   Pulling ${MODEL}  (this may take a while for a 2-bit 35B model)…"
    ollama pull "${MODEL}"
    echo "   Pull complete."
fi

echo ""
echo "✅  Local model ready."
echo "   Base URL: ${OLLAMA_BASE}/v1"
echo "   Model:    ${MODEL}"
echo ""
echo "   Set in .env:"
echo "     LOCAL_MODEL_BASE_URL=${OLLAMA_BASE}/v1"
echo "     LOCAL_MODEL_NAME=${MODEL}"
echo "     EVAL_MODEL=local"
