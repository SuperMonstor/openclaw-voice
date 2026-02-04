#!/usr/bin/env bash
set -euo pipefail

gateway_pid=""
ui_pid=""

cleanup() {
  if [[ -n "${ui_pid}" ]] && kill -0 "${ui_pid}" 2>/dev/null; then
    kill "${ui_pid}" 2>/dev/null || true
  fi
  if [[ -n "${gateway_pid}" ]] && kill -0 "${gateway_pid}" 2>/dev/null; then
    kill "${gateway_pid}" 2>/dev/null || true
  fi
}

trap cleanup EXIT INT TERM

# Start gateway in background (log to /tmp)
openclaw gateway >/tmp/openclaw-gateway.log 2>&1 &
gateway_pid=$!

# Start UI in background
(
  cd /Users/sudarshansk/Documents/dev/openclaw-voice/ui/JarvisOverlay
  swift run JarvisOverlay
) &
ui_pid=$!

# Start voice loop in foreground
cd /Users/sudarshansk/Documents/dev/openclaw-voice/voice
source .venv/bin/activate
python slice0.py
