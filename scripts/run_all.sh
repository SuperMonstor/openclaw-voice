#!/usr/bin/env bash
set -euo pipefail

# Start gateway in background (log to /tmp)
(openclaw gateway >/tmp/openclaw-gateway.log 2>&1 &)

# Start UI in background
(
  cd /Users/sudarshansk/Documents/dev/openclaw-voice/ui/JarvisOverlay
  swift run JarvisOverlay
) &

# Start voice loop in foreground
cd /Users/sudarshansk/Documents/dev/openclaw-voice/voice
source .venv/bin/activate
python slice0.py
