#!/usr/bin/env python3
"""
Probe OpenClaw Gateway to verify protocol understanding.

Protocol (from OpenClaw source):
- Request:  {type:"req", id, method, params}
- Response: {type:"res", id, ok, payload|error}
- Event:    {type:"event", event, payload, seq?}

Key methods:
- chat.send:    {sessionKey, message, idempotencyKey, timeoutMs}
- chat.history: {sessionKey}

Usage:
    python scripts/probe_gateway.py
    python scripts/probe_gateway.py --send "Hello"
"""

import asyncio
import json
import argparse
import uuid
from datetime import datetime
import os
from pathlib import Path


def timestamp():
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


def log(msg):
    print(f"[{timestamp()}] {msg}")


PROTOCOL_VERSION = 3


def load_gateway_token() -> str | None:
    token = os.environ.get("OPENCLAW_GATEWAY_TOKEN")
    if token:
        return token

    config_path = os.environ.get(
        "OPENCLAW_CONFIG_PATH", str(Path.home() / ".openclaw" / "openclaw.json")
    )
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        return config.get("gateway", {}).get("auth", {}).get("token")
    except FileNotFoundError:
        return None
    except Exception:
        return None


async def probe_gateway(url: str, message: str | None = None):
    """Connect to Gateway and test the protocol."""
    try:
        import websockets
    except ImportError:
        print("Install websockets: pip install websockets")
        return

    log(f"Connecting to {url}...")

    try:
        async with websockets.connect(url) as ws:
            log("Connected! Waiting for challenge...")

            # Wait for connect.challenge
            challenge = await asyncio.wait_for(ws.recv(), timeout=5.0)
            challenge_data = json.loads(challenge)
            log(f"Received: {challenge_data.get('event', challenge_data.get('type', 'unknown'))}")
            log(f"Challenge payload: {json.dumps(challenge_data.get('payload', {}))}")

            if challenge_data.get("event") == "connect.challenge":
                challenge_payload = challenge_data.get("payload", {})
                nonce = challenge_payload.get("nonce")
                ts = challenge_payload.get("ts")

                gateway_token = load_gateway_token()
                auth_token = gateway_token
                if not auth_token:
                    log("Missing gateway token. Set OPENCLAW_GATEWAY_TOKEN or run onboarding.")
                    return

                # Respond with connect request (operator role, loopback)
                connect_req = {
                    "type": "req",
                    "id": str(uuid.uuid4()),
                    "method": "connect",
                    "params": {
                        "minProtocol": PROTOCOL_VERSION,
                        "maxProtocol": PROTOCOL_VERSION,
                        "client": {
                            "id": "openclaw-probe",
                            "version": "0.1.0",
                            "platform": "macos",
                            "mode": "probe",
                        },
                        "role": "operator",
                        "scopes": ["operator.admin", "operator.approvals", "operator.pairing"],
                        "caps": [],
                        "commands": [],
                        "permissions": {},
                        "auth": {"token": auth_token},
                        "locale": "en-US",
                        "userAgent": "openclaw-voice/0.1.0",
                    }
                }
                log(f"Sending connect request...")
                await ws.send(json.dumps(connect_req))

                # Wait for hello-ok
                response = await asyncio.wait_for(ws.recv(), timeout=5.0)
                response_data = json.loads(response)
                log(f"Auth response: ok={response_data.get('ok', 'unknown')}")

                if not response_data.get("ok"):
                    log(f"Auth failed: {response_data.get('error')}")
                    return

            # If message provided, send it
            if message:
                session_key = "voice-test"  # Use a test session
                idempotency_key = str(uuid.uuid4())

                log(f"Sending message: {message}")
                send_req = {
                    "type": "req",
                    "id": 2,
                    "method": "chat.send",
                    "params": {
                        "sessionKey": session_key,
                        "message": message,
                        "idempotencyKey": idempotency_key,
                        "timeoutMs": 30000,
                    }
                }
                await ws.send(json.dumps(send_req))

                # Collect responses and events
                log("Listening for response and events...")
                full_response = ""
                start = asyncio.get_event_loop().time()

                while asyncio.get_event_loop().time() - start < 60:
                    try:
                        msg = await asyncio.wait_for(ws.recv(), timeout=2.0)
                        data = json.loads(msg)

                        msg_type = data.get("type")
                        if msg_type == "res":
                            log(f"Response: ok={data.get('ok')} id={data.get('id')}")
                            if data.get("payload"):
                                log(f"  Payload: {json.dumps(data['payload'])[:200]}")
                        elif msg_type == "event":
                            event_name = data.get("event", "unknown")
                            payload = data.get("payload", {})

                            if event_name == "agent":
                                phase = payload.get("phase")
                                log(f"Event: agent phase={phase}")
                                if phase == "end":
                                    log("Agent finished")
                                    break
                            elif "delta" in event_name or "text" in event_name.lower():
                                text = payload.get("text", payload.get("content", ""))
                                if text:
                                    full_response += text
                                    print(text, end="", flush=True)
                            else:
                                log(f"Event: {event_name}")
                        else:
                            log(f"Unknown message type: {msg_type}")

                    except asyncio.TimeoutError:
                        continue

                if full_response:
                    print()  # newline after streamed text
                    log(f"Full response ({len(full_response)} chars)")

            else:
                # Just probe - list sessions
                log("Probing: sessions.list")
                await ws.send(json.dumps({
                    "type": "req",
                    "id": 3,
                    "method": "sessions.list",
                    "params": {}
                }))

                response = await asyncio.wait_for(ws.recv(), timeout=5.0)
                data = json.loads(response)
                log(f"sessions.list response: ok={data.get('ok')}")
                if data.get("payload"):
                    sessions = data["payload"]
                    if isinstance(sessions, list):
                        log(f"  Found {len(sessions)} sessions")
                    else:
                        log(f"  Payload: {json.dumps(sessions)[:200]}")

    except ConnectionRefusedError:
        log(f"Connection refused - is OpenClaw Gateway running at {url}?")
        log("Start it with: openclaw gateway")
    except asyncio.TimeoutError:
        log("Timeout waiting for response")
    except Exception as e:
        log(f"Error: {type(e).__name__}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Probe OpenClaw Gateway protocol")
    parser.add_argument("--url", default="ws://127.0.0.1:18789", help="Gateway WebSocket URL")
    parser.add_argument("--send", help="Message to send")
    args = parser.parse_args()

    asyncio.run(probe_gateway(args.url, args.send))


if __name__ == "__main__":
    main()
