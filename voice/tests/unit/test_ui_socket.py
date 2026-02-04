import json
import socket
import tempfile
import time
import threading
from pathlib import Path

from src.ui_socket import UISocketServer


def _short_socket_path() -> Path:
    tmp_dir = tempfile.mkdtemp(prefix="oc_voice_", dir="/tmp")
    return Path(tmp_dir) / "voice.sock"


def test_ui_socket_broadcast():
    socket_path = _short_socket_path()
    server = UISocketServer(path=str(socket_path))
    server.start()

    deadline = time.time() + 1.0
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    while time.time() < deadline:
        if socket_path.exists():
            try:
                client.connect(str(socket_path))
                break
            except ConnectionRefusedError:
                time.sleep(0.01)
                continue
        time.sleep(0.01)
    else:
        raise RuntimeError("Failed to connect to UI socket")

    server.send_event("state_change", {"state": "listening"})
    data = client.recv(4096)

    client.close()
    server.stop()

    payload = json.loads(data.decode("utf-8"))
    assert payload["event"] == "state_change"
    assert payload["payload"]["state"] == "listening"


def test_ui_socket_command():
    socket_path = _short_socket_path()
    server = UISocketServer(path=str(socket_path))
    received = {}
    received_event = threading.Event()

    def handler(command: str, payload: dict):
        received["command"] = command
        received["payload"] = payload
        received_event.set()

    server.on_command = handler
    server.start()

    deadline = time.time() + 1.0
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    while time.time() < deadline:
        if socket_path.exists():
            try:
                client.connect(str(socket_path))
                break
            except ConnectionRefusedError:
                time.sleep(0.01)
                continue
        time.sleep(0.01)
    else:
        raise RuntimeError("Failed to connect to UI socket")

    message = {"type": "command", "command": "shutdown", "payload": {"source": "ui"}}
    client.sendall((json.dumps(message) + "\n").encode("utf-8"))

    assert received_event.wait(1.0)
    assert received["command"] == "shutdown"
    assert received["payload"]["source"] == "ui"

    client.close()
    server.stop()
