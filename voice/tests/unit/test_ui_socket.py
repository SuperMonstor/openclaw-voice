import json
import socket
import tempfile
from pathlib import Path

from src.ui_socket import UISocketServer


def _short_socket_path() -> Path:
    tmp_dir = tempfile.mkdtemp(prefix="oc_voice_", dir="/tmp")
    return Path(tmp_dir) / "voice.sock"


def test_ui_socket_broadcast():
    socket_path = _short_socket_path()
    server = UISocketServer(path=str(socket_path))
    server.start()

    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client.connect(str(socket_path))

    server.send_event("state_change", {"state": "listening"})
    data = client.recv(4096)

    client.close()
    server.stop()

    payload = json.loads(data.decode("utf-8"))
    assert payload["event"] == "state_change"
    assert payload["payload"]["state"] == "listening"
