"""Unix domain socket server for UI event updates."""

from __future__ import annotations

import json
import os
import queue
import socket
import threading
from typing import Any, Dict, List


class UISocketServer:
    def __init__(self, path: str = "/tmp/openclaw_voice.sock"):
        self.path = path
        self._sock: socket.socket | None = None
        self._clients: List[socket.socket] = []
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None
        self._send_thread: threading.Thread | None = None
        self._send_queue: "queue.Queue[bytes]" = queue.Queue()

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()
        self._send_thread = threading.Thread(target=self._send_loop, daemon=True)
        self._send_thread.start()

    def stop(self):
        self._running = False
        self._send_queue.put(b"")
        if self._sock:
            try:
                self._sock.close()
            except Exception:
                pass
        with self._lock:
            for client in self._clients:
                try:
                    client.close()
                except Exception:
                    pass
            self._clients.clear()
        if os.path.exists(self.path):
            try:
                os.unlink(self.path)
            except Exception:
                pass

    def send_event(self, event: str, payload: Dict[str, Any]):
        message = {"type": "event", "event": event, "payload": payload}
        data = (json.dumps(message) + "\n").encode("utf-8")
        if self._running:
            self._send_queue.put(data)

    def _serve(self):
        if os.path.exists(self.path):
            try:
                os.unlink(self.path)
            except Exception:
                pass
        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._sock.bind(self.path)
        self._sock.listen(5)
        while self._running:
            try:
                client, _ = self._sock.accept()
            except Exception:
                continue
            with self._lock:
                self._clients.append(client)

    def _send_loop(self):
        while self._running:
            try:
                data = self._send_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if not data:
                continue
            dead: List[socket.socket] = []
            with self._lock:
                for client in self._clients:
                    try:
                        client.sendall(data)
                    except Exception:
                        dead.append(client)
                for client in dead:
                    try:
                        client.close()
                    except Exception:
                        pass
                    if client in self._clients:
                        self._clients.remove(client)
