# Repository Guidelines

## Project Structure & Module Organization
- `voice/`: Python voice engine (wake word → STT → gateway → TTS). Source lives in `voice/src/`.
- `voice/scripts/`: one-off utilities (e.g., `probe_gateway.py`).
- `ui/`: SwiftUI overlay app (macOS). Currently a placeholder directory.
- `models/`: local model assets (wake word, whisper, piper).
- Top-level docs: `README.md`, `.plan.md`, `CLAUDE.md`.

## Build, Test, and Development Commands
From repo root:
- `cd voice && python -m venv .venv && source .venv/bin/activate`
  - Create and activate a virtual environment.
- `pip install -e ".[dev]"`
  - Install dependencies and dev tools.
- `python -m src.main`
  - Run the voice engine entrypoint (when implemented).
- `python slice0.py`
  - Run the end-to-end prototype loop.
- `pytest`
  - Run tests.
- `ruff check src/` and `black --check src/`
  - Lint and format checks.

## Coding Style & Naming Conventions
- Python: 4-space indentation, type hints where reasonable.
- Format with `black`; lint with `ruff`.
- Modules in `voice/src/` use `snake_case.py`; classes in `CamelCase`.
- SwiftUI (when added): follow Xcode defaults, `UpperCamelCase` types, 4-space indentation.

## Testing Guidelines
- Framework: `pytest`.
- Prefer unit tests for state logic and pure functions; add functional tests only when hardware/services are required.
- Test files should be named `test_*.py` and live under `voice/tests/`.

## Commit & Pull Request Guidelines
- Commit messages: short, imperative, and scoped (e.g., "Add gateway probe script").
- Keep commits atomic; avoid bundling unrelated changes.
- When implementing a feature, perform atomic commits at regular intervals to keep history clean and allow easy reverts.
- PRs: include a brief summary, test notes, and screenshots for UI changes.
 - Prefer the `auto-commit` skill conventions: `type: description` with types `feat`, `fix`, `refactor`, `docs`, `test`, `chore`, and no Claude/AI footers.

## Collaboration Notes
- Do not ask questions you can look up or answer yourself (how to run app, error logs, etc).

## Configuration & Runtime Notes
- Config is planned via `config.yaml` (see `voice/src/config.py`).
- Gateway expects WebSocket at `ws://127.0.0.1:18789`.
- IPC between UI and engine is planned as a Unix domain socket at `/tmp/openclaw_voice.sock`.
