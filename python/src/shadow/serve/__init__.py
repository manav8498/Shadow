"""Shadow's live-dashboard server.

Small FastAPI app that tails a `.shadow/traces/` directory, re-diffs
baseline vs candidate traces on each change, and streams results to a
minimal HTML+JS UI via WebSocket. Designed for local dev + team
self-hosted instances — no build step, no external services.

Gated behind the `[serve]` extra:

    pip install 'shadow[serve]'
    shadow serve --root .shadow --port 8765
"""

from shadow.serve.app import build_app, serve

__all__ = ["build_app", "serve"]
