"""Live-provider invariant tests.

Every test in this package hits a real LLM provider and is gated by
``SHADOW_RUN_NETWORK_TESTS=1`` plus the relevant API key. Default PR
CI never sets the gate; a nightly workflow (``.github/workflows/
live-tests.yml``) runs the suite and surfaces shape-drift signals
(token-count drift, response-shape changes, streaming-aggregation
correctness, model-id rotation) before they reach users.

Token budget per full run: ~$0.01 across all tests using
``gpt-4o-mini`` and ``claude-haiku-4-5`` with short prompts.
"""
