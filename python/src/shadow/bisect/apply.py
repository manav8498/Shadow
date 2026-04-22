"""Apply a shadow config (YAML-loaded dict) to a baseline chat_request.

The corner scorer needs to synthesize intermediate requests: take the
baseline's user-facing content (user messages, conversation history) but
plug in a different set of config-derived fields (model, system prompt,
sampling params, tool schemas). This module encapsulates that
transformation so both the scorer and the live replay engine use the
same logic.
"""

from __future__ import annotations

from typing import Any

# The four category-level knobs a shadow config exposes. Corner-level
# bisection attributes divergence across these four levers.
CONFIG_CATEGORIES = ("model", "prompt", "params", "tools")


def apply_config_to_request(config: dict[str, Any], request: dict[str, Any]) -> dict[str, Any]:
    """Return a new chat_request payload with config-derived fields swapped.

    Fields touched:
    - `model` ← `config["model"]` (if set)
    - `params` ← `config["params"]` (if set, whole-replace)
    - `tools` ← `config["tools"]` (if set, whole-replace)
    - The FIRST message, if it has `role == "system"`, has its content
      replaced by `config["prompt"]["system"]` (if set). If the baseline
      request had no leading system message but config provides one,
      it's inserted at position 0.

    User/assistant/tool messages after position 0 are preserved verbatim.
    """
    out = dict(request)

    if "model" in config:
        out["model"] = config["model"]
    if "params" in config:
        out["params"] = dict(config["params"])
    if "tools" in config:
        out["tools"] = [dict(t) for t in config["tools"]]

    system_text = (
        config.get("prompt", {}).get("system") if isinstance(config.get("prompt"), dict) else None
    )
    if system_text is not None:
        messages = [dict(m) for m in request.get("messages", [])]
        if messages and messages[0].get("role") == "system":
            messages[0]["content"] = system_text
        else:
            messages.insert(0, {"role": "system", "content": system_text})
        out["messages"] = messages

    return out


def build_intermediate_config(
    config_a: dict[str, Any],
    config_b: dict[str, Any],
    mask: dict[str, bool],
) -> dict[str, Any]:
    """Return a config dict where each [`CONFIG_CATEGORIES`] key comes from
    `config_b` when `mask[cat]` is True, else `config_a`.

    Only the four category-level fields are considered. Anything else in
    the source configs is ignored (the scorer only varies the four
    knobs).
    """
    out: dict[str, Any] = {}
    for cat in CONFIG_CATEGORIES:
        source = config_b if mask.get(cat, False) else config_a
        if cat in source:
            out[cat] = source[cat]
    return out


def active_categories(config_a: dict[str, Any], config_b: dict[str, Any]) -> list[str]:
    """Return the subset of [`CONFIG_CATEGORIES`] that differ between A and B.

    Order is stable and matches [`CONFIG_CATEGORIES`]. Used as the column
    ordering for the bisection design matrix.
    """
    return [c for c in CONFIG_CATEGORIES if config_a.get(c) != config_b.get(c)]
