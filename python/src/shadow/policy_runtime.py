"""Runtime policy enforcement for ``shadow.sdk.Session``.

The post-hoc :func:`shadow.hierarchical.policy_diff` evaluates rules
against a fully recorded trace. Runtime enforcement runs the SAME
rules incrementally as turns are recorded, so a violation can BLOCK
the session before it propagates downstream.

Design follows the canonical guardrails-API shape used by
NeMo Guardrails / Bedrock Guardrails / Guardrails AI:

- a **callback / hook** evaluates each turn as it lands
- the result is a :class:`Verdict` carrying ``allow``, an optional
  ``replacement`` payload, ``reason``, and the underlying violations
- default mode is ``replace`` — the session records a refusal in
  place of the offending response, the trace remains structurally
  valid, and the user's program keeps running
- ``raise`` mode is opt-in for callers who want hard failure
- ``warn`` mode logs and proceeds without modifying the response

The hook is rule-agnostic: every kind in :mod:`shadow.hierarchical`'s
``_POLICY_KINDS`` works at runtime as long as it can be evaluated on
a partial trace. Stateful rules (``must_remain_consistent``,
``must_followup``) and RAG grounding (``must_be_grounded``) are
runtime-checkable; ``must_call_before`` / ``must_call_once`` /
``no_call`` / ``max_turns`` / ``required_stop_reason`` /
``max_total_tokens`` / ``must_include_text`` / ``forbidden_text`` /
``must_match_json_schema`` all work too — the engine just feeds the
in-progress record list to ``policy_diff`` machinery.

A trigger that fires on the LAST recorded pair (``must_followup``)
fires its violation only when the session closes — at runtime, the
rule queues an obligation that the next turn may or may not satisfy.
:meth:`EnforcedSession.__exit__` performs a final pass to flush any
unmet obligations.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from shadow.errors import ShadowError
from shadow.hierarchical import PolicyRule, PolicyViolation, check_policy, load_policy
from shadow.sdk.session import Session

log = logging.getLogger("shadow.policy_runtime")

OnViolation = Literal["replace", "raise", "warn"]


class PolicyViolationError(ShadowError):
    """Raised by :class:`EnforcedSession` when ``on_violation='raise'``."""

    def __init__(self, violations: list[PolicyViolation]) -> None:
        self.violations = violations
        details = "; ".join(f"{v.rule_id} [{v.kind}/{v.severity}]: {v.detail}" for v in violations)
        super().__init__(f"policy violation(s) at runtime: {details}")


@dataclass
class Verdict:
    """The verdict returned by an enforcement check.

    - ``allow=True`` — no new violations; the session proceeds unchanged.
    - ``allow=False`` with a ``replacement`` — return-replacement mode:
      the caller substitutes the replacement payload for the offending
      response/request and records that instead.
    - ``allow=False`` with ``replacement=None`` — caller must decide
      (raise, warn, or surface the verdict to its own UI).
    """

    allow: bool
    replacement: dict[str, Any] | None = None
    reason: str = ""
    violations: list[PolicyViolation] = field(default_factory=list)


# Type alias for a custom replacement-builder callback.
ReplacementBuilder = Callable[[list[PolicyViolation], dict[str, Any]], dict[str, Any]]


def default_replacement_response(
    violations: list[PolicyViolation], original: dict[str, Any]
) -> dict[str, Any]:
    """Build a default refusal-style ``chat_response`` payload that
    replaces an offending response while preserving structural fields
    (model, usage, latency_ms) that downstream renderers expect.
    """
    reasons = [v.detail for v in violations]
    refusal = (
        "[shadow.policy_runtime] response blocked by policy "
        f"({len(violations)} violation(s)): " + "; ".join(reasons)
    )
    out: dict[str, Any] = dict(original) if isinstance(original, dict) else {}
    out["content"] = [{"type": "text", "text": refusal}]
    out["stop_reason"] = "policy_blocked"
    # Preserve usage / latency / model so the trace's downstream axes
    # still have something to align on. Defaults keep the schema valid.
    out.setdefault("model", out.get("model") or "policy-enforced")
    out.setdefault(
        "usage",
        out.get("usage") or {"input_tokens": 0, "output_tokens": 0, "thinking_tokens": 0},
    )
    out.setdefault("latency_ms", out.get("latency_ms") or 0)
    return out


class PolicyEnforcer:
    """Stateful policy evaluator that knows what was already in
    violation, so it surfaces only NEW violations introduced by the
    most recently appended record(s).

    The enforcer is independent of :class:`EnforcedSession` so callers
    who don't use Shadow's :class:`~shadow.sdk.Session` directly (e.g.
    code already integrated with another tracing layer) can still
    enforce. Pass the enforcer the records list as it grows; the
    enforcer reports the delta.
    """

    def __init__(
        self,
        rules: list[PolicyRule],
        *,
        on_violation: OnViolation = "replace",
        replacement_builder: ReplacementBuilder | None = None,
    ) -> None:
        self._rules = list(rules)
        self._on_violation = on_violation
        self._replacement_builder = replacement_builder or default_replacement_response
        # Track violation identities we've already reported so the same
        # whole-trace violation doesn't re-fire on every subsequent turn.
        # Key is (rule_id, pair_index) — NOT including detail. Whole-
        # trace rules (max_turns, must_call_once) embed running counts
        # in their detail string, so detail-keyed dedup let those rules
        # spam a "new" violation every turn after the first crossing.
        # The detail is human-output, not identity.
        self._known: set[tuple[str, int | None]] = set()

    @classmethod
    def from_policy_file(
        cls,
        path: Path | str,
        *,
        on_violation: OnViolation = "replace",
        replacement_builder: ReplacementBuilder | None = None,
    ) -> PolicyEnforcer:
        """Load policy YAML/JSON from disk and construct an enforcer."""
        import json as _json

        from shadow.errors import ShadowConfigError

        p = Path(path)
        text = p.read_text()
        if p.suffix.lower() in (".yaml", ".yml"):
            try:
                import yaml
            except ImportError as e:  # pragma: no cover — ships transitively
                raise ShadowConfigError("pyyaml is required to load YAML policies") from e
            raw = yaml.safe_load(text)
        else:
            raw = _json.loads(text)
        rules = load_policy(raw)
        return cls(
            rules,
            on_violation=on_violation,
            replacement_builder=replacement_builder,
        )

    @property
    def on_violation(self) -> OnViolation:
        return self._on_violation

    def evaluate(self, records: list[dict[str, Any]]) -> Verdict:
        """Evaluate the rules against the trace-so-far and return only
        the violations that are newly introduced since the previous
        ``evaluate`` call. The same enforcer instance can be reused
        across many calls; state is the cumulative known set.

        The returned Verdict is ``allow=True`` when there are no new
        violations. When violations exist, ``replacement`` is built by
        the configured builder unless ``on_violation='warn'`` (warn
        records the original response unchanged).
        """
        all_violations = check_policy(records, self._rules)
        new_violations: list[PolicyViolation] = []
        for v in all_violations:
            key = (v.rule_id, v.pair_index)
            if key in self._known:
                continue
            self._known.add(key)
            new_violations.append(v)
        if not new_violations:
            return Verdict(allow=True)

        # Build replacement only when the caller wants it. `warn` keeps
        # the original; `raise` doesn't need a replacement (the call
        # site won't reach the recording step). `replace` always builds.
        last_response: dict[str, Any] = {}
        for rec in reversed(records):
            if rec.get("kind") == "chat_response":
                last_response = rec.get("payload") or {}
                break
        replacement: dict[str, Any] | None = None
        if self._on_violation == "replace":
            replacement = self._replacement_builder(new_violations, last_response)
        return Verdict(
            allow=False,
            replacement=replacement,
            reason="; ".join(v.detail for v in new_violations),
            violations=new_violations,
        )

    def probe(self, records: list[dict[str, Any]]) -> Verdict:
        """Like :meth:`evaluate` but does NOT mutate ``_known``. Use this
        for pre-dispatch checks where the candidate may or may not end
        up in the trace.

        Pre-tool-call enforcement (see :func:`wrap_tools`) appends a
        synthesised ``tool_call`` record to the records list and calls
        ``probe`` to ask the enforcer "if this tool call happened, would
        any rule fire?" If the verdict denies, the candidate is never
        recorded — so we don't want the probe to remember that
        violation as "already reported." When the probe allows, the
        caller dispatches the tool for real and the next ``evaluate``
        call (e.g. from the next ``record_chat``) sees the actual
        records and correctly reports any violations once.
        """
        snapshot = set(self._known)
        verdict = self.evaluate(records)
        self._known = snapshot
        return verdict


class EnforcedSession(Session):
    """A :class:`Session` that runs policy evaluation incrementally on
    every recorded chat turn. The output trace is the same
    ``.agentlog`` shape — a violation either replaces the offending
    response (``replace`` mode), raises :class:`PolicyViolationError`
    (``raise`` mode), or only logs (``warn`` mode).

    Usage::

        from shadow.policy_runtime import EnforcedSession, PolicyEnforcer

        enforcer = PolicyEnforcer.from_policy_file("policy.yaml")
        with EnforcedSession(output_path="run.agentlog", enforcer=enforcer) as s:
            s.record_chat(request=..., response=...)

    The session's existing ``record_tool_call`` / ``record_tool_result``
    methods are unaffected — they record without invoking the
    enforcer. Tool-side enforcement happens implicitly via
    ``record_chat`` which sees the tool's effect on the next turn.
    """

    def __init__(
        self,
        *,
        enforcer: PolicyEnforcer,
        **session_kwargs: Any,
    ) -> None:
        super().__init__(**session_kwargs)
        self._enforcer = enforcer

    def record_chat(
        self,
        request: dict[str, Any],
        response: dict[str, Any],
        parent_id: str | None = None,
    ) -> tuple[str, str]:
        # Record the chat normally first so the enforcer sees the new
        # turn in context. If the verdict denies, we either swap the
        # last record's payload (replace) or roll back + raise.
        req_id, resp_id = super().record_chat(request, response, parent_id=parent_id)
        verdict = self._enforcer.evaluate(self._records)
        if verdict.allow:
            return req_id, resp_id

        if self._enforcer.on_violation == "raise":
            # Pop the just-recorded chat pair so the trace doesn't
            # carry the violating response. The caller's exception
            # handler will see a trace ending at the previous turn.
            self._records.pop()  # response
            self._records.pop()  # request
            raise PolicyViolationError(verdict.violations)

        if self._enforcer.on_violation == "warn":
            log.warning(
                "shadow.policy_runtime: %d new violation(s); not modifying response: %s",
                len(verdict.violations),
                verdict.reason,
            )
            return req_id, resp_id

        # replace mode (default): swap the recorded response payload
        # with the verdict's replacement. The recorded request stays
        # as-is; the chat_response record's payload + content_id are
        # rebuilt from the replacement so the trace remains valid.
        if verdict.replacement is None:  # defensive — should not happen
            return req_id, resp_id
        from shadow import _core

        replacement_payload = self._redact(verdict.replacement)
        new_resp_id = _core.content_id(replacement_payload)
        new_resp_record = self._envelope(
            "chat_response", replacement_payload, new_resp_id, parent=req_id
        )
        # Replace the last record (the original response) with the new one.
        self._records[-1] = new_resp_record
        return req_id, new_resp_id

    def wrap_tools(
        self,
        tools: dict[str, Callable[..., Any]],
    ) -> dict[str, GuardedTool]:
        """Convenience: wrap a tool registry against this session's
        enforcer. Equivalent to ``wrap_tools(tools, self._enforcer,
        session=self)``."""
        return wrap_tools(tools, self._enforcer, session=self)


# ---- pre-dispatch tool-call enforcement --------------------------------


# Default placeholder returned in `replace` mode when a tool call is
# blocked. Tool functions can return any type, so we hand back a string
# the caller can pattern-match on. Custom replacement-builders can
# override.
_DEFAULT_BLOCKED_TOOL_RETURN = "[shadow.policy_runtime] tool call blocked by policy"


class GuardedTool:
    """A tool function wrapped with pre-dispatch policy enforcement.

    When invoked, the wrapper:

    1. Synthesises a candidate ``tool_call`` record from the call's
       ``args``/``kwargs`` and the wrapper's tool name.
    2. Asks the enforcer to ``probe(records + [candidate])`` —
       non-mutating, so a denied probe doesn't leave state behind.
    3. On ``allow``: calls the underlying tool function, returns its
       result.
    4. On deny: behaviour depends on the enforcer's ``on_violation``:
       - ``raise``: raises :class:`PolicyViolationError`
       - ``replace``: returns a placeholder (custom builder if set)
       - ``warn``: logs and calls the underlying function anyway

    The wrapper does NOT itself record the ``tool_call`` /
    ``tool_result`` pair — that's the caller's job, via
    :meth:`Session.record_tool_call` and :meth:`record_tool_result`.
    Recording on success keeps Shadow's content-addressing pure (the
    real tool-result content-id ends up in the trace, not a synthesised
    probe id).
    """

    def __init__(
        self,
        name: str,
        fn: Callable[..., Any],
        enforcer: PolicyEnforcer,
        *,
        session: Session | None = None,
        records_provider: Callable[[], list[dict[str, Any]]] | None = None,
        blocked_replacement: Any = None,
    ) -> None:
        self.name = name
        self.fn = fn
        self._enforcer = enforcer
        self._session = session
        self._records_provider = records_provider
        self._blocked_replacement = (
            blocked_replacement if blocked_replacement is not None else _DEFAULT_BLOCKED_TOOL_RETURN
        )
        if session is None and records_provider is None:
            raise ValueError(
                "GuardedTool needs either a `session` to read records from "
                "or an explicit `records_provider` callable"
            )

    def _records_so_far(self) -> list[dict[str, Any]]:
        if self._session is not None:
            return list(self._session._records)
        assert self._records_provider is not None  # checked in __init__
        return list(self._records_provider())

    def _build_probe(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
        """Build a synthesised ``tool_call`` record for the candidate
        invocation. ``arguments`` is a best-effort merge of positional
        and keyword args — most tool functions are kwarg-only in
        production code, so positional args are stuffed under
        ``"args"`` and the rule-evaluators fall back gracefully.
        """
        from shadow import _core

        arguments = dict(kwargs)
        if args:
            arguments["args"] = list(args)
        # Synthesise a stable-enough tool_call_id. Real call ids come
        # from the LLM provider; a probe won't have one, so we mint a
        # deterministic-ish placeholder. The id is opaque to rules.
        import uuid

        call_id = f"probe-{uuid.uuid4().hex[:12]}"
        payload = {
            "tool_name": self.name,
            "tool_call_id": call_id,
            "arguments": arguments,
        }
        records = self._records_so_far()
        parent = records[-1]["id"] if records else "sha256:none"
        return {
            "version": "0.1",
            "id": _core.content_id(payload),
            "kind": "tool_call",
            "ts": "1970-01-01T00:00:00.000Z",  # not part of content_id
            "parent": parent,
            "payload": payload,
        }

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        records = self._records_so_far()
        probe_record = self._build_probe(args, kwargs)
        verdict = self._enforcer.probe([*records, probe_record])
        if verdict.allow:
            return self.fn(*args, **kwargs)

        if self._enforcer.on_violation == "raise":
            raise PolicyViolationError(verdict.violations)
        if self._enforcer.on_violation == "warn":
            log.warning(
                "shadow.policy_runtime: tool `%s` would be blocked but warn "
                "mode is set; calling anyway. Violations: %s",
                self.name,
                verdict.reason,
            )
            return self.fn(*args, **kwargs)
        # replace mode: return the placeholder. We do NOT call the real
        # tool. Callers can supply a richer replacement via the
        # `blocked_replacement` constructor arg.
        log.info(
            "shadow.policy_runtime: tool `%s` blocked pre-dispatch: %s",
            self.name,
            verdict.reason,
        )
        return self._blocked_replacement


def wrap_tools(
    tools: dict[str, Callable[..., Any]],
    enforcer: PolicyEnforcer,
    *,
    session: Session | None = None,
    records_provider: Callable[[], list[dict[str, Any]]] | None = None,
    blocked_replacement: Any = None,
) -> dict[str, GuardedTool]:
    """Wrap a tool registry with pre-dispatch policy enforcement.

    Each entry in ``tools`` becomes a :class:`GuardedTool` that
    enforces the policy BEFORE invoking the underlying function —
    catches ``no_call``, ``must_call_before``, ``must_call_once`` and
    similar tool-sequence rules at the dispatch site, not after the
    tool has already executed. Critical for dangerous tools like
    ``issue_refund``, ``send_email``, ``execute_sql``,
    ``delete_user``, ``deploy_service``.

    Either supply a ``session`` (the wrapper reads ``session._records``
    on each call) or an explicit ``records_provider`` callable. The
    convenience method :meth:`EnforcedSession.wrap_tools` passes the
    session automatically.

    ``blocked_replacement`` overrides the default placeholder string
    that ``replace`` mode hands back when a call is blocked.

    Returns a dict mirroring the input — caller dispatches with
    ``guarded[name](*args, **kwargs)``.
    """
    return {
        name: GuardedTool(
            name,
            fn,
            enforcer,
            session=session,
            records_provider=records_provider,
            blocked_replacement=blocked_replacement,
        )
        for name, fn in tools.items()
    }


__all__ = [
    "EnforcedSession",
    "GuardedTool",
    "OnViolation",
    "PolicyEnforcer",
    "PolicyViolationError",
    "ReplacementBuilder",
    "Verdict",
    "default_replacement_response",
    "wrap_tools",
]
