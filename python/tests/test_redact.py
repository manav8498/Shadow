"""Tests for shadow.redact (SPEC §9)."""

from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from shadow.redact import DEFAULT_PATTERNS, Redactor, luhn_valid
from shadow.redact.patterns import CREDIT_CARD_CANDIDATE


def test_openai_key_is_redacted() -> None:
    r = Redactor()
    text = "my key is sk-abcdefghij1234567890xxxxxxxxxxxxxx and that's it"
    out = r.redact_text(text)
    assert "sk-abcdefghij" not in out
    assert "[REDACTED:openai_api_key]" in out


def test_anthropic_key_is_redacted_before_openai_pattern() -> None:
    # Anthropic keys are "sk-ant-..." which would also match the generic
    # OpenAI pattern "sk-...". The anthropic pattern MUST fire first so
    # the REDACTED tag reads "anthropic_api_key", not "openai_api_key".
    r = Redactor()
    text = "token=sk-ant-abcdefghij1234567890xxxxxxxxxxxxxx end"
    out = r.redact_text(text)
    assert "[REDACTED:anthropic_api_key]" in out
    assert "[REDACTED:openai_api_key]" not in out


def test_email_is_redacted() -> None:
    r = Redactor()
    assert r.redact_text("contact me at alice@example.com please") == (
        "contact me at [REDACTED:email] please"
    )


def test_phone_e164_is_redacted() -> None:
    r = Redactor()
    out = r.redact_text("call +14155551234 now")
    assert "[REDACTED:phone]" in out


def test_phone_short_number_is_not_a_refusal() -> None:
    r = Redactor()
    # 5-digit code is below the 10-digit E.164 minimum — don't touch.
    assert "12345" in r.redact_text("code 12345")


def test_credit_card_luhn_valid_is_redacted() -> None:
    r = Redactor()
    # 4111 1111 1111 1111 is a classic Luhn-valid test card.
    assert "[REDACTED:credit_card]" in r.redact_text("card 4111111111111111")


def test_credit_card_luhn_invalid_is_not_redacted() -> None:
    r = Redactor()
    # 13 digits that are NOT Luhn-valid — must pass through untouched.
    text = "number 1234567890123 not a card"
    assert r.redact_text(text) == text


def test_luhn_check_basic() -> None:
    assert luhn_valid("4111111111111111")
    assert luhn_valid("5500000000000004")
    assert not luhn_valid("4111111111111112")  # changed last digit
    assert not luhn_valid("1234")  # too short


def test_redact_value_recurses_into_dict_and_list() -> None:
    r = Redactor()
    payload = {
        "messages": [
            {"role": "user", "content": "email me: alice@example.com"},
        ],
        "nested": {"api": "sk-xxxxxxxxxxxxxxxxxxxxxxxx"},
    }
    out = r.redact_value(payload)
    assert "[REDACTED:email]" in out["messages"][0]["content"]
    assert "[REDACTED:openai_api_key]" in out["nested"]["api"]


def test_allowlist_key_bypasses_redaction() -> None:
    r = Redactor(allowlist_keys=frozenset({"note"}))
    payload = {"note": "alice@example.com", "other": "alice@example.com"}
    out = r.redact_value(payload)
    assert out["note"] == "alice@example.com"  # allowlisted
    assert out["other"] == "[REDACTED:email]"


def test_last_modified_is_true_iff_redaction_fired() -> None:
    r = Redactor()
    r.redact_value({"hello": "world"})
    assert r.last_modified is False
    r.redact_value({"hello": "alice@example.com"})
    assert r.last_modified is True


def test_non_string_scalars_pass_through_unchanged() -> None:
    r = Redactor()
    assert r.redact_value(None) is None
    assert r.redact_value(42) == 42
    assert r.redact_value(3.14) == 3.14
    assert r.redact_value(True) is True


@given(
    st.text(
        alphabet=st.characters(
            blacklist_categories=("Cs",),
            # Exclude digits/@/letters that might accidentally form matching
            # patterns — keeps the property test focused on "random text
            # doesn't trigger spurious redactions."
            blacklist_characters="@0123456789+sk-.ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz",
        ),
        min_size=0,
        max_size=200,
    )
)
def test_property_text_without_pattern_chars_never_redacts(text: str) -> None:
    r = Redactor()
    out = r.redact_text(text)
    for pat in DEFAULT_PATTERNS:
        if pat.name == CREDIT_CARD_CANDIDATE.name:
            continue
        assert f"[REDACTED:{pat.name}]" not in out


def test_metadata_redacted_stamp_reflects_real_work() -> None:
    # Integration-style: Redactor.last_modified is the signal the SDK needs
    # to stamp envelope.meta.redacted = True per SPEC §9.3.
    r = Redactor()
    r.redact_value({"msg": "safe text"})
    assert r.last_modified is False
    r.redact_value({"msg": "call 4111111111111111"})
    assert r.last_modified is True
