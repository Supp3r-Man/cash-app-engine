import json
import sys

import pytest

from cashapp.classify import reason_from_text
from cashapp.remit import LLMRemitParser, RuleRemitParser, validate_llm_json, LLMParseError

P = RuleRemitParser()


def test_classifier_codes_words_and_unknowns():
    assert reason_from_text("INV 1234567 DED DC01 45.00") == "PRICING"
    assert reason_from_text("DMG GOODS") == "DAMAGED"
    assert reason_from_text("SHORT SHIPPED") == "SHORTAGE"
    assert reason_from_text("FRT CHARGE") == "FREIGHT"
    assert reason_from_text("PROMO ALLOW") == "PROMO"
    assert reason_from_text("RMA CREDIT") == "RETURN"
    assert reason_from_text("DEDUCTION DC99") == "OTHER"  # a deduction, but a code nobody told us about
    assert reason_from_text("INV 1234567") is None  # not a deduction at all


def test_rules_parser_reads_listed_amount_and_deduction():
    r = P.parse("ACH-0042918837", "INV 4821937 1,250.00 DED DC01 45.00")
    assert [t.digits for t in r.tokens] == ["4821937"]
    assert r.amounts["4821937"] == 125_000
    assert len(r.deductions) == 1 and r.deductions[0].reason == "PRICING" and r.deductions[0].amount == 4_500


def test_rules_parser_discount_partial_and_on_account():
    assert P.parse("", "INV 4821937 LESS 2% DISC").discount.pct == pytest.approx(0.02)
    assert "partial" in P.parse("", "PARTIAL PMT INV 4821937").hints
    assert "partial" in P.parse("", "1 OF 2 4821937").hints
    assert "final" in P.parse("", "FINAL PMT 4821937").hints
    assert "on_account" in P.parse("", "DEPOSIT").hints
    assert P.parse("", "THANK YOU").tokens == []


def test_deduction_without_amount_and_with_unknown_code():
    r = P.parse("", "INV 4821937 PRICING DIFF")
    assert r.deductions[0].reason == "PRICING" and r.deductions[0].amount is None
    assert P.parse("", "INV 4821937 DED CODE X17 45.00").deductions[0].reason == "OTHER"


GOOD = json.dumps({"invoice_refs": [{"ref": "1999999", "amount": 12.5}], "deductions": [{"reason": "FREIGHT", "amount": 3.0}],
                   "discount": None, "hints": ["partial"]})


def test_llm_parser_uses_a_valid_reply_and_adds_to_rule_references():
    llm = LLMRemitParser(lambda prompt: "Here you go: " + GOOD)
    r = llm.parse("", "INV 4821937 something odd")
    assert r.source == "llm" and llm.llm_ok == 1 and llm.llm_failed == 0
    assert {t.digits for t in r.tokens} == {"4821937", "1999999"}  # the model adds, the rules' references stay
    assert r.amounts["1999999"] == 1250 and r.deductions[0].reason == "FREIGHT" and "partial" in r.hints


@pytest.mark.parametrize("reply", [
    "not json at all",
    '{"invoice_refs": [{"ref": 5}]}',
    '{"deductions": [{"reason": "WHATEVER", "amount": 1}]}',
    '{"hints": ["bogus"]}',
    '{"discount": {"pct": 500}}',
    '{"invoice_refs": [{"ref": "1234567", "amount": "12.50"}]}',
])
def test_llm_parser_falls_back_to_rules_on_any_bad_reply(reply):
    llm = LLMRemitParser(lambda prompt: reply)
    r = llm.parse("", "INV 4821937 DED DC01 45.00")
    assert r.source == "llm_fallback" and llm.llm_failed == 1
    assert r.deductions[0].reason == "PRICING"  # the rules answered


def test_llm_parser_survives_a_model_that_raises():
    def boom(prompt):
        raise TimeoutError("no network")

    r = LLMRemitParser(boom).parse("", "INV 4821937")
    assert r.source == "llm_fallback" and [t.digits for t in r.tokens] == ["4821937"]


def test_validate_rejects_non_objects():
    with pytest.raises(LLMParseError):
        validate_llm_json("[1, 2]")


def test_anthropic_adapter_is_optional():
    # in a fresh interpreter where the SDK cannot be imported, the module still imports and the adapter fails loudly
    import os
    import subprocess
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    code = ("import sys; sys.modules['anthropic'] = None\n"
            "import cashapp.remit as r\n"
            "try:\n    r.anthropic_llm()\nexcept ImportError:\n    print('ok')\n")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=str(root), env={**os.environ, "PYTHONPATH": str(root)})
    assert out.stdout.strip() == "ok", out.stderr
