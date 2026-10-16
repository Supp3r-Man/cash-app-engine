"""Remittance free-text parsing: a pluggable parser with a deterministic offline fallback.

`RuleRemitParser` is a plain regex parser and is what runs by default. `LLMRemitParser` wraps any
function from prompt to text, asks it for JSON, validates the JSON, and falls back to the rules
when the reply is unusable. It only ever adds to what the rules found for invoice references.
`anthropic_llm()` builds such a function for the Anthropic API. It is off by default and has
never been run against the live API here: the model path is tested only with fake model functions.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Callable, Optional

from .classify import reason_from_text
from .refs import MONEY_RE, OCR_FIX, RefToken, extract_tokens


@dataclass
class Deduction:
    reason: str
    amount: Optional[int] = None  # cents


@dataclass
class Discount:
    pct: Optional[float] = None  # 0.02 for 2%
    amount: Optional[int] = None


@dataclass
class ParsedRemit:
    tokens: list = field(default_factory=list)  # list[RefToken]
    amounts: dict = field(default_factory=dict)  # digits -> amount listed next to that reference
    deductions: list = field(default_factory=list)
    discount: Optional[Discount] = None
    hints: set = field(default_factory=set)  # partial | final | on_account
    source: str = "rules"


def money_to_cents(text: str) -> int:
    t = text.replace("$", "").replace(",", "").replace("(", "").replace(")", "").strip()
    neg = t.startswith("-")
    cents = int(round(abs(float(t.lstrip("-"))) * 100))
    return -cents if neg else cents


PARTIAL_RE = re.compile(r"\b(PARTIAL|PART\s+PMT|PART\s+PAYMENT|\d\s+OF\s+[2-9]|INSTALLMENT)\b")
FINAL_RE = re.compile(r"\b(FINAL|BALANCE|BAL\s+DUE|[2-9]\s+OF\s+[2-9])\b")
ON_ACCOUNT_RE = re.compile(r"\b(ON\s+ACCT|ON\s+ACCOUNT|DEPOSIT|PREPAY|PREPAYMENT|ADVANCE)\b")
DISCOUNT_RE = re.compile(r"\bDISC(OUNT)?\b")
PCT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%")


class RuleRemitParser:
    def __init__(self):
        self._cache: dict = {}

    def parse(self, reference: str, memo: str) -> ParsedRemit:
        key = (reference, memo)
        hit = self._cache.get(key)
        if hit is None:
            hit = self._cache[key] = self._parse(reference, memo)
        return hit

    def _parse(self, reference: str, memo: str) -> ParsedRemit:
        text = f"{reference} ; {memo}".upper()
        out = ParsedRemit(tokens=extract_tokens(text))
        for seg in re.split(r"[;|\n]", text):
            seg = seg.strip()
            if not seg:
                continue
            moneys = [(m.start(), m.group()) for m in MONEY_RE.finditer(seg)]
            seg_tokens = extract_tokens(seg)
            if DISCOUNT_RE.search(seg):
                pct = PCT_RE.search(seg)
                amt = [abs(money_to_cents(m)) for _, m in moneys]
                out.discount = Discount(pct=float(pct.group(1)) / 100.0 if pct else None, amount=amt[-1] if amt else None)
                continue
            reason = reason_from_text(seg)
            if reason is not None:
                # the deduction amount is what follows the first deduction keyword
                kw = re.search(r"\b(DED|DEDUCT|DEDUCTION|DEDUCTED|CHARGEBACK|CHGBK|CB|SHORT|LESS|DC\d+|RC\d+|CODE|PRICING|PRICE|DMG|DAMAGED?|FRT|FREIGHT|PROMO|ALLOW(ANCE)?|TAX|VAT|RETURN|RTN)\b", seg)
                start = kw.start() if kw else 0
                after = [abs(money_to_cents(m)) for pos, m in moneys if pos >= start]
                before = [(pos, m) for pos, m in moneys if pos < start]
                out.deductions.append(Deduction(reason, after[0] if after else None))
                moneys = before
            if len(seg_tokens) == 1 and moneys:
                out.amounts[seg_tokens[0].digits] = abs(money_to_cents(moneys[0][1]))
            if PARTIAL_RE.search(seg):
                out.hints.add("partial")
            if FINAL_RE.search(seg):
                out.hints.add("final")
            if ON_ACCOUNT_RE.search(seg):
                out.hints.add("on_account")
        return out


LLM_PROMPT = """You read bank remittance text for an accounts-receivable team.
Return ONLY a JSON object with these keys:
  "invoice_refs": list of {{"ref": digits-only invoice or PO number, "amount": dollars or null}}
  "deductions": list of {{"reason": one of PRICING|DAMAGED|SHORTAGE|FREIGHT|PROMO|TAX|RETURN|OTHER, "amount": dollars or null}}
  "discount": null or {{"pct": percent or null, "amount": dollars or null}}
  "hints": list drawn from ["partial", "final", "on_account"]
Text: {text}
JSON:"""

_REASONS = {"PRICING", "DAMAGED", "SHORTAGE", "FREIGHT", "PROMO", "TAX", "RETURN", "OTHER"}


class LLMParseError(ValueError):
    pass


def _cents(x) -> Optional[int]:
    if x is None:
        return None
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        raise LLMParseError("amount must be a number or null")
    return int(round(abs(float(x)) * 100))


def validate_llm_json(raw: str) -> dict:
    """Strict check of a model reply. Raises LLMParseError for anything that is not exactly the schema."""
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        raise LLMParseError("no JSON object in reply")
    try:
        obj = json.loads(m.group())
    except json.JSONDecodeError as exc:
        raise LLMParseError(f"bad JSON: {exc}") from exc
    if not isinstance(obj, dict):
        raise LLMParseError("reply is not an object")
    refs = []
    for r in obj.get("invoice_refs", []):
        if not isinstance(r, dict) or not isinstance(r.get("ref"), str):
            raise LLMParseError("invoice_refs entries need a string ref")
        digits = r["ref"].upper().translate(OCR_FIX)
        digits = "".join(ch for ch in digits if ch.isdigit())
        if not (4 <= len(digits) <= 8):
            continue  # not an invoice-like number: ignore rather than fail the whole reply
        refs.append((digits, _cents(r.get("amount"))))
    deds = []
    for d in obj.get("deductions", []):
        if not isinstance(d, dict) or d.get("reason") not in _REASONS:
            raise LLMParseError("deductions need a reason from the fixed list")
        deds.append(Deduction(d["reason"], _cents(d.get("amount"))))
    disc = obj.get("discount")
    discount = None
    if disc is not None:
        if not isinstance(disc, dict):
            raise LLMParseError("discount must be an object or null")
        pct = disc.get("pct")
        if pct is not None and (isinstance(pct, bool) or not isinstance(pct, (int, float)) or not 0 < pct < 100):
            raise LLMParseError("discount pct out of range")
        discount = Discount(pct=None if pct is None else float(pct) / 100.0, amount=_cents(disc.get("amount")))
    hints = obj.get("hints", [])
    if not isinstance(hints, list) or any(h not in ("partial", "final", "on_account") for h in hints):
        raise LLMParseError("hints must come from the fixed list")
    return {"refs": refs, "deductions": deds, "discount": discount, "hints": set(hints)}


class LLMRemitParser:
    """Ask a model first; use the rules when its reply fails validation. Counts both outcomes."""

    def __init__(self, llm: Callable[[str], str], fallback: Optional[RuleRemitParser] = None):
        self.llm = llm
        self.fallback = fallback or RuleRemitParser()
        self.llm_ok = 0
        self.llm_failed = 0

    def parse(self, reference: str, memo: str) -> ParsedRemit:
        rules = self.fallback.parse(reference, memo)
        try:
            data = validate_llm_json(self.llm(LLM_PROMPT.format(text=f"{reference} ; {memo}")))
        except Exception:  # a model call can fail in many ways; the rules always answer
            self.llm_failed += 1
            rules.source = "llm_fallback"
            return rules
        self.llm_ok += 1
        out = ParsedRemit(source="llm")
        seen = {t.digits for t in rules.tokens}
        out.tokens = list(rules.tokens)
        out.amounts = dict(rules.amounts)
        for digits, cents in data["refs"]:
            if digits not in seen:
                seen.add(digits)
                out.tokens.append(RefToken(raw=digits, digits=digits))
            if cents is not None:
                out.amounts[digits] = cents
        out.deductions = data["deductions"] or rules.deductions
        out.discount = data["discount"] or rules.discount
        out.hints = data["hints"] | rules.hints
        return out


def anthropic_llm(model: str = "claude-sonnet-5-5", max_tokens: int = 400) -> Callable[[str], str]:
    """A prompt -> text function backed by the Anthropic API. Off by default; NOT run live in this repo."""
    import anthropic  # imported lazily: the rest of the package needs no SDK and no key

    client = anthropic.Anthropic()

    def call(prompt: str) -> str:
        resp = client.messages.create(model=model, max_tokens=max_tokens, messages=[{"role": "user", "content": prompt}])
        return resp.content[0].text

    return call


class TokensOnlyParser:
    """Used by the 'remit' ablation: references are found, nothing else in the text is read."""

    def parse(self, reference: str, memo: str) -> ParsedRemit:
        return ParsedRemit(tokens=extract_tokens(f"{reference} ; {memo}"), source="tokens_only")
