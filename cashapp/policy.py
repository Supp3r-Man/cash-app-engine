"""Auto-apply versus exception queue.

The engine gives every payment a proposal and a confidence. The policy is a threshold on that
confidence, plus the hard exceptions the engine already raised (duplicate, no match). Keeping the
policy separate means one engine run can be scored at many thresholds.
"""
from __future__ import annotations

from dataclasses import replace

EXCEPTION_REASONS = {
    "duplicate": "possible duplicate payment",
    "no_match": "no open invoice explains the payment",
    "refs_closed_invoice": "references an invoice that is already settled",
    "unexplained_short_pay": "short-pay with no reason given",
    "partial_unconfirmed": "partial payment, instalment not confirmed",
    "overpayment": "payment exceeds the invoice",
    "ambiguous_match": "more than one allocation fits",
    "fuzzy_reference": "reference only matched approximately",
    "low_confidence": "confidence below the auto-apply threshold",
}


def _reason(dec, margin_ref: float) -> str:
    if dec.reason:
        return dec.reason
    if dec.kind == "short_pay":
        return "unexplained_short_pay"
    if dec.kind == "partial":
        return "partial_unconfirmed"
    if dec.kind == "overpay":
        return "overpayment"
    if dec.margin < margin_ref:
        return "ambiguous_match"
    if dec.fuzzy_ref:
        return "fuzzy_reference"
    return "low_confidence"


def apply_policy(decisions, threshold: float, margin_ref: float = 0.10) -> list:
    """Copies of the decisions with status and exception reason set at this threshold."""
    out = []
    for d in decisions:
        auto = d.reason is None and bool(d.proposal) and d.confidence >= threshold
        nd = replace(d, explanation=list(d.explanation), proposal=list(d.proposal))
        if auto:
            nd.status, nd.reason = "auto_apply", None
            nd.explanation.append(f"Confidence {d.confidence:.2f} is at or above the threshold {threshold:.2f}: auto-apply.")
        else:
            nd.status, nd.reason = "exception", _reason(d, margin_ref)
            if d.reason is None:
                nd.explanation.append(f"Confidence {d.confidence:.2f} is below the threshold {threshold:.2f}, or the case needs a person: queued ({EXCEPTION_REASONS[nd.reason]}).")
        out.append(nd)
    return out
