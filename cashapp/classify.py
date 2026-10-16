"""Deduction classification: turn the words or code a payer used into a reason."""
from __future__ import annotations

import re
from typing import Optional

# A made-up code table. A real customer would supply theirs, and it is usually per payer.
CODE_TABLE = {
    "DC01": "PRICING", "DC02": "DAMAGED", "DC03": "SHORTAGE", "DC04": "FREIGHT",
    "DC05": "PROMO", "DC06": "TAX", "DC07": "RETURN",
}

KEYWORDS = [
    ("PRICING", r"\b(PRICING|PRICE|PRC|PD)\b"),
    ("DAMAGED", r"\b(DMG|DAMAGE|DAMAGED|BROKEN)\b"),
    ("SHORTAGE", r"\b(SHORT\s*SHIP(PED)?|SHORTAGE|NOT\s+RECEIVED|SHORT\s+RCVD)\b"),
    ("FREIGHT", r"\b(FRT|FREIGHT|SHIPPING)\b"),
    ("PROMO", r"\b(PROMO|ALLOW|ALLOWANCE|COOP|CO-OP|REBATE)\b"),
    ("TAX", r"\b(TAX|VAT|GST)\b"),
    ("RETURN", r"\b(RETURN|RTN|RMA)\b"),
]
GENERIC_SIGNAL = re.compile(r"\b(DED|DEDUCT|DEDUCTION|DEDUCTED|CHARGEBACK|CHGBK|CB|SHORT\s*PAY|CODE\s+[A-Z]?\d{2,3}|(DC|RC)\d{2,3})\b")
CODE_RE = re.compile(r"\b(DC\d{2,3}|RC\d{2,3})\b")


def reason_from_text(segment: str) -> Optional[str]:
    """None when the segment is not a deduction at all, OTHER when it is one with no recognisable reason."""
    s = segment.upper()
    for code in CODE_RE.findall(s):
        if code in CODE_TABLE:
            return CODE_TABLE[code]
    for reason, pat in KEYWORDS:
        if re.search(pat, s):
            return reason
    if GENERIC_SIGNAL.search(s):
        return "OTHER"
    return None
