"""Reference extraction and fuzzy matching against invoice numbers.

A remittance reference is the strongest signal there is, and the most often damaged: digits get
transposed or dropped, a leading part is cut off, OCR turns 0 into O, or the customer quotes its
own PO number. Matching therefore has tiers, each with a stated similarity:

    exact invoice digits            1.00
    exact PO number                 0.90
    one edit away (fuzzy)           0.75   (substitution, insertion, deletion, adjacent swap; 6+ digits)
    truncated to the last 5-6 digits 0.70
"""
from __future__ import annotations

import re
from dataclasses import dataclass

MONEY_RE = re.compile(r"\(?-?\$?\d{1,3}(?:,\d{3})+\.\d{2}\)?|\(?-?\$?\d+\.\d{2}\)?")
DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b|\b20\d{6}\b")
OCR_FIX = str.maketrans("OILSBZ", "011582")
MIN_LEN, MAX_LEN = 4, 8  # longer digit runs are trace ids, not invoice numbers

SIM_EXACT, SIM_PO, SIM_FUZZY, SIM_SUFFIX = 1.0, 0.9, 0.75, 0.70


@dataclass(frozen=True)
class RefToken:
    raw: str
    digits: str


def strip_money_and_dates(text: str) -> str:
    return DATE_RE.sub(" ", MONEY_RE.sub(" ", text.upper()))


def extract_tokens(text: str) -> list:
    """Candidate invoice/PO numbers in free text, de-duplicated, in order of appearance."""
    out, seen = [], set()
    for run in re.split(r"[^A-Z0-9]+", strip_money_and_dates(text)):
        if not run:
            continue
        m = re.match(r"^([A-Z]*)(.*)$", run)
        body = m.group(2) if m and m.group(2) else ""
        if not body:
            continue
        fixed = body.translate(OCR_FIX)
        if not fixed.isdigit() or sum(ch.isdigit() for ch in body) < 3:
            continue
        if not (MIN_LEN <= len(fixed) <= MAX_LEN) or fixed in seen:
            continue
        seen.add(fixed)
        out.append(RefToken(raw=run, digits=fixed))
    return out


def _variants(d: str) -> set:
    """The string, its single deletions and its adjacent swaps: the symmetric-delete neighbourhood."""
    out = {d}
    for i in range(len(d)):
        out.add(d[:i] + d[i + 1:])
    for i in range(len(d) - 1):
        if d[i] != d[i + 1]:
            out.add(d[:i] + d[i + 1] + d[i] + d[i + 2:])
    return out


def edit_distance_le1(a: str, b: str) -> bool:
    """Insert, delete, substitute or swap two adjacent digits."""
    if a == b:
        return True
    la, lb = len(a), len(b)
    if abs(la - lb) > 1:
        return False
    if la == lb:
        diff = [i for i in range(la) if a[i] != b[i]]
        if len(diff) == 1:
            return True
        return len(diff) == 2 and diff[1] == diff[0] + 1 and a[diff[0]] == b[diff[1]] and a[diff[1]] == b[diff[0]]
    if la > lb:
        a, b = b, a
    i = 0
    while i < len(a) and a[i] == b[i]:
        i += 1
    return a[i:] == b[i + 1:]


@dataclass(frozen=True)
class RefHit:
    invoice_id: str
    sim: float
    kind: str  # exact | po | fuzzy | suffix
    token: str


class RefIndex:
    """Index over every invoice, open or not, so a reference to a settled invoice is still seen."""

    def __init__(self, invoices, fuzzy: bool = True):
        self.fuzzy = fuzzy
        self.exact: dict = {}
        self.po: dict = {}
        self.neigh: dict = {}
        self.suffix: dict = {}
        self.digits: dict = {}
        for inv in invoices:
            d = inv.digits
            self.digits[inv.id] = d
            self.exact.setdefault(d, []).append(inv.id)
            if inv.po_number:
                self.po.setdefault("".join(ch for ch in inv.po_number if ch.isdigit()), []).append(inv.id)
            if fuzzy:
                for v in _variants(d):
                    self.neigh.setdefault(v, []).append(inv.id)
                for n in (5, 6):
                    if len(d) > n:
                        self.suffix.setdefault(d[-n:], []).append(inv.id)

    def lookup(self, token: RefToken) -> dict:
        """{invoice_id: RefHit} for one token, keeping the best tier per invoice."""
        d = token.digits
        hits: dict = {}

        def put(inv_id, sim, kind):
            cur = hits.get(inv_id)
            if cur is None or sim > cur.sim:
                hits[inv_id] = RefHit(inv_id, sim, kind, d)

        for inv_id in self.exact.get(d, ()):
            put(inv_id, SIM_EXACT, "exact")
        for inv_id in self.po.get(d, ()):
            put(inv_id, SIM_PO, "po")
        if self.fuzzy:
            if len(d) >= 6:
                for v in _variants(d):
                    for inv_id in self.neigh.get(v, ()):
                        if inv_id not in hits and edit_distance_le1(d, self.digits[inv_id]):
                            put(inv_id, SIM_FUZZY, "fuzzy")
            if len(d) in (5, 6):
                for inv_id in self.suffix.get(d, ()):
                    if inv_id not in hits:
                        put(inv_id, SIM_SUFFIX, "suffix")
        return hits

    def lookup_all(self, tokens) -> dict:
        """{invoice_id: RefHit} over all tokens of a payment."""
        out: dict = {}
        for t in tokens:
            for inv_id, h in self.lookup(t).items():
                cur = out.get(inv_id)
                if cur is None or h.sim > cur.sim:
                    out[inv_id] = h
        return out
