"""A deliberately plain matcher to compare against: what a rules-only cash-application setup does.

It trusts exact things only: an exact invoice-number match, an exact (normalised) payer name, an exact
amount. It applies cash to the referenced invoice(s) without classifying a difference, does not look
for duplicates, and cannot search for combinations of invoices when nothing is referenced.
"""
from __future__ import annotations

from .names import normalize
from .refs import RefIndex, extract_tokens
from .schema import Decision, PropAlloc, convert, money


class BaselineEngine:
    def __init__(self, ds):
        self.ds = ds
        self.cust = {c.id: c for c in ds.customers}
        self.by_name: dict = {}
        for c in ds.customers:
            self.by_name.setdefault(normalize(c.legal_name), []).append(c.id)
        self.all_inv = {i.id: i for i in list(ds.invoices) + list(ds.closed_invoices)}
        self.refidx = RefIndex(self.all_inv.values(), fuzzy=False)
        self.open_by_cust: dict = {}
        for i in ds.invoices:
            self.open_by_cust.setdefault(i.customer_id, []).append(i)

    def run(self) -> list:
        rem = {i.id: i.amount for i in self.ds.invoices}
        out = []
        for p in sorted(self.ds.payments, key=lambda p: (p.date, p.id)):
            d = Decision(payment_id=p.id)
            name_ids = self.by_name.get(normalize(p.payer_name), [])
            hits = [h for h in self.refidx.lookup_all(extract_tokens(f"{p.reference} ; {p.memo}")).values() if h.kind == "exact" and rem.get(h.invoice_id, 0) > 0]
            chosen, conf, why = [], 0.0, ""
            if hits:
                invs = [self.all_inv[h.invoice_id] for h in hits]
                cids = {i.customer_id for i in invs}
                if len(cids) == 1:
                    cid = next(iter(cids))
                    T = convert(p.amount, p.currency, self.cust[cid].currency)
                    total = sum(rem[i.id] for i in invs)
                    named = cid in name_ids
                    if T == total:
                        chosen, conf, why = invs, 0.97 if named else 0.88, f"Exact reference and the amount equals the open balance{' and the payer name matches' if named else ''}."
                    elif len(invs) == 1:
                        chosen, conf, why = invs, 0.60, "Exact reference, but the amount differs from the balance."
            elif len(name_ids) == 1:
                cid = name_ids[0]
                T = convert(p.amount, p.currency, self.cust[cid].currency)
                same = [i for i in self.open_by_cust.get(cid, []) if rem[i.id] == T]
                if len(same) == 1:
                    chosen, conf, why = same, 0.80, "Exact payer name and exactly one open invoice with this amount."
            if chosen:
                d.kind = "exact"
                d.confidence = conf
                d.proposal = [PropAlloc(i.id, rem[i.id], rem[i.id], "clean") for i in chosen]
                d.explanation = [why]
                for i in chosen:
                    rem[i.id] = 0
            else:
                d.reason, d.kind = "no_match", "no_match"
                d.explanation = ["No exact reference, name and amount combination found."]
            out.append(d)
        return out
