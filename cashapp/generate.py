"""Seeded synthetic data with ground truth. Everything here is invented; no real company or customer data.

The generator builds a customer master, a snapshot of open invoices, some already-settled invoices, a
history of how each payer name has appeared before, and a batch of incoming payments. Each payment
is produced by a "case builder" for one hard-case category and gets a ground-truth label saying
exactly which invoices it settles and how (clean, discount, deduction with a reason, short-pay,
partial instalment, FX or rounding difference), or that it is a duplicate or has nothing to apply to.
"""
from __future__ import annotations

import datetime as dt
import math
import random
from typing import Optional

from .names import ABBREVIATIONS
from .schema import (
    CATEGORIES, FX_TO_USD, Customer, CustomerHistory, Dataset, Invoice, Payment, Truth, TruthAlloc, convert, money,
)

START = dt.date(2025, 9, 1)
DAYS = 75
END = START + dt.timedelta(days=DAYS - 1)

FIRST = ("Acme Apex Atlas Beacon Birch Bluewater Cardinal Cascade Cedar Crescent Delta Eagle Everest Falcon Frontier Garnet "
         "Granite Harbor Horizon Iron Juniper Keystone Lakeside Lumen Maple Meridian Nimbus Northstar Oak Orion Pinnacle "
         "Quarry Redwood Ridge Sierra Summit Sterling Tidewater Union Vanguard Westbrook Zenith").split()
SECOND = ("Manufacturing International Distribution Services Systems Industries Logistics Foods Supply Technologies Holdings "
          "Group Partners Electric Associates Packaging Healthcare Materials").split()
SUFFIX = ["Inc", "LLC", "Corp", "Ltd", "Co"]
STRANGER_FIRST = "Zephyr Quill Marlow Thorne Obsidian Pelican Rowan Tamarack Wren Yarrow".split()
STRANGER_SECOND = "Dynamics Ventures Enterprises Resources Capital Works".split()

REVERSE_ABBR: dict = {}
for _short, _long in ABBREVIATIONS.items():
    REVERSE_ABBR.setdefault(_long, []).append(_short)

REASON_TEXT = {
    "PRICING": ["PRICING DIFF", "PRICE DIFFERENCE", "DED DC01", "PRICE ADJ"],
    "DAMAGED": ["DMG GOODS", "DAMAGED", "DED DC02", "BROKEN ITEMS"],
    "SHORTAGE": ["SHORT SHIPPED", "SHORTAGE", "DED DC03", "NOT RECEIVED"],
    "FREIGHT": ["FRT CHARGE", "FREIGHT DEDUCT", "DED DC04"],
    "PROMO": ["PROMO ALLOW", "ALLOWANCE", "DED DC05", "CO-OP"],
    "TAX": ["TAX ADJ", "VAT DEDUCT", "DED DC06"],
    "RETURN": ["RETURN CREDIT", "RTN GOODS", "DED DC07", "RMA CREDIT"],
}
UNKNOWN_CODES = ["DED CODE X17", "DEDUCTION DC99", "CB 4471", "DED RC88"]
GENERIC_MEMO = ["ACH PAYMENT", "THANK YOU", "PMT", "INVOICE PAYMENT", "REMITTANCE ADVICE ATTACHED", "", "", "PAYMENT"]

DEFAULT_MIX = {
    "exact": 0.22, "multi_invoice": 0.12, "split_payment": 0.08, "short_pay": 0.06, "early_discount": 0.06,
    "deduction": 0.08, "rounding_fx": 0.06, "garbled_ref": 0.08, "name_variant": 0.08, "duplicate": 0.05, "unapplied": 0.08,
}


def fmt_money(cents: int) -> str:
    return f"{cents / 100:,.2f}"


class _Cust:
    def __init__(self, cid, name, ccy, net, disc, lag):
        self.id, self.name, self.ccy, self.net, self.disc, self.lag = cid, name, ccy, net, disc, lag
        self.canonical = name.upper()
        self.hist_variants: list = []


class Generator:
    def __init__(self, seed: int = 7, n_customers: int = 120, n_payments: int = 1600, mix: Optional[dict] = None):
        self.seed = seed
        self.rng = random.Random(seed)
        self.n_customers = n_customers
        self.n_payments = n_payments
        self.mix = dict(DEFAULT_MIX if mix is None else mix)
        self.cust: dict = {}
        self.invoices: dict = {}  # id -> Invoice
        self.rem: dict = {}  # id -> remaining cents
        self.claimed: set = set()
        self.closed: list = []
        self.history: dict = {}
        self.digits_used: set = set()
        self.payments: list = []
        self.truth: dict = {}
        self._pid = 0
        self._iid = 0

    # ---- building blocks -----------------------------------------------------------------
    def _digits(self) -> str:
        while True:
            d = str(self.rng.randrange(1_000_000, 9_999_999))
            if d not in self.digits_used:
                self.digits_used.add(d)
                return d

    def _new_invoice(self, c: _Cust, issue: dt.date, amount: int, closed: bool = False) -> Invoice:
        self._iid += 1
        d = self._digits()
        po = str(self.rng.randrange(100_000, 999_999)) if self.rng.random() < 0.4 else None
        disc_pct, disc_days = c.disc if c.disc else (0.0, 0)
        inv = Invoice(id=f"I{self._iid:05d}", number=f"INV-{d}", customer_id=c.id, amount=amount, currency=c.ccy,
                      issue_date=issue, due_date=issue + dt.timedelta(days=c.net), discount_pct=disc_pct,
                      discount_days=disc_days, po_number=po)
        return inv

    def _amount(self) -> int:
        return int(round(math.exp(self.rng.uniform(math.log(150), math.log(25000))) * 100))

    def build_master(self) -> None:
        rng = self.rng
        used = set()
        for k in range(self.n_customers):
            while True:
                a, b = rng.choice(FIRST), rng.choice(SECOND)
                if (a, b) not in used:
                    used.add((a, b))
                    break
            name = f"{a} {b} {rng.choice(SUFFIX)}"
            ccy = rng.choices(["USD", "EUR", "GBP", "CAD"], [0.72, 0.10, 0.09, 0.09])[0]
            net = rng.choice([30, 30, 30, 45, 60])
            disc = (rng.choice([0.01, 0.02, 0.03]), rng.choice([10, 15])) if rng.random() < 0.35 else None
            c = _Cust(f"C{k + 1:03d}", name, ccy, net, disc, rng.gauss(3, 7))
            self.cust[c.id] = c
            # open invoice snapshot
            prev = None
            for _ in range(rng.randint(22, 38)):
                issue = START - dt.timedelta(days=100) + dt.timedelta(days=rng.randint(0, 100 + DAYS - 10))
                amount = prev if (prev is not None and rng.random() < 0.06) else self._amount()
                prev = amount
                inv = self._new_invoice(c, issue, amount)
                self.invoices[inv.id] = inv
                self.rem[inv.id] = amount
            # settled before the batch
            for _ in range(8):
                issue = START - dt.timedelta(days=rng.randint(60, 200))
                inv = self._new_invoice(c, issue, self._amount(), closed=True)
                self.closed.append(inv)
            # history of payer strings and payment lag
            n_hist = 0 if rng.random() < 0.10 else rng.randint(5, 11)
            strings: dict = {}
            if n_hist:
                strings[c.canonical] = max(1, n_hist - 3)
                for _ in range(rng.choice([0, 1, 1, 2])):
                    v = self.variant(c)
                    c.hist_variants.append(v)
                    strings[v] = strings.get(v, 0) + rng.randint(1, 3)
            self.history[c.id] = CustomerHistory(payer_strings=strings, lag_mean=(c.lag + rng.gauss(0, 2)) if n_hist else None, n=n_hist)

    # ---- payer names -------------------------------------------------------------------------
    def variant(self, c: _Cust) -> str:
        rng = self.rng
        words = c.canonical.split()
        core = words[:-1] if words[-1] in {s.upper() for s in SUFFIX} else words
        for _ in range(20):
            kind = rng.choice(["drop_suffix", "abbr", "truncate", "bank_prefix", "the", "typo", "long_suffix", "first_only", "case_punct", "initials", "one_word"])
            if kind == "drop_suffix":
                v = " ".join(core)
            elif kind == "abbr":
                v = " ".join(rng.choice(REVERSE_ABBR[w]) if w in REVERSE_ABBR and rng.random() < 0.9 else w for w in core) + " " + words[-1]
            elif kind == "truncate":
                v = c.canonical[: rng.randint(16, 21)].rstrip()
            elif kind == "bank_prefix":
                v = rng.choice(["ACH CREDIT ", "WIRE FROM ", "ORIG CO NAME ", "EFT "]) + c.canonical
            elif kind == "the":
                v = "THE " + " ".join(core) + " CO"
            elif kind == "typo":
                w = list(c.canonical)
                i = rng.randrange(3, len(w) - 1)
                if w[i] == " ":
                    continue
                if rng.random() < 0.5:
                    del w[i]
                else:
                    w[i] = rng.choice("AEIOUNRST")
                v = "".join(w)
            elif kind == "long_suffix":
                v = " ".join(core) + " " + {"INC": "INCORPORATED", "CORP": "CORPORATION", "CO": "COMPANY", "LTD": "LIMITED", "LLC": "L.L.C."}[words[-1]]
            elif kind == "initials":
                v = "".join(w[0] for w in core) + " " + words[-1] if len(core) >= 2 else c.canonical[:6]
            elif kind == "one_word":
                v = core[0]
            elif kind == "first_only":
                v = core[0] + " " + core[1][:3] if len(core) > 1 and rng.random() < 0.4 else " ".join(core[:2])
            else:
                v = ", ".join(core).replace(",", "") + "."
            if v and v != c.canonical:
                return v
        return c.canonical[:18]

    def payer(self, c: _Cust, force_variant: bool = False, p_variant: float = 0.18) -> tuple:
        """(payer string, tags)."""
        rng = self.rng
        if not force_variant and rng.random() >= p_variant:
            return c.canonical, []
        seen = c.hist_variants
        if seen and rng.random() < 0.4:
            return rng.choice(seen), ["name_variant_seen"]
        return self.variant(c), ["name_variant_new"]

    # ---- references and memos ----------------------------------------------------------------
    def disp_ref(self, d: str) -> str:
        return self.rng.choice(["INV {d}", "INV# {d}", "INV-{d}", "{d}", "INVOICE {d}", "IN {d}"]).format(d=d)

    def noisy(self, inv: Invoice) -> tuple:
        """(damaged reference text, operation). Falls back to a swap when the operation does not apply."""
        rng = self.rng
        d = inv.digits
        op = rng.choice(["transpose", "drop", "subst", "trunc", "ocr", "po"])
        if op == "po" and not inv.po_number:
            op = "transpose"
        if op == "po":
            return f"PO {inv.po_number}", op
        if op in ("transpose", "drop", "subst") and rng.random() < 0.30:
            # two separate corruptions: past what one-edit fuzzy matching can repair
            first, _ = self._single_edit(d, op)
            second, _ = self._single_edit(first, rng.choice(["transpose", "drop", "subst"]))
            return second, "double_" + op
        if op == "po":
            return f"PO {inv.po_number}", op
        if op == "ocr":
            idx = [i for i, ch in enumerate(d) if ch in "01582"]
            if not idx:
                op = "transpose"
            else:
                i = rng.choice(idx)
                return d[:i] + {"0": "O", "1": "I", "5": "S", "8": "B", "2": "Z"}[d[i]] + d[i + 1:], op
        if op == "trunc":
            return d[-5:], op
        if op == "drop":
            i = rng.randrange(len(d))
            return d[:i] + d[i + 1:], op
        if op == "subst":
            i = rng.randrange(len(d))
            return d[:i] + rng.choice([x for x in "0123456789" if x != d[i]]) + d[i + 1:], op
        for _ in range(10):
            i = rng.randrange(len(d) - 1)
            if d[i] != d[i + 1]:
                return d[:i] + d[i + 1] + d[i] + d[i + 2:], "transpose"
        return d[:-1], "drop"

    def _single_edit(self, d: str, op: str) -> tuple:
        rng = self.rng
        if op == "drop" and len(d) > 4:
            i = rng.randrange(len(d))
            return d[:i] + d[i + 1:], op
        if op == "subst":
            i = rng.randrange(len(d))
            return d[:i] + rng.choice([x for x in "0123456789" if x != d[i]]) + d[i + 1:], op
        for _ in range(10):
            i = rng.randrange(len(d) - 1)
            if d[i] != d[i + 1]:
                return d[:i] + d[i + 1] + d[i] + d[i + 2:], "transpose"
        return d, op

    def trace(self) -> str:
        r = self.rng.random()
        if r < 0.5:
            return "ACH-" + "".join(str(self.rng.randrange(10)) for _ in range(10))
        if r < 0.6:
            return "CHK " + "".join(str(self.rng.randrange(10)) for _ in range(6))
        if r < 0.75:
            return "TRN " + "".join(str(self.rng.randrange(10)) for _ in range(9))
        return ""

    def new_payment(self, payer, amount, ccy, date, reference, memo) -> Payment:
        self._pid += 1
        return Payment(id=f"P{self._pid:05d}", payer_name=payer, amount=amount, currency=ccy, date=date, reference=reference, memo=memo)

    def pay_date(self, invs, c: _Cust, lo: Optional[dt.date] = None, hi: Optional[dt.date] = None) -> Optional[dt.date]:
        rng = self.rng
        earliest = max(START, max(i.issue_date for i in invs) + dt.timedelta(days=1), lo or START)
        latest = hi or END
        if earliest > latest:
            return None
        centre = max(i.due_date for i in invs) + dt.timedelta(days=round(c.lag))
        if centre < earliest:  # already past due when the batch starts: spread the late payers out
            centre = earliest + dt.timedelta(days=rng.randint(0, 30))
        d = centre + dt.timedelta(days=round(rng.gauss(0, 5)))
        return min(max(d, earliest), latest)

    # ---- picking invoices --------------------------------------------------------------------
    def avail(self, cid: str, pred=None) -> list:
        return [i for i in self.invoices.values() if i.customer_id == cid and i.id not in self.claimed and self.rem[i.id] > 0 and i.issue_date < END and (pred is None or pred(i))]

    def pick(self, k: int = 1, pred=None, cust_pred=None) -> Optional[tuple]:
        rng = self.rng
        cids = list(self.cust)
        rng.shuffle(cids)
        for cid in cids:
            c = self.cust[cid]
            if cust_pred and not cust_pred(c):
                continue
            pool = self.avail(cid, pred)
            if len(pool) >= k and len(self.avail(cid)) >= k + 3:
                return c, rng.sample(pool, k)
        return None

    def claim(self, invs) -> None:
        for i in invs:
            self.claimed.add(i.id)

    def _truth(self, p: Payment, kind: str, category: str, allocs=None, on_account: int = 0, tags=None, customer_id: str = "") -> Truth:
        if allocs:
            customer_id = self.invoices[allocs[0].invoice_id].customer_id
        t = Truth(payment_id=p.id, kind=kind, category=category, allocations=allocs or [], on_account=on_account, tags=list(tags or []), customer_id=customer_id)
        self.truth[p.id] = t
        return t

    def _maybe_noise_refs(self, invs, tags):
        """For categories that are not about references: a small share still have a damaged one."""
        if self.rng.random() < 0.06:
            tags.append("garbled_ref")
            return {i.id: self.noisy(i)[0] for i in invs[:1]}
        return {}

    # ---- case builders -----------------------------------------------------------------------
    def _single_text(self, inv: Invoice, ref_text: Optional[str], amount_in_memo: bool = False, extra: str = "") -> tuple:
        """(reference field, memo) for one invoice."""
        rng = self.rng
        shown = self.disp_ref(inv.digits) if ref_text is None else ref_text
        if amount_in_memo:
            shown = f"{shown} {fmt_money(inv.amount)}"
        memo = f"{shown} {extra}".strip()
        if rng.random() < 0.3 and not extra and not amount_in_memo:
            return shown, rng.choice(GENERIC_MEMO)
        return self.trace(), memo

    def case_exact(self) -> list:
        rng = self.rng
        # a pair of look-alike invoices (same payer, same amount, different references) that must not be mistaken for a duplicate
        if rng.random() < 0.12:
            twin = self.pick(2, pred=None)
            if twin:
                c, invs = twin
                amt = invs[0].amount
                for i in invs:
                    self.invoices[i.id].amount = amt
                    self.rem[i.id] = amt
                out = []
                d1 = self.pay_date(invs[:1], c)
                if d1 is None:
                    return []
                self.claim(invs)
                payer, tags = self.payer(c)
                for k, inv in enumerate(invs):
                    d = d1 + dt.timedelta(days=rng.randint(0, 4) + 0 * k)
                    d = min(max(d, inv.issue_date + dt.timedelta(days=1)), END)
                    ref, memo = self._single_text(inv, None)
                    p = self.new_payment(payer, amt, c.ccy, d, ref, memo)
                    self._truth(p, "apply", "exact", [TruthAlloc(inv.id, amt, amt, "clean")], tags=tags + ["lookalike"])
                    out.append(p)
                return out
        got = self.pick(1)
        if not got:
            return []
        c, (inv,) = got
        d = self.pay_date([inv], c)
        if d is None:
            return []
        self.claim([inv])
        payer, tags = self.payer(c)
        ref_text = None
        if rng.random() < 0.06:
            tags = tags + ["garbled_ref"]
            ref_text = self.noisy(inv)[0]
        extra = 0
        if rng.random() < 0.05:  # pays the invoice plus a round amount on account
            extra = rng.choice([5_000, 10_000, 25_000, 50_000])
            tags = tags + ["overpay"]
        ref, memo = self._single_text(inv, ref_text, amount_in_memo=rng.random() < 0.3, extra="ON ACCT" if extra and rng.random() < 0.6 else "")
        if not extra and rng.random() < 0.10:  # nothing usable in the remittance: payer, amount and dates only
            ref, memo, tags = self.trace(), rng.choice(GENERIC_MEMO), tags + ["no_ref"]
        p = self.new_payment(payer, inv.amount + extra, c.ccy, d, ref, memo)
        self._truth(p, "apply", "exact", [TruthAlloc(inv.id, inv.amount, inv.amount, "clean")], on_account=extra, tags=tags)
        return [p]

    def case_multi_invoice(self) -> list:
        rng = self.rng
        k = rng.choice([2, 2, 3, 3, 4, 5, 6])
        got = self.pick(k)
        if not got:
            return []
        c, invs = got
        d = self.pay_date(invs, c)
        if d is None:
            return []
        self.claim(invs)
        total = sum(i.amount for i in invs)
        style = rng.choices(["amounts", "plain", "partial_refs", "none"], [0.35, 0.25, 0.15, 0.25])[0]
        tags = [f"refs_{style}"]
        if style == "amounts":
            memo = "; ".join(f"{self.disp_ref(i.digits)} {fmt_money(i.amount)}" for i in invs)
        elif style == "plain":
            memo = "PAYMENT FOR " + " ".join(self.disp_ref(i.digits) for i in invs)
        elif style == "partial_refs":
            shown = rng.sample(invs, rng.randint(1, max(1, len(invs) - 1)))
            memo = "INVS " + " ".join(i.digits for i in shown)
        else:
            memo = rng.choice(GENERIC_MEMO)
        payer, ptags = self.payer(c)
        p = self.new_payment(payer, total, c.ccy, d, self.trace(), memo)
        self._truth(p, "apply", "multi_invoice", [TruthAlloc(i.id, i.amount, i.amount, "clean") for i in invs], tags=tags + ptags)
        return [p]

    def case_split_payment(self) -> list:
        rng = self.rng
        got = self.pick(1, pred=lambda i: i.amount >= 200_000)
        if not got:
            return []
        c, (inv,) = got
        n = rng.choice([2, 2, 3])
        equal = rng.random() < 0.25
        if equal and inv.amount % n:
            equal = False
        if equal:
            parts = [inv.amount // n] * n
        else:
            cuts = sorted(rng.uniform(0.2, 0.8) for _ in range(n - 1))
            edges = [0.0] + cuts + [1.0]
            parts = [int(round((edges[k + 1] - edges[k]) * inv.amount)) for k in range(n)]
            parts[-1] = inv.amount - sum(parts[:-1])
            if min(parts) < 30_000 or (n == 2 and parts[0] / inv.amount > 0.88):
                return []
        d = self.pay_date([inv], c, hi=END - dt.timedelta(days=6 * (n - 1)))
        if d is None:
            return []
        self.claim([inv])
        payer, ptags = self.payer(c)
        out = []
        for k, part in enumerate(parts):
            if k:
                d = d + dt.timedelta(days=rng.randint(4, 16))
                if d > END:
                    d = END
            last = k == n - 1
            show_ref = rng.random() < 0.85
            hint = rng.random() < 0.55
            bits = []
            if show_ref:
                bits.append(self.disp_ref(inv.digits))
            if hint:
                bits.append(rng.choice(["FINAL PMT", "BALANCE"]) if last else rng.choice(["PARTIAL PMT", f"{k + 1} OF {n}", "PART PAYMENT"]))
            p = self.new_payment(payer, part, c.ccy, d, self.trace(), " ".join(bits) if bits else rng.choice(GENERIC_MEMO))
            cls = "clean" if last else "partial"
            self._truth(p, "apply", "split_payment", [TruthAlloc(inv.id, part, part, cls)], tags=ptags + (["equal_installments"] if equal else []))
            out.append(p)
        return out

    def case_short_pay(self) -> list:
        rng = self.rng
        k = 1 if rng.random() < 0.9 else 2
        got = self.pick(k)
        if not got:
            return []
        c, invs = got
        d = self.pay_date(invs, c)
        if d is None:
            return []
        self.claim(invs)
        total = sum(i.amount for i in invs)
        short = int(round(total * rng.uniform(0.01, 0.09)))
        if short < 150:
            return []
        big = max(invs, key=lambda i: i.amount)
        allocs = [TruthAlloc(i.id, i.amount - (short if i is big else 0), i.amount, "short_pay" if i is big else "clean") for i in invs]
        memo = " ".join(self.disp_ref(i.digits) for i in invs) if rng.random() > 0.15 else rng.choice(GENERIC_MEMO)
        payer, ptags = self.payer(c)
        p = self.new_payment(payer, total - short, c.ccy, d, self.trace(), memo)
        self._truth(p, "apply", "short_pay", allocs, tags=ptags)
        return [p]

    def case_early_discount(self) -> list:
        rng = self.rng
        got = self.pick(1, pred=lambda i: i.discount_pct > 0 and i.issue_date >= START - dt.timedelta(days=1) and i.issue_date <= END - dt.timedelta(days=3))
        if not got:
            return []
        c, (inv,) = got
        d = inv.issue_date + dt.timedelta(days=rng.randint(2, inv.discount_days))
        d = min(max(d, START), END)
        if d > inv.issue_date + dt.timedelta(days=inv.discount_days) or d <= inv.issue_date:
            return []
        self.claim([inv])
        paid = int(round(inv.amount * (1 - inv.discount_pct)))
        bits = [self.disp_ref(inv.digits)]
        if rng.random() < 0.5:
            bits.append(rng.choice([f"LESS {inv.discount_pct * 100:g}% DISC", f"{inv.discount_pct * 100:g}% DISC TAKEN", f"DISC {fmt_money(inv.amount - paid)}"]))
        payer, ptags = self.payer(c)
        p = self.new_payment(payer, paid, c.ccy, d, self.trace(), " ".join(bits))
        self._truth(p, "apply", "early_discount", [TruthAlloc(inv.id, paid, inv.amount, "discount")], tags=ptags)
        return [p]

    def case_deduction(self) -> list:
        rng = self.rng
        k = 1 if rng.random() < 0.85 else 2
        got = self.pick(k)
        if not got:
            return []
        c, invs = got
        d = self.pay_date(invs, c)
        if d is None:
            return []
        total = sum(i.amount for i in invs)
        ded = int(round(total * rng.uniform(0.02, 0.15)))
        if ded < 500:
            return []
        self.claim(invs)
        if rng.random() < 0.12:
            reason, text = "OTHER", rng.choice(UNKNOWN_CODES)
        else:
            reason = rng.choice(list(REASON_TEXT))
            text = rng.choice(REASON_TEXT[reason])
        listed = rng.random() < 0.7
        refs = " ".join(self.disp_ref(i.digits) for i in invs)
        memo = f"{refs} {text}" + (f" {fmt_money(ded)}" if listed else "")
        big = max(invs, key=lambda i: i.amount)
        allocs = [TruthAlloc(i.id, i.amount - (ded if i is big else 0), i.amount, f"deduction:{reason}" if i is big else "clean") for i in invs]
        payer, ptags = self.payer(c)
        p = self.new_payment(payer, total - ded, c.ccy, d, self.trace(), memo)
        self._truth(p, "apply", "deduction", allocs, tags=ptags + [f"reason_{reason}"])
        return [p]

    def case_rounding_fx(self) -> list:
        rng = self.rng
        fx = rng.random() < 0.5
        if fx:
            got = self.pick(1, cust_pred=lambda c: c.ccy != "USD")
        else:
            got = self.pick(1, pred=lambda i: i.amount % 100 != 0)
        if not got:
            return []
        c, (inv,) = got
        d = self.pay_date([inv], c)
        if d is None:
            return []
        self.claim([inv])
        if fx:
            actual = FX_TO_USD[c.ccy] * (1 + rng.choice([-1, 1]) * rng.uniform(0.001, 0.012))
            amount, ccy, tag = int(round(inv.amount * actual)), "USD", "fx"
        else:
            amount, ccy, tag = int(round(inv.amount / 100.0)) * 100, c.ccy, "rounding"
            if amount == inv.amount:
                return []
        payer, ptags = self.payer(c)
        p = self.new_payment(payer, amount, ccy, d, self.trace(), self.disp_ref(inv.digits))
        self._truth(p, "apply", "rounding_fx", [TruthAlloc(inv.id, inv.amount, inv.amount, "fx_rounding")], tags=ptags + [tag])
        return [p]

    def case_garbled_ref(self) -> list:
        rng = self.rng
        got = self.pick(1)
        if not got:
            return []
        c, (inv,) = got
        d = self.pay_date([inv], c)
        if d is None:
            return []
        self.claim([inv])
        text, op = self.noisy(inv)
        shown = text if op == "po" else self.disp_ref(text) if text.isdigit() else f"INV {text}"
        payer, ptags = self.payer(c)
        p = self.new_payment(payer, inv.amount, c.ccy, d, self.trace(), shown)
        self._truth(p, "apply", "garbled_ref", [TruthAlloc(inv.id, inv.amount, inv.amount, "clean")], tags=ptags + [f"noise_{op}"])
        return [p]

    def case_name_variant(self) -> list:
        rng = self.rng
        got = self.pick(1)
        if not got:
            return []
        c, (inv,) = got
        d = self.pay_date([inv], c)
        if d is None:
            return []
        self.claim([inv])
        payer, ptags = self.payer(c, force_variant=True)
        r = rng.random()
        if r < 0.6:
            memo, tags = rng.choice(GENERIC_MEMO), ["no_ref"]
        elif r < 0.8:
            memo, tags = self.noisy(inv)[0], ["garbled_ref"]
        else:
            memo, tags = self.disp_ref(inv.digits), []
        p = self.new_payment(payer, inv.amount, c.ccy, d, self.trace(), memo)
        self._truth(p, "apply", "name_variant", [TruthAlloc(inv.id, inv.amount, inv.amount, "clean")], tags=ptags + tags)
        return [p]

    def case_unapplied(self) -> list:
        rng = self.rng
        sub = rng.choices(["unknown_payer", "prepay", "stale"], [0.35, 0.40, 0.25])[0]
        d = START + dt.timedelta(days=rng.randrange(DAYS))
        if sub == "unknown_payer":
            name = f"{rng.choice(STRANGER_FIRST)} {rng.choice(STRANGER_SECOND)} {rng.choice(SUFFIX)}".upper()
            memo = self.disp_ref(str(rng.randrange(1_000_000, 9_999_999))) if rng.random() < 0.5 else rng.choice(GENERIC_MEMO)
            p = self.new_payment(name, self._amount(), "USD", d, self.trace(), memo)
        elif sub == "prepay":
            c = self.cust[rng.choice(list(self.cust))]
            name, _ = self.payer(c)
            p = self.new_payment(name, self._amount(), c.ccy, d, self.trace(), rng.choice(["DEPOSIT", "PREPAY", "ON ACCOUNT", "", "ADVANCE PAYMENT"]))
        else:
            c = self.cust[rng.choice(list(self.cust))]
            old = rng.choice([i for i in self.closed if i.customer_id == c.id])
            name, _ = self.payer(c)
            p = self.new_payment(name, old.amount, c.ccy, d, self.trace(), self.disp_ref(old.digits))
        self._truth(p, "unapplied", "unapplied", [], tags=[sub], customer_id="" if sub == "unknown_payer" else c.id)
        return [p]

    def make_duplicates(self, count: int) -> None:
        rng = self.rng
        pool = [p for p in self.payments if self.truth[p.id].kind == "apply" and self.truth[p.id].category not in ("split_payment",) and "lookalike" not in self.truth[p.id].tags]
        rng.shuffle(pool)
        made = 0
        for src in pool:
            if made >= count:
                break
            d = src.date + dt.timedelta(days=rng.randint(1, 6))
            if d > END:
                continue
            ref = src.reference if rng.random() < 0.7 else self.trace()
            dup = self.new_payment(src.payer_name, src.amount, src.currency, d, ref, src.memo)
            self._truth(dup, "duplicate", "duplicate", [], tags=[src.id], customer_id=self.truth[src.id].customer_id)
            self.payments.append(dup)
            made += 1

    # ---- driver ------------------------------------------------------------------------------
    def generate(self) -> Dataset:
        self.build_master()
        quota = {k: int(round(v * self.n_payments)) for k, v in self.mix.items() if k != "duplicate"}
        counts = {k: 0 for k in quota}
        stalls = 0
        while any(counts[k] < quota[k] for k in quota) and stalls < 400:
            k = self.rng.choice([k for k in quota if counts[k] < quota[k]])
            made = getattr(self, f"case_{k}")()
            if not made:
                stalls += 1
                continue
            self.payments.extend(made)
            counts[k] += len(made)
        self.make_duplicates(int(round(self.mix.get("duplicate", 0) * self.n_payments)))
        self.payments.sort(key=lambda p: (p.date, p.id))
        customers = [Customer(c.id, c.name, c.ccy) for c in self.cust.values()]
        meta = {"seed": self.seed, "n_customers": self.n_customers, "n_payments_requested": self.n_payments,
                "n_payments": len(self.payments), "batch_start": START.isoformat(), "batch_end": END.isoformat(),
                "synthetic": True}
        return Dataset(customers=customers, invoices=list(self.invoices.values()), closed_invoices=self.closed,
                       payments=self.payments, truth=self.truth, history=self.history, meta=meta)


def generate(seed: int = 7, n_customers: int = 120, n_payments: int = 1600, mix: Optional[dict] = None) -> Dataset:
    return Generator(seed, n_customers, n_payments, mix).generate()
