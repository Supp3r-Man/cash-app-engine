"""Data model: customers, invoices, payments, ground truth and engine decisions.

All money is integer cents. Invoices are in their customer's currency; a payment can
arrive in a different currency (that is the FX case).
"""
from __future__ import annotations

import datetime as dt
import json
from dataclasses import asdict, dataclass, field
from typing import Optional

# Fixed, made-up rate table (units of USD per 1 unit of currency). The generator adds noise
# around it; the engine only knows this table, never the noisy rate a bank actually used.
FX_TO_USD = {"USD": 1.0, "EUR": 1.08, "GBP": 1.27, "CAD": 0.73}

CATEGORIES = [
    "exact",
    "multi_invoice",
    "split_payment",
    "short_pay",
    "early_discount",
    "deduction",
    "rounding_fx",
    "garbled_ref",
    "name_variant",
    "duplicate",
    "unapplied",
]

REASONS = ["PRICING", "DAMAGED", "SHORTAGE", "FREIGHT", "PROMO", "TAX", "RETURN", "OTHER"]


def convert(cents: int, src: str, dst: str) -> int:
    """Convert at the table rate."""
    if src == dst:
        return cents
    return int(round(cents * FX_TO_USD[src] / FX_TO_USD[dst]))


def usd(cents: int, ccy: str) -> float:
    return cents * FX_TO_USD[ccy] / 100.0


def money(cents: int, ccy: str = "") -> str:
    sign = "-" if cents < 0 else ""
    text = f"{sign}{abs(cents) / 100:,.2f}"
    return f"{text} {ccy}".strip()


@dataclass(frozen=True)
class Customer:
    id: str
    legal_name: str
    currency: str = "USD"


@dataclass
class Invoice:
    id: str
    number: str  # display number, e.g. INV-4821937; the digits are the matching key
    customer_id: str
    amount: int  # cents, in the customer's currency
    currency: str
    issue_date: dt.date
    due_date: dt.date
    discount_pct: float = 0.0  # early-pay discount, e.g. 0.02
    discount_days: int = 0  # discount window from the issue date
    po_number: Optional[str] = None

    @property
    def digits(self) -> str:
        return "".join(ch for ch in self.number if ch.isdigit())


@dataclass
class Payment:
    id: str
    payer_name: str  # as printed by the bank: messy
    amount: int  # cents
    currency: str
    date: dt.date
    reference: str = ""  # the bank's reference / end-to-end field
    memo: str = ""  # free-text remittance


@dataclass
class CustomerHistory:
    """What the engine is allowed to remember from past, already-applied payments."""

    payer_strings: dict = field(default_factory=dict)  # raw payer string -> times seen
    lag_mean: Optional[float] = None  # average days paid after the due date
    n: int = 0


@dataclass
class TruthAlloc:
    invoice_id: str
    cash: int  # cash applied, in the invoice's currency
    settled: int  # open balance this payment removed from the invoice
    cls: str  # clean | discount | deduction:<REASON> | short_pay | partial | fx_rounding


@dataclass
class Truth:
    payment_id: str
    kind: str  # apply | duplicate | unapplied
    category: str
    allocations: list = field(default_factory=list)  # list[TruthAlloc]
    on_account: int = 0  # cash left over after the allocations, in the invoice currency
    tags: list = field(default_factory=list)
    customer_id: str = ""  # who really paid ("" for a payer that is not a customer); used only to split the data

    def invoice_set(self) -> frozenset:
        return frozenset(a.invoice_id for a in self.allocations)

    def class_signature(self) -> tuple:
        return tuple(sorted(a.cls for a in self.allocations if a.cls != "clean"))


@dataclass
class Dataset:
    customers: list
    invoices: list  # open at the start of the batch
    closed_invoices: list  # already settled before the batch (the engine knows them)
    payments: list
    truth: dict  # payment_id -> Truth. The engine never reads this.
    history: dict  # customer_id -> CustomerHistory
    meta: dict = field(default_factory=dict)

    # ---- serialisation -------------------------------------------------------------
    def to_json(self) -> str:
        def enc(o):
            if isinstance(o, dt.date):
                return o.isoformat()
            raise TypeError(type(o))

        body = {
            "customers": [asdict(c) for c in self.customers],
            "invoices": [asdict(i) for i in self.invoices],
            "closed_invoices": [asdict(i) for i in self.closed_invoices],
            "payments": [asdict(p) for p in self.payments],
            "truth": {k: asdict(v) for k, v in self.truth.items()},
            "history": {k: asdict(v) for k, v in self.history.items()},
            "meta": self.meta,
        }
        return json.dumps(body, default=enc, indent=1)

    @staticmethod
    def from_json(text: str) -> "Dataset":
        d = json.loads(text)

        def inv(x):
            x = dict(x)
            x["issue_date"] = dt.date.fromisoformat(x["issue_date"])
            x["due_date"] = dt.date.fromisoformat(x["due_date"])
            return Invoice(**x)

        def pay(x):
            x = dict(x)
            x["date"] = dt.date.fromisoformat(x["date"])
            return Payment(**x)

        def truth(x):
            x = dict(x)
            x["allocations"] = [TruthAlloc(**a) for a in x["allocations"]]
            return Truth(**x)

        return Dataset(
            customers=[Customer(**c) for c in d["customers"]],
            invoices=[inv(i) for i in d["invoices"]],
            closed_invoices=[inv(i) for i in d["closed_invoices"]],
            payments=[pay(p) for p in d["payments"]],
            truth={k: truth(v) for k, v in d["truth"].items()},
            history={k: CustomerHistory(**v) for k, v in d["history"].items()},
            meta=d.get("meta", {}),
        )

    def save(self, path) -> None:
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(self.to_json())

    @staticmethod
    def load(path) -> "Dataset":
        with open(path, encoding="utf-8") as fh:
            return Dataset.from_json(fh.read())

    def without_truth(self) -> "Dataset":
        return Dataset(self.customers, self.invoices, self.closed_invoices, self.payments, {}, self.history, self.meta)


@dataclass
class PropAlloc:
    invoice_id: str
    cash: int
    settled: int
    cls: str


@dataclass
class Decision:
    payment_id: str
    status: str = "exception"  # auto_apply | exception (set by the policy)
    reason: Optional[str] = None  # why it is an exception
    proposal: list = field(default_factory=list)  # list[PropAlloc]
    on_account: int = 0
    confidence: float = 0.0
    margin: float = 1.0
    kind: str = ""  # exact | discount | deduction | fx_rounding | short_pay | partial | overpay | ...
    explanation: list = field(default_factory=list)
    runner_up: str = ""
    fuzzy_ref: bool = False

    def invoice_set(self) -> frozenset:
        return frozenset(a.invoice_id for a in self.proposal)

    def class_signature(self) -> tuple:
        return tuple(sorted(a.cls for a in self.proposal if a.cls != "clean"))

    def to_dict(self) -> dict:
        return asdict(self)
