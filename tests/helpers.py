"""Tiny hand-built datasets for engine tests."""
import datetime as dt

from cashapp.schema import Customer, CustomerHistory, Dataset, Invoice, Payment


def d(s: str) -> dt.date:
    return dt.date.fromisoformat(s)


def customer(cid="C1", name="Acme Manufacturing Inc", ccy="USD"):
    return Customer(cid, name, ccy)


def invoice(iid, digits, amount, cid="C1", issue="2025-08-01", net=30, disc=(0.0, 0), po=None, ccy="USD"):
    issue_d = d(issue)
    return Invoice(id=iid, number=f"INV-{digits}", customer_id=cid, amount=amount, currency=ccy, issue_date=issue_d,
                   due_date=issue_d + dt.timedelta(days=net), discount_pct=disc[0], discount_days=disc[1], po_number=po)


def payment(pid, payer, amount, date="2025-09-05", ref="", memo="", ccy="USD"):
    return Payment(id=pid, payer_name=payer, amount=amount, currency=ccy, date=d(date), reference=ref, memo=memo)


def dataset(customers, invoices, payments, closed=(), history=None):
    hist = history if history is not None else {c.id: CustomerHistory({c.legal_name.upper(): 5}, 0.0, 5) for c in customers}
    return Dataset(customers=list(customers), invoices=list(invoices), closed_invoices=list(closed), payments=list(payments),
                   truth={}, history=hist, meta={})


def three_invoice_ds(payments, **kw):
    invs = [invoice("I1", "1111111", 100_000), invoice("I2", "2222222", 25_050), invoice("I3", "3333333", 7_525), invoice("I4", "4444444", 31_010)]
    return dataset([customer()], invs, payments, **kw)
