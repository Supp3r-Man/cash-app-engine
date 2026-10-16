import collections

from cashapp.generate import generate
from cashapp.schema import CATEGORIES, Dataset


def test_same_seed_same_data_different_seed_different_data():
    a, b, c = generate(3, 20, 150), generate(3, 20, 150), generate(4, 20, 150)
    assert a.to_json() == b.to_json()
    assert a.to_json() != c.to_json()


def test_every_hard_case_category_is_produced():
    ds = generate(5, 60, 700)
    seen = collections.Counter(t.category for t in ds.truth.values())
    assert set(seen) == set(CATEGORIES)
    classes = {a.cls for t in ds.truth.values() for a in t.allocations}
    assert {"clean", "discount", "short_pay", "partial", "fx_rounding"} <= classes
    assert any(c.startswith("deduction:") for c in classes)
    assert any(t.on_account for t in ds.truth.values())
    assert any("lookalike" in t.tags for t in ds.truth.values())


def test_truth_is_consistent_with_the_invoices():
    ds = generate(6, 40, 400)
    inv = {i.id: i for i in ds.invoices}
    settled = collections.Counter()
    for p in ds.payments:
        t = ds.truth[p.id]
        assert t.payment_id == p.id
        if t.kind != "apply":
            assert not t.allocations
            continue
        for a in t.allocations:
            assert a.invoice_id in inv and a.cash > 0 and a.settled >= a.cash - 1
            settled[a.invoice_id] += a.settled
        assert len({inv[a.invoice_id].customer_id for a in t.allocations}) == 1
    # no invoice is settled for more than it is worth (instalments add up to exactly the invoice)
    for iid, total in settled.items():
        assert total <= inv[iid].amount


def test_duplicates_copy_a_real_earlier_payment():
    ds = generate(7, 40, 400)
    by_id = {p.id: p for p in ds.payments}
    dups = [p for p in ds.payments if ds.truth[p.id].kind == "duplicate"]
    assert dups
    for p in dups:
        src = by_id[ds.truth[p.id].tags[0]]
        assert (p.payer_name, p.amount, p.currency, p.memo) == (src.payer_name, src.amount, src.currency, src.memo)
        assert 1 <= (p.date - src.date).days <= 6
        assert ds.truth[src.id].kind == "apply"


def test_payments_are_never_dated_before_their_invoice_exists():
    ds = generate(8, 40, 400)
    inv = {i.id: i for i in ds.invoices}
    for p in ds.payments:
        for a in ds.truth[p.id].allocations:
            assert p.date > inv[a.invoice_id].issue_date


def test_json_round_trip():
    ds = generate(9, 12, 100)
    back = Dataset.from_json(ds.to_json())
    assert back.to_json() == ds.to_json()
    assert back.payments[0].date == ds.payments[0].date


def test_data_is_visibly_synthetic():
    ds = generate(10, 10, 60)
    assert ds.meta["synthetic"] is True
    assert all(c.id.startswith("C") for c in ds.customers)
