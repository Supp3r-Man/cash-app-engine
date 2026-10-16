import itertools
import random

from cashapp.engine import Config, Engine, subset_sum_solutions
from cashapp.generate import generate
from cashapp.policy import apply_policy
from helpers import customer, dataset, invoice, payment, three_invoice_ds


def run(ds, cfg=None, threshold=0.8):
    cfg = cfg or Config()
    decs = apply_policy(Engine(ds, cfg).run(), threshold, cfg.margin_ref)
    return {x.payment_id: x for x in decs}


def ids(dec):
    return {a.invoice_id for a in dec.proposal}


def test_exact_single_invoice_with_reference():
    ds = three_invoice_ds([payment("P1", "ACME MANUFACTURING INC", 25_050, ref="ACH-1", memo="INV 2222222")])
    x = run(ds)["P1"]
    assert x.status == "auto_apply" and ids(x) == {"I2"} and x.kind == "exact"
    text = " ".join(x.explanation)
    assert "INV-2222222" in text and "exactly" in text and "Reference 2222222 matches" in text


def test_payer_and_amount_alone_find_the_invoice_without_any_reference():
    ds = three_invoice_ds([payment("P1", "ACH CREDIT ACME MFG", 7_525, memo="THANK YOU")])
    x = run(ds)["P1"]
    assert ids(x) == {"I3"} and "No usable invoice reference" in " ".join(x.explanation)


def test_one_payment_for_many_invoices_found_by_subset_sum():
    ds = three_invoice_ds([payment("P1", "ACME MANUFACTURING INC", 25_050 + 31_010 + 7_525, memo="PMT")])
    x = run(ds)["P1"]
    assert ids(x) == {"I2", "I3", "I4"} and x.kind == "exact"
    assert "to the cent" in " ".join(x.explanation)


def test_references_that_name_only_some_invoices_still_find_the_whole_set():
    ds = three_invoice_ds([payment("P1", "ACME MANUFACTURING INC", 25_050 + 31_010, memo="INVS 2222222")])
    x = run(ds)["P1"]
    assert ids(x) == {"I2", "I4"}


def test_early_pay_discount_is_recognised():
    invs = [invoice("I1", "1111111", 100_000, issue="2025-09-01", disc=(0.02, 10)), invoice("I2", "2222222", 98_000)]
    ds = dataset([customer()], invs, [payment("P1", "ACME MANUFACTURING INC", 98_000, date="2025-09-06", memo="INV 1111111")])
    x = run(ds)["P1"]
    assert ids(x) == {"I1"} and x.kind == "discount" and x.proposal[0].cls == "discount"
    assert "early-pay discount" in " ".join(x.explanation)


def test_discount_is_not_granted_after_the_window():
    invs = [invoice("I1", "1111111", 100_000, issue="2025-08-01", disc=(0.02, 10))]
    ds = dataset([customer()], invs, [payment("P1", "ACME MANUFACTURING INC", 98_000, date="2025-09-20", memo="INV 1111111")])
    assert run(ds)["P1"].kind != "discount"


def test_listed_deduction_is_classified_by_reason():
    ds = three_invoice_ds([payment("P1", "ACME MANUFACTURING INC", 100_000 - 4_500, memo="INV 1111111 DED DC01 45.00")])
    x = run(ds)["P1"]
    assert ids(x) == {"I1"} and x.proposal[0].cls == "deduction:PRICING" and x.kind == "deduction"
    off = run(ds, Config(disable=frozenset({"classifier"})))["P1"]
    assert off.proposal[0].cls == "short_pay"  # without the classifier the reason is lost


def test_unknown_deduction_code_is_other():
    ds = three_invoice_ds([payment("P1", "ACME MANUFACTURING INC", 100_000 - 4_500, memo="INV 1111111 DED CODE X17 45.00")])
    assert run(ds)["P1"].proposal[0].cls == "deduction:OTHER"


def test_unexplained_short_pay_is_proposed_but_queued():
    ds = three_invoice_ds([payment("P1", "ACME MANUFACTURING INC", 95_000, memo="INV 1111111")])
    x = run(ds, threshold=0.95)["P1"]
    assert ids(x) == {"I1"} and x.proposal[0].cls == "short_pay"
    assert x.status == "exception" and x.reason == "unexplained_short_pay"
    exact = run(three_invoice_ds([payment("P1", "ACME MANUFACTURING INC", 100_000, memo="INV 1111111")]), threshold=0.95)["P1"]
    assert exact.status == "auto_apply" and exact.confidence > x.confidence


def test_instalments_use_the_balance_left_by_the_first_payment():
    pays = [payment("P1", "ACME MANUFACTURING INC", 40_000, date="2025-09-05", memo="INV 1111111 PARTIAL PMT"),
            payment("P2", "ACME MANUFACTURING INC", 60_000, date="2025-09-12", memo="INV 1111111 FINAL PMT")]
    out = run(three_invoice_ds(pays), Config(ledger_min=0.0))
    assert out["P1"].proposal[0].cls == "partial"
    assert out["P2"].kind == "exact" and out["P2"].proposal[0].cls == "clean" and ids(out["P2"]) == {"I1"}
    no_ledger = run(three_invoice_ds(pays), Config(disable=frozenset({"ledger"})))
    assert no_ledger["P2"].kind != "exact"  # without the ledger the second payment looks like a short-pay


def test_low_confidence_proposals_do_not_touch_the_ledger():
    pays = [payment("P1", "ACME MANUFACTURING INC", 40_000, date="2025-09-05", memo="THANK YOU"),
            payment("P2", "ACME MANUFACTURING INC", 100_000, date="2025-09-12", memo="INV 1111111")]
    out = run(three_invoice_ds(pays), Config(ledger_min=0.99))
    assert out["P2"].kind == "exact" and ids(out["P2"]) == {"I1"}


def test_repeat_of_a_settled_payment_is_a_duplicate_but_a_lookalike_is_not():
    invs = [invoice("I1", "1111111", 50_000), invoice("I2", "2222222", 50_000), invoice("I3", "3333333", 12_345)]
    pays = [payment("P1", "ACME MANUFACTURING INC", 50_000, date="2025-09-05", memo="INV 1111111"),
            payment("P2", "ACME MANUFACTURING INC", 50_000, date="2025-09-07", memo="INV 2222222"),   # same amount, other invoice
            payment("P3", "ACME MANUFACTURING INC", 50_000, date="2025-09-08", memo="INV 1111111")]  # a real repeat
    out = run(dataset([customer()], invs, pays))
    assert out["P2"].status == "auto_apply" and ids(out["P2"]) == {"I2"}
    assert out["P3"].reason == "duplicate" and not out["P3"].proposal and out["P3"].status == "exception"
    assert "P1" in " ".join(out["P3"].explanation)
    off = run(dataset([customer()], invs, pays), Config(disable=frozenset({"dup"})))
    assert off["P3"].reason != "duplicate"


def test_unknown_payer_and_closed_invoice_references_are_not_applied():
    closed = [invoice("X1", "9999999", 20_000)]
    pays = [payment("P1", "ZEPHYR VENTURES LLC", 33_333, memo="THANK YOU"),
            payment("P2", "ACME MANUFACTURING INC", 20_000, memo="INV 9999999")]
    out = run(three_invoice_ds(pays, closed=closed))
    assert out["P1"].reason == "no_match" and not out["P1"].proposal
    assert out["P2"].status == "exception" and not out["P2"].proposal
    assert out["P2"].reason == "refs_closed_invoice" and "already settled" in " ".join(out["P2"].explanation)


def test_damaged_reference_is_repaired_by_fuzzy_matching_and_flagged():
    ds = three_invoice_ds([payment("P1", "ACME MANUFACTURING INC", 25_050, memo="INV 2222272")])
    x = run(ds)["P1"]
    assert ids(x) == {"I2"} and x.fuzzy_ref and "one digit off" in " ".join(x.explanation)
    off = run(ds, Config(disable=frozenset({"fuzzy_ref"})))["P1"]
    assert not off.fuzzy_ref  # still found by payer and amount, but not through the reference


def test_foreign_currency_payment_within_fx_tolerance_matches():
    cust = customer("C1", "Acme Manufacturing Inc", "EUR")
    inv = invoice("I1", "1111111", 100_000, ccy="EUR")
    other = invoice("I2", "2222222", 55_555, ccy="EUR")
    usd_paid = int(round(100_000 * 1.08 * 1.007))  # the bank used a rate 0.7% off the table
    ds = dataset([cust], [inv, other], [payment("P1", "ACME MANUFACTURING INC", usd_paid, memo="INV 1111111", ccy="USD")])
    x = run(ds)["P1"]
    assert ids(x) == {"I1"} and x.proposal[0].cls == "fx_rounding"
    ds2 = dataset([cust], [inv, other], [payment("P1", "ACME MANUFACTURING INC", int(100_000 * 1.08 * 1.05), memo="INV 1111111", ccy="USD")])
    assert run(ds2)["P1"].proposal[0].cls != "fx_rounding"  # 5% is a short or over payment, not FX noise


def test_overpayment_leaves_cash_on_account():
    ds = three_invoice_ds([payment("P1", "ACME MANUFACTURING INC", 100_000 + 25_000, memo="INV 1111111 ON ACCT")])
    x = run(ds)["P1"]
    assert ids(x) == {"I1"} and x.on_account == 25_000 and x.kind == "overpay"
    assert "surplus held on account" in " ".join(x.explanation)
    silent = run(three_invoice_ds([payment("P1", "ACME MANUFACTURING INC", 100_000 + 25_000, memo="INV 1111111")]), threshold=0.9)["P1"]
    assert silent.status == "exception" and silent.reason == "overpayment" and silent.confidence < x.confidence


def test_names_that_fit_two_customers_are_not_auto_applied():
    cs = [customer("C1", "Acme Manufacturing Inc"), customer("C2", "Acme Industries LLC")]
    invs = [invoice("I1", "1111111", 50_000, cid="C1"), invoice("I2", "2222222", 50_000, cid="C2")]
    ds = dataset(cs, invs, [payment("P1", "ACME", 50_000, memo="PMT")], history={})
    x = run(ds)["P1"]
    assert x.status == "exception" and x.margin < 0.1


def test_every_decision_has_an_explanation_and_the_engine_never_reads_truth():
    ds = generate(11, 15, 120)
    with_truth = [x.to_dict() for x in Engine(ds).run()]
    without = [x.to_dict() for x in Engine(ds.without_truth()).run()]
    assert with_truth == without
    assert all(x["explanation"] for x in with_truth)


def test_engine_is_deterministic():
    ds = generate(12, 15, 120)
    a = [x.to_dict() for x in Engine(ds).run()]
    b = [x.to_dict() for x in Engine(ds).run()]
    assert a == b


def test_subset_sum_matches_brute_force_and_respects_limits():
    rng = random.Random(3)
    for _ in range(60):
        vals = tuple(rng.randrange(1, 40) for _ in range(rng.randrange(2, 11)))
        target = rng.randrange(5, 80)
        for max_size in (2, 4):
            want = {c for k in range(1, max_size + 1) for c in itertools.combinations(range(len(vals)), k) if sum(vals[i] for i in c) == target}
            got = set(subset_sum_solutions(vals, (target,), max_size, cap=10_000))
            assert got == want
    assert len(subset_sum_solutions(tuple([1] * 20), (3,), 6, cap=5)) <= 5  # the cap bounds the output


def test_ablation_flags_change_behaviour():
    ds = three_invoice_ds([payment("P1", "ACME MFG", 25_050, memo="INV 2222222")])
    base = run(ds)["P1"]
    no_name = run(ds, Config(disable=frozenset({"name"})))["P1"]
    assert "name similarity" in " ".join(base.explanation) and "name similarity" not in " ".join(no_name.explanation)
