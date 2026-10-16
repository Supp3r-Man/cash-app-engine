import copy
import random

import pytest

from cashapp.engine import Config
from cashapp.evaluate import bootstrap, by_category, choose_threshold, pair_counts, policy_metrics, proposal_correct, raw_metrics
from cashapp.generate import generate
from cashapp.schema import Decision, PropAlloc, Truth, TruthAlloc
from cashapp.tune import TOLERANCE, run_experiment, split_dataset, tune
from helpers import customer, dataset, invoice, payment


def _mini():
    """Three payments of $100, $300 and $600; the engine is right on the first, wrong on the second, unsure on the third."""
    invs = [invoice(f"I{k}", f"{k}000000", 10_000) for k in range(1, 7)]
    pays = [payment("P1", "ACME MANUFACTURING INC", 10_000), payment("P2", "ACME MANUFACTURING INC", 30_000), payment("P3", "ACME MANUFACTURING INC", 60_000)]
    ds = dataset([customer()], invs, pays)
    ds.truth = {
        "P1": Truth("P1", "apply", "exact", [TruthAlloc("I1", 10_000, 10_000, "clean")]),
        "P2": Truth("P2", "apply", "multi_invoice", [TruthAlloc(f"I{k}", 10_000, 10_000, "clean") for k in (2, 3, 4)]),
        "P3": Truth("P3", "unapplied", "unapplied", []),
    }
    decs = [
        Decision("P1", proposal=[PropAlloc("I1", 10_000, 10_000, "clean")], confidence=0.95, kind="exact"),
        Decision("P2", proposal=[PropAlloc("I2", 10_000, 10_000, "clean"), PropAlloc("I3", 10_000, 10_000, "clean"), PropAlloc("I5", 10_000, 10_000, "clean")], confidence=0.90, kind="exact"),
        Decision("P3", proposal=[PropAlloc("I6", 10_000, 10_000, "clean")], confidence=0.50, kind="overpay"),
    ]
    return ds, decs


def test_proposal_correctness_rules():
    ds, decs = _mini()
    t = ds.truth
    assert proposal_correct(decs[0], t["P1"])
    assert not proposal_correct(decs[1], t["P2"])  # one invoice wrong
    assert not proposal_correct(decs[2], t["P3"])  # anything proposed for unapplied cash is wrong
    assert proposal_correct(Decision("P3"), t["P3"])  # proposing nothing is right
    wrong_class = Decision("P1", proposal=[PropAlloc("I1", 10_000, 10_000, "short_pay")])
    assert not proposal_correct(wrong_class, t["P1"])


def test_pair_counts_hand_computed():
    ds, decs = _mini()
    assert pair_counts(decs[0], ds.truth["P1"]) == (1, 1, 1)
    assert pair_counts(decs[1], ds.truth["P2"]) == (2, 3, 3)
    assert pair_counts(decs[2], ds.truth["P3"]) == (0, 1, 0)
    raw = raw_metrics(ds, decs)
    assert raw["precision"] == pytest.approx(3 / 5) and raw["recall"] == pytest.approx(3 / 4)
    assert raw["payment_correct_rate"] == pytest.approx(1 / 3)


def test_dollar_weighted_policy_metrics_hand_computed():
    ds, decs = _mini()
    m = policy_metrics(ds, decs, 0.8)
    # P1 ($100, right) and P2 ($300, wrong) are auto-applied; P3 ($600) is queued
    assert m["auto_apply_rate_dollars"] == pytest.approx(400 / 1000)
    assert m["misapplied_rate_dollars"] == pytest.approx(300 / 400)
    assert m["n_auto"] == 2 and m["n_misapplied"] == 1
    assert m["wrong_proposals_caught"] == pytest.approx(1 / 2)  # of the two wrong proposals, P3's was queued
    m95 = policy_metrics(ds, decs, 0.93)
    assert m95["auto_apply_rate_dollars"] == pytest.approx(100 / 1000) and m95["misapplied_rate_dollars"] == 0.0


def test_choose_threshold_is_the_lowest_one_meeting_the_target():
    ds, decs = _mini()
    assert choose_threshold(ds, decs, target=0.01) == pytest.approx(0.91)  # first step above P2's 0.90
    assert choose_threshold(ds, decs, target=1.0) == pytest.approx(0.30)


def test_bootstrap_interval_brackets_the_point_estimate():
    ds, decs = _mini()
    ci = bootstrap(ds, decs, 0.8, n_boot=300)
    point = policy_metrics(ds, decs, 0.8)["auto_apply_rate_dollars"]
    lo, hi = ci["auto_apply_rate_dollars_ci"]
    assert lo <= point <= hi


def test_by_category_rows_cover_the_categories_present():
    ds, decs = _mini()
    rows = by_category(ds, decs, 0.8)
    assert set(rows) == {"exact", "multi_invoice", "unapplied"}
    assert rows["exact"]["proposal_correct_rate"] == 1.0 and rows["multi_invoice"]["misapplied_rate_dollars"] == 1.0


def test_split_is_by_customer_reproducible_and_disjoint():
    ds = generate(21, 30, 260)
    a, b = split_dataset(ds, 5)
    a2, b2 = split_dataset(ds, 5)
    assert [p.id for p in a.payments] == [p.id for p in a2.payments]
    assert {c.id for c in a.customers}.isdisjoint({c.id for c in b.customers})
    assert {p.id for p in a.payments}.isdisjoint({p.id for p in b.payments})
    assert len(a.payments) + len(b.payments) == len(ds.payments)
    for half in (a, b):
        cids = {c.id for c in half.customers}
        assert {i.customer_id for i in half.invoices} <= cids and {i.customer_id for i in half.closed_invoices} <= cids
        inv_owner = {i.id: i.customer_id for i in half.invoices}
        for p in half.payments:
            for al in half.truth[p.id].allocations:
                assert inv_owner[al.invoice_id] in cids  # a payment's invoices are on its own side
    c, _ = split_dataset(ds, 6)
    assert {p.id for p in c.payments} != {p.id for p in a.payments}  # a different seed gives a different split


def test_tune_keeps_a_change_only_if_it_beats_the_tolerance():
    ds = generate(22, 16, 130)
    a, _ = split_dataset(ds, 5)
    grid = [("margin_pen", [0.0, 0.5])]
    res = tune(a, grid=grid, passes=1)
    start = res.trials[0]["auto"]
    assert len(res.trials) == 3 and res.trials[0]["stage"] == "start"
    if res.config.margin_pen != Config().margin_pen:
        assert res.objective["auto"] > start + TOLERANCE
    else:
        assert all(t["auto"] <= start + TOLERANCE for t in res.trials[1:])


def _scramble_held_out_truth(ds, held_payment_ids, seed=0):
    """Swap the truth labels between held-out payments, so held-out scores change but nothing else does."""
    bad = copy.copy(ds)
    bad.truth = dict(ds.truth)
    ids = sorted(held_payment_ids)
    shuffled = ids[:]
    random.Random(seed).shuffle(shuffled)
    for src, dst in zip(ids, shuffled):
        t = copy.deepcopy(ds.truth[src])
        t.payment_id = dst
        t.customer_id = ds.truth[dst].customer_id  # ownership decides which half a payment is in: leave it alone
        bad.truth[dst] = t
    return bad


def test_held_out_labels_cannot_influence_tuning():
    ds = generate(23, 24, 200)
    _, held = split_dataset(ds, 9)
    clean = run_experiment(ds, split_seed=9, quick=True)
    poisoned = run_experiment(_scramble_held_out_truth(ds, [p.id for p in held.payments]), split_seed=9, quick=True)
    # everything chosen during tuning is identical ...
    assert clean["tuning"] == poisoned["tuning"]
    # ... and only the held-out scores move
    assert clean["held_out"]["tuned"]["raw"] != poisoned["held_out"]["tuned"]["raw"]
