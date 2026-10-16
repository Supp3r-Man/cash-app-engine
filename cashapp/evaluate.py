"""Scoring: pair precision/recall, dollars-weighted auto-apply against misapplied, per category.

Definitions used everywhere in this repo:

* A proposal is *correct* when the engine names exactly the invoices the truth settles, the set of
  non-clean classes (discount, deduction:<REASON>, short_pay, partial, fx_rounding) is the same, and
  any cash left on account is within one currency unit. For a duplicate or unapplied payment the only
  correct proposal is none.
* A payment is *misapplied* when the policy auto-applied it and its proposal is not correct.
* Dollars are payment amounts converted to USD at the table rate.
"""
from __future__ import annotations

import random

from .policy import apply_policy
from .schema import CATEGORIES, usd

ON_ACCOUNT_TOL = 100


def proposal_correct(dec, truth) -> bool:
    if truth.kind != "apply":
        return not dec.proposal
    if not dec.proposal:
        return False
    return (dec.invoice_set() == truth.invoice_set() and dec.class_signature() == truth.class_signature()
            and abs(dec.on_account - truth.on_account) <= ON_ACCOUNT_TOL)


def pair_counts(dec, truth) -> tuple:
    prop = dec.invoice_set()
    gold = truth.invoice_set() if truth.kind == "apply" else frozenset()
    return len(prop & gold), len(prop), len(gold)


def _prf(tp: int, n_prop: int, n_gold: int) -> dict:
    p = tp / n_prop if n_prop else 1.0
    r = tp / n_gold if n_gold else 1.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": p, "recall": r, "f1": f}


def _rows(ds, decisions):
    pays = {p.id: p for p in ds.payments}
    for d in decisions:
        p = pays[d.payment_id]
        yield d, ds.truth[d.payment_id], usd(p.amount, p.currency)


def policy_metrics(ds, decisions, threshold: float, margin_ref: float = 0.10) -> dict:
    """Headline numbers at one threshold."""
    dec = apply_policy(decisions, threshold, margin_ref)
    total = auto = mis = 0.0
    n = n_auto = n_mis = 0
    wrong = caught = 0
    for d, t, v in _rows(ds, dec):
        n += 1
        total += v
        ok = proposal_correct(d, t)
        if d.proposal and not ok:
            wrong += 1
            caught += d.status != "auto_apply"
        if d.status == "auto_apply":
            n_auto += 1
            auto += v
            if not ok:
                n_mis += 1
                mis += v
    return {
        "threshold": threshold, "n": n, "n_auto": n_auto, "n_misapplied": n_mis,
        "auto_apply_rate_dollars": auto / total if total else 0.0,
        "auto_apply_rate_count": n_auto / n if n else 0.0,
        "misapplied_rate_dollars": mis / auto if auto else 0.0,
        "misapplied_rate_count": n_mis / n_auto if n_auto else 0.0,
        "wrong_proposals_caught": caught / wrong if wrong else 1.0,
        "total_dollars": total, "auto_dollars": auto, "misapplied_dollars": mis,
    }


def raw_metrics(ds, decisions) -> dict:
    """Threshold-free matching quality: pair precision/recall and the share of payments proposed correctly."""
    tp = npr = ng = 0
    ok = tot = 0
    ok_apply = tot_apply = 0
    for d, t, _ in _rows(ds, decisions):
        a, b, c = pair_counts(d, t)
        tp, npr, ng = tp + a, npr + b, ng + c
        good = proposal_correct(d, t)
        ok += good
        tot += 1
        if t.kind == "apply":
            tot_apply += 1
            ok_apply += good
    out = _prf(tp, npr, ng)
    out.update({"payment_correct_rate": ok / tot if tot else 0.0,
                "payment_correct_rate_apply_only": ok_apply / tot_apply if tot_apply else 0.0, "n": tot})
    return out


def sweep(ds, decisions, thresholds, margin_ref: float = 0.10) -> list:
    return [policy_metrics(ds, decisions, t, margin_ref) for t in thresholds]


def choose_threshold(ds, decisions, target: float = 0.01, grid=None, margin_ref: float = 0.10) -> float:
    """Lowest threshold whose misapplied-dollar rate is at or under the target (that maximises automation).

    Misapplied rate is not monotone in the threshold on a finite sample, so every threshold on the grid is
    checked and the lowest one that satisfies the target wins. If none does, the highest threshold is used.
    """
    grid = grid or [round(0.30 + 0.01 * k, 2) for k in range(0, 71)]
    best = None
    for t in grid:
        m = policy_metrics(ds, decisions, t, margin_ref)
        if m["misapplied_rate_dollars"] <= target and m["n_auto"] > 0:
            best = t
            break
    return best if best is not None else grid[-1]


def target_met(ds, decisions, threshold: float, target: float, margin_ref: float = 0.10) -> bool:
    """Whether this threshold keeps misapplied dollars at or under the target while auto-applying anything at all."""
    m = policy_metrics(ds, decisions, threshold, margin_ref)
    return m["n_auto"] > 0 and m["misapplied_rate_dollars"] <= target


def by_category(ds, decisions, threshold: float, margin_ref: float = 0.10) -> dict:
    dec = apply_policy(decisions, threshold, margin_ref)
    out = {}
    for cat in CATEGORIES:
        rows = [(d, t, v) for d, t, v in _rows(ds, dec) if t.category == cat]
        if not rows:
            continue
        tp = npr = ng = 0
        ok = 0
        total = auto = mis = 0.0
        for d, t, v in rows:
            a, b, c = pair_counts(d, t)
            tp, npr, ng = tp + a, npr + b, ng + c
            good = proposal_correct(d, t)
            ok += good
            total += v
            if d.status == "auto_apply":
                auto += v
                if not good:
                    mis += v
        row = _prf(tp, npr, ng)
        n_auto = sum(d.status == "auto_apply" for d, _, _ in rows)
        n_mis = sum(d.status == "auto_apply" and not proposal_correct(d, t) for d, t, _ in rows)
        row.update({"n": len(rows), "n_gold_pairs": ng, "n_auto": n_auto, "n_misapplied": n_mis, "proposal_correct_rate": ok / len(rows),
                    "auto_apply_rate_dollars": auto / total if total else 0.0,
                    "misapplied_rate_dollars": mis / auto if auto else 0.0})
        out[cat] = row
    return out


def bootstrap(ds, decisions, threshold: float, n_boot: int = 1000, seed: int = 11, margin_ref: float = 0.10) -> dict:
    """95% intervals for the auto-apply and misapplied dollar rates, resampling payments."""
    dec = apply_policy(decisions, threshold, margin_ref)
    rows = [(v, d.status == "auto_apply", d.status == "auto_apply" and not proposal_correct(d, t)) for d, t, v in _rows(ds, dec)]
    rng = random.Random(seed)
    autos, miss = [], []
    n = len(rows)
    for _ in range(n_boot):
        total = auto = mis = 0.0
        for _i in range(n):
            v, a, m = rows[rng.randrange(n)]
            total += v
            if a:
                auto += v
                if m:
                    mis += v
        autos.append(auto / total if total else 0.0)
        miss.append(mis / auto if auto else 0.0)
    autos.sort()
    miss.sort()
    lo, hi = int(0.025 * n_boot), int(0.975 * n_boot) - 1
    return {"auto_apply_rate_dollars_ci": [autos[lo], autos[hi]], "misapplied_rate_dollars_ci": [miss[lo], miss[hi]]}
