"""The matching engine: resolve the payer, find invoice candidates, search allocations, explain.

For each payment (in date order) the engine:

1. parses the remittance text (references, listed amounts, deduction words, discount, hints);
2. resolves the payer to a few candidate customers (name similarity, alias memory, and any customer
   that owns an invoice the references point at);
3. builds *hypotheses* per candidate customer: a set of open invoices plus how the cash is explained
   (clean, FX/rounding, early-pay discount, listed deduction, unexplained short-pay, partial
   instalment, overpayment). Sets come from three places: single invoices, subsets of the invoices the
   references point at, and an exact subset-sum search (meet in the middle, bounded) over the
   customer's open invoices when nothing anchors the payment;
4. scores each hypothesis from explainable features and picks the best, with the margin over the
   runner-up as an ambiguity signal;
5. checks for duplicates and updates a tentative ledger so a later instalment sees the balance left.

The engine never reads ground truth: it gets a Dataset and uses customers, invoices, closed invoices,
payments and history only.
"""
from __future__ import annotations

import datetime as dt
import math
from bisect import bisect_left
from dataclasses import dataclass, field
from typing import Optional

from .classify import reason_from_text  # noqa: F401  (re-exported for callers)
from .names import NameIndex, normalize
from .refs import RefIndex
from .remit import RuleRemitParser, TokensOnlyParser
from .schema import Decision, PropAlloc, convert, money

DEFAULT_WEIGHTS = {"ref": 3.0, "name": 2.0, "amount": 2.0, "date": 0.5, "history": 1.0}
DEFAULT_EVIDENCE = {
    "exact": 1.0, "fx_rounding": 0.95, "rounding": 0.85, "discount": 0.90, "deduction_explained": 0.90, "deduction_reason": 0.75,
    "deduction_mismatch": 0.60, "short_pay": 0.45, "partial_hint": 0.75, "partial": 0.40, "overpay": 0.50, "overpay_hint": 0.60,
}
ABLATIONS = ["ref", "fuzzy_ref", "name", "amount", "date", "history", "subset", "remit", "classifier", "dup", "ledger"]


@dataclass
class Config:
    weights: dict = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    evidence: dict = field(default_factory=lambda: dict(DEFAULT_EVIDENCE))
    margin_pen: float = 0.30
    margin_ref: float = 0.10
    name_floor: float = 0.45
    top_customers: int = 3
    unapplied_floor: float = 0.40
    strong_ref: float = 0.75
    mitm_pool: int = 30
    ledger_min: float = 0.80  # a proposal below this confidence does not reduce the tentative ledger
    max_set: int = 6
    round_tol: int = 100  # cents: customers rounding to whole currency units
    fx_tol: float = 0.015  # 1.5% FX spread
    dup_days: int = 10
    dup_override: float = 0.85
    disable: frozenset = frozenset()

    def to_dict(self) -> dict:
        return {"weights": self.weights, "evidence": self.evidence, "margin_pen": self.margin_pen, "margin_ref": self.margin_ref,
                "name_floor": self.name_floor, "top_customers": self.top_customers, "unapplied_floor": self.unapplied_floor,
                "strong_ref": self.strong_ref, "mitm_pool": self.mitm_pool, "ledger_min": self.ledger_min, "max_set": self.max_set, "round_tol": self.round_tol,
                "fx_tol": self.fx_tol, "dup_days": self.dup_days, "dup_override": self.dup_override, "disable": sorted(self.disable)}

    @staticmethod
    def from_dict(d: dict) -> "Config":
        d = dict(d)
        d["disable"] = frozenset(d.get("disable", ()))
        return Config(**d)

    def replace(self, **kw) -> "Config":
        d = self.to_dict()
        d.update(kw)
        if "disable" in kw:
            d["disable"] = frozenset(kw["disable"])
        else:
            d["disable"] = frozenset(d["disable"])
        d["weights"] = dict(d["weights"])
        d["evidence"] = dict(d["evidence"])
        return Config(**d)


# ---- bounded exact subset-sum (meet in the middle) ---------------------------------------------
_MITM_CACHE: dict = {}


def subset_sum_solutions(values: tuple, targets: tuple, max_size: int, cap: int = 40) -> list:
    """Index sets (as tuples) whose values sum to exactly one of the targets, 1..max_size items.

    Meet in the middle: enumerate subsets of each half (size-capped), sort one side, binary-search the
    complement. Bounded: at most 2^(n/2) sums per side, at most `cap` solutions returned.
    Results are cached because tuning re-runs the engine on identical states many times.
    """
    key = (values, targets, max_size, cap)
    hit = _MITM_CACHE.get(key)
    if hit is not None:
        return hit
    n = len(values)
    h = n // 2

    def enum(vals, offset):
        subs = [(0, (), 0)]
        for i, v in enumerate(vals):
            subs.extend([(s + v, idx + (offset + i,), k + 1) for s, idx, k in subs if k < max_size])
        return subs

    left = enum(values[:h], 0)
    right = enum(values[h:], h)
    right.sort(key=lambda x: x[0])
    rsums = [x[0] for x in right]
    found: dict = {}
    for tgt in targets:
        for s, idx, k in left:
            need = tgt - s
            if need < 0:
                continue
            j = bisect_left(rsums, need)
            while j < len(rsums) and rsums[j] == need:
                s2, idx2, k2 = right[j]
                if 0 < k + k2 <= max_size:
                    found[tuple(sorted(idx + idx2))] = True
                j += 1
            if len(found) >= cap:
                break
    out = list(found)[:cap]
    if len(_MITM_CACHE) > 200_000:
        _MITM_CACHE.clear()
    _MITM_CACHE[key] = out
    return out


# ---- hypotheses ---------------------------------------------------------------------------------
@dataclass
class Hyp:
    cid: str
    invs: tuple  # Invoice objects
    kind: str  # exact | fx_rounding | discount | deduction | short_pay | partial | overpay
    cls: dict  # invoice id -> class label
    evidence: float
    cash: dict  # invoice id -> (cash, settled)
    on_account: int = 0
    note: str = ""
    score: float = 0.0
    features: dict = field(default_factory=dict)

    @property
    def sig(self) -> tuple:
        return tuple(sorted(c for c in self.cls.values() if c != "clean"))

    @property
    def key(self) -> tuple:
        return (self.cid, frozenset(i.id for i in self.invs), self.sig, self.on_account > 0)


@dataclass
class Ctx:
    p: object
    parsed: object
    hits: dict  # invoice id -> RefHit
    has_refs: bool
    name_sims: dict
    alias: dict  # customer id -> times this payer string was seen


class Engine:
    def __init__(self, ds, cfg: Optional[Config] = None, parser=None):
        self.ds = ds
        self.cfg = cfg or Config()
        dis = self.cfg.disable
        self.parser = parser or (TokensOnlyParser() if "remit" in dis else RuleRemitParser())
        self.names = NameIndex(ds.customers)
        self.cust = {c.id: c for c in ds.customers}
        self.all_inv = {i.id: i for i in list(ds.invoices) + list(ds.closed_invoices)}
        self.refidx = RefIndex(self.all_inv.values(), fuzzy="fuzzy_ref" not in dis)
        self.open_by_cust: dict = {}
        for inv in ds.invoices:
            self.open_by_cust.setdefault(inv.customer_id, []).append(inv)
        self.alias: dict = {}  # normalised payer -> {customer id: count}
        self.lag: dict = {}
        self.hist_n: dict = {}
        if "history" not in dis:
            for cid, h in ds.history.items():
                self.hist_n[cid] = h.n
                if h.lag_mean is not None:
                    self.lag[cid] = h.lag_mean
                for s, n in h.payer_strings.items():
                    self.alias.setdefault(normalize(s), {})[cid] = self.alias.get(normalize(s), {}).get(cid, 0) + n
        self.rem: dict = {}

    # ---- features ----------------------------------------------------------------------------
    def _date_feature(self, p, inv) -> float:
        if p.date <= inv.issue_date:
            return 0.0
        d = (p.date - inv.due_date).days
        mu = self.lag.get(inv.customer_id, 0.0)
        return math.exp(-0.5 * ((d - mu) / 20.0) ** 2)

    def _ref_feature(self, ctx: Ctx, inv) -> Optional[float]:
        if "ref" in self.cfg.disable or not ctx.has_refs:
            return None
        h = ctx.hits.get(inv.id)
        if h is None:
            return 0.0
        sim = h.sim
        listed = ctx.parsed.amounts.get(h.token)
        if listed is not None and listed in (self.rem.get(inv.id), inv.amount):
            sim = max(sim, 0.95)  # an amount that matches confirms a damaged reference
        return sim

    def _score_inv(self, ctx: Ctx, inv, evidence: Optional[float], lenient_ref: bool = False) -> float:
        cfg, dis = self.cfg, self.cfg.disable
        cid = inv.customer_id
        ref = self._ref_feature(ctx, inv)
        if lenient_ref and ref == 0.0:
            ref = None  # an invoice the remittance does not name, inside an exact-sum set that includes every one it does name
        feats = {
            "ref": ref,
            "name": None if "name" in dis else ctx.name_sims.get(cid, 0.0),
            "amount": None if "amount" in dis else evidence,
            "date": None if "date" in dis else self._date_feature(ctx.p, inv),
            "history": None,
        }
        if "history" not in dis and self.hist_n.get(cid, 0) > 0:
            feats["history"] = 1.0 if ctx.alias.get(cid, 0) > 0 else 0.0
        num = den = 0.0
        for k, v in feats.items():
            if v is None:
                continue
            w = cfg.weights.get(k, 0.0)
            num += w * v
            den += w
        return num / den if den else 0.0

    def _score_hyp(self, ctx: Ctx, h: Hyp, strong_open: set) -> float:
        ids = {i.id for i in h.invs}
        lenient = h.evidence >= 0.9 and len(h.invs) > 1 and bool(strong_open) and strong_open <= ids
        per = [self._score_inv(ctx, i, h.evidence, lenient) for i in h.invs]
        score = sum(per) / len(per)
        if strong_open:
            missed = len(strong_open - {i.id for i in h.invs})
            score -= 0.15 * missed / len(strong_open)
        score -= 0.01 * (len(h.invs) - 1)
        return max(0.0, min(1.0, score))

    # ---- one set of invoices -> hypotheses ------------------------------------------------------
    def _eval_set(self, ctx: Ctx, cid: str, invs, T: int, fx: bool, anchored: bool, residual: str) -> list:
        """residual: 'none' (exact/discount only), 'explained' (+ listed deductions), 'all'."""
        cfg, p, dis = self.cfg, ctx.p, self.cfg.disable
        rem = [self.rem[i.id] for i in invs]
        rs = sum(rem)
        diff = T - rs
        tol = max(cfg.round_tol, int(cfg.fx_tol * T)) if fx else cfg.round_tol
        if not (anchored or len(invs) == 1):
            tol = 0
        big = max(invs, key=lambda i: self.rem[i.id])
        ev = cfg.evidence
        out = []

        def hyp(kind, cls_big, evidence, on_account=0, note="", cash_big_delta=0, partial=False):
            cls = {i.id: "clean" for i in invs}
            if cls_big != "clean":
                cls[big.id] = cls_big
            if partial:
                cash = {invs[0].id: (T, T)}
            else:
                cash = {i.id: (self.rem[i.id] - (cash_big_delta if i is big else 0), self.rem[i.id]) for i in invs}
            return Hyp(cid, tuple(invs), kind, cls, evidence, cash, on_account, note)

        if abs(diff) <= tol:
            if diff == 0 and not fx:
                out.append(hyp("exact", "clean", ev["exact"]))
            else:
                out.append(hyp("fx_rounding", "fx_rounding", ev["fx_rounding" if fx else "rounding"], note=f"diff={diff}", cash_big_delta=max(0, -diff)))
        # early-pay discount: all discount-eligible invoices taken at their discount
        elig = [i for i in invs if i.discount_pct > 0 and p.date <= i.issue_date + dt.timedelta(days=i.discount_days + 3)]
        if elig:
            disc_amt = sum(int(round(self.rem[i.id] * i.discount_pct)) for i in elig)
            if abs(T - (rs - disc_amt)) <= tol and diff < -tol:
                e = ev["discount"] + (0.05 if ctx.parsed.discount else 0.0)
                out.append(hyp("discount", "discount", min(1.0, e), note=f"pct={elig[0].discount_pct}", cash_big_delta=disc_amt))
        if residual != "none" and diff < -tol:
            short = -diff
            deds = ctx.parsed.deductions
            classify = "classifier" not in dis
            if deds:
                explained = [d for d in deds if d.amount is not None and abs(d.amount - short) <= max(tol, cfg.round_tol)]
                d0 = explained[0] if explained else deds[0]
                label = f"deduction:{d0.reason}" if classify else "short_pay"
                if explained:
                    e = ev["deduction_explained"]
                elif d0.amount is None:
                    e = ev["deduction_reason"]
                else:
                    e = ev["deduction_mismatch"]
                if explained or residual == "all":
                    out.append(hyp("deduction", label, e, note=f"reason={d0.reason}", cash_big_delta=short))
            elif residual == "all":
                if short / rs <= 0.12:
                    out.append(hyp("short_pay", "short_pay", ev["short_pay"], cash_big_delta=short))
                if len(invs) == 1 and T / rs <= 0.93:
                    e = ev["partial_hint"] if "partial" in ctx.parsed.hints else ev["partial"]
                    out.append(hyp("partial", "partial", e, partial=True))
        if residual == "all" and diff > tol and (anchored or len(invs) == 1):
            e = ev["overpay_hint"] if "on_account" in ctx.parsed.hints else ev["overpay"]
            out.append(hyp("overpay", "clean", e, on_account=diff))
        return out

    # ---- hypotheses for one customer -------------------------------------------------------------
    def _customer_hyps(self, ctx: Ctx, cid: str) -> list:
        cfg, p, dis = self.cfg, ctx.p, self.cfg.disable
        cust = self.cust[cid]
        T = convert(p.amount, p.currency, cust.currency)
        fx = p.currency != cust.currency
        pool = [i for i in self.open_by_cust.get(cid, []) if self.rem[i.id] > 0]
        if not pool or T <= 0:
            return []
        strong = [i for i in pool if i.id in ctx.hits and ctx.hits[i.id].sim >= cfg.strong_ref and "ref" not in dis]
        pre = {i.id: self._score_inv(ctx, i, 0.5) for i in pool}
        hyps: list = []
        # 1) single invoices: exact-style fits for all, residual explanations for the most plausible
        focus = {i.id for i in sorted(pool, key=lambda i: -pre[i.id])[:4]} | {i.id for i in strong}
        for inv in pool:
            hyps += self._eval_set(ctx, cid, [inv], T, fx, True, "all" if inv.id in focus else "none")
        if "subset" not in dis:
            # 2) subsets of the invoices the references point at
            if 2 <= len(strong) <= 8:
                n = len(strong)
                for mask in range(1, 1 << n):
                    sub = [strong[b] for b in range(n) if mask >> b & 1]
                    if len(sub) >= 2:
                        hyps += self._eval_set(ctx, cid, sub, T, fx, True, "all")
            # 3) exact subset-sum over the customer's open invoices when nothing anchors a complete answer
            anchored_exact = any(h.kind == "exact" and len(h.invs) >= 1 and {i.id for i in strong} <= {i.id for i in h.invs} and strong for h in hyps)
            if not fx and len(pool) >= 2 and not anchored_exact:
                targets = [T]
                deds = [d.amount for d in ctx.parsed.deductions if d.amount]
                if deds:
                    targets.append(T + sum(deds))
                # an invoice bigger than the payment (plus any listed deduction) cannot be in an exact set
                fits = [i for i in pool if self.rem[i.id] <= max(targets)]
                ranked = sorted(fits, key=lambda i: (i.id not in {s.id for s in strong}, -pre[i.id]))[: cfg.mitm_pool]
                ranked.sort(key=lambda i: self.rem[i.id])
                vals = tuple(self.rem[i.id] for i in ranked)
                sols = subset_sum_solutions(vals, tuple(targets), cfg.max_set)
                elig_idx = [k for k, i in enumerate(ranked) if i.discount_pct > 0 and p.date <= i.issue_date + dt.timedelta(days=i.discount_days + 3)]
                if elig_idx:
                    dv = tuple(self.rem[i.id] - (int(round(self.rem[i.id] * i.discount_pct)) if k in elig_idx else 0) for k, i in enumerate(ranked))
                    sols_d = subset_sum_solutions(dv, (T,), cfg.max_set)
                    for s in sols_d:
                        if any(k in elig_idx for k in s) and len(s) >= 2:
                            sols = sols + [s]
                seen_sets = set()
                for s in sols:
                    sub = [ranked[k] for k in s]
                    key = frozenset(i.id for i in sub)
                    if key in seen_sets or len(sub) < 2:
                        continue
                    seen_sets.add(key)
                    hyps += self._eval_set(ctx, cid, sub, T, fx, False, "explained")
        return hyps

    # ---- one payment --------------------------------------------------------------------------------
    def _decide(self, p, recent: dict) -> Decision:
        cfg, dis = self.cfg, self.cfg.disable
        parsed = self.parser.parse(p.reference, p.memo)
        hits = {} if "ref" in dis else self.refidx.lookup_all(parsed.tokens)
        has_refs = bool(parsed.tokens) and "ref" not in dis
        name_sims = {} if "name" in dis else self.names.scores(p.payer_name)
        alias = {} if "history" in dis else self.alias.get(normalize(p.payer_name), {})
        ctx = Ctx(p, parsed, hits, has_refs, name_sims, alias)

        cands: dict = {}
        if "name" not in dis:
            for cid, s in sorted(name_sims.items(), key=lambda x: (-x[1], x[0]))[: cfg.top_customers]:
                if s >= cfg.name_floor:
                    cands[cid] = s
        for cid in alias:
            cands.setdefault(cid, name_sims.get(cid, 0.0))
        strong_open_all = {}
        for inv_id, h in hits.items():
            inv = self.all_inv[inv_id]
            if h.sim >= cfg.strong_ref and self.rem.get(inv_id, 0) > 0:
                cands.setdefault(inv.customer_id, name_sims.get(inv.customer_id, 0.0))
                strong_open_all.setdefault(inv.customer_id, set()).add(inv_id)

        hyps: list = []
        for cid in cands:
            hs = self._customer_hyps(ctx, cid)
            strong = strong_open_all.get(cid, set())
            for h in hs:
                h.score = self._score_hyp(ctx, h, strong)
            hyps += hs
        hyps.sort(key=lambda h: (-h.score, len(h.invs), h.cid, tuple(i.id for i in h.invs)))

        dec = Decision(payment_id=p.id)
        closed_hits = [self.all_inv[i] for i, h in hits.items() if h.sim >= cfg.strong_ref and self.rem.get(i, 0) <= 0]
        best = hyps[0] if hyps else None

        # duplicate: same payer, amount and settled-invoice references as an earlier payment
        if "dup" not in dis:
            fp = self._fingerprint(p, hits)
            for q in reversed(recent.get(fp, ())):
                if 0 <= (p.date - q["date"]).days <= cfg.dup_days:
                    override = best is not None and best.kind == "exact" and best.score >= cfg.dup_override and {i.id for i in best.invs} != q["invs"]
                    # the earlier payment's invoice still has the balance to take this one: a further instalment, not a repeat
                    absorbs = bool(q["invs"]) and any(
                        self.rem[i] >= convert(p.amount, p.currency, self.all_inv[i].currency) - cfg.round_tol for i in q["invs"])
                    if not override and not absorbs:
                        dec.reason, dec.kind = "duplicate", "duplicate"
                        dec.explanation = [
                            f"Same payer, amount ({money(p.amount, p.currency)}) and references as payment {q['id']} on {q['date']}.",
                            "The invoices it would pay are already settled, so this looks like a re-presented duplicate. Held for review, nothing applied.",
                        ]
                        recent.setdefault(fp, []).append({"id": p.id, "date": p.date, "invs": set()})
                        return dec
                    break

        # a remittance that names only settled invoices is a repeat or stale payment, whatever else fits
        names_only_settled = bool(closed_hits) and not strong_open_all
        if best is None or best.score < cfg.unapplied_floor or names_only_settled:
            dec.reason, dec.kind = "no_match", "no_match"
            dec.confidence = best.score if best else 0.0
            ex = ["No open invoice explains this payment well enough."]
            top = self.names.rank(p.payer_name, 1)
            if top and top[0][1] >= cfg.name_floor and "name" not in dis:
                ex.insert(0, f"Payer '{p.payer_name}' looks like {self.cust[top[0][0]].legal_name} (name similarity {top[0][1]:.2f}).")
            else:
                ex.insert(0, f"Payer '{p.payer_name}' does not resemble any customer in the master.")
            if closed_hits:
                ex.append("The remittance references " + ", ".join(i.number for i in closed_hits[:3]) + ", which is already settled: possible repeat payment. Held as unapplied cash.")
                dec.reason = "refs_closed_invoice"
            elif has_refs:
                ex.append("The reference in the remittance matches no open invoice.")
            dec.explanation = ex
            recent.setdefault(self._fingerprint(p, hits), []).append({"id": p.id, "date": p.date, "invs": set()})
            return dec

        second = next((h for h in hyps[1:] if h.key != best.key), None)
        margin = best.score - second.score if second else 1.0
        conf = best.score * (1.0 - cfg.margin_pen * max(0.0, 1.0 - margin / cfg.margin_ref))
        dec.kind = best.kind
        dec.confidence = max(0.0, min(1.0, conf))
        dec.margin = margin
        dec.proposal = [PropAlloc(i.id, best.cash[i.id][0], best.cash[i.id][1], best.cls[i.id]) for i in best.invs]
        dec.on_account = best.on_account
        dec.fuzzy_ref = any(ctx.hits.get(i.id) is not None and ctx.hits[i.id].kind != "exact" for i in best.invs)
        dec.explanation = self._explain(ctx, best, second, margin, hyps)
        dec.runner_up = self._describe(second) if second else ""
        if "ledger" not in dis and dec.confidence >= cfg.ledger_min:
            for i in best.invs:
                self.rem[i.id] -= best.cash[i.id][1]
        recent.setdefault(self._fingerprint(p, hits), []).append({"id": p.id, "date": p.date, "invs": {i.id for i in best.invs}})
        return dec

    def _fingerprint(self, p, hits) -> tuple:
        return (normalize(p.payer_name), p.amount, p.currency, frozenset(i for i, h in hits.items() if h.sim >= self.cfg.strong_ref))

    @staticmethod
    def _describe(h: Hyp) -> str:
        return f"{', '.join(i.number for i in h.invs)} as {h.kind} (score {h.score:.2f})"

    def _explain(self, ctx: Ctx, h: Hyp, second: Optional[Hyp], margin: float, hyps: list) -> list:
        p = ctx.p
        cust = self.cust[h.cid]
        lines = []
        ns = ctx.name_sims.get(h.cid)
        seen = ctx.alias.get(h.cid, 0)
        hist = self.hist_n.get(h.cid, 0)
        who = f"Payer '{p.payer_name}' resolves to {cust.legal_name}"
        if ns is not None:
            who += f" (name similarity {ns:.2f}"
            who += f"; this payer string seen {seen}x before" if seen else ("; this payer string is new for the customer" if hist else "; no payment history")
            who += ")"
        lines.append(who + ".")
        if ctx.has_refs and "ref" not in self.cfg.disable:
            for inv in h.invs:
                hit = ctx.hits.get(inv.id)
                if hit:
                    how = {"exact": "exact", "po": "the customer's PO number", "fuzzy": "one digit off", "suffix": "last digits only"}[hit.kind]
                    lines.append(f"Reference {hit.token} matches {inv.number} ({how}).")
                else:
                    lines.append(f"{inv.number} is not named in the remittance.")
        else:
            lines.append("No usable invoice reference in the remittance: matched on amount, payer and dates.")
        nums = ", ".join(i.number for i in h.invs)
        T = convert(p.amount, p.currency, cust.currency)
        rs = sum(self.rem[i.id] for i in h.invs)
        pay = money(p.amount, p.currency)
        k = h.kind
        if k == "exact":
            others = sum(1 for x in hyps if x.kind == "exact" and x.key != h.key and x.cid == h.cid)
            if len(h.invs) == 1:
                lines.append(f"Payment {pay} equals the open balance of {nums} exactly.")
            else:
                lines.append(f"Payment {pay} equals the open balances of {len(h.invs)} invoices ({nums}) to the cent"
                             + (f"; {others} other combination(s) also fit." if others else "; no other combination fits."))
        elif k == "fx_rounding":
            lines.append(f"Payment {pay} is {money(T, cust.currency)} at the table rate against a balance of {money(rs, cust.currency)}: "
                         f"difference {money(T - rs, cust.currency)} ({abs(T - rs) / rs * 100:.2f}%) is within tolerance and written off as FX/rounding.")
        elif k == "discount":
            lines.append(f"Payment is {(rs - T) / rs * 100:.1f}% under the balance of {nums}, matching its early-pay discount ({h.invs[0].discount_pct * 100:g}% within {h.invs[0].discount_days} days).")
        elif k == "deduction":
            lines.append(f"Payment is short of {nums} by {money(rs - T, cust.currency)}; the remittance names a {h.note.split('=')[1]} deduction. Recorded as {', '.join(sorted(set(h.cls.values()) - {'clean'}))}.")
        elif k == "short_pay":
            lines.append(f"Payment is short of {nums} by {money(rs - T, cust.currency)} ({(rs - T) / rs * 100:.1f}%) with no reason in the remittance: unexplained short-pay, needs a deduction code.")
        elif k == "partial":
            hint = "; the remittance says partial" if "partial" in ctx.parsed.hints else "; no instalment wording in the remittance"
            lines.append(f"Payment is {T / rs * 100:.0f}% of the open balance of {nums}{hint}: applied as a partial payment, {money(rs - T, cust.currency)} stays open.")
        elif k == "overpay":
            lines.append(f"Payment exceeds the open balance of {nums} by {money(h.on_account, cust.currency)}: surplus held on account.")
        if second is not None:
            lines.append(f"Runner-up: {self._describe(second)}; margin {margin:.2f}" + (" (ambiguous)." if margin < self.cfg.margin_ref else "."))
        return lines

    def run(self) -> list:
        """Decisions in processing order (date, id). Status is set later by the policy."""
        self.rem = {i.id: i.amount for i in self.ds.invoices}
        for i in self.ds.closed_invoices:
            self.rem[i.id] = 0
        recent: dict = {}
        out = []
        for p in sorted(self.ds.payments, key=lambda p: (p.date, p.id)):
            out.append(self._decide(p, recent))
        return out
