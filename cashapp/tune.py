"""Tuning and the held-out evaluation, with the discipline of the sibling rag-tuning-lab project.

* The data is split once, by customer, with a fixed seed: a tuning half and a held-out half. A
  customer's invoices, history and payments all stay on one side.
* `tune()` receives only the tuning half. Nothing in it can read the held-out data or labels.
* Settings are changed one at a time; a change is kept only if it beats the current setting by
  more than a tolerance (ties keep what is there).
* The held-out half is scored after tuning is finished: the tuned engine, the untuned engine, the
  plain baseline, and one engine per ablation. The threshold for each of them is chosen on the tuning
  half, never on the held-out half.
* Every configuration tried is written to a trials file.
"""
from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from typing import Optional

from .baseline import BaselineEngine
from .engine import ABLATIONS, Config, Engine
from .evaluate import bootstrap, by_category, choose_threshold, policy_metrics, raw_metrics, sweep, target_met
from .schema import Dataset

TARGET_MISAPPLIED = 0.01  # at most 1% of auto-applied dollars may be wrong
TOLERANCE = 0.003  # a change must beat the current setting by more than this auto-apply share
SWEEP_THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]

GRID = [
    ("weights.ref", [1.0, 2.0, 3.0, 4.0, 6.0]),
    ("weights.name", [0.5, 1.0, 2.0, 3.0, 4.0]),
    ("weights.amount", [0.5, 1.0, 2.0, 3.0, 4.0]),
    ("weights.date", [0.0, 0.25, 0.5, 1.0, 2.0]),
    ("weights.history", [0.0, 0.5, 1.0, 2.0]),
    ("margin_pen", [0.0, 0.15, 0.30, 0.50]),
    ("margin_ref", [0.05, 0.10, 0.20]),
    ("unapplied_floor", [0.30, 0.40, 0.50]),
    ("ledger_min", [0.0, 0.6, 0.8, 0.9]),
]
QUICK_GRID = [("weights.ref", [2.0, 3.0]), ("weights.name", [1.0, 2.0]), ("margin_pen", [0.15, 0.30])]


def split_dataset(ds: Dataset, seed: int = 2024, frac: float = 0.5) -> tuple:
    """(tuning half, held-out half), split by customer. Made once; the same seed gives the same split."""
    cids = sorted(c.id for c in ds.customers)
    rng = random.Random(seed)
    rng.shuffle(cids)
    k = int(round(len(cids) * frac))
    side = {cid: 0 for cid in cids[:k]}
    side.update({cid: 1 for cid in cids[k:]})
    # a payer that is not a customer has no owner: assign it to a side, seeded, by payment id
    orphan = random.Random(seed + 1)

    def owner_side(pid: str) -> int:
        t = ds.truth[pid]
        return side[t.customer_id] if t.customer_id else orphan_side[pid]

    orphan_side = {p.id: orphan.randrange(2) for p in sorted(ds.payments, key=lambda p: p.id) if not ds.truth[p.id].customer_id}
    halves = []
    for s in (0, 1):
        members = {cid for cid, v in side.items() if v == s}
        halves.append(Dataset(
            customers=[c for c in ds.customers if c.id in members],
            invoices=[i for i in ds.invoices if i.customer_id in members],
            closed_invoices=[i for i in ds.closed_invoices if i.customer_id in members],
            payments=[p for p in ds.payments if owner_side(p.id) == s],
            truth={pid: t for pid, t in ds.truth.items() if owner_side(pid) == s},
            history={cid: h for cid, h in ds.history.items() if cid in members},
            meta=dict(ds.meta, split_seed=seed, half="tuning" if s == 0 else "held_out"),
        ))
    return halves[0], halves[1]


def _get(cfg: Config, path: str):
    if "." in path:
        a, b = path.split(".")
        return getattr(cfg, a)[b]
    return getattr(cfg, path)


def _set(cfg: Config, path: str, value) -> Config:
    if "." in path:
        a, b = path.split(".")
        d = dict(getattr(cfg, a))
        d[b] = value
        return cfg.replace(**{a: d})
    return cfg.replace(**{path: value})


def objective(ds: Dataset, cfg: Config, target: float = TARGET_MISAPPLIED) -> dict:
    """Auto-apply share of dollars at the lowest threshold that keeps misapplied dollars at or under the target."""
    decs = Engine(ds, cfg).run()
    t = choose_threshold(ds, decs, target, margin_ref=cfg.margin_ref)
    m = policy_metrics(ds, decs, t, cfg.margin_ref)
    return {"threshold": t, "auto": m["auto_apply_rate_dollars"], "misapplied": m["misapplied_rate_dollars"], "f1": raw_metrics(ds, decs)["f1"]}


@dataclass
class TuneResult:
    config: Config
    threshold: float
    objective: dict
    trials: list


def tune(tune_ds: Dataset, base: Optional[Config] = None, grid=None, passes: int = 2, log_path: Optional[str] = None, verbose: bool = False) -> TuneResult:
    """Coordinate ascent on the tuning half only. The only data it can see is the one argument."""
    cfg = base or Config()
    grid = grid or GRID
    trials: list = []
    cur = objective(tune_ds, cfg)
    trials.append({"stage": "start", "param": None, "value": None, **cur, "config": cfg.to_dict()})
    log = open(log_path, "w", encoding="utf-8", newline="
") if log_path else None
    try:
        if log:
            log.write(json.dumps(trials[-1]) + "\n")
        for ps in range(passes):
            changed = False
            for path, values in grid:
                best_val, best = _get(cfg, path), cur
                for v in values:
                    if v == _get(cfg, path):
                        continue
                    trial_cfg = _set(cfg, path, v)
                    res = objective(tune_ds, trial_cfg)
                    rec = {"stage": f"pass{ps + 1}", "param": path, "value": v, **res, "config": trial_cfg.to_dict()}
                    trials.append(rec)
                    if log:
                        log.write(json.dumps(rec) + "\n")
                        log.flush()
                    if verbose:
                        print(f"  {path}={v}: auto {res['auto']:.3f} at t={res['threshold']:.2f} (mis {res['misapplied']:.3f})", flush=True)
                    if res["auto"] > best["auto"] + TOLERANCE:
                        best_val, best = v, res
                if best is not cur:
                    cfg = _set(cfg, path, best_val)
                    cur = best
                    changed = True
            if not changed:
                break
    finally:
        if log:
            log.close()
    return TuneResult(cfg, cur["threshold"], cur, trials)


def _evaluate(ds: Dataset, decisions, threshold: float, margin_ref: float, boot: bool = True) -> dict:
    out = {
        "at_threshold": policy_metrics(ds, decisions, threshold, margin_ref),
        "raw": raw_metrics(ds, decisions),
        "sweep": sweep(ds, decisions, SWEEP_THRESHOLDS, margin_ref),
        "by_category": by_category(ds, decisions, threshold, margin_ref),
    }
    if boot:
        out["ci"] = bootstrap(ds, decisions, threshold, margin_ref=margin_ref)
    return out


def run_experiment(ds: Dataset, split_seed: int = 2024, quick: bool = False, log_path: Optional[str] = None,
                   verbose: bool = False, ablations=None) -> dict:
    """Split, tune on the tuning half, score the held-out half. Returns everything needed for the report."""
    t0 = time.time()
    tune_ds, held_ds = split_dataset(ds, split_seed)
    tuned = tune(tune_ds, grid=QUICK_GRID if quick else GRID, passes=1 if quick else 2, log_path=log_path, verbose=verbose)
    cfg, thr = tuned.config, tuned.threshold
    out: dict = {
        "setup": {
            "n_customers": len(ds.customers), "n_payments": len(ds.payments), "n_invoices_open": len(ds.invoices),
            "tuning": {"customers": len(tune_ds.customers), "payments": len(tune_ds.payments)},
            "held_out": {"customers": len(held_ds.customers), "payments": len(held_ds.payments)},
            "split_seed": split_seed, "data_seed": ds.meta.get("seed"), "target_misapplied_dollars": TARGET_MISAPPLIED,
            "tolerance": TOLERANCE, "quick": quick,
        },
        "tuning": {"threshold": thr, "objective": tuned.objective, "trials": len(tuned.trials), "config": cfg.to_dict()},
    }
    # held-out, tuned engine
    decs = Engine(held_ds, cfg).run()
    out["held_out"] = {"tuned": _evaluate(held_ds, decs, thr, cfg.margin_ref)}
    # untuned engine; its threshold is also chosen on the tuning half
    out["held_out"]["tuned"]["threshold"] = thr
    out["held_out"]["tuned"]["target_met_on_tuning"] = target_met(tune_ds, Engine(tune_ds, cfg).run(), thr, TARGET_MISAPPLIED, cfg.margin_ref)
    d_cfg = Config()
    d_tune = Engine(tune_ds, d_cfg).run()
    d_thr = choose_threshold(tune_ds, d_tune, TARGET_MISAPPLIED, margin_ref=d_cfg.margin_ref)
    out["held_out"]["default"] = _evaluate(held_ds, Engine(held_ds, d_cfg).run(), d_thr, d_cfg.margin_ref, boot=False)
    out["held_out"]["default"]["threshold"] = d_thr
    out["held_out"]["default"]["target_met_on_tuning"] = target_met(tune_ds, d_tune, d_thr, TARGET_MISAPPLIED, d_cfg.margin_ref)
    # plain baseline
    b_tune = BaselineEngine(tune_ds).run()
    b_thr = choose_threshold(tune_ds, b_tune, TARGET_MISAPPLIED)
    out["held_out"]["baseline"] = _evaluate(held_ds, BaselineEngine(held_ds).run(), b_thr, 0.10, boot=False)
    out["held_out"]["baseline"]["threshold"] = b_thr
    out["held_out"]["baseline"]["target_met_on_tuning"] = target_met(tune_ds, b_tune, b_thr, TARGET_MISAPPLIED)
    # ablations of the tuned engine; each one gets its own threshold chosen on the tuning half
    abl = {}
    for name in (ablations if ablations is not None else ([] if quick else ABLATIONS)):
        a_cfg = cfg.replace(disable=[name])
        a_tune = Engine(tune_ds, a_cfg).run()
        a_thr = choose_threshold(tune_ds, a_tune, TARGET_MISAPPLIED, margin_ref=a_cfg.margin_ref)
        a_dec = Engine(held_ds, a_cfg).run()
        ev = _evaluate(held_ds, a_dec, a_thr, a_cfg.margin_ref, boot=False)
        abl[name] = {"threshold": a_thr, "target_met_on_tuning": target_met(tune_ds, a_tune, a_thr, TARGET_MISAPPLIED, a_cfg.margin_ref),
                     "at_threshold": ev["at_threshold"], "at_tuned_threshold": policy_metrics(held_ds, a_dec, thr, a_cfg.margin_ref),
                     "raw": ev["raw"], "by_category": ev["by_category"]}
        if verbose:
            print(f"  ablation {name}: auto {ev['at_threshold']['auto_apply_rate_dollars']:.3f}, mis {ev['at_threshold']['misapplied_rate_dollars']:.3f}, f1 {ev['raw']['f1']:.3f}", flush=True)
    out["ablations"] = abl
    out["seconds"] = round(time.time() - t0, 1)
    return out
