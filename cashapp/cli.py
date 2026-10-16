"""Command line: demo | generate | match | eval | report."""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

from .engine import Config, Engine
from .evaluate import policy_metrics, proposal_correct, raw_metrics
from .generate import generate
from .policy import EXCEPTION_REASONS, apply_policy
from .schema import Dataset, money

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"


def _fmt_pay(p) -> str:
    return f"{p.id} {p.date} {p.payer_name!r} {money(p.amount, p.currency)}  ref={p.reference!r} memo={p.memo!r}"


def _show(p, d, ds, inv) -> None:
    print("  " + _fmt_pay(p))
    t = ds.truth.get(p.id)
    print(f"    -> {d.status.upper()}" + (f" ({EXCEPTION_REASONS[d.reason]})" if d.reason else "") + f"  confidence {d.confidence:.2f}"
          + ("  [proposal matches the synthetic truth]" if t is not None and proposal_correct(d, t) else "  [proposal differs from the synthetic truth]" if t is not None else ""))
    if d.proposal:
        print("       applies to " + ", ".join(f"{inv[a.invoice_id].number} ({a.cls})" for a in d.proposal) + (f", {money(d.on_account)} on account" if d.on_account else ""))
    for line in d.explanation:
        print("       - " + line)


def cmd_demo(args) -> int:
    ds = generate(seed=3, n_customers=16, n_payments=170)
    cfg = Config()
    threshold = 0.85
    decs = apply_policy(Engine(ds, cfg).run(), threshold)
    inv = {i.id: i for i in ds.invoices + ds.closed_invoices}
    pays = {p.id: p for p in ds.payments}
    m = policy_metrics(ds, Engine(ds, cfg).run(), threshold)
    raw = raw_metrics(ds, Engine(ds, cfg).run())
    print(f"Synthetic batch: {len(ds.customers)} customers, {len(ds.invoices)} open invoices, {len(ds.payments)} payments (offline, no model, default settings, threshold {threshold}).")
    print(f"Auto-applied {m['n_auto']} of {m['n']} payments = {m['auto_apply_rate_dollars'] * 100:.1f}% of dollars; {m['n_misapplied']} of those were wrong against the synthetic truth.")
    print(f"Pair precision {raw['precision']:.3f}, recall {raw['recall']:.3f}. Exception queue by reason:")
    for r, n in collections.Counter(d.reason for d in decs if d.status == "exception").most_common():
        print(f"  {n:4d}  {EXCEPTION_REASONS[r]}")
    print("\nA few decisions, each with its explanation:")
    shown = set()
    wanted = [("multi_invoice", "auto_apply"), ("deduction", "auto_apply"), ("early_discount", "auto_apply"), ("name_variant", "auto_apply"),
              ("split_payment", "auto_apply"), ("short_pay", "exception"), ("duplicate", "exception"), ("unapplied", "exception"), ("garbled_ref", "auto_apply")]
    for cat, status in wanted:
        for d in decs:
            t = ds.truth[d.payment_id]
            if t.category == cat and d.status == status and cat not in shown:
                print(f"\n[{cat}]")
                _show(pays[d.payment_id], d, ds, inv)
                shown.add(cat)
                break
    print("\nThe remittance parser is pluggable: see cashapp/remit.py (LLMRemitParser, off by default, never run live here).")
    return 0


def cmd_generate(args) -> int:
    ds = generate(seed=args.seed, n_customers=args.customers, n_payments=args.payments)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ds.save(out)
    print(f"wrote {out}: {len(ds.customers)} customers, {len(ds.invoices)} open invoices, {len(ds.payments)} payments")
    return 0


def cmd_match(args) -> int:
    ds = Dataset.load(args.data) if args.data else generate(seed=args.seed, n_customers=args.customers, n_payments=args.payments)
    cfg = Config.from_dict(json.loads(Path(args.config).read_text(encoding="utf-8"))["config"]) if args.config else Config()
    raw = Engine(ds, cfg).run()
    decs = apply_policy(raw, args.threshold, cfg.margin_ref)
    if args.out:
        with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
            for d in decs:
                fh.write(json.dumps(d.to_dict()) + "\n")
        print(f"wrote {len(decs)} decisions to {args.out}")
    c = collections.Counter(d.status for d in decs)
    print(f"{c['auto_apply']} auto-applied, {c['exception']} in the exception queue (threshold {args.threshold})")
    for r, n in collections.Counter(d.reason for d in decs if d.status == "exception").most_common():
        print(f"  {n:4d}  {EXCEPTION_REASONS[r]}")
    if ds.truth:
        m = policy_metrics(ds, raw, args.threshold, cfg.margin_ref)
        print(f"Against the synthetic truth: {m['auto_apply_rate_dollars'] * 100:.1f}% of dollars auto-applied, {m['misapplied_rate_dollars'] * 100:.2f}% of those misapplied")
    return 0


def cmd_eval(args) -> int:
    from .report import make_charts
    from .tune import run_experiment

    RESULTS.mkdir(exist_ok=True)
    ds = generate(seed=args.seed, n_customers=args.customers, n_payments=args.payments)
    print(f"generated {len(ds.customers)} customers, {len(ds.payments)} payments; tuning (this takes several minutes)...", flush=True)
    summary = run_experiment(ds, split_seed=args.split_seed, quick=args.quick, log_path=str(RESULTS / "trials.jsonl"), verbose=True)
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8", newline="\n")
    (RESULTS / "tuned_config.json").write_text(json.dumps({"config": summary["tuning"]["config"], "threshold": summary["tuning"]["threshold"]}, indent=1), encoding="utf-8", newline="\n")
    make_charts(summary, RESULTS)
    t = summary["held_out"]["tuned"]["at_threshold"]
    print(f"held-out: {t['auto_apply_rate_dollars'] * 100:.1f}% of dollars auto-applied at threshold {t['threshold']:.2f}, {t['misapplied_rate_dollars'] * 100:.2f}% of those misapplied")
    print(f"wrote results to {RESULTS}")
    return 0


def cmd_report(args) -> int:
    from .report import load_summary, make_charts, write_readme

    summary = load_summary(RESULTS / "summary.json")
    make_charts(summary, RESULTS)
    if args.write_readme:
        write_readme(ROOT / "README.md", summary)
        print("README results block updated")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="cashapp", description="Cash application engine: match payments to open invoices, with explanations.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("demo", help="small synthetic batch, offline, a few seconds").set_defaults(fn=cmd_demo)
    g = sub.add_parser("generate", help="write a synthetic dataset with ground truth")
    g.add_argument("--seed", type=int, default=7)
    g.add_argument("--customers", type=int, default=120)
    g.add_argument("--payments", type=int, default=1600)
    g.add_argument("--out", default=str(ROOT / "data" / "synth.json"))
    g.set_defaults(fn=cmd_generate)
    m = sub.add_parser("match", help="run the engine over a dataset and write decisions")
    m.add_argument("--data", help="dataset JSON from `generate` (default: generate one in memory)")
    m.add_argument("--seed", type=int, default=7)
    m.add_argument("--customers", type=int, default=40)
    m.add_argument("--payments", type=int, default=400)
    m.add_argument("--threshold", type=float, default=0.85)
    m.add_argument("--config", help="results/tuned_config.json from `eval`")
    m.add_argument("--out", help="write decisions as JSON lines")
    m.set_defaults(fn=cmd_match)
    e = sub.add_parser("eval", help="split, tune, score held-out, ablate; writes results/")
    e.add_argument("--seed", type=int, default=7)
    e.add_argument("--customers", type=int, default=120)
    e.add_argument("--payments", type=int, default=1600)
    e.add_argument("--split-seed", type=int, default=2024)
    e.add_argument("--quick", action="store_true", help="tiny grid, no ablations (for a smoke test)")
    e.set_defaults(fn=cmd_eval)
    r = sub.add_parser("report", help="rebuild charts, and the README results block, from results/summary.json")
    r.add_argument("--write-readme", action="store_true")
    r.set_defaults(fn=cmd_report)
    args = ap.parse_args(argv)
    return args.fn(args)
