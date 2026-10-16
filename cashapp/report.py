"""Tables and charts, generated from results/summary.json so the README never holds a typed-in number."""
from __future__ import annotations

import json
import re
from pathlib import Path

START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"

ABLATION_LABELS = {
    "ref": "reference matching (all)", "fuzzy_ref": "fuzzy reference matching", "name": "payer-name matching", "amount": "amount-fit evidence",
    "date": "date proximity", "history": "alias and lag history", "subset": "subset-sum / multi-invoice search", "remit": "remittance text parsing",
    "classifier": "deduction reason classifier", "dup": "duplicate detection", "ledger": "tentative ledger (instalments)",
}
CATEGORY_LABELS = {
    "exact": "exact", "multi_invoice": "one payment, many invoices", "split_payment": "many payments, one invoice", "short_pay": "short-pay",
    "early_discount": "early-pay discount", "deduction": "deduction with reason", "rounding_fx": "rounding / FX", "garbled_ref": "garbled reference",
    "name_variant": "payer-name variant", "duplicate": "duplicate payment", "unapplied": "unapplied cash",
}


def pct(x: float, d: int = 1) -> str:
    return f"{100 * x:.{d}f}%"


def make_charts(summary: dict, outdir) -> list:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir = Path(outdir)
    ho = summary["held_out"]
    blue, orange, grey = "#2a6fbb", "#d9822b", "#7a7a7a"
    files = []

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for key, label, col, mk in (("tuned", "tuned engine", blue, "o"), ("default", "untuned engine", orange, "s"), ("baseline", "exact-match baseline", grey, "^")):
        pts = [(m["auto_apply_rate_dollars"] * 100, m["misapplied_rate_dollars"] * 100) for m in ho[key]["sweep"]]
        pts.sort()
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker=mk, color=col, label=label, lw=1.5, ms=5)
    t = ho["tuned"]["at_threshold"]
    ax.scatter([t["auto_apply_rate_dollars"] * 100], [t["misapplied_rate_dollars"] * 100], s=140, facecolors="none", edgecolors=blue, lw=2, label=f"chosen threshold {t['threshold']:.2f}", zorder=5)
    ax.axhline(summary["setup"]["target_misapplied_dollars"] * 100, color="#555", ls=":", lw=1)
    top = 10.0
    ax.set_ylim(-0.4, top)
    ax.text(ax.get_xlim()[1], summary["setup"]["target_misapplied_dollars"] * 100 + 0.15, "1% misapplied target ", fontsize=8, color="#555", ha="right")
    worst = max(m["misapplied_rate_dollars"] * 100 for m in ho["baseline"]["sweep"])
    if worst > top:
        ax.text(0.99, 0.62, f"exact-match baseline reaches {worst:.0f}%" + chr(10) + "misapplied at its lowest threshold (off chart)", transform=ax.transAxes, fontsize=8, color=grey, ha="right")
    ax.set_xlabel("share of payment dollars auto-applied (%)")
    ax.set_ylabel("share of auto-applied dollars that were wrong (%)")
    ax.set_title("Held-out: automation against error", fontsize=11)
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    f = outdir / "tradeoff.png"
    fig.savefig(f, dpi=140)
    plt.close(fig)
    files.append(f)

    cats = [c for c in CATEGORY_LABELS if c in ho["tuned"]["by_category"]]
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ys = range(len(cats))
    h = 0.38
    ax.barh([y + h / 2 for y in ys], [ho["baseline"]["by_category"][c]["proposal_correct_rate"] * 100 for c in cats], height=h, color=grey, label="exact-match baseline")
    ax.barh([y - h / 2 for y in ys], [ho["tuned"]["by_category"][c]["proposal_correct_rate"] * 100 for c in cats], height=h, color=blue, label="tuned engine")
    ax.set_yticks(list(ys))
    ax.set_yticklabels([CATEGORY_LABELS[c] for c in cats], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("payments whose proposal is exactly right (%), before the auto-apply policy")
    ax.set_title("Held-out: by hard-case category", fontsize=11)
    ax.legend(fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    f = outdir / "by_category.png"
    fig.savefig(f, dpi=140)
    plt.close(fig)
    files.append(f)

    if summary.get("ablations"):
        base_mis = t["misapplied_rate_dollars"]
        base_f1 = ho["tuned"]["raw"]["f1"]
        names = list(summary["ablations"])
        d_mis = [(summary["ablations"][n]["at_tuned_threshold"]["misapplied_rate_dollars"] - base_mis) * 100 for n in names]
        d_f1 = [(summary["ablations"][n]["raw"]["f1"] - base_f1) * 100 for n in names]
        order = sorted(range(len(names)), key=lambda i: d_mis[i])
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 4.6), sharey=True)
        a1.barh(range(len(names)), [d_mis[i] for i in order], color=orange)
        a2.barh(range(len(names)), [d_f1[i] for i in order], color=blue)
        a1.set_yticks(range(len(names)))
        a1.set_yticklabels([ABLATION_LABELS.get(names[i], names[i]) for i in order], fontsize=8)
        for ax_, vals in ((a1, d_mis), (a2, d_f1)):
            lo, hi = min(vals + [0]), max(vals + [0])
            pad = (hi - lo) * 0.14 or 1.0
            ax_.set_xlim(lo - pad, hi + pad)
            for row, i in enumerate(order):
                ax_.text(vals[i], row, f" {vals[i]:+.1f} ", va="center", ha="right" if vals[i] < 0 else "left", fontsize=7)
        a1.set_xlabel("change in misapplied share of auto-applied dollars (points)")
        a2.set_xlabel("change in pair F1 (points)")
        a1.set_title("At the tuned threshold: more wrong applications", fontsize=10)
        a2.set_title("Matching quality", fontsize=10)
        for a in (a1, a2):
            a.axvline(0, color="#333", lw=0.8)
            a.grid(axis="x", alpha=0.25)
        fig.tight_layout()
        f = outdir / "ablation.png"
        fig.savefig(f, dpi=140)
        plt.close(fig)
        files.append(f)
    return files


def results_markdown(summary: dict) -> str:
    s, ho, tu = summary["setup"], summary["held_out"], summary["tuning"]
    t = ho["tuned"]["at_threshold"]
    ci = ho["tuned"]["ci"]
    L = []
    L.append("![Automation against error](results/tradeoff.png)\n")
    L.append("![By category](results/by_category.png)\n")
    if summary.get("ablations"):
        L.append("![Ablations](results/ablation.png)\n")
    L.append("### Setup\n")
    L.append("| | |\n|---|---|")
    L.append(f"| Data | synthetic, seed {s['data_seed']}: {s['n_customers']} customers, {s['n_invoices_open']:,} open invoices, {s['n_payments']:,} payments |")
    L.append(f"| Split | by customer, seed {s['split_seed']}, made once. Tuning half: {s['tuning']['customers']} customers, {s['tuning']['payments']} payments. Held-out half: {s['held_out']['customers']} customers, {s['held_out']['payments']} payments |")
    L.append(f"| Policy target | at most {pct(s['target_misapplied_dollars'], 0)} of auto-applied dollars may be misapplied; the threshold is the lowest one that meets this on the tuning half |")
    L.append(f"| Tuning | {tu['trials']} configurations tried, written to `results/trials.jsonl` |")
    L.append(f"| Run time | {summary['seconds']} s on one laptop CPU, pure Python |")
    L.append("")
    L.append("### Held-out results\n")
    L.append("The threshold for every row was chosen on the tuning half. Held-out payments were scored after tuning finished.\n")
    L.append("| System | Threshold | Auto-applied dollars | Misapplied share of auto-applied dollars | Misapplied / auto-applied payments | 1% target met on tuning half | Pair precision | Pair recall | Payments proposed exactly right |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for key, label in (("baseline", "exact-match baseline"), ("default", "engine, default settings"), ("tuned", "engine, tuned")):
        e = ho[key]
        m = e["at_threshold"]
        thr = e["threshold"] if key != "tuned" else tu["threshold"]
        met = "yes" if e.get("target_met_on_tuning", True) else "no"
        L.append(f"| {label} | {thr:.2f} | {pct(m['auto_apply_rate_dollars'])} | {pct(m['misapplied_rate_dollars'], 2)} | {m['n_misapplied']} / {m['n_auto']} | {met} | {pct(e['raw']['precision'])} | {pct(e['raw']['recall'])} | {pct(e['raw']['payment_correct_rate'])} |")
    L.append("")
    L.append(f"Tuned engine, 95% bootstrap interval over payments: auto-applied dollars {pct(ci['auto_apply_rate_dollars_ci'][0])} to {pct(ci['auto_apply_rate_dollars_ci'][1])}; "
             f"misapplied share {pct(ci['misapplied_rate_dollars_ci'][0], 2)} to {pct(ci['misapplied_rate_dollars_ci'][1], 2)}. "
             f"At the chosen threshold {t['n_auto']} of {t['n']} held-out payments were auto-applied and {t['n_misapplied']} of those {'was' if t['n_misapplied'] == 1 else 'were'} wrong; "
             f"{pct(t['wrong_proposals_caught'], 0)} of all wrong proposals were sent to the exception queue instead.\n")
    L.append("### Threshold sweep (tuned engine, held-out)\n")
    L.append("| Threshold | Auto-applied dollars | Misapplied share of auto-applied dollars | Payments auto-applied |")
    L.append("|---|---|---|---|")
    for m in ho["tuned"]["sweep"]:
        L.append(f"| {m['threshold']:.2f} | {pct(m['auto_apply_rate_dollars'])} | {pct(m['misapplied_rate_dollars'], 2)} | {m['n_auto']} of {m['n']} |")
    L.append("")
    L.append("### By hard-case category (tuned engine, held-out, at the chosen threshold)\n")
    L.append("Proposal-correct is before the policy: did the engine name exactly the right invoices and classes (for a duplicate or unapplied payment: nothing).\n")
    L.append("| Category | Payments | Proposal correct | Baseline proposal correct | Pair F1 | Auto-applied dollars | Misapplied / auto-applied payments |")
    L.append("|---|---|---|---|---|---|---|")
    for c, label in CATEGORY_LABELS.items():
        r = ho["tuned"]["by_category"].get(c)
        if not r:
            continue
        b = ho["baseline"]["by_category"].get(c, {}).get("proposal_correct_rate", 0.0)
        mis = f"{r['n_misapplied']} / {r['n_auto']}"
        f1 = f"{r['f1']:.2f}" if r.get("n_gold_pairs", 1) else "n/a"
        L.append(f"| {label} | {r['n']} | {pct(r['proposal_correct_rate'])} | {pct(b)} | {f1} | {pct(r['auto_apply_rate_dollars'])} | {mis} |")
    L.append("")
    if summary.get("ablations"):
        L.append("### Ablations (held-out)\n")
        L.append("One piece switched off at a time, everything else as tuned. Two views. **Own threshold:** the ablated engine gets its own threshold, chosen on the tuning half by the same rule; "
                 "a threshold of 1.00 means nothing short of full confidence kept errors inside the 1% budget on the tuning half. "
                 "**At the tuned threshold:** the ablated engine is scored at the full engine's threshold, which isolates what the piece does to errors.\n")
        L.append("| Removed | Own threshold | Auto-applied dollars | Misapplied share | At the tuned threshold: auto-applied | At the tuned threshold: misapplied share | Pair F1 | Payments proposed exactly right |")
        L.append("|---|---|---|---|---|---|---|---|")
        full = ho["tuned"]
        L.append(f"| nothing (tuned engine) | {tu['threshold']:.2f} | {pct(t['auto_apply_rate_dollars'])} | {pct(t['misapplied_rate_dollars'], 2)} | {pct(t['auto_apply_rate_dollars'])} | {pct(t['misapplied_rate_dollars'], 2)} | {full['raw']['f1']:.3f} | {pct(full['raw']['payment_correct_rate'])} |")
        for name, a in sorted(summary["ablations"].items(), key=lambda kv: kv[1]["at_threshold"]["auto_apply_rate_dollars"]):
            m, f = a["at_threshold"], a["at_tuned_threshold"]
            L.append(f"| {ABLATION_LABELS.get(name, name)} | {a['threshold']:.2f} | {pct(m['auto_apply_rate_dollars'])} | {pct(m['misapplied_rate_dollars'], 2)} | {pct(f['auto_apply_rate_dollars'])} | {pct(f['misapplied_rate_dollars'], 2)} | {a['raw']['f1']:.3f} | {pct(a['raw']['payment_correct_rate'])} |")
        L.append("")
    if summary.get("ablations"):
        base_mis = t["misapplied_rate_dollars"]
        base_auto = t["auto_apply_rate_dollars"]
        abl = summary["ablations"]
        worst = sorted(abl, key=lambda n: -(abl[n]["at_tuned_threshold"]["misapplied_rate_dollars"] - base_mis))
        flat = [n for n in abl if abs(abl[n]["at_tuned_threshold"]["misapplied_rate_dollars"] - base_mis) < 0.005
                and abs(abl[n]["at_tuned_threshold"]["auto_apply_rate_dollars"] - base_auto) < 0.03 and abs(abl[n]["raw"]["f1"] - ho["tuned"]["raw"]["f1"]) < 0.01]
        L.append("Reading the ablations (generated from the table above): at the tuned threshold the misapplied share rises most when these are removed: "
                 + ", ".join(f"{ABLATION_LABELS.get(n, n)} ({pct(abl[n]['at_tuned_threshold']['misapplied_rate_dollars'], 1)})" for n in worst[:4])
                 + f", against {pct(base_mis, 2)} for the full engine. "
                 + ("Removing these made no measurable difference on this data: " + ", ".join(ABLATION_LABELS.get(n, n) for n in flat) + ". " if flat else "")
                 + "A single run on one synthetic dataset cannot separate a small effect from noise, so \"no measurable difference\" is not \"useless\".\n")
    cfg = tu["config"]
    L.append("### Tuned settings\n")
    L.append("Feature weights: " + ", ".join(f"{k} {v:g}" for k, v in cfg["weights"].items()) + f". Margin penalty {cfg['margin_pen']:g} over a reference margin of {cfg['margin_ref']:g}; payments scoring under {cfg['unapplied_floor']:g} are treated as having no match.\n")
    return "\n".join(L)


def write_readme(readme: Path, summary: dict) -> None:
    text = Path(readme).read_text(encoding="utf-8")
    block = f"{START}\n\n{results_markdown(summary)}\n{END}"
    if START in text and END in text:
        text = re.sub(re.escape(START) + r".*?" + re.escape(END), lambda m: block, text, flags=re.S)
    else:
        raise SystemExit("README has no RESULTS markers")
    Path(readme).write_text(text, encoding="utf-8", newline="\n")


def load_summary(path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
