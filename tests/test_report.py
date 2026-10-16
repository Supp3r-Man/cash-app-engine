from cashapp.generate import generate
from cashapp.report import END, START, make_charts, results_markdown, write_readme
from cashapp.tune import run_experiment


def _summary():
    return run_experiment(generate(14, 24, 200), quick=True, ablations=["dup"])


def test_charts_and_tables_are_built_from_the_summary(tmp_path):
    s = _summary()
    files = make_charts(s, tmp_path)
    assert {f.name for f in files} == {"tradeoff.png", "by_category.png", "ablation.png"}
    assert all(f.stat().st_size > 5_000 for f in files)
    md = results_markdown(s)
    t = s["held_out"]["tuned"]["at_threshold"]
    assert f"{100 * t['auto_apply_rate_dollars']:.1f}%" in md  # the README number is the summary number
    assert "| exact |" in md and "duplicate detection" in md


def test_write_readme_replaces_only_the_marked_block(tmp_path):
    s = _summary()
    readme = tmp_path / "README.md"
    readme.write_text(f"before\n{START}\nold numbers\n{END}\nafter\n", encoding="utf-8")
    write_readme(readme, s)
    text = readme.read_text(encoding="utf-8")
    assert text.startswith("before\n") and text.endswith("after\n") and "old numbers" not in text and "Held-out results" in text
