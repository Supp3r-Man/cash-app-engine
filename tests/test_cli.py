import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run(*args):
    return subprocess.run([sys.executable, "-m", "cashapp", *args], capture_output=True, text=True, cwd=str(ROOT),
                          env={**os.environ, "PYTHONPATH": str(ROOT)}, timeout=300)


def test_demo_runs_offline_and_prints_explanations():
    out = run("demo")
    assert out.returncode == 0, out.stderr
    assert "Auto-applied" in out.stdout and "Exception queue by reason" in out.stdout
    assert "[multi_invoice]" in out.stdout and "Confidence" in out.stdout


def test_generate_then_match(tmp_path):
    data = tmp_path / "d.json"
    out = run("generate", "--seed", "4", "--customers", "10", "--payments", "80", "--out", str(data))
    assert out.returncode == 0, out.stderr
    dec = tmp_path / "decisions.jsonl"
    out = run("match", "--data", str(data), "--threshold", "0.8", "--out", str(dec))
    assert out.returncode == 0, out.stderr
    rows = [json.loads(line) for line in dec.read_text(encoding="utf-8").splitlines()]
    assert rows and all(r["status"] in ("auto_apply", "exception") and r["explanation"] for r in rows)
    assert "Against the synthetic truth" in out.stdout


def test_sample_dataset_in_the_repo_loads():
    from cashapp.schema import Dataset

    ds = Dataset.load(ROOT / "sample" / "sample_batch.json")
    assert ds.meta["synthetic"] is True and ds.payments
