# Changelog

## 0.1.0 - 2026-11-18

Initial release.

- `annotrust scan` reads CSV, TSV, JSONL, and JSON annotation tables and writes a text report, `--json`, and a self-contained `--html` file. `--fail-on low|medium|high` exits 1 when an annotator's severity is at or above that level.
- Dawid-Skene consensus and confusion matrices, Cohen's kappa collusion scores, targeted-flip checks, and speed and streak checks when timestamps exist.
- Item-level quality scores. Suspect items are reported and do not change the exit code.
- `annotrust simulate` plants honest, careless, colluding, and poisoning annotators.
- `scripts/benchmark.py` writes `docs/benchmark.md` from a real run.
- Best-effort, unofficial readers for Encord JSON exports and Label Studio JSON exports.
- Not published to PyPI (`Private :: Do Not Upload`).
