## What changed

## How it was tested

- [ ] `ruff check .` and `ruff format --check .`
- [ ] `mypy src`
- [ ] `pytest` (coverage stays at or above 90%)

If a threshold or the simulator changed, `python scripts/benchmark.py` was re-run and `docs/benchmark.md` is the script's output.
