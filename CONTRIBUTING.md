# Contributing

Annotrust is maintained by Sai Pranay Kowshik Chowdary Sakhamuri ([@koushikchowdary6](https://github.com/koushikchowdary6)).

## Setup

Python 3.11 or newer.

```bash
python -m pip install -e ".[dev]"
ruff check .
ruff format --check .
mypy src
pytest
```

`pytest` fails under 90% coverage. CI runs those four commands on Python 3.11, 3.12, and 3.13.

## What a change needs

- A test that fails before the change and passes after it, when the change is a bugfix or a new check.
- No new required dependency. numpy, pandas, and scipy are the runtime set. Cleanlab stays optional.
- No network calls. `tests/test_offline.py` reads the sources for HTTP clients.
- Threshold changes belong in `ScanConfig` and in `docs/engineering-decisions.md`. Re-run `python scripts/benchmark.py` and commit the regenerated `docs/benchmark.md` and `docs/benchmark.json`. Do not type numbers into that markdown by hand.

## Publishing

Do not upload this package to PyPI. `pyproject.toml` sets the `Private :: Do Not Upload` classifier on purpose.

## Decisions left for the author

`docs/engineering-decisions.md` records the problem, the alternatives, the choice in the code, and how it was tested. The sections "Why I chose this" and "Open questions" are intentionally empty.
