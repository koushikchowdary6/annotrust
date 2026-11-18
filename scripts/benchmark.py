"""Measure detection on the simulator and write docs/benchmark.md.

The markdown and JSON in docs/ are this script's output. Re-run it instead of
editing the tables.

    python scripts/benchmark.py
"""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

import numpy
import pandas
import scipy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from annotrust import ScanConfig, __version__, scan, simulate  # noqa: E402
from annotrust.metrics import classification_counts  # noqa: E402

SEEDS = (0, 1, 2, 3, 4)
ROLES = ("careless", "colluding", "poisoning")


def main() -> None:
    reference = run_group(SEEDS)
    poison_sweep = [(rate, run_group(SEEDS, poison_rate=rate)) for rate in (0.15, 0.35, 0.55, 0.9)]
    collude_sweep = [(rate, run_group(SEEDS, collude_rate=rate)) for rate in (0.25, 0.5, 0.7, 0.95)]
    honest_copy = run_group(SEEDS, leader_accuracy=0.95, collude_rate=1.0)
    weak_rates = flip_rates(SEEDS, poison_rate=0.15)
    majority = majority_case(SEEDS)
    payload = {
        "annotrust": __version__,
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "scipy": scipy.__version__,
        "seeds": list(SEEDS),
        "reference": reference,
        "poison_sweep": [{"poison_rate": rate, **group} for rate, group in poison_sweep],
        "collude_sweep": [{"collude_rate": rate, **group} for rate, group in collude_sweep],
        "honest_copy": honest_copy,
        "weak_flip_rates": weak_rates,
        "majority": majority,
        "config": ScanConfig().__dict__ if hasattr(ScanConfig(), "__dict__") else _config_dict(),
    }
    # Frozen dataclass has no __dict__ values we want to hand-maintain. Store fields.
    payload["config"] = _config_dict()
    out_json = ROOT / "docs" / "benchmark.json"
    out_md = ROOT / "docs" / "benchmark.md"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    out_md.write_text(render_markdown(payload, poison_sweep, collude_sweep), encoding="utf-8")
    print(out_md.read_text(encoding="utf-8"))


def run_group(seeds: tuple[int, ...], **overrides: object) -> dict[str, object]:
    pooled = {role: {"tp": 0, "fp": 0, "fn": 0} for role in ROLES}
    per_seed = []
    planted = {role: 0 for role in ROLES}
    for seed in seeds:
        simulation = simulate(seed=seed, **overrides)  # type: ignore[arg-type]
        result = scan(simulation.frame)
        roles = simulation.roles()
        seed_row: dict[str, object] = {"seed": seed}
        for role in ROLES:
            truth = {annotator_id for annotator_id, planted_role in roles.items() if planted_role == role}
            predicted = {row.annotator_id for row in result.annotators if row.primary == role}
            counts = classification_counts(truth, predicted)
            planted[role] += len(truth)
            for key in ("tp", "fp", "fn"):
                pooled[role][key] += int(counts[key])  # type: ignore[arg-type]
            seed_row[role] = counts["recall"]
        per_seed.append(seed_row)
    for role in ROLES:
        if pooled[role]["tp"] + pooled[role]["fn"] != planted[role]:
            raise SystemExit(f"{role} tally does not match the planted count")
    summary = {}
    for role in ROLES:
        tp = pooled[role]["tp"]
        fp = pooled[role]["fp"]
        fn = pooled[role]["fn"]
        summary[role] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": (tp / (tp + fp)) if tp + fp else None,
            "recall": (tp / (tp + fn)) if tp + fn else None,
        }
    return {"pooled": summary, "per_seed": per_seed}


def flip_rates(seeds: tuple[int, ...], poison_rate: float) -> list[dict[str, float]]:
    rows = []
    for seed in seeds:
        simulation = simulate(seed=seed, poison_rate=poison_rate, timestamps=False)
        result = scan(simulation.frame)
        poisoner = next(row for row in result.annotators if row.annotator_id == "poisoner_00")
        source = "class_0"
        target = "class_1"
        source_index = result.classes.index(source)
        target_index = result.classes.index(target)
        rows.append(
            {
                "seed": seed,
                "rate_to_target": poisoner.confusion[source_index][target_index],
                "flagged": poisoner.primary == "poisoning",
            }
        )
    return rows


def majority_case(seeds: tuple[int, ...]) -> list[dict[str, object]]:
    rows = []
    for seed in seeds:
        simulation = simulate(
            seed=seed,
            timestamps=False,
            n_items=200,
            labels_per_item=6,
            n_honest=2,
            n_careless=0,
            n_colluders=0,
            n_poisoners=5,
            poison_rate=1.0,
        )
        result = scan(simulation.frame)
        truth = {annotator.annotator_id for annotator in simulation.annotators if annotator.role == "poisoning"}
        predicted = {row.annotator_id for row in result.annotators if row.primary == "poisoning"}
        counts = classification_counts(truth, predicted)
        if counts["tp"] + counts["fn"] != len(truth):
            raise SystemExit("majority tally does not match the planted count")
        source_index = result.classes.index("class_0")
        rows.append(
            {
                "seed": seed,
                "precision": counts["precision"],
                "recall": counts["recall"],
                "tp": counts["tp"],
                "fp": counts["fp"],
                "fn": counts["fn"],
                "predicted": sorted(predicted),
                "accused_honest": sorted(predicted - truth),
                "source_prior": result.prior[source_index],
            }
        )
    return rows


def render_markdown(
    payload: dict[str, object],
    poison_sweep: list[tuple[float, dict[str, object]]],
    collude_sweep: list[tuple[float, dict[str, object]]],
) -> str:
    reference = payload["reference"]
    assert isinstance(reference, dict)
    lines = [
        "# Benchmark",
        "",
        "Generated by `python scripts/benchmark.py`. The tables are that command's output.",
        "Do not edit them by hand.",
        "",
        "Each run plants annotators with `annotrust.simulate`, audits the frame with `scan`,",
        "and scores the primary label against the planted role. Precision and recall are",
        "pooled across seeds: true positives divided by the pooled predictions, not a mean of",
        "per-seed ratios. Precision is `n/a` when that attack type was never flagged.",
        "",
        "## Setup",
        "",
        f"- Annotrust {payload['annotrust']}",
        f"- Python {payload['python']}, numpy {payload['numpy']}, pandas {payload['pandas']}, scipy {payload['scipy']}",
        "- Seeds 0, 1, 2, 3, and 4",
        "- Default panel, unless a row says otherwise: 200 items, 4 classes, 7 labels per item,",
        "  6 honest annotators at accuracy 0.9, 2 careless (one uniform, one constant label),",
        "  one colluding pair (leader accuracy 0.6, copy rate 0.95), one poisoner",
        "  (`class_0` to `class_1` at the stated rate). Colluders and poisoners are on every",
        "  item when the panel still has room for someone else. Careless and honest annotators",
        "  fill the remaining seats.",
        "- Thresholds are the `ScanConfig` defaults in `src/annotrust/config.py`. They were not",
        "  fit by a search over these seeds.",
        "- A primary label of `colluding` or `poisoning` or `careless` is the prediction.",
        "  An annotator can carry a lower-severity speed flag and still be scored as careless.",
        "",
        "## Reference",
        "",
        "Defaults: poison rate 0.9, collude rate 0.95, leader accuracy 0.6.",
        "",
        _role_table(reference),
        "",
        "Per-seed recall:",
        "",
        _seed_table(reference),
        "",
        _reference_sentence(reference),
        "",
        "## Poison rate",
        "",
        "Collude rate stays at 0.95. The poisoner's other labels use honest accuracy 0.9,",
        "so a stated flip rate of 0.15 is not a pure 0.15: the honest mistakes add a little",
        "mass on the target class. The rate column below is the planted probability, not the",
        "fitted matrix.",
        "",
        _sweep_table("poison rate", poison_sweep, "poisoning"),
        "",
        _weak_sentence(payload),
        "",
        "## Collusion rate",
        "",
        "Poison rate stays at 0.9. The partner copies the leader with the stated probability",
        "and otherwise labels at honest accuracy 0.9.",
        "",
        _sweep_table("collude rate", collude_sweep, "colluding"),
        "",
        _collude_sentence(collude_sweep),
        "",
        "## Where detection fails",
        "",
        "### A weak targeted flip",
        "",
        _weak_detail(payload),
        "",
        "### Collusion that mostly tells the truth",
        "",
        _low_collude_detail(collude_sweep),
        "",
        "### Copying an accurate leader",
        "",
        _honest_copy_detail(payload),
        "",
        "### A majority of every panel flips one class",
        "",
        _majority_detail(payload),
        "",
        "## What this does not measure",
        "",
        "These are simulated annotators with independent honest noise, one shared flip, and",
        "pairs who were assigned the same items. A vendor file will have task difficulty,",
        "class imbalance, and partial overlap that this script does not reproduce. Cleanlab",
        "was not installed for this run. The numbers are not a false-positive rate on real",
        "disagreement, and they are not an audit of the detector.",
        "",
    ]
    return "\n".join(lines)


def _reference_sentence(group: dict[str, object]) -> str:
    pooled = group["pooled"]
    assert isinstance(pooled, dict)
    perfect = all(_recall(group, role) == 1.0 and _precision(group, role) == 1.0 for role in ROLES)
    if perfect:
        return (
            "On this panel, at these strengths, every planted attacker was the primary label "
            "and no honest annotator was."
        )
    return "On this panel the pooled rates above are the result. They are not all perfect."


def _weak_sentence(payload: dict[str, object]) -> str:
    rates = payload["weak_flip_rates"]
    assert isinstance(rates, list)
    mean_rate = sum(float(row["rate_to_target"]) for row in rates) / len(rates)
    flagged = sum(1 for row in rates if row["flagged"])
    return (
        f"At poison rate 0.15 the fitted P(label class_1 | estimated class class_0) for "
        f"`poisoner_00` averaged {mean_rate:.3f} across the five seeds "
        f"({flagged} of 5 seeds flagged the poisoner). "
        f"The default `poison_min_rate` is {ScanConfig.poison_min_rate:.2f}, so this sits under the cutoff."
    )


def _weak_detail(payload: dict[str, object]) -> str:
    sweep = payload["poison_sweep"]
    assert isinstance(sweep, list)
    weak = next(row for row in sweep if row["poison_rate"] == 0.15)
    pooled = weak["pooled"]["poisoning"]
    return (
        f"Pooled poisoning precision {_fmt(pooled['precision'])}, recall {_fmt(pooled['recall'])} "
        f"(tp={pooled['tp']}, fp={pooled['fp']}, fn={pooled['fn']}). "
        "Detection fails here. The flip is real in the generator and too small, next to honest "
        "noise on four classes, for the confusion-matrix rule to call it targeted. "
        "Raising recall on this row means lowering `poison_min_rate`, which is the knob that "
        "also admits ordinary mistakes."
    )


def _collude_sentence(sweep: list[tuple[float, dict[str, object]]]) -> str:
    low = dict(sweep)[0.25]
    mid = dict(sweep)[0.5]
    high = dict(sweep)[0.7]
    return (
        f"Copy rate 0.25 recall {_fmt(_recall(low, 'colluding'))}, "
        f"copy rate 0.50 recall {_fmt(_recall(mid, 'colluding'))}, "
        f"copy rate 0.70 recall {_fmt(_recall(high, 'colluding'))}. "
        "The score needs the pair to agree with each other more than the better of them agrees "
        "with the consensus, and to share several wrong answers. A partner who copies on a "
        "quarter or half of items and is otherwise honest does not clear that."
    )


def _low_collude_detail(sweep: list[tuple[float, dict[str, object]]]) -> str:
    group = dict(sweep)[0.25]
    pooled = group["pooled"]["colluding"]  # type: ignore[index]
    return (
        f"At collude rate 0.25, pooled colluding precision {_fmt(pooled['precision'])}, "
        f"recall {_fmt(pooled['recall'])} "
        f"(tp={pooled['tp']}, fp={pooled['fp']}, fn={pooled['fn']}). "
        "Detection fails. The pair is planted and mostly not visible, because the partner's "
        "labels are usually the honest ones. Two accurate annotators also agree with each "
        "other; the score is built not to call that collusion."
    )


def _honest_copy_detail(payload: dict[str, object]) -> str:
    group = payload["honest_copy"]
    assert isinstance(group, dict)
    pooled = group["pooled"]["colluding"]  # type: ignore[index]
    return (
        "Leader accuracy 0.95 and copy rate 1.0: the partner repeats the leader, including "
        "the leader's rare mistakes. "
        f"Pooled colluding precision {_fmt(pooled['precision'])}, recall {_fmt(pooled['recall'])} "
        f"(tp={pooled['tp']}, fp={pooled['fp']}, fn={pooled['fn']}). "
        "Detection fails. Excess kappa is kappa(pair) minus the better kappa against the "
        "consensus. Both annotators match the consensus, so the excess stays small. "
        "Copying someone who is already right is not separable from two careful annotators."
    )


def _majority_detail(payload: dict[str, object]) -> str:
    rows = payload["majority"]
    assert isinstance(rows, list)
    tp = sum(int(row["tp"]) for row in rows)
    fp = sum(int(row["fp"]) for row in rows)
    fn = sum(int(row["fn"]) for row in rows)
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    lines = [
        "Five poisoners and two honest annotators, six labels per item, poison rate 1.0, no",
        "careless annotators and no colluders. Every item is labeled by all five poisoners and",
        "one of the two honest annotators. The poisoners always emit `class_1` when the true",
        "class is `class_0`. Dawid-Skene assumes annotators are independent given the true class",
        "and that the class prior can be told from the labels. A bloc that outvotes the honest",
        "annotator on the attacked class can move the latent class.",
        "",
        f"Pooled poisoning precision {_fmt(precision)}, recall {_fmt(recall)} (tp={tp}, fp={fp}, fn={fn}).",
        "",
        "| seed | precision | recall | accused honest annotators | prior of class_0 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        accused = ", ".join(row["accused_honest"]) if row["accused_honest"] else "(none)"
        prior = f"{float(row['source_prior']):.3f}"
        lines.append(f"| {row['seed']} | {_fmt(row['precision'])} | {_fmt(row['recall'])} | {accused} | {prior} |")
    lines.extend(
        [
            "",
            "Detection fails on this panel. On the seeds where recall is 0, the poisoners are",
            "labeled honest and at least one honest annotator is labeled poisoning. The prior of",
            "`class_0` falls well below the other classes: the flipped label has been absorbed",
            "into the consensus. Seeds where the poisoners are still flagged also accuse the",
            "honest annotators. A majority coordinated flip is outside what this detector can",
            "separate from the truth.",
        ]
    )
    return "\n".join(lines)


def _role_table(group: dict[str, object]) -> str:
    pooled = group["pooled"]
    assert isinstance(pooled, dict)
    lines = [
        "| attack | precision | recall | tp | fp | fn |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for role in ROLES:
        row = pooled[role]
        lines.append(
            f"| {role} | {_fmt(row['precision'])} | {_fmt(row['recall'])} | {row['tp']} | {row['fp']} | {row['fn']} |"
        )
    return "\n".join(lines)


def _seed_table(group: dict[str, object]) -> str:
    per_seed = group["per_seed"]
    assert isinstance(per_seed, list)
    lines = [
        "| seed | careless recall | colluding recall | poisoning recall |",
        "| --- | --- | --- | --- |",
    ]
    for row in per_seed:
        lines.append(
            f"| {row['seed']} | {_fmt(row['careless'])} | {_fmt(row['colluding'])} | {_fmt(row['poisoning'])} |"
        )
    return "\n".join(lines)


def _sweep_table(label: str, sweep: list[tuple[float, dict[str, object]]], role: str) -> str:
    lines = [
        f"| {label} | precision | recall | tp | fp | fn |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for rate, group in sweep:
        pooled = group["pooled"]
        assert isinstance(pooled, dict)
        row = pooled[role]
        lines.append(
            f"| {rate:.2f} | {_fmt(row['precision'])} | {_fmt(row['recall'])} | "
            f"{row['tp']} | {row['fp']} | {row['fn']} |"
        )
    return "\n".join(lines)


def _recall(group: dict[str, object], role: str) -> float | None:
    pooled = group["pooled"]
    assert isinstance(pooled, dict)
    value = pooled[role]["recall"]
    return None if value is None else float(value)


def _precision(group: dict[str, object], role: str) -> float | None:
    pooled = group["pooled"]
    assert isinstance(pooled, dict)
    value = pooled[role]["precision"]
    return None if value is None else float(value)


def _fmt(value: object) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.2f}"


def _config_dict() -> dict[str, object]:
    config = ScanConfig()
    return {field: getattr(config, field) for field in config.__dataclass_fields__}


if __name__ == "__main__":
    main()
