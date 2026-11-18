"""Plant honest, careless, colluding, and poisoning annotators.

The generator is a measurement tool. It is not a claim about how a vendor's
workforce behaves. Colluding pairs are pinned onto every item, and so are
poisoners, as long as ``labels_per_item`` still has room for at least one
annotator from outside that group. If the planted attackers already fill the
panel, annotators are drawn uniformly instead. That fallback is what the
majority-poisoning benchmark uses.

A careless annotator at an odd index always submits the first class name.
Even indexes draw a uniform label. Colluding partners copy the pair leader
with probability ``collude_rate`` on items the leader also labeled, and
otherwise label with ``honest_accuracy``. Poisoners emit ``poison_target``
on items whose true class is ``poison_source`` with probability
``poison_rate``, and otherwise label with ``honest_accuracy``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from annotrust.errors import AnnotrustError

_FAST_GAP = (0.15, 0.8)
_NORMAL_GAP = (6.0, 30.0)


@dataclass(frozen=True)
class AnnotatorSpec:
    """Ground truth for one simulated annotator."""

    annotator_id: str
    role: str
    style: str | None = None
    leader: bool = False
    partner_id: str | None = None
    source: str | None = None
    target: str | None = None
    rate: float | None = None

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "annotator_id": self.annotator_id,
            "role": self.role,
        }
        if self.style is not None:
            payload["style"] = self.style
        if self.role == "colluding":
            payload["leader"] = self.leader
            payload["partner_id"] = self.partner_id
        if self.role == "poisoning":
            payload["source"] = self.source
            payload["target"] = self.target
            payload["rate"] = self.rate
        return payload


@dataclass(frozen=True)
class Simulation:
    """Annotations plus the planted ground truth."""

    frame: pd.DataFrame
    classes: tuple[str, ...]
    item_labels: dict[str, str]
    annotators: tuple[AnnotatorSpec, ...]
    seed: int
    params: dict[str, str | int | float | bool | None]

    def roles(self) -> dict[str, str]:
        return {annotator.annotator_id: annotator.role for annotator in self.annotators}

    def truth_dict(self) -> dict[str, object]:
        return {
            "seed": self.seed,
            "classes": list(self.classes),
            "params": self.params,
            "items": [{"item_id": item_id, "label": label} for item_id, label in self.item_labels.items()],
            "annotators": [annotator.to_dict() for annotator in self.annotators],
        }


def simulate(
    *,
    n_items: int = 200,
    n_classes: int | None = None,
    class_names: list[str] | None = None,
    labels_per_item: int = 7,
    n_honest: int = 6,
    n_careless: int = 2,
    n_colluders: int = 2,
    n_poisoners: int = 1,
    honest_accuracy: float = 0.9,
    leader_accuracy: float = 0.6,
    collude_rate: float = 0.95,
    poison_rate: float = 0.9,
    poison_source: str | None = None,
    poison_target: str | None = None,
    timestamps: bool = True,
    seed: int = 0,
) -> Simulation:
    """Return a labeled annotation table and the planted roles."""
    classes = _classes(n_classes, class_names)
    _check_rate("honest_accuracy", honest_accuracy)
    _check_rate("leader_accuracy", leader_accuracy)
    _check_rate("collude_rate", collude_rate)
    _check_rate("poison_rate", poison_rate)
    if n_items < 1:
        raise AnnotrustError("n_items must be at least 1")
    if min(n_honest, n_careless, n_colluders, n_poisoners) < 0:
        raise AnnotrustError("annotator counts must be non-negative")
    if n_colluders % 2 != 0:
        raise AnnotrustError("n_colluders must be a non-negative even number (they are planted in pairs)")
    if labels_per_item < 1:
        raise AnnotrustError("labels_per_item must be at least 1")

    specs = _specs(
        n_honest,
        n_careless,
        n_colluders,
        n_poisoners,
        classes,
        poison_source,
        poison_target,
        poison_rate,
    )
    if not specs:
        raise AnnotrustError("simulate needs at least one annotator")
    if labels_per_item > len(specs):
        raise AnnotrustError(f"labels_per_item={labels_per_item} is larger than the annotator pool ({len(specs)})")

    rng = np.random.default_rng(seed)
    width = max(4, len(str(n_items - 1)))
    item_ids = [f"item_{index:0{width}d}" for index in range(n_items)]
    truth_index = rng.integers(0, len(classes), size=n_items)
    item_labels = {item_id: classes[int(truth_index[index])] for index, item_id in enumerate(item_ids)}

    colluder_ids = [spec.annotator_id for spec in specs if spec.role == "colluding"]
    poisoner_ids = [spec.annotator_id for spec in specs if spec.role == "poisoning"]
    outside_ids = [spec.annotator_id for spec in specs if spec.role not in {"colluding", "poisoning"}]
    everyone = [spec.annotator_id for spec in specs]

    assignments: dict[str, list[str]] = {spec.annotator_id: [] for spec in specs}
    pinned = colluder_ids + poisoner_ids
    use_pinning = len(pinned) < labels_per_item and labels_per_item - len(pinned) <= len(outside_ids)
    for item_id in item_ids:
        panel = _panel(rng, pinned, outside_ids, everyone, labels_per_item, use_pinning)
        for annotator_id in panel:
            assignments[annotator_id].append(item_id)

    label_of: dict[tuple[str, str], str] = {}
    leaders = [spec for spec in specs if spec.role == "colluding" and spec.leader]
    partners = [spec for spec in specs if spec.role == "colluding" and not spec.leader]
    ordered = (
        [spec for spec in specs if spec.role == "honest"]
        + [spec for spec in specs if spec.role == "careless"]
        + leaders
        + partners
        + [spec for spec in specs if spec.role == "poisoning"]
    )
    for spec in ordered:
        for item_id in assignments[spec.annotator_id]:
            truth = item_labels[item_id]
            label_of[(spec.annotator_id, item_id)] = _draw_label(
                rng,
                spec,
                item_id,
                truth,
                classes,
                label_of,
                honest_accuracy,
                leader_accuracy,
                collude_rate,
            )

    timestamps_of: dict[tuple[str, str], pd.Timestamp] = {}
    if timestamps:
        start = pd.Timestamp("2026-01-01T00:00:00Z")
        for spec in specs:
            owned = assignments[spec.annotator_id]
            if not owned:
                continue
            order = rng.permutation(len(owned))
            low, high = _FAST_GAP if spec.role == "careless" else _NORMAL_GAP
            gaps = rng.uniform(low, high, size=max(len(owned) - 1, 0))
            cumulative = np.concatenate([np.array([0.0]), np.cumsum(gaps)]) if len(owned) else np.array([])
            for position, item_pos in enumerate(order):
                timestamps_of[(spec.annotator_id, owned[int(item_pos)])] = start + pd.to_timedelta(
                    float(cumulative[position]), unit="s"
                )

    rows: list[dict[str, object]] = []
    for spec in specs:
        for item_id in assignments[spec.annotator_id]:
            row: dict[str, object] = {
                "item_id": item_id,
                "annotator_id": spec.annotator_id,
                "label": label_of[(spec.annotator_id, item_id)],
            }
            if timestamps:
                row["timestamp"] = timestamps_of[(spec.annotator_id, item_id)]
            rows.append(row)
    frame = pd.DataFrame(rows).sort_values(["item_id", "annotator_id"], kind="mergesort").reset_index(drop=True)
    source = _resolve_class(poison_source, list(classes), 0) if n_poisoners else None
    target = _resolve_class(poison_target, list(classes), 1) if n_poisoners else None
    params: dict[str, str | int | float | bool | None] = {
        "n_items": n_items,
        "labels_per_item": labels_per_item,
        "n_honest": n_honest,
        "n_careless": n_careless,
        "n_colluders": n_colluders,
        "n_poisoners": n_poisoners,
        "honest_accuracy": honest_accuracy,
        "leader_accuracy": leader_accuracy,
        "collude_rate": collude_rate,
        "poison_rate": poison_rate,
        "poison_source": source,
        "poison_target": target,
        "timestamps": timestamps,
        "attackers_pinned_to_every_item": use_pinning,
        "seed": seed,
    }
    return Simulation(
        frame=frame,
        classes=tuple(classes),
        item_labels=item_labels,
        annotators=tuple(specs),
        seed=seed,
        params=params,
    )


def write_simulation(simulation: Simulation, path: Path, truth_path: Path | None = None) -> None:
    """Write annotations and, optionally, the ground-truth JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = simulation.frame.copy()
    if "timestamp" in frame.columns:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True).dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    if path.suffix.lower() == ".jsonl":
        records = frame.to_dict(orient="records")
        with path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, separators=(",", ":")) + "\n")
    else:
        frame.to_csv(path, index=False)
    if truth_path is not None:
        truth_path.parent.mkdir(parents=True, exist_ok=True)
        truth_path.write_text(json.dumps(simulation.truth_dict(), indent=2) + "\n", encoding="utf-8")


def _check_rate(name: str, value: float) -> None:
    if not 0.0 <= float(value) <= 1.0:
        raise AnnotrustError(f"{name} must be between 0 and 1")


def _classes(n_classes: int | None, class_names: list[str] | None) -> list[str]:
    if class_names is not None:
        names = [name.strip() for name in class_names]
        if any(not name for name in names):
            raise AnnotrustError("class names must be non-empty")
        if len(set(names)) != len(names):
            raise AnnotrustError("class names must be unique")
        if len(names) < 2:
            raise AnnotrustError("class names need at least two classes")
        if len(names) > 200:
            raise AnnotrustError("n_classes must be at most 200")
        if n_classes is not None and n_classes != len(names):
            raise AnnotrustError("n_classes does not match class_names")
        return names
    count = 4 if n_classes is None else n_classes
    if count < 2:
        raise AnnotrustError("n_classes must be at least 2")
    if count > 200:
        raise AnnotrustError("n_classes must be at most 200")
    return [f"class_{index}" for index in range(count)]


def _specs(
    n_honest: int,
    n_careless: int,
    n_colluders: int,
    n_poisoners: int,
    classes: list[str],
    poison_source: str | None,
    poison_target: str | None,
    poison_rate: float,
) -> list[AnnotatorSpec]:
    specs: list[AnnotatorSpec] = [
        AnnotatorSpec(annotator_id=f"honest_{index:02d}", role="honest") for index in range(n_honest)
    ]
    for index in range(n_careless):
        specs.append(
            AnnotatorSpec(
                annotator_id=f"careless_{index:02d}",
                role="careless",
                style="constant" if index % 2 else "random",
            )
        )
    source = _resolve_class(poison_source, classes, 0) if n_poisoners else None
    target = _resolve_class(poison_target, classes, 1) if n_poisoners else None
    if source is not None and source == target:
        raise AnnotrustError("poison source and target must be different classes")
    for index in range(0, n_colluders, 2):
        leader = f"colluder_{index:02d}"
        partner = f"colluder_{index + 1:02d}"
        specs.append(AnnotatorSpec(annotator_id=leader, role="colluding", leader=True, partner_id=partner))
        specs.append(AnnotatorSpec(annotator_id=partner, role="colluding", leader=False, partner_id=leader))
    for index in range(n_poisoners):
        specs.append(
            AnnotatorSpec(
                annotator_id=f"poisoner_{index:02d}",
                role="poisoning",
                source=source,
                target=target,
                rate=poison_rate,
            )
        )
    return specs


def _resolve_class(name: str | None, classes: list[str], default_index: int) -> str:
    if name is None:
        return classes[default_index]
    if name in classes:
        return name
    if name.isdigit():
        index = int(name)
        if 0 <= index < len(classes):
            return classes[index]
    known = ", ".join(classes)
    raise AnnotrustError(f"Unknown class {name!r}. Known classes: {known}")


def _panel(
    rng: np.random.Generator,
    pinned: list[str],
    outside: list[str],
    everyone: list[str],
    labels_per_item: int,
    use_pinning: bool,
) -> list[str]:
    if use_pinning:
        rest = labels_per_item - len(pinned)
        pool = np.array(outside, dtype=object)
        chosen = rng.choice(pool, size=rest, replace=False)
        return list(pinned) + [str(item) for item in np.atleast_1d(chosen)]
    pool = np.array(everyone, dtype=object)
    chosen = rng.choice(pool, size=labels_per_item, replace=False)
    return [str(item) for item in np.atleast_1d(chosen)]


def _draw_label(
    rng: np.random.Generator,
    spec: AnnotatorSpec,
    item_id: str,
    truth: str,
    classes: list[str],
    label_of: dict[tuple[str, str], str],
    honest_accuracy: float,
    leader_accuracy: float,
    collude_rate: float,
) -> str:
    if spec.role == "careless":
        if spec.style == "constant":
            return classes[0]
        return classes[int(rng.integers(0, len(classes)))]
    if spec.role == "colluding" and not spec.leader and spec.partner_id is not None:
        leader_label = label_of.get((spec.partner_id, item_id))
        if leader_label is not None and float(rng.random()) < collude_rate:
            return leader_label
        return _honest(rng, truth, classes, honest_accuracy)
    if spec.role == "poisoning" and spec.source is not None and spec.target is not None and spec.rate is not None:
        if truth == spec.source and float(rng.random()) < spec.rate:
            return spec.target
        return _honest(rng, truth, classes, honest_accuracy)
    accuracy = leader_accuracy if spec.leader else honest_accuracy
    return _honest(rng, truth, classes, accuracy)


def _honest(rng: np.random.Generator, truth: str, classes: list[str], accuracy: float) -> str:
    if float(rng.random()) < accuracy:
        return truth
    others = [label for label in classes if label != truth]
    return others[int(rng.integers(0, len(others)))]
