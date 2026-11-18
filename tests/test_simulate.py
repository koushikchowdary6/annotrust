"""The simulator plants the roles it claims to plant."""

from __future__ import annotations

from pathlib import Path

import pytest

from annotrust.errors import AnnotrustError
from annotrust.simulate import simulate, write_simulation


def test_roles_and_determinism() -> None:
    first = simulate(seed=3, timestamps=False)
    second = simulate(seed=3, timestamps=False)
    assert first.frame.equals(second.frame)
    roles = first.roles()
    assert list(roles.values()).count("honest") == 6
    assert list(roles.values()).count("careless") == 2
    assert list(roles.values()).count("colluding") == 2
    assert list(roles.values()).count("poisoning") == 1
    assert first.params["attackers_pinned_to_every_item"] is True


def test_copy_rate_one_matches_the_leader_on_shared_items() -> None:
    simulation = simulate(seed=1, collude_rate=1.0, timestamps=False, n_items=30, labels_per_item=5)
    frame = simulation.frame
    leader = frame[frame["annotator_id"] == "colluder_00"].set_index("item_id")["label"]
    partner = frame[frame["annotator_id"] == "colluder_01"].set_index("item_id")["label"]
    shared = leader.index.intersection(partner.index)
    assert len(shared) > 0
    assert leader.loc[shared].equals(partner.loc[shared])


def test_poison_rate_one_flips_only_the_source_class() -> None:
    simulation = simulate(
        seed=2,
        poison_rate=1.0,
        honest_accuracy=1.0,
        n_careless=0,
        n_colluders=0,
        n_poisoners=1,
        n_honest=3,
        labels_per_item=3,
        timestamps=False,
        class_names=["cat", "dog", "bird"],
        poison_source="cat",
        poison_target="dog",
    )
    poison = simulation.frame[simulation.frame["annotator_id"] == "poisoner_00"]
    saw_source = False
    for item_id, label in zip(poison["item_id"], poison["label"], strict=True):
        truth = simulation.item_labels[item_id]
        if truth == "cat":
            saw_source = True
            assert label == "dog"
        else:
            assert label == truth
    assert saw_source


def test_constant_careless_and_fast_gaps() -> None:
    simulation = simulate(seed=0, n_items=40, timestamps=True)
    constant = simulation.frame[simulation.frame["annotator_id"] == "careless_01"]["label"]
    assert constant.nunique() == 1
    careless = simulation.frame[simulation.frame["annotator_id"] == "careless_00"].sort_values("timestamp")
    gaps = careless["timestamp"].diff().dt.total_seconds().dropna()
    honest = simulation.frame[simulation.frame["annotator_id"] == "honest_00"].sort_values("timestamp")
    honest_gaps = honest["timestamp"].diff().dt.total_seconds().dropna()
    assert float(gaps.max()) < 2
    assert float(honest_gaps.min()) > 2


def test_perfect_honest_panel_matches_truth() -> None:
    simulation = simulate(
        seed=0,
        n_honest=3,
        n_careless=0,
        n_colluders=0,
        n_poisoners=0,
        n_items=20,
        labels_per_item=3,
        honest_accuracy=1.0,
        timestamps=False,
    )
    for item_id, label in zip(simulation.frame["item_id"], simulation.frame["label"], strict=True):
        assert label == simulation.item_labels[item_id]


def test_rejects_bad_requests() -> None:
    with pytest.raises(AnnotrustError, match="even"):
        simulate(n_colluders=1)
    with pytest.raises(AnnotrustError, match="different"):
        simulate(poison_source="class_0", poison_target="class_0")
    with pytest.raises(AnnotrustError, match="between 0 and 1"):
        simulate(poison_rate=1.5)
    with pytest.raises(AnnotrustError, match="at least 2"):
        simulate(n_classes=1)


def test_truth_file_roundtrip(tmp_path: Path) -> None:
    simulation = simulate(seed=4, n_items=15, timestamps=True)
    out = tmp_path / "data.csv"
    truth = tmp_path / "truth.json"
    write_simulation(simulation, out, truth)
    text = out.read_text(encoding="utf-8")
    assert "item_id,annotator_id,label,timestamp" in text
    assert "2026-01-01T" in text
    saved = truth.read_text(encoding="utf-8")
    assert "poisoner_00" in saved
    jsonl = tmp_path / "data.jsonl"
    write_simulation(simulation, jsonl)
    assert jsonl.read_text(encoding="utf-8").count("\n") == len(simulation.frame)
