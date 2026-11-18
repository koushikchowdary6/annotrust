"""CLI exit codes and the two subcommands."""

from __future__ import annotations

import json
import runpy
import subprocess
import sys
from pathlib import Path

import pytest

from annotrust.cli import cli_entrypoint, main
from annotrust.config import ScanConfig
from annotrust.errors import AnnotrustError


def test_version_and_help() -> None:
    assert main(["--version"]) == 0
    assert main(["--help"]) == 0
    assert main(["scan", "--help"]) == 0


def test_scan_fail_on_and_reports(tmp_path: Path) -> None:
    out = tmp_path / "demo.csv"
    truth = tmp_path / "demo.json"
    assert (
        main(
            [
                "simulate",
                "--out",
                str(out),
                "--truth",
                str(truth),
                "--seed",
                "0",
                "--items",
                "80",
                "--class-names",
                "cat,dog,bird,fish",
                "--poison-source",
                "cat",
                "--poison-target",
                "dog",
            ]
        )
        == 0
    )
    report = tmp_path / "report.json"
    html = tmp_path / "report.html"
    code = main(
        [
            "scan",
            str(out),
            "--json",
            str(report),
            "--html",
            str(html),
            "--fail-on",
            "high",
        ]
    )
    assert code == 1
    body = json.loads(report.read_text(encoding="utf-8"))
    assert body["summary"]["annotator_flags"]["high"] >= 1
    assert "flip cat->dog" in report.read_text(encoding="utf-8") or "poisoning" in report.read_text(encoding="utf-8")
    page = html.read_text(encoding="utf-8")
    assert page.startswith("<!DOCTYPE html>")
    assert "http://" not in page
    planted = json.loads(truth.read_text(encoding="utf-8"))
    assert planted["annotators"]


def test_honest_file_passes_fail_on_high(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "honest.csv"
    assert (
        main(
            [
                "simulate",
                "--out",
                str(out),
                "--seed",
                "1",
                "--items",
                "40",
                "--honest",
                "4",
                "--careless",
                "0",
                "--colluders",
                "0",
                "--poisoners",
                "0",
                "--labels-per-item",
                "3",
                "--no-timestamps",
            ]
        )
        == 0
    )
    assert main(["scan", str(out), "--fail-on", "high"]) == 0
    assert "fail-on high: pass" in capsys.readouterr().out


def test_missing_file_and_bad_simulate_args() -> None:
    assert main(["scan", "does-not-exist.csv"]) == 2
    assert main(["simulate", "--out", "x.csv", "--colluders", "1"]) == 2
    assert main(["simulate", "--out", "x.csv", "--class-names", "onlyone"]) == 2


def test_entrypoint_version(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["annotrust", "--version"])
    with pytest.raises(SystemExit) as exc:
        cli_entrypoint()
    assert exc.value.code == 0


def test_module_invocation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["annotrust", "scan", "--help"])
    with pytest.raises(SystemExit) as exc:
        runpy.run_module("annotrust", run_name="__main__")
    assert exc.value.code == 0


def test_subprocess_module() -> None:
    root = Path(__file__).resolve().parents[1]
    env = dict(**{key: value for key, value in __import__("os").environ.items()})
    env["PYTHONPATH"] = str(root / "src")
    completed = subprocess.run(
        [sys.executable, "-m", "annotrust", "--version"],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert completed.returncode == 0
    assert "annotrust" in completed.stdout


def test_config_validation() -> None:
    with pytest.raises(AnnotrustError):
        ScanConfig(max_iter=0).validate()
    with pytest.raises(AnnotrustError):
        ScanConfig(smoothing=-0.1).validate()
    ScanConfig().validate()


def test_bad_fail_level() -> None:
    from annotrust.scan import scan
    from annotrust.simulate import simulate

    result = scan(
        simulate(
            seed=0,
            n_items=20,
            n_honest=3,
            n_careless=0,
            n_colluders=0,
            n_poisoners=0,
            labels_per_item=3,
            timestamps=False,
        ).frame
    )
    with pytest.raises(AnnotrustError, match="fail-on"):
        result.fails("critical")
