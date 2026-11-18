"""Command line for ``annotrust scan`` and ``annotrust simulate``."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

from annotrust.config import ScanConfig
from annotrust.errors import AnnotrustError
from annotrust.report import render_html, render_json, render_text
from annotrust.scan import scan
from annotrust.simulate import simulate, write_simulation
from annotrust.version import __version__


def main(argv: list[str] | None = None) -> int:
    """Run the CLI. Returns a process exit code."""
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        code = exc.code
        return int(code) if isinstance(code, int) else 2
    try:
        if args.command == "scan":
            return _scan(args)
        return _simulate(args)
    except AnnotrustError as exc:
        print(f"annotrust: {exc}", file=sys.stderr)
        return 2


def cli_entrypoint() -> None:
    """Setuptools console-script target."""
    raise SystemExit(main())


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        prog="annotrust",
        description="Detect suspicious annotators and poisoned labels in training data.",
    )
    parser.add_argument("--version", action="version", version=f"annotrust {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    scan_parser = commands.add_parser("scan", help="Audit an annotation file")
    scan_parser.add_argument("path", help="CSV, TSV, JSONL, JSON records, Encord JSON, or Label Studio JSON")
    scan_parser.add_argument(
        "--format",
        dest="file_format",
        default="auto",
        choices=["auto", "csv", "tsv", "jsonl", "json", "encord", "label-studio"],
        help="Input format. auto uses the suffix and, for JSON, a few public export shapes.",
    )
    scan_parser.add_argument("--json", dest="json_path", help="Write a JSON report to this path")
    scan_parser.add_argument("--html", dest="html_path", help="Write a self-contained HTML report to this path")
    scan_parser.add_argument(
        "--fail-on",
        choices=["low", "medium", "high"],
        default=None,
        help="Exit 1 when an annotator's severity is at or above this level. Items do not affect the exit code.",
    )
    scan_parser.add_argument("--min-labels", type=int, default=ScanConfig.min_labels)
    scan_parser.add_argument("--min-pair-overlap", type=int, default=ScanConfig.min_pair_overlap)
    scan_parser.add_argument("--poison-min-rate", type=float, default=ScanConfig.poison_min_rate)
    scan_parser.add_argument("--fast-seconds", type=float, default=ScanConfig.fast_seconds)
    scan_parser.add_argument(
        "--cleanlab",
        action="store_true",
        help="Compare item scores with cleanlab when model_prob_<class> columns exist. Requires the optional extra.",
    )

    simulate_parser = commands.add_parser("simulate", help="Generate a dataset with planted annotators")
    simulate_parser.add_argument("-o", "--out", required=True, help="CSV or JSONL path for the annotations")
    simulate_parser.add_argument("--truth", help="Write the planted ground truth JSON here")
    simulate_parser.add_argument("--seed", type=int, default=0)
    simulate_parser.add_argument("--items", type=int, default=200)
    simulate_parser.add_argument("--classes", type=int, default=None)
    simulate_parser.add_argument(
        "--class-names",
        help="Comma-separated class names. Overrides --classes when the lengths match.",
    )
    simulate_parser.add_argument("--labels-per-item", type=int, default=7)
    simulate_parser.add_argument("--honest", type=int, default=6)
    simulate_parser.add_argument("--careless", type=int, default=2)
    simulate_parser.add_argument("--colluders", type=int, default=2)
    simulate_parser.add_argument("--poisoners", type=int, default=1)
    simulate_parser.add_argument("--honest-accuracy", type=float, default=0.9)
    simulate_parser.add_argument("--leader-accuracy", type=float, default=0.6)
    simulate_parser.add_argument("--collude-rate", type=float, default=0.95)
    simulate_parser.add_argument("--poison-rate", type=float, default=0.9)
    simulate_parser.add_argument("--poison-source", default=None)
    simulate_parser.add_argument("--poison-target", default=None)
    simulate_parser.add_argument("--timestamps", action=argparse.BooleanOptionalAction, default=True)
    return parser


def _scan(args: argparse.Namespace) -> int:
    config = ScanConfig(
        min_labels=args.min_labels,
        min_pair_overlap=args.min_pair_overlap,
        poison_min_rate=args.poison_min_rate,
        fast_seconds=args.fast_seconds,
        use_cleanlab=bool(args.cleanlab),
    )
    result = scan(args.path, config, file_format=args.file_format)
    text = render_text(result)
    sys.stdout.write(text if text.endswith("\n") else text + "\n")
    if args.json_path:
        Path(args.json_path).write_text(render_json(result), encoding="utf-8")
        print(f"wrote {args.json_path}", file=sys.stderr)
    if args.html_path:
        Path(args.html_path).write_text(render_html(result), encoding="utf-8")
        print(f"wrote {args.html_path}", file=sys.stderr)
    if args.fail_on:
        counts = result.severity_counts()
        if result.fails(args.fail_on):
            print(
                f"fail-on {args.fail_on}: FAIL (high={counts['high']}, medium={counts['medium']}, low={counts['low']})",
            )
            return 1
        print(f"fail-on {args.fail_on}: pass")
    return 0


def _simulate(args: argparse.Namespace) -> int:
    names = _class_names(args.class_names)
    simulation = simulate(
        n_items=args.items,
        n_classes=args.classes,
        class_names=names,
        labels_per_item=args.labels_per_item,
        n_honest=args.honest,
        n_careless=args.careless,
        n_colluders=args.colluders,
        n_poisoners=args.poisoners,
        honest_accuracy=args.honest_accuracy,
        leader_accuracy=args.leader_accuracy,
        collude_rate=args.collude_rate,
        poison_rate=args.poison_rate,
        poison_source=args.poison_source,
        poison_target=args.poison_target,
        timestamps=bool(args.timestamps),
        seed=args.seed,
    )
    truth_path = Path(args.truth) if args.truth else None
    write_simulation(simulation, Path(args.out), truth_path)
    counts = Counter(annotator.role for annotator in simulation.annotators)
    poison = next((annotator for annotator in simulation.annotators if annotator.role == "poisoning"), None)
    flip = ""
    if poison is not None:
        flip = f" ({poison.source} -> {poison.target})"
    print(
        f"wrote {args.out} ({len(simulation.frame)} annotations, "
        f"{simulation.frame['item_id'].nunique()} items, "
        f"{simulation.frame['annotator_id'].nunique()} annotators)"
    )
    print(
        "planted: "
        f"{counts['honest']} honest, {counts['careless']} careless, "
        f"{counts['colluding']} colluding, {counts['poisoning']} poisoning{flip}"
    )
    if truth_path is not None:
        print(f"wrote {truth_path}")
    return 0


def _class_names(value: str | None) -> list[str] | None:
    if value is None:
        return None
    names = [part.strip() for part in value.split(",") if part.strip()]
    if len(names) < 2:
        raise AnnotrustError("--class-names needs at least two comma-separated names")
    return names
