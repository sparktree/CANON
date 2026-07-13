"""Evaluate the required CANON configurations and ablations with one contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import config
from adjudication import sample as sample_adjudication
from error_analysis import sample_errors
from phase4_evaluate import evaluate


REQUIRED_CONFIGS = {"independent", "joint", "joint_soft", "csp"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", action="append", default=[],
                        help="NAME=PREDICTIONS:ASSIGNMENT; repeat for configurations/ablations")
    parser.add_argument("--gold", default=str(config.PHASE2_SPLITS_DIR / "test.jsonl"))
    parser.add_argument("--output-dir", default=str(config.OUTPUTS_ROOT / "phase4"))
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--error-sample-size", type=int, default=100,
                        help="Number of CSP error documents to prepare for Phase 4.6 review (50-100).")
    parser.add_argument("--adjudication-size", type=int, default=200,
                        help="Number of Tier-1 escalations to prepare for dual review.")
    args = parser.parse_args()
    if not 50 <= args.error_sample_size <= 100:
        raise SystemExit("--error-sample-size must be between 50 and 100")
    specs = {}
    for item in args.config:
        name, value = item.split("=", 1)
        path, assignment = value.rsplit(":", 1)
        specs[name] = (Path(path), assignment)
    missing = REQUIRED_CONFIGS - set(specs)
    if missing and not args.allow_partial:
        raise SystemExit(f"missing required Phase 4 configurations: {sorted(missing)}")
    output_dir = Path(args.output_dir)
    results = {}
    for name, (path, assignment) in specs.items():
        if not path.exists():
            raise FileNotFoundError(path)
        results[name] = evaluate(path, Path(args.gold), output_dir / f"{name}.json", assignment)
        if name == "csp":
            results[name]["error_review"] = sample_errors(
                path, Path(args.gold), output_dir / "error_review.jsonl",
                size=args.error_sample_size, assignment=assignment)
            results[name]["tier1_adjudication_sampled"] = sample_adjudication(
                path, output_dir / "tier1_adjudication.jsonl", args.adjudication_size)
    (output_dir / "comparison.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
