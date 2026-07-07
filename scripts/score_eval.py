"""
Offline scorer for agent eval results.

Reads eval_results/agent_run_latest.json (or a specified file),
computes detection correctness (confusion matrix, P/R/F1) and
case-file quality metrics, and writes a summary JSON alongside.

Rationale for separating scoring from running: standard ML-eval
practice (per scikit-learn docs) — persist predictions once, re-score
many times. This lets you re-compute metrics after changing scoring
criteria (e.g. reclassifying under_review) without burning Groq/Claude
tokens on a re-run.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


_TAKEN_SERIOUSLY = {"escalated", "closed_sar_filed", "under_review"}


def _load_records(path: Path) -> list[dict]:
    """Load records from an eval run JSON file."""
    data = json.loads(path.read_text())
    return data.get("records", [])


def _score_detection(records: list[dict]) -> dict[str, float | int]:
    """Confusion matrix + P/R/F1 over completed (non-ERROR) records."""
    tp = fp = fn = tn = 0
    scored = [r for r in records if r["applied_status"] != "ERROR"]
    for r in scored:
        truth = r["ground_truth_laundering"]
        positive = r["applied_status"] in _TAKEN_SERIOUSLY
        if truth and positive:
            tp += 1
        elif truth and not positive:
            fn += 1
        elif not truth and positive:
            fp += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "scored_count": len(scored),
    }


def _score_quality(records: list[dict]) -> dict[str, float]:
    """Case-file quality metrics — orthogonal to detection correctness."""
    n = len(records) or 1
    parse_ok = sum(1 for r in records if r["case_file_parse_ok"])
    complete = sum(1 for r in records if r["stopped_reason"] == "complete")
    error = sum(1 for r in records if r["applied_status"] == "ERROR")
    verdicts = [r["guardrail_verdict"] for r in records]
    return {
        "parse_ok_rate": round(parse_ok / n, 4),
        "complete_rate": round(complete / n, 4),
        "error_rate": round(error / n, 4),
        "guardrail_pass_rate": round(verdicts.count("pass") / n, 4),
        "guardrail_review_rate": round(verdicts.count("requires_review") / n, 4),
        "guardrail_block_rate": round(verdicts.count("block") / n, 4),
    }


def _score_cost(records: list[dict]) -> dict[str, float | int]:
    """Token usage summary across the run."""
    totals = [r.get("total_tokens", 0) for r in records]
    completed = [t for t, r in zip(totals, records) if r["applied_status"] != "ERROR"]
    return {
        "total_tokens": sum(totals),
        "mean_tokens_per_alert": round(sum(completed) / len(completed), 1) if completed else 0,
        "max_tokens_alert": max(completed) if completed else 0,
        "min_tokens_alert": min(completed) if completed else 0,
    }


def _score_latency(records: list[dict]) -> dict[str, float]:
    """Elapsed-time summary across completed alerts."""
    times = [r["elapsed_sec"] for r in records if r["applied_status"] != "ERROR"]
    if not times:
        return {"mean_sec": 0, "max_sec": 0, "min_sec": 0}
    return {
        "mean_sec": round(sum(times) / len(times), 2),
        "max_sec": round(max(times), 2),
        "min_sec": round(min(times), 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Score agent eval results offline.")
    parser.add_argument(
        "--input", type=Path, default=Path("eval_results/agent_run_latest.json"),
        help="Path to the eval run JSON file.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("eval_results"),
        help="Directory for the summary JSON.",
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: {args.input} not found.", file=sys.stderr)
        sys.exit(1)

    records = _load_records(args.input)
    detection = _score_detection(records)
    quality = _score_quality(records)
    cost = _score_cost(records)
    latency = _score_latency(records)

    summary = {
        "scored_at": datetime.now(timezone.utc).isoformat(),
        "source_file": str(args.input),
        "sample_size": len(records),
        "detection": detection,
        "quality": quality,
        "cost": cost,
        "latency": latency,
    }

    # Persist summary.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "eval_summary.json"
    out_path.write_text(json.dumps(summary, indent=2))

    # Print report.
    print(f"\n{'='*50}")
    print(f"  AGENT EVAL SUMMARY")
    print(f"  Source: {args.input}")
    print(f"  Alerts: {len(records)}")
    print(f"{'='*50}\n")

    print("  Detection Correctness")
    print(f"  {'─'*30}")
    print(f"  TP: {detection['tp']:>4}   FN: {detection['fn']:>4}")
    print(f"  FP: {detection['fp']:>4}   TN: {detection['tn']:>4}")
    print(f"  Precision: {detection['precision']:.4f}")
    print(f"  Recall:    {detection['recall']:.4f}")
    print(f"  F1:        {detection['f1']:.4f}\n")

    print("  Case File Quality")
    print(f"  {'─'*30}")
    for k, v in quality.items():
        print(f"  {k:<26} {v:.4f}")

    print(f"\n  Token Cost")
    print(f"  {'─'*30}")
    print(f"  Total tokens:     {cost['total_tokens']:>10,}")
    print(f"  Mean per alert:   {cost['mean_tokens_per_alert']:>10,.1f}")
    print(f"  Max:              {cost['max_tokens_alert']:>10,}")
    print(f"  Min:              {cost['min_tokens_alert']:>10,}")

    print(f"\n  Latency")
    print(f"  {'─'*30}")
    print(f"  Mean: {latency['mean_sec']:.2f}s  Max: {latency['max_sec']:.2f}s  Min: {latency['min_sec']:.2f}s")

    print(f"\n  Summary persisted to: {out_path}\n")


if __name__ == "__main__":
    main()