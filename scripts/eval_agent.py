"""
CLI: evaluate agent detection quality against AMLSim ground truth.

Checkpoint/resume with sleep-and-retry on rate limits.
"""

import argparse
import json
import logging
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from anthropic import RateLimitError as AnthropicRateLimitError
from sqlalchemy import select

from aml_agent.agent.orchestrator import investigate_alert
from aml_agent.db.models import Alert, Transaction
from aml_agent.db.session import get_db
from aml_agent.eval.labels import sample_alerts_stratified


_TAKEN_SERIOUSLY = {"escalated", "closed_sar_filed", "under_review"}
_DISMISSED = {"closed_false_positive", "closed_no_action"}
_RETRY_AFTER_RE = re.compile(r"try again in (?:(\d+)h)?(?:(\d+)m)?(?:([\d.]+)s)?")
_DEFAULT_WAIT_SEC = 20
_MAX_RETRIES_PER_ALERT = 5


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )


def _parse_retry_after_seconds(error_message: str) -> int:
    """Extract wait time from rate-limit error body."""
    match = _RETRY_AFTER_RE.search(error_message)
    if not match:
        return _DEFAULT_WAIT_SEC
    hours = int(match.group(1) or 0)
    minutes = int(match.group(2) or 0)
    seconds = float(match.group(3) or 0)
    total = int(hours * 3600 + minutes * 60 + seconds) + 30
    return max(total, 30)


def _load_ground_truth_for_alerts(db, alert_ids: list[int]) -> dict[int, bool]:
    rows = db.execute(
        select(Alert.alert_id, Transaction.is_laundering_ground_truth)
        .join(Transaction, Transaction.transaction_id == Alert.transaction_id)
        .where(Alert.alert_id.in_(alert_ids))
    ).all()
    return {r[0]: bool(r[1]) for r in rows}


def _score_detection(records: list[dict]) -> dict[str, float | int]:
    tp = fp = fn = tn = 0
    for r in records:
        truth = r["ground_truth_laundering"]
        took_seriously = r["applied_status"] in _TAKEN_SERIOUSLY
        if truth and took_seriously:
            tp += 1
        elif truth and not took_seriously:
            fn += 1
        elif not truth and took_seriously:
            fp += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "recall": recall, "f1": f1}


def _score_quality(records: list[dict]) -> dict[str, float]:
    n = len(records) or 1
    parse_ok = sum(1 for r in records if r["case_file_parse_ok"])
    complete = sum(1 for r in records if r["stopped_reason"] == "complete")
    verdicts = [r["guardrail_verdict"] for r in records]
    return {
        "parse_ok_rate": parse_ok / n,
        "complete_rate": complete / n,
        "guardrail_pass_rate": verdicts.count("pass") / n,
        "guardrail_review_rate": verdicts.count("requires_review") / n,
        "guardrail_block_rate": verdicts.count("block") / n,
    }


def _load_existing_results(out_path: Path) -> list[dict]:
    if not out_path.exists():
        return []
    try:
        data = json.loads(out_path.read_text())
        return data.get("records", [])
    except (json.JSONDecodeError, OSError) as e:
        logging.getLogger("aml_agent.eval.agent").warning(
            "could not load %s (%r); starting fresh", out_path, e
        )
        return []


def _write_results(out_path: Path, records: list[dict]) -> None:
    out_path.write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sample_size": len(records),
        "records": records,
    }, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate agent detection quality on stratified sample."
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--target-size", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, default=Path("eval_results"))
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()

    _configure_logging()
    logger = logging.getLogger("aml_agent.eval.agent")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "agent_run_latest.json"

    with get_db() as db:
        logger.info("Building stratified sample")
        alert_ids = sample_alerts_stratified(db, target_size=args.target_size)
        if args.limit is not None:
            alert_ids = alert_ids[:args.limit]
        logger.info("Sample size: %d", len(alert_ids))

        truth_map = _load_ground_truth_for_alerts(db, alert_ids)

        records: list[dict] = [] if args.fresh else _load_existing_results(out_path)
        done_ids = {
            r["alert_id"] for r in records if r["applied_status"] != "ERROR"
        }
        if done_ids:
            logger.info("Resuming: %d already done, skipping", len(done_ids))

        remaining_ids = [a for a in alert_ids if a not in done_ids]
        records = [r for r in records if r["alert_id"] not in remaining_ids]

        total = len(alert_ids)
        for i, alert_id in enumerate(remaining_ids, 1):
            start = time.time()
            logger.info(
                "[%d/%d remaining, %d/%d overall] alert %d",
                i, len(remaining_ids), len(done_ids) + i, total, alert_id,
            )

            retries = 0
            outcome = None

            # --- Eval idempotency: snapshot alert status before investigation ---
            # investigate_alert() commits status changes to the DB as a
            # side effect (it must, for production use). But eval runs
            # should be non-destructive — otherwise repeated runs shift
            # alerts out of 'open' and LifecycleStateGuardrail blocks
            # all subsequent runs. Snapshot here, restore after.
            alert_row = db.execute(
                select(Alert).where(Alert.alert_id == alert_id)
            ).scalar_one_or_none()
            original_status = (
                alert_row.status.value if hasattr(alert_row.status, "value")
                else str(alert_row.status)
            ) if alert_row else "open"
            original_narrative = alert_row.narrative if alert_row else None

            while True:
                try:
                    outcome = investigate_alert(db, alert_id)
                    break
                except AnthropicRateLimitError as e:
                    retries += 1
                    if retries > _MAX_RETRIES_PER_ALERT:
                        logger.error(
                            "alert %d: exceeded %d retries, recording ERROR",
                            alert_id, _MAX_RETRIES_PER_ALERT,
                        )
                        break
                    wait_sec = _parse_retry_after_seconds(str(e))
                    _write_results(out_path, records)
                    logger.warning(
                        "Rate limit at alert %d. Sleeping %ds (retry %d/%d)...",
                        alert_id, wait_sec, retries, _MAX_RETRIES_PER_ALERT,
                    )
                    time.sleep(wait_sec)
                    continue
                    logger.exception("alert %d raised: %r", alert_id, e)
                    records.append({
                        "alert_id": alert_id,
                        "ground_truth_laundering": truth_map.get(alert_id, False),
                        "applied_status": "ERROR",
                        "agent_action": None,
                        "guardrail_verdict": None,
                        "stopped_reason": "exception",
                        "case_file_parse_ok": False,
                        "tool_call_count": 0,
                        "elapsed_sec": time.time() - start,
                        "diagnostic": repr(e),
                        "token_usage": [],
                        "total_tokens": 0,
                    })
                    _write_results(out_path, records)
                    outcome = None
                    break

            # --- Restore original alert status so eval is non-destructive ---
            if alert_row is not None:
                alert_row.status = original_status
                alert_row.narrative = original_narrative
                db.commit()

            if outcome is None:
                if retries > _MAX_RETRIES_PER_ALERT:
                    records.append({
                        "alert_id": alert_id,
                        "ground_truth_laundering": truth_map.get(alert_id, False),
                        "applied_status": "ERROR",
                        "agent_action": None,
                        "guardrail_verdict": None,
                        "stopped_reason": "rate_limit_exhausted",
                        "case_file_parse_ok": False,
                        "tool_call_count": 0,
                        "elapsed_sec": time.time() - start,
                        "diagnostic": f"rate limit after {_MAX_RETRIES_PER_ALERT} retries",
                        "token_usage": [],
                        "total_tokens": 0,
                    })
                    _write_results(out_path, records)
                continue

            total_tokens = sum(u["total_tokens"] for u in outcome.token_usage)
            records.append({
                "alert_id": outcome.alert_id,
                "ground_truth_laundering": truth_map.get(alert_id, False),
                "applied_status": outcome.applied_status,
                "agent_action": (
                    outcome.agent_decision.action.value
                    if outcome.agent_decision else None
                ),
                "guardrail_verdict": (
                    outcome.guardrail_verdict.value
                    if outcome.guardrail_verdict else None
                ),
                "stopped_reason": outcome.stopped_reason,
                "case_file_parse_ok": outcome.case_file_parse_ok,
                "tool_call_count": len(outcome.tool_calls),
                "elapsed_sec": time.time() - start,
                "diagnostic": outcome.diagnostic,
                "token_usage": outcome.token_usage,
                "total_tokens": total_tokens,
            })
            _write_results(out_path, records)

    detection = _score_detection(records)
    quality = _score_quality(records)
    run_total_tokens = sum(r.get("total_tokens", 0) for r in records)

    print(f"\nSample size: {len(records)}")
    print(f"Results persisted to: {out_path}")
    print(f"Total tokens: {run_total_tokens}\n")

    print("=== Detection correctness ===")
    print(f"  TP: {detection['tp']:>4}   FN: {detection['fn']:>4}")
    print(f"  FP: {detection['fp']:>4}   TN: {detection['tn']:>4}")
    print(f"  Precision: {detection['precision']:.3f}")
    print(f"  Recall:    {detection['recall']:.3f}")
    print(f"  F1:        {detection['f1']:.3f}\n")

    print("=== Case file quality ===")
    for k, v in quality.items():
        print(f"  {k:<24} {v:.3f}")

    if len(records) == total:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        snapshot_path = args.output_dir / f"agent_run_{timestamp}.json"
        snapshot_path.write_text(out_path.read_text())
        print(f"\nFull run complete — snapshot: {snapshot_path}")


if __name__ == "__main__":
    main()