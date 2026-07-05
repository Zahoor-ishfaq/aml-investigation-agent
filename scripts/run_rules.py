"""
CLI entry point: execute all registered rules against the DB.

Usage:
    python scripts/run_rules.py                  # run and persist alerts
    python scripts/run_rules.py --dry-run        # count candidates, no writes
"""

import argparse
import logging
import sys

from aml_agent.db.session import get_db
from aml_agent.rule_engine.runner import run_all_rules


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AML rule engine over ingested transactions.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Count candidate alerts per rule without writing to the database.",
    )
    args = parser.parse_args()

    _configure_logging()
    with get_db() as db:
        counts = run_all_rules(db, dry_run=args.dry_run)

    total = sum(counts.values())
    verb = "would insert" if args.dry_run else "inserted"
    print(f"\nRule engine: {verb} {total} alerts total")
    for code, n in counts.items():
        print(f"  {code}: {n}")


if __name__ == "__main__":
    main()