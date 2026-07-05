"""
AMLSim → Postgres ingestion script.

Reads the CSVs produced by AMLSim (accounts.csv, transactions.csv) and
loads them into customers/accounts/transactions. Writes audit_log entries
at start and end of every run — architecture requires every pipeline
step to be logged, and data ingestion is a pipeline step.

Usage:
    python scripts/run_ingestion.py                  # ingest all CSVs
    python scripts/run_ingestion.py --reset          # truncate first, then ingest
    python scripts/run_ingestion.py --dry-run        # validate CSVs, no writes
"""

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sqlalchemy import insert, text

from aml_agent.config import settings
from aml_agent.db.models import Account, AuditLog, Customer, Transaction
from aml_agent.db.session import get_db

logger = logging.getLogger("aml_agent.ingestion")

# Bulk insert batch size. 5000 balances round-trip overhead against memory
# and Postgres's max query size. At 10K rows total this is effectively 2
# batches for accounts, ~18-20 for transactions. Tune upward for larger runs.
BATCH_SIZE = 5000


def _configure_logging() -> None:
    """Basic stderr logging with timestamps. Batch script, not a service —
    structured JSON logging would be overkill."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )


def _write_audit(db, action: str, details: dict) -> None:
    """Append an audit_log entry for this ingestion run. entity_type is
    'ingestion_run' since ingestion runs aren't first-class entities in
    our schema (no runs table) — timestamp-int gives a stable identifier."""
    db.add(AuditLog(
        entity_type="ingestion_run",
        entity_id=int(datetime.now(timezone.utc).timestamp() * 1000),
        action=action,
        actor="ingestion_pipeline",
        details=details,
    ))
    db.commit()


def _reset_tables(db) -> None:
    """Truncate ingestion target tables, resetting sequences. CASCADE handles
    FK dependencies (alerts, if any) automatically. Used with --reset for
    dev iteration; unsafe once real alerts or audit history exist beyond
    ingestion — hence opt-in, not default."""
    logger.warning("Resetting tables: transactions, accounts, customers")
    db.execute(text("TRUNCATE transactions, accounts, customers RESTART IDENTITY CASCADE"))
    db.commit()


# ---------------------------------------------------------------------------
# Substep 1.6.3 — customers + accounts
# ---------------------------------------------------------------------------

def _map_customer_type(amlsim_type: str) -> str:
    """AMLSim uses 'I' (individual) / 'B' (business). Our customer_type
    enum values are 'individual' / 'business'. Unknown types default to
    'individual' — safer than crashing since AMLSim occasionally uses
    other letters for edge cases."""
    return "business" if amlsim_type == "B" else "individual"


def _map_account_type(amlsim_type: str) -> str:
    """Map AMLSim I/B to our account_type enum. Retail individuals default
    to 'checking' since AMLSim doesn't distinguish checking/savings; loses
    granularity but explicit and reversible if the eval harness needs it."""
    return "business" if amlsim_type == "B" else "checking"


def _map_account_status(amlsim_stat: str) -> str:
    """AMLSim 'acct_stat' is typically 'A' (active) or blank. Our enum
    values: active / dormant / frozen / closed. Default to active — the
    close_dt column signals actual closure, not status alone."""
    return "active"


def _load_customers_and_accounts(db, src_dir: Path) -> tuple[int, int]:
    """Read AMLSim accounts.csv → insert customers + accounts (1:1).

    Bulk insert via SQLAlchemy Core insert().values(list_of_dicts) rather
    than ORM session.add_all(): skips per-row ORM instrumentation, roughly
    10-50x faster at this row count.

    Returns (customers_inserted, accounts_inserted).
    """
    path = src_dir / "accounts.csv"
    logger.info("Loading accounts from %s", path)
    df = pd.read_csv(path)
    logger.info("Read %d rows from accounts.csv", len(df))

    # --- Customer rows ---
    # external_ref uses AMLSim's acct_id (unique per row). full_name concat
    # of first_name + last_name; blank names get placeholder to satisfy
    # NOT NULL — real banks have KYC-required names, but AMLSim leaves some
    # business accounts empty here.
    customer_rows = []
    for _, r in df.iterrows():
        first = str(r.get("first_name", "") or "").strip()
        last = str(r.get("last_name", "") or "").strip()
        full_name = f"{first} {last}".strip() or f"UNKNOWN-{r['acct_id']}"

        dob = r.get("birth_date")
        # AMLSim writes 'NaN' for missing DOB (businesses); pandas parses
        # to float nan. Convert to None so the DB stores NULL, not the
        # string 'nan'.
        if pd.isna(dob):
            dob = None

        customer_rows.append({
            "external_ref": str(r["acct_id"]),
            "customer_type": _map_customer_type(str(r["type"])),
            "full_name": full_name,
            "date_of_birth": dob,
            "country_code": str(r["country"])[:2],
            "risk_rating": "low",   # baseline; rule engine may update later
            "onboarding_date": pd.to_datetime(r["open_dt"]).date(),
        })

    # Bulk insert in batches, letting Postgres assign customer_id via
    # BIGSERIAL. We need the mapping {external_ref -> customer_id} for
    # FK linkage on accounts, so we re-query after insert rather than
    # relying on RETURNING (RETURNING with bulk insert has driver quirks
    # in psycopg2 v2).
    for i in range(0, len(customer_rows), BATCH_SIZE):
        db.execute(insert(Customer), customer_rows[i:i + BATCH_SIZE])
    db.commit()
    logger.info("Inserted %d customers", len(customer_rows))

    # Map external_ref -> customer_id for FK linkage on accounts.
    result = db.execute(text("SELECT external_ref, customer_id FROM customers"))
    ext_to_cust_id = {row[0]: row[1] for row in result}

    # --- Account rows ---
    account_rows = []
    for _, r in df.iterrows():
        acct_ext = str(r["acct_id"])
        close_dt = r.get("close_dt")
        # AMLSim emits close_dt as either blank OR 1970-01-01 (unix epoch)
        # for still-open accounts. Both must map to NULL to satisfy the
        # chk_closed_consistency constraint (status=active requires closed_at IS NULL).
        if pd.isna(close_dt) or str(close_dt).strip() == "":
            close_dt = None
        else:
            close_dt = pd.to_datetime(close_dt)
            if close_dt.year <= 1970:
                close_dt = None

        account_rows.append({
            "external_ref": acct_ext,
            "customer_id": ext_to_cust_id[acct_ext],
            "account_type": _map_account_type(str(r["type"])),
            "status": _map_account_status(str(r.get("acct_stat", ""))),
            "opened_at": pd.to_datetime(r["open_dt"]),
            "closed_at": close_dt,
            "currency": str(r.get("acct_rptng_crncy", "USD"))[:3],
        })

    for i in range(0, len(account_rows), BATCH_SIZE):
        db.execute(insert(Account), account_rows[i:i + BATCH_SIZE])
    db.commit()
    logger.info("Inserted %d accounts", len(account_rows))

    return len(customer_rows), len(account_rows)


# ---------------------------------------------------------------------------
# Substep 1.6.4 — transactions
# ---------------------------------------------------------------------------

def _map_channel(amlsim_tx_type: str) -> str:
    """AMLSim's tx_type is typically 'TRANSFER'. Map to our transaction_channel
    enum: TRANSFER → wire (bank-to-bank). Extend when AMLSim configs add
    other types (CASH, DEPOSIT, WITHDRAWAL)."""
    mapping = {
        "TRANSFER": "wire",
        "DEPOSIT": "cash",
        "WITHDRAWAL": "cash",
        "PAYMENT": "ach",
    }
    return mapping.get(amlsim_tx_type.upper(), "wire")


def _load_transactions(db, src_dir: Path) -> int:
    """Read AMLSim transactions.csv → insert into transactions.

    Two-phase FK resolution: load {acct_external_ref → account_id} into
    memory once, then use dict lookups per row. Alternative (join on
    external_ref per row) would issue ~90K queries; the map costs one
    query for 10K rows and ~1MB RAM.
    """
    path = src_dir / "transactions.csv"
    logger.info("Loading transactions from %s", path)
    df = pd.read_csv(path)
    logger.info("Read %d rows from transactions.csv", len(df))

    result = db.execute(text("SELECT external_ref, account_id FROM accounts"))
    ext_to_acct_id = {row[0]: row[1] for row in result}

    tx_rows = []
    skipped = 0
    for _, r in df.iterrows():
        orig = str(r["orig_acct"])
        bene = str(r["bene_acct"])
        if orig not in ext_to_acct_id or bene not in ext_to_acct_id:
            # Defensive skip: shouldn't happen if accounts.csv was loaded
            # first, but AMLSim occasionally emits transactions referencing
            # accounts filtered out earlier. Logging count avoids silent loss.
            skipped += 1
            continue
        if orig == bene:
            # chk_not_self_transfer constraint would reject anyway; skip
            # here so batch insert doesn't fail whole batch on one bad row.
            skipped += 1
            continue

        # AMLSim is_sar is 'True'/'False' string or bool; normalize.
        is_sar_raw = r.get("is_sar", False)
        is_sar = str(is_sar_raw).strip().lower() == "true" if isinstance(is_sar_raw, str) else bool(is_sar_raw)

        tx_rows.append({
            "external_ref": str(r["tran_id"]),
            "sender_account_id": ext_to_acct_id[orig],
            "receiver_account_id": ext_to_acct_id[bene],
            "amount": float(r["base_amt"]),
            "currency": "USD",  # AMLSim doesn't emit per-tx currency; inherit account default
            "channel": _map_channel(str(r["tx_type"])),
            "executed_at": pd.to_datetime(r["tran_timestamp"]),
            "is_laundering_ground_truth": is_sar,
        })

    if skipped:
        logger.warning("Skipped %d transactions (missing FK or self-transfer)", skipped)

    for i in range(0, len(tx_rows), BATCH_SIZE):
        db.execute(insert(Transaction), tx_rows[i:i + BATCH_SIZE])
    db.commit()
    logger.info("Inserted %d transactions", len(tx_rows))

    return len(tx_rows)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_ingestion(dry_run: bool = False, reset: bool = False) -> None:
    src_dir = settings.amlsim_output_dir
    logger.info("Ingestion source: %s", src_dir)

    if not src_dir.exists():
        logger.error("AMLSim output directory does not exist: %s", src_dir)
        sys.exit(1)

    if dry_run:
        logger.info("Dry run — no DB writes will occur")
        # Still exercise CSV parse so schema drift surfaces here.
        pd.read_csv(src_dir / "accounts.csv")
        pd.read_csv(src_dir / "transactions.csv")
        logger.info("Dry run OK: both CSVs parseable")
        return

    with get_db() as db:
        _write_audit(db, "ingestion_started", {"source_dir": str(src_dir), "reset": reset})

        if reset:
            _reset_tables(db)

        n_cust, n_acct = _load_customers_and_accounts(db, src_dir)
        n_tx = _load_transactions(db, src_dir)

        _write_audit(db, "ingestion_completed", {
            "source_dir": str(src_dir),
            "customers_inserted": n_cust,
            "accounts_inserted": n_acct,
            "transactions_inserted": n_tx,
        })
        logger.info("Ingestion complete: %d customers, %d accounts, %d transactions",
                    n_cust, n_acct, n_tx)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest AMLSim output CSVs into Postgres.")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs without writing.")
    parser.add_argument("--reset", action="store_true", help="Truncate target tables first.")
    args = parser.parse_args()

    _configure_logging()
    run_ingestion(dry_run=args.dry_run, reset=args.reset)


if __name__ == "__main__":
    main()