from aml_agent.db.session import get_db
from aml_agent.eval.labels import ground_truth_account_days, alert_account_days, sample_alerts_stratified

with get_db() as db:
    truth = ground_truth_account_days(db)
    alerts = alert_account_days(db)
    sample = sample_alerts_stratified(db, target_size=100)

print(f"Ground truth (account, day) pairs: {len(truth)}")
print(f"Alert coverage per rule:")
for code, s in alerts.items():
    print(f"  {code}: {len(s)} distinct (account, day) pairs")
print(f"Stratified sample size: {len(sample)}")