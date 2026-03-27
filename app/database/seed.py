import random
from app.database.db import SessionLocal, engine, Base
from app.database.models import User, UsageLog, Transaction
from data.synthetic.generate_data import build_dataset



HIGH_RISK_FLOOR = 0.60      # high-risk users must be AT LEAST this
HIGH_RISK_CEIL  = 0.95      # cap so we don't crowd at 1.0
MEDIUM_RISK_FLOOR = 0.30    # medium-risk users land here
MEDIUM_RISK_CEIL  = 0.59
LOW_RISK_FLOOR  = 0.05      # low-risk users never collapse to 0
LOW_RISK_CEIL   = 0.29


HIGH_RISK_RAW_THRESHOLD = 0.55   # raw values above this → high-risk bucket


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _calibrate_churn_probability(raw: float, risk_level: str | None = None) -> float:
    """
    Map a raw synthetic churn probability into a calibrated range.

    - High-risk  → [0.60, 0.95]
    - Medium-risk → [0.30, 0.59]
    - Low-risk   → [0.05, 0.29]

    If `risk_level` is provided (e.g. 'high', 'medium', 'low') it takes
    precedence over the raw threshold heuristic.
    """
    if risk_level is not None:
        level = risk_level.lower()
    elif raw >= HIGH_RISK_RAW_THRESHOLD:
        level = "high"
    elif raw >= 0.30:
        level = "medium"
    else:
        level = "low"

    if level == "high":
       
        span = HIGH_RISK_CEIL - HIGH_RISK_FLOOR
        relative = (raw - HIGH_RISK_RAW_THRESHOLD) / (1.0 - HIGH_RISK_RAW_THRESHOLD + 1e-9)
        calibrated = HIGH_RISK_FLOOR + relative * span
       
        calibrated += random.uniform(-0.02, 0.02)
        return round(_clamp(calibrated, HIGH_RISK_FLOOR, HIGH_RISK_CEIL), 4)

    elif level == "medium":
        span = MEDIUM_RISK_CEIL - MEDIUM_RISK_FLOOR
        relative = (raw - 0.30) / (HIGH_RISK_RAW_THRESHOLD - 0.30 + 1e-9)
        calibrated = MEDIUM_RISK_FLOOR + relative * span
        calibrated += random.uniform(-0.02, 0.02)
        return round(_clamp(calibrated, MEDIUM_RISK_FLOOR, MEDIUM_RISK_CEIL), 4)

    else:  # low
        span = LOW_RISK_CEIL - LOW_RISK_FLOOR
        relative = raw / (0.30 + 1e-9)
        calibrated = LOW_RISK_FLOOR + relative * span
        calibrated += random.uniform(-0.01, 0.01)
        return round(_clamp(calibrated, LOW_RISK_FLOOR, LOW_RISK_CEIL), 4)


def run_seed(n_users: int = 2000, days: int = 90):
    Base.metadata.create_all(bind=engine)

    db = SessionLocal()
    try:
        users, usage_logs, transactions, churn_labels = build_dataset(
            n_users=n_users, days=days
        )

        
        label_map: dict[str, float] = {}
        for c in churn_labels:
            raw = float(c.get("churn_probability", 0.0))
            risk_level = c.get("risk_level")          # optional explicit bucket
            label_map[c["user_id"]] = _calibrate_churn_probability(raw, risk_level)

        
        existing_emails = {
            email for (email,) in db.query(User.email).all()
        }

        user_objs: list[User] = []
        new_user_email_to_obj: dict[str, User] = {}

        for u in users:
            email = u.get("email")
            if not email or email in existing_emails:
                continue

            user_obj = User(**u)
            user_objs.append(user_obj)
            new_user_email_to_obj[email] = user_obj
            existing_emails.add(email)

        if user_objs:
            db.add_all(user_objs)
            db.commit()
            for user_obj in user_objs:
                db.refresh(user_obj)

       
        new_users_only = [u for u in users if u.get("email") in new_user_email_to_obj]

        for original_user, inserted_user in zip(new_users_only, user_objs):
            synthetic_to_real_user_id[original_user["id"]] = inserted_user.id

       
        usage_log_objs: list[UsageLog] = []
        for row in usage_logs:
            real_user_id = synthetic_to_real_user_id.get(row.get("user_id"))
            if real_user_id is None:
                continue
            row_copy = {**row, "user_id": real_user_id}
            usage_log_objs.append(UsageLog(**row_copy))

        if usage_log_objs:
            db.add_all(usage_log_objs)
            db.commit()

       
        transaction_objs: list[Transaction] = []
        for row in transactions:
            real_user_id = synthetic_to_real_user_id.get(row.get("user_id"))
            if real_user_id is None:
                continue
            row_copy = {**row, "user_id": real_user_id}
            transaction_objs.append(Transaction(**row_copy))

        if transaction_objs:
            db.add_all(transaction_objs)
            db.commit()

        
        high_risk_count = medium_risk_count = low_risk_count = 0

        for original_user in users:
            email = original_user.get("email")
            user_obj = new_user_email_to_obj.get(email)
            if not user_obj:
                continue

            prob = label_map.get(original_user.get("id"))

            if prob is None:
               
                risk_level = original_user.get("risk_level", "low")
                defaults = {
                    "high":   random.uniform(HIGH_RISK_FLOOR, HIGH_RISK_CEIL),
                    "medium": random.uniform(MEDIUM_RISK_FLOOR, MEDIUM_RISK_CEIL),
                    "low":    random.uniform(LOW_RISK_FLOOR, LOW_RISK_CEIL),
                }
                prob = round(defaults.get(risk_level.lower(), random.uniform(LOW_RISK_FLOOR, LOW_RISK_CEIL)), 4)

            user_obj.churn_probability = prob

            
            if prob >= HIGH_RISK_FLOOR:
                high_risk_count += 1
            elif prob >= MEDIUM_RISK_FLOOR:
                medium_risk_count += 1
            else:
                low_risk_count += 1

        db.commit()

        print(
            f"Seed complete.\n"
            f"  New users       : {len(user_objs)}\n"
            f"  Usage logs      : {len(usage_log_objs)}\n"
            f"  Transactions    : {len(transaction_objs)}\n"
            f"  High-risk  (≥{HIGH_RISK_FLOOR}) : {high_risk_count}\n"
            f"  Medium-risk     : {medium_risk_count}\n"
            f"  Low-risk        : {low_risk_count}"
        )

    finally:
        db.close()


if __name__ == "__main__":
    run_seed()