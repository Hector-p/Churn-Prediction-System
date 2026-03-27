from app.database.db import SessionLocal, engine, Base
from app.database.models import User, UsageLog, Transaction
from data.synthetic.generate_data import build_dataset


def run_seed(n_users: int = 2000, days: int = 90):
    Base.metadata.create_all(bind=engine)

    db = SessionLocal()
    try:
        users, usage_logs, transactions, churn_labels = build_dataset(
            n_users=n_users, days=days
        )

        # Build quick lookup maps
        label_map = {
            c["user_id"]: float(c["churn_probability"])
            for c in churn_labels
        }

        # Get all existing user emails already in the database
        existing_emails = {
            email for (email,) in db.query(User.email).all()
        }

        # Insert only users that do not already exist
        user_objs = []
        new_user_email_to_obj = {}

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

            # Refresh to get generated IDs from DB
            for user_obj in user_objs:
                db.refresh(user_obj)

        # Map synthetic user_id -> real DB user_id for only newly inserted users
        # Assumes users and churn_labels/usage_logs/transactions were generated together
        synthetic_to_real_user_id = {}
        for original_user, inserted_user in zip(
            [u for u in users if u.get("email") in new_user_email_to_obj],
            user_objs,
        ):
            synthetic_to_real_user_id[original_user["id"]] = inserted_user.id

        # Insert usage logs only for newly inserted users
        usage_log_objs = []
        for row in usage_logs:
            original_user_id = row.get("user_id")
            real_user_id = synthetic_to_real_user_id.get(original_user_id)
            if real_user_id is None:
                continue

            row_copy = row.copy()
            row_copy["user_id"] = real_user_id
            usage_log_objs.append(UsageLog(**row_copy))

        if usage_log_objs:
            db.add_all(usage_log_objs)
            db.commit()

        # Insert transactions only for newly inserted users
        transaction_objs = []
        for row in transactions:
            original_user_id = row.get("user_id")
            real_user_id = synthetic_to_real_user_id.get(original_user_id)
            if real_user_id is None:
                continue

            row_copy = row.copy()
            row_copy["user_id"] = real_user_id
            transaction_objs.append(Transaction(**row_copy))

        if transaction_objs:
            db.add_all(transaction_objs)
            db.commit()

        # Update churn_probability only for newly inserted users
        for original_user in users:
            email = original_user.get("email")
            user_obj = new_user_email_to_obj.get(email)
            if not user_obj:
                continue

            original_user_id = original_user.get("id")
            user_obj.churn_probability = label_map.get(original_user_id, 0.0)

        db.commit()

        print(
            f"Seed complete. Added {len(user_objs)} new users, "
            f"{len(usage_log_objs)} usage logs, {len(transaction_objs)} transactions."
        )
    finally:
        db.close()


if __name__ == "__main__":
    run_seed()