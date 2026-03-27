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

        
        existing_emails = {email for (email,) in db.query(User.email).all()}

        user_objs = []
        synthetic_to_user_obj = {}

      
        for synthetic_id, u in enumerate(users, start=1):
            email = u.get("email")

            if not email:
                continue

            if email in existing_emails:
                continue

            user_obj = User(**u)
            user_objs.append(user_obj)
            synthetic_to_user_obj[synthetic_id] = user_obj
            existing_emails.add(email)

      
        if user_objs:
            db.add_all(user_objs)
            db.commit()

           
            for user_obj in user_objs:
                db.refresh(user_obj)

        
        synthetic_to_real_user_id = {
            synthetic_id: user_obj.id
            for synthetic_id, user_obj in synthetic_to_user_obj.items()
        }

        
        usage_log_objs = []
        for row in usage_logs:
            synthetic_user_id = row.get("user_id")
            real_user_id = synthetic_to_real_user_id.get(synthetic_user_id)

            if real_user_id is None:
                continue

            usage_log_objs.append(
                UsageLog(
                    user_id=real_user_id,
                    day=row["day"],
                    sessions=row["sessions"],
                    minutes_spent=row["minutes_spent"],
                    actions_count=row["actions_count"],
                )
            )

        if usage_log_objs:
            db.add_all(usage_log_objs)
            db.commit()

        
        transaction_objs = []
        for row in transactions:
            synthetic_user_id = row.get("user_id")
            real_user_id = synthetic_to_real_user_id.get(synthetic_user_id)

            if real_user_id is None:
                continue

            transaction_objs.append(
                Transaction(
                    user_id=real_user_id,
                    day=row["day"],
                    amount=row["amount"],
                    successful=row["successful"],
                )
            )

        if transaction_objs:
            db.add_all(transaction_objs)
            db.commit()

        
        churn_map = {
            row["user_id"]: float(row["churn_probability"])
            for row in churn_labels
        }

        
        for synthetic_id, user_obj in synthetic_to_user_obj.items():
            user_obj.churn_probability = churn_map.get(synthetic_id, 0.0)

        db.commit()

        print(
            f"Seed complete. Added {len(user_objs)} new users, "
            f"{len(usage_log_objs)} usage logs, "
            f"{len(transaction_objs)} transactions."
        )

    except Exception as e:
        db.rollback()
        print(f"Seeding failed: {e}")
        raise
    finally:
        db.close()


if __name__ == "__main__":
    run_seed()