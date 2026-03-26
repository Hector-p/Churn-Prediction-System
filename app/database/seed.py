from app.database.db import SessionLocal, engine, Base
from app.database.models import User, UsageLog, Transaction
from data.synthetic.generate_data import build_dataset

def run_seed(n_users=2000, days=90):
    Base.metadata.create_all(bind=engine)

    db = SessionLocal()
    try:
        users, usage_logs, transactions, churn_labels = build_dataset(n_users=n_users, days=days)

        # Insert Users
        user_objs = []
        for u in users:
            user_objs.append(User(**u))
        db.add_all(user_objs)
        db.commit()

        # Insert usage logs
        db.add_all([UsageLog(**row) for row in usage_logs])
        db.commit()

        # Insert transactions
        db.add_all([Transaction(**row) for row in transactions])
        db.commit()

        # Update churn_probability field on users (optional for now)
        label_map = {c["user_id"]: c["churn_probability"] for c in churn_labels}
        for user in db.query(User).all():
            user.churn_probability = float(label_map.get(user.id, 0.0))
        db.commit()

        print(f"Seeded: {n_users} users, {len(usage_logs)} usage logs, {len(transactions)} transactions.")
    finally:
        db.close()

if __name__ == "__main__":
    run_seed()