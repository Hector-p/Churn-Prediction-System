from datetime import datetime, timedelta
import pandas as pd
from sqlalchemy.orm import Session
from app.database.db import SessionLocal
from app.database.models import User, UsageLog, Transaction


def build_features():
    db: Session = SessionLocal()

    try:
        users = pd.read_sql(db.query(User).statement, db.bind)
        usage = pd.read_sql(db.query(UsageLog).statement, db.bind)
        tx = pd.read_sql(db.query(Transaction).statement, db.bind)

        today = datetime.utcnow()

        # -------------------------
        # Feature 1: Customer tenure
        # -------------------------
        users["signup_date"] = pd.to_datetime(users["signup_date"])
        users["tenure_days"] = (today - users["signup_date"]).dt.days

        # -------------------------
        # Last 14 days usage
        # -------------------------
        usage["day"] = pd.to_datetime(usage["day"])
        last_14 = usage[usage["day"] >= today - timedelta(days=14)]

        sessions_14 = (
            last_14.groupby("user_id")["sessions"]
            .mean()
            .reset_index()
            .rename(columns={
                "user_id": "id",
                "sessions": "avg_sessions_14d"
            })
        )

        # -------------------------
        # Last 30 days usage
        # -------------------------
        last_30 = usage[usage["day"] >= today - timedelta(days=30)]

        sessions_30 = (
            last_30.groupby("user_id")["sessions"]
            .mean()
            .reset_index()
            .rename(columns={
                "user_id": "id",
                "sessions": "avg_sessions_30d"
            })
        )

        minutes_30 = (
            last_30.groupby("user_id")["minutes_spent"]
            .sum()
            .reset_index()
            .rename(columns={
                "user_id": "id",
                "minutes_spent": "total_minutes_30d"
            })
        )

        # -------------------------
        # Transactions in last 30 days
        # -------------------------
        tx["day"] = pd.to_datetime(tx["day"])
        tx_30 = tx[tx["day"] >= today - timedelta(days=30)]

        failed_tx = (
            tx_30[tx_30["successful"] == False]
            .groupby("user_id")
            .size()
            .reset_index(name="failed_payments_30d")
            .rename(columns={"user_id": "id"})
        )

        revenue = (
            tx_30[tx_30["successful"] == True]
            .groupby("user_id")["amount"]
            .sum()
            .reset_index()
            .rename(columns={
                "user_id": "id",
                "amount": "revenue_30d"
            })
        )

        # -------------------------
        # Merge all features on id
        # -------------------------
        df = users.merge(sessions_14, on="id", how="left")
        df = df.merge(sessions_30, on="id", how="left")
        df = df.merge(minutes_30, on="id", how="left")
        df = df.merge(failed_tx, on="id", how="left")
        df = df.merge(revenue, on="id", how="left")

        # Fill missing values for users with no activity/payments
        df.fillna(0, inplace=True)

        features = df[
            [
                "id",
                "tenure_days",
                "avg_sessions_14d",
                "avg_sessions_30d",
                "total_minutes_30d",
                "failed_payments_30d",
                "revenue_30d",
                "subscription_plan",
                "churn_probability",
            ]
        ]

        return features

    finally:
        db.close()



def build_single_user_features(user_id: int):
    df = build_features()
    user_df = df[df["id"] == user_id].copy()

    if user_df.empty:
        return None

    return user_df[
        [
            "tenure_days",
            "avg_sessions_14d",
            "avg_sessions_30d",
            "total_minutes_30d",
            "failed_payments_30d",
            "revenue_30d",
            "subscription_plan",
        ]
    ]