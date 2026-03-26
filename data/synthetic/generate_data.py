from datetime import datetime, timedelta
import random
from faker import Faker

fake = Faker()

PLANS = ["free", "beta", "premium"]
REGIONS = ["NG-North", "NG-South", "NG-East", "NG-West", "GH", "KE", "ZA"]
DEVICES = ["android", "ios", "web"]

def generate_users(n=2000):
    users = []
    for i in range(n):
        name = fake.name()
        email = f"user{i}_{fake.user_name()}@example.com"
        plan = random.choices(PLANS, weights=[0.7, 0.2, 0.1])[0]
        region = random.choice(REGIONS)
        device = random.choice(DEVICES)
        signup_date = datetime.utcnow() - timedelta(days=random.randint(1, 365))

        users.append({
            "name": name,
            "email": email,
            "subscription_plan": plan,
            "monthly_spend": 0.0,  # will be derived later
            "region": region,
            "device_type": device,
            "signup_date": signup_date,
        })
    return users

def generate_activity_and_transactions(user_id, plan, signup_date, days=90):
    usage_logs = []
    transactions = []

    start = max(signup_date, datetime.utcnow() - timedelta(days=days))
    for d in range(days):
        day = start + timedelta(days=d)

        # base activity by plan
        base_sessions = {"free": 1, "beta": 2, "premium": 3}[plan]
        sessions = max(0, int(random.gauss(base_sessions, 1)))

        minutes_spent = max(0, int(random.gauss(10 * base_sessions, 8)))
        actions_count = max(0, int(random.gauss(20 * base_sessions, 15)))

        usage_logs.append({
            "user_id": user_id,
            "day": day,
            "sessions": sessions,
            "minutes_spent": minutes_spent,
            "actions_count": actions_count,
        })

        # transactions: more likely for paid plans
        if plan in ["beta", "premium"] and random.random() < (0.25 if plan == "beta" else 0.45):
            amount = 5000 if plan == "beta" else 20000
            # some failures to simulate payment issues
            successful = random.random() > (0.10 if plan == "beta" else 0.06)
            transactions.append({
                "user_id": user_id,
                "day": day,
                "amount": float(amount),
                "successful": successful,
            })

    return usage_logs, transactions

def churn_probability_from_behavior(last_14_days_sessions, failed_payments_30d, plan):
    # simple rule: low activity + failures => higher churn
    p = 0.15
    if plan == "free":
        p += 0.10
    if last_14_days_sessions < 5:
        p += 0.35
    if failed_payments_30d >= 2:
        p += 0.30
    return min(max(p, 0.02), 0.95)

def build_dataset(n_users=2000, days=90, seed=42):
    random.seed(seed)
    Faker.seed(seed)

    users = generate_users(n_users)

    all_usage = []
    all_tx = []
    churn_labels = []  # per user

    for idx, u in enumerate(users, start=1):
        usage, tx = generate_activity_and_transactions(idx, u["subscription_plan"], u["signup_date"], days=days)
        all_usage.extend(usage)
        all_tx.extend(tx)

        # compute behavior signals
        last_14 = [row for row in usage if row["day"] >= (datetime.utcnow() - timedelta(days=14))]
        last_14_sessions = sum(r["sessions"] for r in last_14)

        last_30_tx = [t for t in tx if t["day"] >= (datetime.utcnow() - timedelta(days=30))]
        failed_30 = sum(1 for t in last_30_tx if not t["successful"])

        p = churn_probability_from_behavior(last_14_sessions, failed_30, u["subscription_plan"])
        churned = (random.random() < p)

        churn_labels.append({
            "user_id": idx,
            "churn_probability": p,
            "churned": churned,
        })

        # derive monthly_spend estimate (avg successful tx in last 30d)
        success_30 = [t for t in last_30_tx if t["successful"]]
        u["monthly_spend"] = float(sum(t["amount"] for t in success_30))

    return users, all_usage, all_tx, churn_labels