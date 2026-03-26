"""
ROI Simulation Service — estimates the return on investment
for churn-prevention interventions targeting high-risk users.
"""
from __future__ import annotations

from sqlalchemy.orm import Session
from app.database.models import User


def simulate_roi(
    db: Session,
    intervention_cost_per_user: float,
    expected_retention_rate: float,
    avg_ltv: float,
    high_risk_users: list[int],
) -> dict:
    """
    Simulate the ROI of running a retention campaign on a set of
    high-risk users.

    Args:
        db: Active database session.
        intervention_cost_per_user: Cost (₦) to intervene per user.
        expected_retention_rate: Fraction of targeted users expected
            to be retained (0.0 – 1.0).
        avg_ltv: Average lifetime value (₦) per retained user.
        high_risk_users: List of User IDs to target.

    Returns:
        Dictionary with ROI simulation results.
    """
    if not high_risk_users:
        return _empty_result()

    # Fetch user records for the supplied IDs
    users = (
        db.query(User)
        .filter(User.id.in_(high_risk_users))
        .all()
    )

    if not users:
        return _empty_result()

    total_targeted = len(users)
    total_monthly_spend = sum(u.monthly_spend or 0.0 for u in users)
    annual_revenue_at_risk = total_monthly_spend * 12
    avg_churn_probability = (
        sum(u.churn_probability or 0.0 for u in users) / total_targeted
    )

    # --- Core calculations ---
    expected_saved_users = total_targeted * expected_retention_rate
    saved_revenue = expected_saved_users * avg_ltv
    total_cost = total_targeted * intervention_cost_per_user
    net_roi = saved_revenue - total_cost
    roi_pct = (net_roi / total_cost * 100) if total_cost > 0 else 0.0
    cost_per_saved = (
        total_cost / expected_saved_users if expected_saved_users > 0 else 0.0
    )

    # --- Per-plan breakdown ---
    plan_breakdown: dict[str, dict] = {}
    for u in users:
        plan = u.subscription_plan or "unknown"
        entry = plan_breakdown.setdefault(plan, {
            "user_count": 0,
            "monthly_spend": 0.0,
            "avg_churn_probability": 0.0,
        })
        entry["user_count"] += 1
        entry["monthly_spend"] += u.monthly_spend or 0.0
        entry["avg_churn_probability"] += u.churn_probability or 0.0

    for plan, entry in plan_breakdown.items():
        count = entry["user_count"]
        entry["avg_churn_probability"] = (
            entry["avg_churn_probability"] / count if count else 0.0
        )
        entry["estimated_saved"] = round(count * expected_retention_rate, 1)
        entry["plan_saved_revenue"] = entry["estimated_saved"] * avg_ltv
        entry["plan_cost"] = count * intervention_cost_per_user
        entry["plan_net_roi"] = entry["plan_saved_revenue"] - entry["plan_cost"]

    return {
        # Headline metrics
        "saved_revenue": round(saved_revenue, 2),
        "net_roi": round(net_roi, 2),
        "roi_pct": round(roi_pct, 1),
        # Extra detail
        "total_targeted_users": total_targeted,
        "expected_saved_users": round(expected_saved_users, 1),
        "total_intervention_cost": round(total_cost, 2),
        "cost_per_saved_user": round(cost_per_saved, 2),
        "annual_revenue_at_risk": round(annual_revenue_at_risk, 2),
        "avg_churn_probability": round(avg_churn_probability, 4),
        "plan_breakdown": plan_breakdown,
    }


def _empty_result() -> dict:
    """Return a zeroed-out result when there are no users to simulate."""
    return {
        "saved_revenue": 0.0,
        "net_roi": 0.0,
        "roi_pct": 0.0,
        "total_targeted_users": 0,
        "expected_saved_users": 0,
        "total_intervention_cost": 0.0,
        "cost_per_saved_user": 0.0,
        "annual_revenue_at_risk": 0.0,
        "avg_churn_probability": 0.0,
        "plan_breakdown": {},
    }
