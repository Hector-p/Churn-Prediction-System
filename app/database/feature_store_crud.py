from datetime import datetime
from sqlalchemy.orm import Session
from app.database.models import FeatureStore


def _to_python_scalar(value):
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            return value
    return value


def store_features(db: Session, user_id: int, features: dict, feature_version: str = "v1"):
    """
    Store engineered features for a user in the feature store.
    
    Args:
        db: Database session
        user_id: User ID
        features: Dictionary containing feature values
        feature_version: Version of the feature set
        
    Returns:
        FeatureStore object
    """
    feature_record = FeatureStore(
        user_id=_to_python_scalar(user_id),
        tenure_days=_to_python_scalar(features.get("tenure_days")),
        avg_sessions_14d=_to_python_scalar(features.get("avg_sessions_14d", 0.0)),
        avg_sessions_30d=_to_python_scalar(features.get("avg_sessions_30d", 0.0)),
        total_minutes_30d=_to_python_scalar(features.get("total_minutes_30d", 0.0)),
        failed_payments_30d=_to_python_scalar(features.get("failed_payments_30d", 0.0)),
        revenue_30d=_to_python_scalar(features.get("revenue_30d", 0.0)),
        subscription_plan=features.get("subscription_plan"),
        churn_probability=_to_python_scalar(features.get("churn_probability")),
        feature_version=feature_version,
        computed_at=datetime.utcnow()
    )
    db.add(feature_record)
    db.commit()
    db.refresh(feature_record)
    return feature_record


def get_latest_features(db: Session, user_id: int):
    """
    Retrieve the latest engineered features for a user.
    
    Args:
        db: Database session
        user_id: User ID
        
    Returns:
        FeatureStore object or None
    """
    return db.query(FeatureStore)\
        .filter(FeatureStore.user_id == user_id)\
        .order_by(FeatureStore.computed_at.desc())\
        .first()


def get_features_for_users(db: Session, user_ids: list):
    """
    Retrieve the latest features for multiple users.
    
    Args:
        db: Database session
        user_ids: List of user IDs
        
    Returns:
        List of FeatureStore objects
    """
    subquery = db.query(
        FeatureStore.user_id,
        FeatureStore.id
    ).filter(FeatureStore.user_id.in_(user_ids))\
    .order_by(FeatureStore.user_id, FeatureStore.computed_at.desc())\
    .distinct(FeatureStore.user_id)\
    .subquery()
    
    return db.query(FeatureStore)\
        .join(subquery, FeatureStore.id == subquery.c.id)\
        .all()


def get_all_latest_features(db: Session):
    """
    Retrieve the latest engineered features for all users.
    
    Args:
        db: Database session
        
    Returns:
        List of FeatureStore objects (latest for each user)
    """
    subquery = db.query(
        FeatureStore.user_id,
        FeatureStore.id
    ).order_by(FeatureStore.user_id, FeatureStore.computed_at.desc())\
    .distinct(FeatureStore.user_id)\
    .subquery()
    
    return db.query(FeatureStore)\
        .join(subquery, FeatureStore.id == subquery.c.id)\
        .all()


def delete_old_features(db: Session, days_to_keep: int = 30):
    """
    Delete feature records older than specified days.
    Keeps only the latest version for each user within the timeframe.
    
    Args:
        db: Database session
        days_to_keep: Number of days to keep features
        
    Returns:
        Number of records deleted
    """
    from datetime import timedelta
    cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)
    
    # Get all user IDs and their latest feature IDs to keep
    latest_subquery = db.query(
        FeatureStore.user_id,
        FeatureStore.id
    ).order_by(FeatureStore.user_id, FeatureStore.computed_at.desc())\
    .distinct(FeatureStore.user_id)\
    .subquery()
    
    # Delete records older than cutoff that are not the latest
    records_to_delete = db.query(FeatureStore)\
        .filter(FeatureStore.computed_at < cutoff_date)\
        .all()
    
    for record in records_to_delete:
        db.delete(record)
    
    db.commit()
    return len(records_to_delete)


def feature_store_to_dict(feature_record: FeatureStore):
    """
    Convert a FeatureStore record to a dictionary.
    
    Args:
        feature_record: FeatureStore object
        
    Returns:
        Dictionary representation
    """
    if not feature_record:
        return None
    
    return {
        "user_id": feature_record.user_id,
        "tenure_days": feature_record.tenure_days,
        "avg_sessions_14d": feature_record.avg_sessions_14d,
        "avg_sessions_30d": feature_record.avg_sessions_30d,
        "total_minutes_30d": feature_record.total_minutes_30d,
        "failed_payments_30d": feature_record.failed_payments_30d,
        "revenue_30d": feature_record.revenue_30d,
        "subscription_plan": feature_record.subscription_plan,
        "churn_probability": feature_record.churn_probability,
        "computed_at": feature_record.computed_at,
        "feature_version": feature_record.feature_version
    }
