from app.database.db import SessionLocal
from app.database.models import FeatureStore, ModelPrediction, User
from app.services.churn_service import predict_user_churn
from app.services.user_service import create_user, UserCreate


def _delete_user_with_dependencies(db, email: str):
    existing = db.query(User).filter(User.email == email).first()
    if not existing:
        return

    db.query(FeatureStore).filter(FeatureStore.user_id == existing.id).delete()
    db.query(ModelPrediction).filter(ModelPrediction.user_id == existing.id).delete()
    db.delete(existing)
    db.commit()

def test_prediction_range():
    db = SessionLocal()
    try:
        _delete_user_with_dependencies(db, "predict_test@example.com")

        user = create_user(db, UserCreate(
            name="Test User",
            email="predict_test@example.com",
            subscription_plan="basic",
            monthly_spend=100.0,
            region="Lagos",
            device_type="mobile"
        ))
        result = predict_user_churn(user.id, db)
        assert result is not None
        assert 0 <= result["churn_probability"] <= 1
    finally:
        db.close()
