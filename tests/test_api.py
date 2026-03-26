from fastapi.testclient import TestClient
from app.main import app
from app.database.db import SessionLocal
from app.database.models import FeatureStore, ModelPrediction, User

client = TestClient(app)


def _delete_user_with_dependencies(db, email: str):
    existing = db.query(User).filter(User.email == email).first()
    if not existing:
        return

    db.query(FeatureStore).filter(FeatureStore.user_id == existing.id).delete()
    db.query(ModelPrediction).filter(ModelPrediction.user_id == existing.id).delete()
    db.delete(existing)
    db.commit()

def test_churn_prediction():
    # Clean up from previous runs
    db = SessionLocal()
    _delete_user_with_dependencies(db, "test@example.com")
    db.close()

    user_response = client.post("/users/", json={
        "name": "Test User",
        "email": "test@example.com",
        "subscription_plan": "basic",
        "monthly_spend": 100.0,
        "region": "Lagos",
        "device_type": "mobile"
    })
    user_id = user_response.json()["id"]

    response = client.post(f"/churn/predict/{user_id}")
    assert response.status_code == 200
    data = response.json()
    assert "churn_probability" in data
    assert 0.0 <= data["churn_probability"] <= 1.0
