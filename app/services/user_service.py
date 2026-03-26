# app/services/user_service.py
from sqlalchemy.orm import Session
from app.database.models import User

class UserCreate:
    def __init__(self, name, email, subscription_plan, monthly_spend, region, device_type):
        self.name = name
        self.email = email
        self.subscription_plan = subscription_plan
        self.monthly_spend = monthly_spend
        self.region = region
        self.device_type = device_type

def create_user(db: Session, user_data: UserCreate) -> User:
    user = User(
        name=user_data.name,
        email=user_data.email,
        subscription_plan=user_data.subscription_plan,
        monthly_spend=user_data.monthly_spend,
        region=user_data.region,
        device_type=user_data.device_type
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user