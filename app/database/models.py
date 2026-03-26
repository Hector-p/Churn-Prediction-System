from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, ForeignKey
from .db import Base
from datetime import datetime


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    subscription_plan = Column(String, nullable=False, default="free")
    monthly_spend = Column(Float, nullable=False, default=0.0)

    region = Column(String, nullable=True)
    device_type = Column(String, nullable=True)
    signup_date = Column(DateTime, default=datetime.utcnow)

    churn_probability = Column(Float, nullable=False, default=0.0)
    is_active = Column(Boolean, nullable=False, default=True)


class UsageLog(Base):
    __tablename__ = "usage_logs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True, nullable=False)
    day = Column(DateTime, index=True, nullable=False)

    sessions = Column(Integer, nullable=False, default=0)
    minutes_spent = Column(Integer, nullable=False, default=0)
    actions_count = Column(Integer, nullable=False, default=0)


class Transaction(Base):
    __tablename__ = "transactions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True, nullable=False)
    day = Column(DateTime, index=True, nullable=False)

    amount = Column(Float, nullable=False, default=0.0)
    successful = Column(Boolean, nullable=False, default=True)


class ModelPrediction(Base):
    __tablename__ = "model_predictions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    prediction = Column(Integer, nullable=False)
    churn_probability = Column(Float, nullable=False)
    model_version = Column(String, nullable=False, default="v1")
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class FeatureStore(Base):
    __tablename__ = "feature_store"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    
    # Tenure features
    tenure_days = Column(Float, nullable=True)
    
    # Usage features (14 days)
    avg_sessions_14d = Column(Float, nullable=True, default=0.0)
    
    # Usage features (30 days)
    avg_sessions_30d = Column(Float, nullable=True, default=0.0)
    total_minutes_30d = Column(Float, nullable=True, default=0.0)
    
    # Transaction features (30 days)
    failed_payments_30d = Column(Float, nullable=True, default=0.0)
    revenue_30d = Column(Float, nullable=True, default=0.0)
    
    # Categorical features
    subscription_plan = Column(String, nullable=True)
    
    # Target variable / label
    churn_probability = Column(Float, nullable=True)
    
    # Metadata
    computed_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    feature_version = Column(String, nullable=False, default="v1")