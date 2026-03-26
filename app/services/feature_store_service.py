"""
Feature Store Service - Manages feature computation, storage, and retrieval
"""
from datetime import datetime
from sqlalchemy.orm import Session
import pandas as pd

from app.database.db import SessionLocal
from app.database.models import User, UsageLog, Transaction, FeatureStore
from training.feature_pipeline import build_features, build_single_user_features
from app.database.feature_store_crud import (
    store_features,
    get_latest_features,
    get_all_latest_features,
    feature_store_to_dict,
    delete_old_features
)


class FeatureStoreService:
    """Service for managing feature store operations"""
    
    @staticmethod
    def compute_and_store_all_features(db: Session = None, feature_version: str = "v1"):
        """
        Compute features for all users and store in feature store.
        
        Args:
            db: Database session (creates new if not provided)
            feature_version: Version identifier for features
            
        Returns:
            List of stored FeatureStore records
        """
        owns_session = db is None
        if db is None:
            db = SessionLocal()
        
        try:
            # Get all features using the existing pipeline
            features_df = build_features()
            
            stored_records = []
            for _, row in features_df.iterrows():
                user_id = int(row['id'])
                features_dict = {
                    "tenure_days": row['tenure_days'],
                    "avg_sessions_14d": row['avg_sessions_14d'],
                    "avg_sessions_30d": row['avg_sessions_30d'],
                    "total_minutes_30d": row['total_minutes_30d'],
                    "failed_payments_30d": row['failed_payments_30d'],
                    "revenue_30d": row['revenue_30d'],
                    "subscription_plan": row['subscription_plan'],
                    "churn_probability": row['churn_probability'],
                }
                
                record = store_features(db, user_id, features_dict, feature_version)
                stored_records.append(record)
            
            return stored_records
        finally:
            if owns_session and db:
                db.close()
    
    @staticmethod
    def compute_and_store_single_user_features(user_id: int, db: Session = None, feature_version: str = "v1"):
        """
        Compute and store features for a single user.
        
        Args:
            user_id: User ID
            db: Database session
            feature_version: Version identifier for features
            
        Returns:
            FeatureStore record
        """
        owns_session = db is None
        if db is None:
            db = SessionLocal()
        
        try:
            user_features_df = build_single_user_features(user_id)
            
            if user_features_df is None or user_features_df.empty:
                return None
            
            row = user_features_df.iloc[0]
            features_dict = {
                "tenure_days": row['tenure_days'],
                "avg_sessions_14d": row['avg_sessions_14d'],
                "avg_sessions_30d": row['avg_sessions_30d'],
                "total_minutes_30d": row['total_minutes_30d'],
                "failed_payments_30d": row['failed_payments_30d'],
                "revenue_30d": row['revenue_30d'],
                "subscription_plan": row['subscription_plan'],
            }
            
            # Get current churn probability from user record
            user = db.query(User).filter(User.id == user_id).first()
            if user:
                features_dict["churn_probability"] = user.churn_probability
            
            record = store_features(db, user_id, features_dict, feature_version)
            return record
        finally:
            if owns_session and db:
                db.close()
    
    @staticmethod
    def get_features(user_id: int, db: Session = None):
        """
        Get the latest features for a user.
        
        Args:
            user_id: User ID
            db: Database session
            
        Returns:
            Dictionary of features or None
        """
        owns_session = db is None
        if db is None:
            db = SessionLocal()
        
        try:
            features = get_latest_features(db, user_id)
            return feature_store_to_dict(features)
        finally:
            if owns_session and db:
                db.close()
    
    @staticmethod
    def get_all_features(db: Session = None):
        """
        Get latest features for all users.
        
        Args:
            db: Database session
            
        Returns:
            List of feature dictionaries
        """
        owns_session = db is None
        if db is None:
            db = SessionLocal()
        
        try:
            features_list = get_all_latest_features(db)
            return [feature_store_to_dict(f) for f in features_list]
        finally:
            if owns_session and db:
                db.close()
    
    @staticmethod
    def get_features_as_dataframe(db: Session = None):
        """
        Get all latest features as a pandas DataFrame.
        
        Args:
            db: Database session
            
        Returns:
            Pandas DataFrame
        """
        owns_session = db is None
        if db is None:
            db = SessionLocal()
        
        try:
            features_list = get_all_latest_features(db)
            data = [feature_store_to_dict(f) for f in features_list]
            return pd.DataFrame(data)
        finally:
            if owns_session and db:
                db.close()
    
    @staticmethod
    def cleanup_old_features(days_to_keep: int = 30, db: Session = None):
        """
        Delete old feature records.
        
        Args:
            days_to_keep: Number of days to keep
            db: Database session
            
        Returns:
            Number of records deleted
        """
        owns_session = db is None
        if db is None:
            db = SessionLocal()
        
        try:
            return delete_old_features(db, days_to_keep)
        finally:
            if owns_session and db:
                db.close()
