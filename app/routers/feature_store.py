"""
Feature Store API Router - Endpoints for feature store operations
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from datetime import datetime
from typing import List, Optional

from app.database.db import SessionLocal
from app.services.feature_store_service import FeatureStoreService


router = APIRouter(
    prefix="/api/feature-store",
    tags=["Feature Store"]
)


def get_db():
    """Dependency for database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.post("/compute-all")
async def compute_and_store_all_features(feature_version: str = "v1", db: Session = Depends(get_db)):
    """
    Compute and store features for all users.
    
    Args:
        feature_version: Version identifier for the features
        
    Returns:
        Count of features stored
    """
    try:
        records = FeatureStoreService.compute_and_store_all_features(db, feature_version)
        return {
            "status": "success",
            "message": f"Stored features for {len(records)} users",
            "count": len(records),
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/compute/{user_id}")
async def compute_and_store_user_features(
    user_id: int,
    feature_version: str = "v1",
    db: Session = Depends(get_db)
):
    """
    Compute and store features for a single user.
    
    Args:
        user_id: User ID
        feature_version: Version identifier for the features
        
    Returns:
        Stored feature details
    """
    try:
        record = FeatureStoreService.compute_and_store_single_user_features(
            user_id, db, feature_version
        )
        
        if not record:
            raise HTTPException(status_code=404, detail=f"User {user_id} not found")
        
        return {
            "status": "success",
            "user_id": user_id,
            "message": "Features computed and stored",
            "computed_at": record.computed_at.isoformat(),
            "feature_version": record.feature_version
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/features/{user_id}")
async def get_user_features(user_id: int, db: Session = Depends(get_db)):
    """
    Retrieve the latest features for a user.
    
    Args:
        user_id: User ID
        
    Returns:
        User's latest features
    """
    features = FeatureStoreService.get_features(user_id, db)
    
    if not features:
        raise HTTPException(status_code=404, detail=f"Features not found for user {user_id}")
    
    return {
        "status": "success",
        "data": features
    }


@router.get("/features")
async def get_all_features(db: Session = Depends(get_db)):
    """
    Retrieve latest features for all users.
    
    Returns:
        List of all users' latest features
    """
    features = FeatureStoreService.get_all_features(db)
    
    return {
        "status": "success",
        "count": len(features),
        "data": features
    }


@router.post("/cleanup")
async def cleanup_old_features(
    days_to_keep: int = 30,
    db: Session = Depends(get_db)
):
    """
    Delete old feature records.
    
    Args:
        days_to_keep: Number of days of features to keep
        
    Returns:
        Number of records deleted
    """
    try:
        deleted_count = FeatureStoreService.cleanup_old_features(days_to_keep, db)
        
        return {
            "status": "success",
            "message": f"Deleted {deleted_count} old feature records",
            "deleted_count": deleted_count,
            "days_kept": days_to_keep,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
