from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func

from app.database.db import get_db
from app.database.models import ModelPrediction
from app.services.monitoring_service import MonitoringService

router = APIRouter(prefix="/monitoring", tags=["Monitoring"])


@router.get("/predictions")
def get_prediction_logs(limit: int = 50, db: Session = Depends(get_db)):
    """Get recent prediction logs"""
    logs = (
        db.query(ModelPrediction)
        .order_by(ModelPrediction.created_at.desc())
        .limit(limit)
        .all()
    )
    return {
        "count": len(logs),
        "data": logs,
    }


@router.get("/summary")
def get_monitoring_summary(days: int = 7, db: Session = Depends(get_db)):
    """Get prediction summary for the last N days"""
    summary = MonitoringService.get_prediction_summary(db, days=days)
    return {
        "status": "success",
        "data": summary,
    }


@router.get("/daily-metrics")
def get_daily_metrics(days: int = 30, db: Session = Depends(get_db)):
    """Get daily prediction metrics"""
    metrics = MonitoringService.get_daily_metrics(db, days=days)
    return {
        "status": "success",
        "days_analyzed": days,
        "count": len(metrics),
        "data": metrics,
    }


@router.get("/drift-detection")
def detect_drift(baseline_days: int = 30, recent_days: int = 7, db: Session = Depends(get_db)):
    """
    Detect prediction drift.
    Compares recent predictions with baseline to identify distribution shifts.
    """
    drift_analysis = MonitoringService.detect_drift(
        db, 
        baseline_days=baseline_days, 
        recent_days=recent_days
    )
    
    return {
        "status": "success",
        "data": drift_analysis,
    }


@router.get("/high-risk-users")
def get_high_risk_users(threshold: float = 0.7, limit: int = 50, db: Session = Depends(get_db)):
    """Get users with high churn probability for targeted campaigns"""
    users = MonitoringService.get_high_risk_users(db, threshold=threshold, limit=limit)
    
    return {
        "status": "success",
        "threshold": threshold,
        "count": len(users),
        "data": users,
    }


@router.get("/feature-statistics")
def get_feature_statistics(days: int = 30, db: Session = Depends(get_db)):
    """Get statistics on engineered features"""
    stats = MonitoringService.get_feature_statistics(db, days=days)
    
    return {
        "status": "success",
        "data": stats,
    }


@router.get("/health")
def health_check(db: Session = Depends(get_db)):
    """Health check endpoint with prediction counts"""
    total_predictions = db.query(ModelPrediction).count()
    
    avg_probability = db.query(
        func.avg(ModelPrediction.churn_probability)
    ).scalar()

    high_risk_count = db.query(ModelPrediction).filter(
        ModelPrediction.churn_probability >= 0.7
    ).count()

    return {
        "status": "healthy",
        "total_predictions": total_predictions,
        "average_churn_probability": float(avg_probability) if avg_probability is not None else 0.0,
        "high_risk_count": high_risk_count,
    }