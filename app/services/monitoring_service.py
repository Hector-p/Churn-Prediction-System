"""
Monitoring Service - Tracks model predictions, performance, and drift
"""
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import func
import pandas as pd
from app.database.models import ModelPrediction, FeatureStore


class MonitoringService:
    """Service for monitoring prediction quality and drift"""
    
    @staticmethod
    def get_prediction_summary(db: Session, days: int = 7):
        """
        Get prediction summary for the last N days.
        
        Args:
            db: Database session
            days: Number of days to look back
            
        Returns:
            Monitoring metrics dictionary
        """
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        recent_predictions = db.query(ModelPrediction)\
            .filter(ModelPrediction.created_at >= cutoff_date)\
            .all()
        
        if not recent_predictions:
            return {
                "days_analyzed": days,
                "total_predictions": 0,
                "churn_distribution": {},
                "model_versions_used": {},
            }
        
        predictions_df = pd.DataFrame([
            {
                "churn_probability": p.churn_probability,
                "prediction": p.prediction,
                "model_version": p.model_version,
                "created_at": p.created_at,
            }
            for p in recent_predictions
        ])
        
        # Churn distribution
        churn_distribution = {
            "high_risk_70+": len(predictions_df[predictions_df['churn_probability'] >= 0.7]),
            "medium_risk_40_70": len(predictions_df[
                (predictions_df['churn_probability'] >= 0.4) & 
                (predictions_df['churn_probability'] < 0.7)
            ]),
            "low_risk_0_40": len(predictions_df[predictions_df['churn_probability'] < 0.4]),
        }
        
        return {
            "days_analyzed": days,
            "total_predictions": len(predictions_df),
            "churn_distribution": churn_distribution,
            "avg_churn_probability": float(predictions_df['churn_probability'].mean()),
            "median_churn_probability": float(predictions_df['churn_probability'].median()),
            "std_churn_probability": float(predictions_df['churn_probability'].std()),
            "model_versions_used": predictions_df['model_version'].value_counts().to_dict(),
        }
    
    @staticmethod
    def get_daily_metrics(db: Session, days: int = 30):
        """
        Get daily prediction metrics.
        
        Args:
            db: Database session
            days: Number of days to look back
            
        Returns:
            List of daily metrics
        """
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        # Query predictions by day
        daily_data = db.query(
            func.date(ModelPrediction.created_at).label('date'),
            func.count(ModelPrediction.id).label('count'),
            func.avg(ModelPrediction.churn_probability).label('avg_probability'),
        ).filter(ModelPrediction.created_at >= cutoff_date)\
        .group_by(func.date(ModelPrediction.created_at))\
        .order_by(func.date(ModelPrediction.created_at))\
        .all()
        
        return [
            {
                "date": str(d[0]),
                "prediction_count": d[1],
                "avg_churn_probability": float(d[2]) if d[2] else 0.0,
            }
            for d in daily_data
        ]
    
    @staticmethod
    def detect_drift(db: Session, baseline_days: int = 30, recent_days: int = 7):
        """
        Detect prediction drift by comparing recent predictions with baseline.
        
        Args:
            db: Database session
            baseline_days: Days to use for baseline
            recent_days: Recent days to compare
            
        Returns:
            Drift detection results
        """
        now = datetime.utcnow()
        baseline_start = now - timedelta(days=baseline_days + recent_days)
        baseline_cutoff = now - timedelta(days=recent_days)
        recent_start = now - timedelta(days=recent_days)
        
        # Get baseline predictions
        baseline_predictions = db.query(ModelPrediction.churn_probability)\
            .filter(
                (ModelPrediction.created_at >= baseline_start) &
                (ModelPrediction.created_at < baseline_cutoff)
            ).all()
        
        # Get recent predictions
        recent_predictions = db.query(ModelPrediction.churn_probability)\
            .filter(ModelPrediction.created_at >= recent_start)\
            .all()
        
        if not baseline_predictions or not recent_predictions:
            return {
                "drift_detected": False,
                "reason": "Insufficient data for comparison",
                "baseline_count": len(baseline_predictions),
                "recent_count": len(recent_predictions),
            }
        
        baseline_mean = sum(p[0] for p in baseline_predictions) / len(baseline_predictions)
        recent_mean = sum(p[0] for p in recent_predictions) / len(recent_predictions)
        
        # Calculate percentage change
        pct_change = abs((recent_mean - baseline_mean) / baseline_mean * 100) if baseline_mean > 0 else 0
        
        # Drift threshold: 10% change in mean
        drift_detected = pct_change > 10.0
        
        return {
            "drift_detected": drift_detected,
            "baseline_avg_probability": float(baseline_mean),
            "recent_avg_probability": float(recent_mean),
            "change_percentage": float(pct_change),
            "baseline_period_days": baseline_days,
            "recent_period_days": recent_days,
            "severity": "HIGH" if pct_change > 15 else "MEDIUM" if pct_change > 10 else "LOW",
        }
    
    @staticmethod
    def get_high_risk_users(db: Session, threshold: float = 0.7, limit: int = 50):
        """
        Get high-risk users for churn campaigns.
        
        Args:
            db: Database session
            threshold: Probability threshold for high-risk
            limit: Maximum users to return
            
        Returns:
            List of high-risk users
        """
        high_risk_predictions = db.query(ModelPrediction)\
            .filter(ModelPrediction.churn_probability >= threshold)\
            .order_by(ModelPrediction.churn_probability.desc())\
            .limit(limit)\
            .all()
        
        return [
            {
                "user_id": p.user_id,
                "churn_probability": float(p.churn_probability),
                "prediction": int(p.prediction),
                "predicted_at": p.created_at.isoformat(),
            }
            for p in high_risk_predictions
        ]
    
    @staticmethod
    def get_feature_statistics(db: Session, days: int = 30):
        """
        Get statistics on engineered features.
        
        Args:
            db: Database session
            days: Number of days to look back
            
        Returns:
            Feature statistics
        """
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        recent_features = db.query(FeatureStore)\
            .filter(FeatureStore.computed_at >= cutoff_date)\
            .all()
        
        if not recent_features:
            return {"total_records": 0, "statistics": {}}
        
        features_df = pd.DataFrame([
            {
                "tenure_days": f.tenure_days,
                "avg_sessions_14d": f.avg_sessions_14d,
                "avg_sessions_30d": f.avg_sessions_30d,
                "total_minutes_30d": f.total_minutes_30d,
                "failed_payments_30d": f.failed_payments_30d,
                "revenue_30d": f.revenue_30d,
            }
            for f in recent_features
        ])
        
        stats = {}
        for col in features_df.columns:
            stats[col] = {
                "mean": float(features_df[col].mean()),
                "median": float(features_df[col].median()),
                "std": float(features_df[col].std()),
                "min": float(features_df[col].min()),
                "max": float(features_df[col].max()),
            }
        
        return {
            "days_analyzed": days,
            "total_records": len(recent_features),
            "statistics": stats,
        }
