from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.database.db import get_db
from app.services.churn_service import predict_user_churn, predict_all_users_churn

router = APIRouter(prefix="/churn", tags=["Churn Prediction"])


@router.post("/predict/{user_id}")
def predict_churn(user_id: int, refresh_features: bool = False, db: Session = Depends(get_db)):
    """
    Predict churn for a single user.
    
    Args:
        user_id: User ID
        refresh_features: If True, compute fresh features before prediction
        db: Database session
        
    Returns:
        Prediction result with probability and features
    """
    result = predict_user_churn(user_id, db, refresh_features=refresh_features)

    if result is None:
        raise HTTPException(status_code=404, detail="User not found or features unavailable")

    return result


@router.post("/predict-all")
def predict_all_users(refresh_features: bool = False, db: Session = Depends(get_db)):
    """
    Predict churn for all users in batch.
    
    Args:
        refresh_features: If True, compute fresh features for all users before prediction
        db: Database session
        
    Returns:
        Batch prediction results
    """
    try:
        result = predict_all_users_churn(db, refresh_features=refresh_features)
        return {
            "status": "success",
            "message": f"Predicted {result['successful']} users successfully",
            "total_users": result['total'],
            "successful": result['successful'],
            "failed": result['failed'],
            "summary": {
                "high_risk": sum(1 for p in result['predictions'] if p['churn_probability'] >= 0.7),
                "medium_risk": sum(1 for p in result['predictions'] if 0.4 <= p['churn_probability'] < 0.7),
                "low_risk": sum(1 for p in result['predictions'] if p['churn_probability'] < 0.4),
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))