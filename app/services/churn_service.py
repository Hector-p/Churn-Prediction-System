import joblib
import pandas as pd
from sqlalchemy.orm import Session
from app.database.models import User, ModelPrediction
from app.services.feature_store_service import FeatureStoreService
from app.model_loader import get_current_model, get_model_version, get_model_mlflow_run_id


MODEL_FEATURE_COLUMNS = [
    "tenure_days",
    "avg_sessions_14d",
    "avg_sessions_30d",
    "total_minutes_30d",
    "failed_payments_30d",
    "revenue_30d",
    "subscription_plan",
]


def _build_model_input(features_dict: dict) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "tenure_days": features_dict["tenure_days"],
                "avg_sessions_14d": features_dict["avg_sessions_14d"],
                "avg_sessions_30d": features_dict["avg_sessions_30d"],
                "total_minutes_30d": features_dict["total_minutes_30d"],
                "failed_payments_30d": features_dict["failed_payments_30d"],
                "revenue_30d": features_dict["revenue_30d"],
                "subscription_plan": features_dict["subscription_plan"],
            }
        ],
        columns=MODEL_FEATURE_COLUMNS,
    )


def predict_user_churn(user_id: int, db: Session, refresh_features: bool = False):
    """
    Predict churn for a single user using feature store.
    
    Args:
        user_id: User ID
        db: Database session
        refresh_features: If True, compute and store fresh features before prediction
        
    Returns:
        Prediction result dict or None
    """
    user = db.query(User).filter(User.id == user_id).first()

    if not user:
        return None

    # Refresh features if requested
    if refresh_features:
        FeatureStoreService.compute_and_store_single_user_features(user_id, db)

    # Get features from feature store
    features_dict = FeatureStoreService.get_features(user_id, db)

    if not features_dict:
        FeatureStoreService.compute_and_store_single_user_features(user_id, db)
        features_dict = FeatureStoreService.get_features(user_id, db)

    if not features_dict:
        return None

    model_input = _build_model_input(features_dict)
    
    # Load model and get current version
    model = get_current_model()
    model_version = get_model_version()
    mlflow_run_id = get_model_mlflow_run_id()
    
    # Make prediction
    prediction = model.predict(model_input)[0]
    probability = model.predict_proba(model_input)[0][1]

    # Update user table with latest prediction
    user.churn_probability = float(probability)

    # Log prediction event
    prediction_log = ModelPrediction(
        user_id=user.id,
        prediction=int(prediction),
        churn_probability=float(probability),
        model_version=model_version,
    )

    db.add(prediction_log)
    db.commit()
    db.refresh(user)

    return {
        "user_id": user.id,
        "prediction": int(prediction),
        "churn_probability": float(probability),
        "subscription_plan": user.subscription_plan,
        "tenure_days": features_dict['tenure_days'],
        "model_version": model_version,
        "mlflow_run_id": mlflow_run_id,
        "features_computed_at": features_dict['computed_at'].isoformat() if features_dict['computed_at'] else None,
    }


def predict_all_users_churn(db: Session, refresh_features: bool = False):
    """
    Predict churn for all users in batch.
    
    Args:
        db: Database session
        refresh_features: If True, compute fresh features for all users before prediction
        
    Returns:
        Dictionary with results summary
    """
    # Refresh all features if requested
    if refresh_features:
        FeatureStoreService.compute_and_store_all_features(db)
    
    # Get all users
    users = db.query(User).all()
    
    if not users:
        return {"total": 0, "successful": 0, "failed": 0, "predictions": []}
    
    # Get all latest features
    features_list = FeatureStoreService.get_all_features(db)
    features_dict_map = {f['user_id']: f for f in features_list}
    
    # Load model once and get version
    model = get_current_model()
    model_version = get_model_version()
    
    predictions = []
    successful_count = 0
    failed_count = 0
    
    # Prepare batch prediction data
    for user in users:
        features_dict = features_dict_map.get(user.id)
        
        if not features_dict:
            failed_count += 1
            continue
        
        try:
            model_input = _build_model_input(features_dict)
            
            # Make prediction
            prediction = model.predict(model_input)[0]
            probability = model.predict_proba(model_input)[0][1]
            
            # Update user churn probability
            user.churn_probability = float(probability)
            
            # Create prediction log
            prediction_log = ModelPrediction(
                user_id=user.id,
                prediction=int(prediction),
                churn_probability=float(probability),
                model_version=model_version,
            )
            
            db.add(prediction_log)
            
            predictions.append({
                "user_id": user.id,
                "prediction": int(prediction),
                "churn_probability": float(probability),
                "subscription_plan": user.subscription_plan,
            })
            
            successful_count += 1
            
        except Exception as e:
            failed_count += 1
            print(f"Error predicting user {user.id}: {str(e)}")
            continue
    
    # Commit all changes
    db.commit()
    
    return {
        "total": len(users),
        "successful": successful_count,
        "failed": failed_count,
        "predictions": predictions,
    }
