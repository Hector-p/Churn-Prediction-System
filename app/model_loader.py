"""
Model Loader - Handles model loading and versioning
"""
import os
import json
import joblib


MODEL_REGISTRY_PATH = "app/models"
CURRENT_MODEL_INFO_PATH = "app/models/current_model.json"
DEFAULT_MODEL_PATH = "app/models/churn_model.pkl"


def get_current_model_info():
    """
    Get the current model metadata including version and run ID.
    
    Returns:
        Dictionary with model metadata
    """
    if os.path.exists(CURRENT_MODEL_INFO_PATH):
        try:
            with open(CURRENT_MODEL_INFO_PATH, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading model info: {e}")
    
    # Return default if not found
    return {
        "version": "v1",
        "baseline_metrics": {},
        "mlflow_run_id": None,
        "promoted_at": None,
    }


def get_current_model():
    """
    Load the current trained model.
    
    Returns:
        Loaded model object
    """
    if not os.path.exists(DEFAULT_MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {DEFAULT_MODEL_PATH}. "
            "Please train a model first using training/train_churn.py or "
            "training/retraining_pipeline.py"
        )
    
    return joblib.load(DEFAULT_MODEL_PATH)


def get_model_version():
    """
    Get the current model version.
    
    Returns:
        Version string (e.g., "v1", "v2")
    """
    info = get_current_model_info()
    return info.get("version", "v1")


def get_model_mlflow_run_id():
    """
    Get the MLflow run ID for the current model.
    
    Returns:
        MLflow run ID or None
    """
    info = get_current_model_info()
    return info.get("mlflow_run_id")


def get_model_metrics():
    """
    Get the baseline metrics for the current model.
    
    Returns:
        Dictionary of metrics
    """
    info = get_current_model_info()
    return info.get("baseline_metrics", {})
