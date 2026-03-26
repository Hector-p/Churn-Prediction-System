"""
Retraining Pipeline - Complete ML workflow with model comparison and promotion
"""
import os
import json
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier

from app.database.db import SessionLocal
from app.services.feature_store_service import FeatureStoreService


MODEL_REGISTRY_PATH = "app/models"
CURRENT_MODEL_INFO_PATH = "app/models/current_model.json"


class RetryingModelComparer:
    """Handles model training, comparison, and promotion"""
    
    def __init__(self, force_promote: bool = False):
        """
        Initialize the model comparer.
        
        Args:
            force_promote: If True, always promote new model even if worse
        """
        self.force_promote = force_promote
        self.db = SessionLocal()
    
    def load_current_model_info(self):
        """Load current model metadata"""
        if os.path.exists(CURRENT_MODEL_INFO_PATH):
            with open(CURRENT_MODEL_INFO_PATH, 'r') as f:
                return json.load(f)
        return {"version": "v0", "baseline_metrics": {}}
    
    def save_model_info(self, version: str, metrics: dict, mlflow_run_id: str):
        """Save model metadata"""
        info = {
            "version": version,
            "promoted_at": datetime.utcnow().isoformat(),
            "baseline_metrics": metrics,
            "mlflow_run_id": mlflow_run_id,
        }
        
        os.makedirs(MODEL_REGISTRY_PATH, exist_ok=True)
        with open(CURRENT_MODEL_INFO_PATH, 'w') as f:
            json.dump(info, f, indent=2)
        
        return info
    
    def train_models(self, test_size: float = 0.2, random_state: int = 42):
        """
        Train both LogisticRegression and XGBoost models.
        
        Returns:
            Dictionary with training results
        """
        print("=" * 60)
        print("RETRAINING PIPELINE: Feature Materialization & Model Training")
        print("=" * 60)
        
        # Step 1: Materialize features
        print("\n[1/4] Materializing features from feature store...")
        try:
            feature_records = FeatureStoreService.compute_and_store_all_features(
                self.db, 
                feature_version="v2"
            )
            print(f"✓ Materialized {len(feature_records)} user features")
        except Exception as e:
            print(f"✗ Feature materialization failed: {e}")
            return {"success": False, "error": str(e)}
        
        # Step 2: Load features as DataFrame
        print("\n[2/4] Loading features...")
        try:
            features_df = FeatureStoreService.get_features_as_dataframe(self.db)
            print(f"✓ Loaded {len(features_df)} feature records")
        except Exception as e:
            print(f"✗ Failed to load features: {e}")
            return {"success": False, "error": str(e)}
        
        # Prepare data
        feature_columns = [
            "tenure_days",
            "avg_sessions_14d",
            "avg_sessions_30d",
            "total_minutes_30d",
            "failed_payments_30d",
            "revenue_30d",
        ]
        
        X = features_df[feature_columns]
        y = (features_df["churn_probability"] >= 0.5).astype(int)
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"✓ Train: {len(X_train)}, Test: {len(X_test)}")
        
        # Define preprocessor
        categorical_features = []
        numeric_features = feature_columns
        
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", "passthrough", numeric_features),
            ]
        )
        
        # Step 3: Train models with MLflow tracking
        print("\n[3/4] Training models...")
        
        mlflow.set_tracking_uri("file:./mlruns")
        mlflow.set_experiment("bank_churn_retraining")
        
        models_results = {}
        
        # Train LogisticRegression
        with mlflow.start_run(run_name="LogisticRegression_retrain"):
            lr_model = LogisticRegression(max_iter=1000)
            lr_pipeline = Pipeline([
                ("preprocessor", preprocessor),
                ("classifier", lr_model),
            ])
            
            lr_pipeline.fit(X_train, y_train)
            y_pred_lr = lr_pipeline.predict(X_test)
            
            lr_metrics = {
                "accuracy": accuracy_score(y_test, y_pred_lr),
                "precision": precision_score(y_test, y_pred_lr, zero_division=0),
                "recall": recall_score(y_test, y_pred_lr, zero_division=0),
                "f1": f1_score(y_test, y_pred_lr, zero_division=0),
            }
            
            mlflow.log_params({"model": "LogisticRegression", "max_iter": 1000})
            for metric_name, metric_value in lr_metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            mlflow.sklearn.log_model(lr_pipeline, artifact_path="model")
            
            lr_run_id = mlflow.active_run().info.run_id
            models_results["LogisticRegression"] = {
                "model": lr_pipeline,
                "metrics": lr_metrics,
                "run_id": lr_run_id,
            }
            
            print(f"✓ LogisticRegression - Accuracy: {lr_metrics['accuracy']:.4f}, "
                  f"F1: {lr_metrics['f1']:.4f}")
        
        # Train XGBoost
        with mlflow.start_run(run_name="XGBoost_retrain"):
            xgb_model = XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                eval_metric="logloss",
                verbose=0,
            )
            xgb_pipeline = Pipeline([
                ("preprocessor", preprocessor),
                ("classifier", xgb_model),
            ])
            
            xgb_pipeline.fit(X_train, y_train)
            y_pred_xgb = xgb_pipeline.predict(X_test)
            
            xgb_metrics = {
                "accuracy": accuracy_score(y_test, y_pred_xgb),
                "precision": precision_score(y_test, y_pred_xgb, zero_division=0),
                "recall": recall_score(y_test, y_pred_xgb, zero_division=0),
                "f1": f1_score(y_test, y_pred_xgb, zero_division=0),
            }
            
            mlflow.log_params({
                "model": "XGBoost",
                "n_estimators": 200,
                "max_depth": 6,
            })
            for metric_name, metric_value in xgb_metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            mlflow.sklearn.log_model(xgb_pipeline, artifact_path="model")
            
            xgb_run_id = mlflow.active_run().info.run_id
            models_results["XGBoost"] = {
                "model": xgb_pipeline,
                "metrics": xgb_metrics,
                "run_id": xgb_run_id,
            }
            
            print(f"✓ XGBoost - Accuracy: {xgb_metrics['accuracy']:.4f}, "
                  f"F1: {xgb_metrics['f1']:.4f}")
        
        # Step 4: Compare and promote best model
        print("\n[4/4] Model comparison & promotion...")
        
        # Select best model by F1 score
        best_model_name = max(
            models_results.keys(),
            key=lambda x: models_results[x]["metrics"]["f1"]
        )
        
        best_model = models_results[best_model_name]["model"]
        best_metrics = models_results[best_model_name]["metrics"]
        best_run_id = models_results[best_model_name]["run_id"]
        
        # Load current model info
        current_info = self.load_current_model_info()
        current_metrics = current_info.get("baseline_metrics", {})
        
        # Compare with current model
        f1_improvement = best_metrics["f1"] - current_metrics.get("f1", 0)
        precision_improvement = best_metrics["precision"] - current_metrics.get("precision", 0)
        
        print(f"\nBest Model: {best_model_name}")
        print(f"  F1 Score: {best_metrics['f1']:.4f} (Δ {f1_improvement:+.4f})")
        print(f"  Precision: {best_metrics['precision']:.4f} (Δ {precision_improvement:+.4f})")
        print(f"  Recall: {best_metrics['recall']:.4f}")
        
        # Promote if better or force_promote
        should_promote = (
            self.force_promote or 
            f1_improvement > 0.01  # Promote if F1 improves by >1%
        )
        
        if should_promote:
            # Determine version
            current_version = current_info.get("version", "v0")
            new_version = f"v{int(current_version[1:]) + 1}"
            
            # Save model
            model_path = f"{MODEL_REGISTRY_PATH}/churn_model.pkl"
            joblib.dump(best_model, model_path)
            
            # Save metadata
            self.save_model_info(new_version, best_metrics, best_run_id)
            
            print(f"\n✓ MODEL PROMOTED: {best_model_name} → {new_version}")
            print(f"  Saved to: {model_path}")
            
            return {
                "success": True,
                "promoted": True,
                "model_name": best_model_name,
                "version": new_version,
                "metrics": best_metrics,
                "improvement": {
                    "f1": f1_improvement,
                    "precision": precision_improvement,
                },
            }
        else:
            print(f"\n✗ MODEL NOT PROMOTED (F1 improvement: {f1_improvement:.4f} < 0.01)")
            return {
                "success": True,
                "promoted": False,
                "model_name": best_model_name,
                "metrics": best_metrics,
                "improvement": {
                    "f1": f1_improvement,
                    "precision": precision_improvement,
                },
            }
    
    def close(self):
        """Close database session"""
        if self.db:
            self.db.close()


def run_retraining_pipeline(force_promote: bool = False):
    """
    Execute the complete retraining pipeline.
    
    Args:
        force_promote: If True, always promote new model
        
    Returns:
        Pipeline results
    """
    comparer = RetryingModelComparer(force_promote=force_promote)
    try:
        return comparer.train_models()
    finally:
        comparer.close()


if __name__ == "__main__":
    import sys
    force_promote = "--force" in sys.argv
    result = run_retraining_pipeline(force_promote=force_promote)
    print("\n" + "=" * 60)
    print("PIPELINE RESULTS")
    print("=" * 60)
    print(json.dumps(result, indent=2, default=str))
