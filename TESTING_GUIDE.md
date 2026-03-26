"""
COMPLETE ML PIPELINE TESTING & USAGE GUIDE

This guide covers all 5 components of your production ML pipeline.
"""

# ==============================================================================
# STEP 0: Database Setup (One-time)
# ==============================================================================

"""
Initialize database tables:
    python init_db.py

Verify tables created:
    - users
    - usage_logs
    - transactions
    - model_predictions
    - feature_store
"""

# ==============================================================================
# STEP 1: Populate Sample Data (Optional but Recommended)
# ==============================================================================

"""
Generate synthetic data:
    python data/synthetic/generate_data.py

Seed database:
    python app/database/seed.py
"""

# ==============================================================================
# STEP 2: Train Initial Model
# ==============================================================================

"""
Run training pipeline:
    python training/train_churn.py

Output:
    - Model saved to: app/models/churn_model.pkl
    - Runs logged to MLflow at: http://localhost:5000
"""

# ==============================================================================
# STEP 3: Start Services (in separate terminals)
# ==============================================================================

"""
Terminal 1 - Start FastAPI server:
    uvicorn app.main:app --reload --port 8000
    
Terminal 2 - Start MLflow UI:
    mlflow ui --backend-store-uri ./mlruns --port 5000
    
Terminal 3 - Start Streamlit Dashboard:
    streamlit run dashboard.py --server.port 8501

Then open:
    - API Docs: http://localhost:8000/docs
    - MLflow UI: http://localhost:5000
    - Dashboard: http://localhost:8501
"""

# ==============================================================================
# STEP 4: Feature Store Operations
# ==============================================================================

"""
Compute and store features for all users:
    POST http://localhost:8000/api/feature-store/compute-all

Compute features for single user:
    POST http://localhost:8000/api/feature-store/compute/123

Get features for user:
    GET http://localhost:8000/api/feature-store/features/123

Get all features:
    GET http://localhost:8000/api/feature-store/features

Clean up old features (>30 days):
    POST http://localhost:8000/api/feature-store/cleanup?days_to_keep=30
"""

# ==============================================================================
# STEP 5: Prediction Pipeline
# ==============================================================================

"""
Single user prediction:
    POST http://localhost:8000/churn/predict/123
    
Single user prediction with fresh features:
    POST http://localhost:8000/churn/predict/123?refresh_features=true

Batch predict all users:
    POST http://localhost:8000/churn/predict-all
    
Batch predict with fresh features:
    POST http://localhost:8000/churn/predict-all?refresh_features=true

Response example:
{
  "status": "success",
  "message": "Predicted 500 users successfully",
  "total_users": 500,
  "successful": 500,
  "failed": 0,
  "summary": {
    "high_risk": 45,
    "medium_risk": 120,
    "low_risk": 335
  }
}
"""

# ==============================================================================
# STEP 6: Monitoring & Drift Detection
# ==============================================================================

"""
Prediction summary (last 7 days):
    GET http://localhost:8000/monitoring/summary?days=7

Daily metrics (last 30 days):
    GET http://localhost:8000/monitoring/daily-metrics?days=30

Drift detection:
    GET http://localhost:8000/monitoring/drift-detection?baseline_days=30&recent_days=7

High-risk users for campaigns:
    GET http://localhost:8000/monitoring/high-risk-users?threshold=0.7&limit=50

Feature statistics:
    GET http://localhost:8000/monitoring/feature-statistics?days=30

Health check:
    GET http://localhost:8000/monitoring/health
"""

# ==============================================================================
# STEP 7: Retraining Pipeline
# ==============================================================================

"""
Run retraining (only promotes if beats current model):
    python training/retraining_pipeline.py

Run retraining with force promotion:
    python training/retraining_pipeline.py --force

The pipeline:
    1. Materializes fresh features from feature store
    2. Trains LogisticRegression + XGBoost models
    3. Compares via F1 score
    4. Automatically promotes best model if it improves by >1%
    5. Saves metadata with version and MLflow run ID
    6. Updates app/models/current_model.json

Expected output:
    ============================================================
    RETRAINING PIPELINE: Feature Materialization & Model Training
    ============================================================
    
    [1/4] Materializing features from feature store...
    ✓ Materialized 500 user features
    [2/4] Loading features...
    ✓ Loaded 500 feature records
    [3/4] Training models...
    ✓ LogisticRegression - Accuracy: 0.9800, F1: 0.8500
    ✓ XGBoost - Accuracy: 0.9775, F1: 0.8620
    [4/4] Model comparison & promotion...
    
    Best Model: XGBoost
      F1 Score: 0.8620 (Δ +0.0120)
    
    ✓ MODEL PROMOTED: XGBoost → v2
"""

# ==============================================================================
# STEP 8: Complete Workflow Example
# ==============================================================================

"""
Full production workflow:

1. Schedule feature materialization (e.g., hourly):
   curl -X POST "http://localhost:8000/api/feature-store/compute-all"

2. Schedule batch predictions (e.g., daily):
   curl -X POST "http://localhost:8000/churn/predict-all?refresh_features=true"

3. Monitor drift (e.g., daily):
   curl -X GET "http://localhost:8000/monitoring/drift-detection"
   
   If drift detected (>10% change):
   curl -X POST "http://localhost:8000/churn/predict-all?refresh_features=true"
   python training/retraining_pipeline.py

4. Schedule weekly retraining:
   python training/retraining_pipeline.py

5. View results in Streamlit Dashboard:
   - Model version and metrics
   - Drift status
   - Churn distribution
   - Risk categories
   - Feature statistics
   - Prediction logs
"""

# ==============================================================================
# STEP 9: Using the CLI Script (Optional)
# ==============================================================================

"""
Create run_feature_store.py and run:

Compute all features:
    python run_feature_store.py compute-all

Compute user features:
    python run_feature_store.py compute-user --user-id 5

Clean up old features:
    python run_feature_store.py cleanup --days 30

Export features to CSV:
    python run_feature_store.py export --output my_features.csv
"""

# ==============================================================================
# TROUBLESHOOTING
# ==============================================================================

"""
Error: "relation feature_store does not exist"
    Solution: Run init_db.py to create tables
    python init_db.py

Error: "ModuleNotFoundError" for streamlit
    Solution: Install requirements
    pip install -r requirements.txt

Error: "Failed to load dashboard data"
    Solution: Check PostgreSQL is running and DATABASE_URL is set
    echo $DATABASE_URL  # verify env var

Model not moving to feature store:
    Solution: First populate feature store
    POST http://localhost:8000/api/feature-store/compute-all

No drift detected (always False):
    Solution: Need at least 30+ days of prediction data
    Run predictions for a while first
"""

# ==============================================================================
# KEY ENDPOINTS SUMMARY
# ==============================================================================

"""
FEATURE STORE:
  POST   /api/feature-store/compute-all                 Compute all features
  POST   /api/feature-store/compute/{user_id}          Compute single user
  GET    /api/feature-store/features/{user_id}         Get user features
  GET    /api/feature-store/features                    Get all features
  POST   /api/feature-store/cleanup                     Delete old features

PREDICTIONS:
  POST   /churn/predict/{user_id}                       Single prediction
  POST   /churn/predict-all                             Batch prediction

MONITORING:
  GET    /monitoring/summary                            7-day summary
  GET    /monitoring/daily-metrics                      Daily metrics
  GET    /monitoring/drift-detection                    Drift analysis
  GET    /monitoring/high-risk-users                    Campaign targets
  GET    /monitoring/feature-statistics                 Feature stats
  GET    /monitoring/health                             Health check

LEGACY:
  GET    /monitoring/predictions                        Raw prediction logs
"""
