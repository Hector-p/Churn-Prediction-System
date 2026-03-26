import os
import joblib
import mlflow
import mlflow.sklearn


mlflow.set_tracking_uri("file:./mlruns")

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from xgboost import XGBClassifier

from training.feature_pipeline import build_features


def run_experiment(model, model_name, X_train, X_test, y_train, y_test, preprocessor):
    with mlflow.start_run(run_name=model_name):
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", model),
            ]
        )

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        print(f"\nModel: {model_name}")
        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1 Score:", f1)

        mlflow.log_param("model_type", model_name)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        mlflow.sklearn.log_model(pipeline, artifact_path="model")

        return pipeline, accuracy


def train_models():
    mlflow.set_experiment("bank_churn_experiments")

    df = build_features()
    df["churn_label"] = (df["churn_probability"] >= 0.5).astype(int)

    feature_columns = [
        "tenure_days",
        "avg_sessions_14d",
        "avg_sessions_30d",
        "total_minutes_30d",
        "failed_payments_30d",
        "revenue_30d",
        "subscription_plan",
    ]

    X = df[feature_columns]
    y = df["churn_label"]

    categorical_features = ["subscription_plan"]
    numeric_features = [
        "tenure_days",
        "avg_sessions_14d",
        "avg_sessions_30d",
        "total_minutes_30d",
        "failed_payments_30d",
        "revenue_30d",
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("num", "passthrough", numeric_features),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    lr_model = LogisticRegression(max_iter=1000)
    lr_pipeline, lr_score = run_experiment(
        lr_model,
        "LogisticRegression",
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor,
    )

    xgb_model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        eval_metric="logloss",
    )
    xgb_pipeline, xgb_score = run_experiment(
        xgb_model,
        "XGBoost",
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor,
    )

    if xgb_score > lr_score:
        best_model = xgb_pipeline
        best_name = "XGBoost"
    else:
        best_model = lr_pipeline
        best_name = "LogisticRegression"

    os.makedirs("app/models", exist_ok=True)
    joblib.dump(best_model, "app/models/churn_model.pkl")
    print(f"\nBest model saved: {best_name}")


if __name__ == "__main__":
    train_models()