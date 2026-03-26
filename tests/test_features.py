from training.feature_pipeline import build_features

def test_feature_pipeline_runs():

    df = build_features()

    assert df is not None
    assert len(df) > 0

    required_columns = [
        "tenure_days",
        "avg_sessions_14d",
        "avg_sessions_30d",
        "total_minutes_30d",
        "failed_payments_30d",
        "revenue_30d",
    ]

    for col in required_columns:
        assert col in df.columns