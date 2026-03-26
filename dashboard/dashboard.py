import os
import sys


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from datetime import datetime, timezone

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

from app.database.db import SessionLocal
from app.model_loader import get_current_model_info
from app.services.monitoring_service import MonitoringService
from app.services.roi_service import simulate_roi

load_dotenv()

st.set_page_config(
    page_title="Bank Churn Intelligence Dashboard",
    page_icon="",
    layout="wide",
)

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    st.error("DATABASE_URL is not set in your environment.")
    st.stop()

if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

engine = create_engine(DATABASE_URL, pool_pre_ping=True)

st.markdown(
    """
<style>
html, body, [class*="st-"] { font-family: 'Inter', sans-serif; }

header[data-testid="stHeader"] {
    background: linear-gradient(90deg, #0a1628, #132743, #1a3a5c) !important;
}

div[data-testid="stMetric"] {
    background: rgba(255, 255, 255, 0.05);
    border-left: 4px solid #60a5fa;
    border-radius: 10px;
    padding: 14px 18px;
    box-shadow: 0 2px 8px rgba(0,0,0,.25);
}
div[data-testid="stMetric"] label {
    font-weight: 600 !important;
    color: #94a3b8 !important;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    font-weight: 700 !important;
    color: #f1f5f9 !important;
}
div[data-testid="stMetric"] [data-testid="stMetricDelta"] {
    color: #94a3b8 !important;
}

button[data-baseweb="tab"] {
    color: #94a3b8 !important;
    font-weight: 500 !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #60a5fa !important;
    font-weight: 700 !important;
}

.stDataFrame { border-radius: 10px; overflow: hidden; }

.roi-positive {
    text-align: center;
    background: rgba(16, 185, 129, 0.12);
    border: 1px solid rgba(52, 211, 153, 0.30);
    border-radius: 12px;
    padding: 20px 10px;
}
.roi-negative {
    text-align: center;
    background: rgba(239, 68, 68, 0.12);
    border: 1px solid rgba(248, 113, 113, 0.30);
    border-radius: 12px;
    padding: 20px 10px;
}
.roi-positive .val { font-size: 1.5rem; font-weight: 700; color: #6ee7b7; }
.roi-negative .val { font-size: 1.5rem; font-weight: 700; color: #fca5a5; }
.roi-positive .lbl, .roi-negative .lbl {
    font-size: 0.82rem; color: #cbd5e1; margin-top: 6px;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f1b2d 0%, #162236 100%) !important;
}

button[kind="primary"] {
    background: linear-gradient(135deg, #3b82f6, #2563eb) !important;
    border: none !important;
    color: #fff !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
}
</style>
""",
    unsafe_allow_html=True,
)

@st.cache_data(ttl=60)
def load_users() -> pd.DataFrame:
    return pd.read_sql("SELECT * FROM users", engine)

@st.cache_data(ttl=60)
def load_feature_store() -> pd.DataFrame:
    return pd.read_sql("SELECT * FROM feature_store ORDER BY computed_at DESC", engine)

@st.cache_data(ttl=60)
def load_prediction_logs() -> pd.DataFrame:
    return pd.read_sql("SELECT * FROM model_predictions ORDER BY created_at DESC", engine)

@st.cache_data(ttl=60)
def load_plan_summary() -> pd.DataFrame:
    query = """
        SELECT
            subscription_plan,
            COUNT(*) AS user_count,
            AVG(churn_probability) AS avg_churn_probability,
            AVG(monthly_spend) AS avg_monthly_spend
        FROM users
        GROUP BY subscription_plan
        ORDER BY subscription_plan
    """
    return pd.read_sql(query, engine)

@st.cache_data(ttl=60)
def load_high_risk_users(limit: int = 20) -> pd.DataFrame:
    query = text(
        """
        SELECT id, name, email, subscription_plan, monthly_spend,
               churn_probability, region, device_type
        FROM users
        ORDER BY churn_probability DESC, monthly_spend DESC
        LIMIT :limit_value
    """
    )
    with engine.connect() as conn:
        return pd.read_sql(query, conn, params={"limit_value": limit})

@st.cache_data(ttl=60)
def load_revenue_at_risk() -> float:
    query = """
        SELECT COALESCE(SUM(monthly_spend), 0) AS revenue_at_risk
        FROM users WHERE churn_probability >= 0.5
    """
    with engine.connect() as conn:
        result = conn.execute(text(query)).scalar()
    return float(result or 0.0)

@st.cache_data(ttl=60)
def load_revenue_at_risk_by_plan(threshold: float = 0.5) -> pd.DataFrame:
    query = text(
        """
        SELECT subscription_plan,
               COUNT(*) AS at_risk_users,
               COALESCE(SUM(monthly_spend), 0) AS revenue_at_risk,
               AVG(churn_probability) AS avg_churn_probability
        FROM users
        WHERE churn_probability >= :threshold
        GROUP BY subscription_plan
        ORDER BY revenue_at_risk DESC
    """
    )
    with engine.connect() as conn:
        return pd.read_sql(query, conn, params={"threshold": threshold})

@st.cache_data(ttl=60)
def load_monitoring_daily() -> pd.DataFrame:
    query = """
        SELECT DATE(created_at) AS prediction_day,
               COUNT(*) AS prediction_count,
               AVG(churn_probability) AS avg_churn_probability
        FROM model_predictions
        GROUP BY DATE(created_at)
        ORDER BY prediction_day ASC
    """
    return pd.read_sql(query, engine)

@st.cache_data(ttl=60)
def get_drift_analysis():
    db = SessionLocal()
    try:
        return MonitoringService.detect_drift(db, baseline_days=30, recent_days=7)
    except Exception as e:
        return {"drift_detected": False, "error": str(e)}
    finally:
        db.close()

@st.cache_data(ttl=60)
def get_model_version_info():
    try:
        return get_current_model_info()
    except Exception as e:
        return {"version": "unknown", "error": str(e)}

@st.cache_data(ttl=60)
def get_feature_stats():
    db = SessionLocal()
    try:
        return MonitoringService.get_feature_statistics(db, days=30)
    except Exception as e:
        return {"error": str(e)}
    finally:
        db.close()

def build_risk_segments(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "churn_probability" not in df.columns:
        return pd.DataFrame(columns=["risk_band", "count"])

    segmented = df.copy()
    segmented["risk_band"] = pd.cut(
        segmented["churn_probability"],
        bins=[-0.01, 0.30, 0.60, 1.00],
        labels=["Low Risk", "Medium Risk", "High Risk"],
    )

    return (
        segmented["risk_band"]
        .value_counts(sort=False)
        .rename_axis("risk_band")
        .reset_index(name="count")
    )

def build_prediction_label_counts(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    if df.empty or "churn_probability" not in df.columns:
        return pd.DataFrame(columns=["prediction_label", "count"])

    pred_df = df.copy()
    pred_df["prediction_label"] = pred_df["churn_probability"].apply(
        lambda x: "Churn" if x >= threshold else "Non-Churn"
    )

    return (
        pred_df["prediction_label"]
        .value_counts()
        .rename_axis("prediction_label")
        .reset_index(name="count")
    )

def build_drift_comparison_df(drift_info: dict) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "period": ["Baseline (30d)", "Recent (7d)"],
            "avg_probability": [
                drift_info.get("baseline_avg_probability", 0),
                drift_info.get("recent_avg_probability", 0),
            ],
        }
    )

def make_time_features(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "prediction_day" not in df.columns:
        return df

    temp = df.copy()
    temp["prediction_day"] = pd.to_datetime(temp["prediction_day"])
    temp = temp.sort_values("prediction_day")

    if "prediction_count" in temp.columns:
        temp["prediction_count_rolling_3"] = temp["prediction_count"].rolling(3, min_periods=1).mean()

    if "avg_churn_probability" in temp.columns:
        temp["avg_churn_probability_rolling_3"] = temp["avg_churn_probability"].rolling(3, min_periods=1).mean()

    return temp

def format_naira(value: float) -> str:
    return f"N{value:,.2f}"

st.title("Bank Churn Intelligence Dashboard")
st.caption("Business analytics and ML monitoring for the churn prediction platform")

try:
    users_df = load_users()
    features_df = load_feature_store()
    logs_df = load_prediction_logs()
    plan_df = load_plan_summary()
    load_high_risk_users()
    revenue_at_risk = load_revenue_at_risk()
    monitoring_daily_df = load_monitoring_daily()
    drift_info = get_drift_analysis()
    model_info = get_model_version_info()
    feature_stats = get_feature_stats()
except Exception as exc:
    st.error(f"Failed to load dashboard data: {exc}")
    st.stop()

if users_df.empty:
    st.warning("No users found in the database yet.")
    st.stop()

if "created_at" in logs_df.columns:
    logs_df["created_at"] = pd.to_datetime(logs_df["created_at"], errors="coerce")

monitoring_daily_df = make_time_features(monitoring_daily_df)

st.sidebar.header("Filters")

selected_plan = st.sidebar.selectbox(
    "Subscription Plan",
    options=["All"] + sorted(users_df["subscription_plan"].dropna().unique().tolist()),
)

region_options = sorted(users_df["region"].dropna().unique().tolist()) if "region" in users_df.columns else []
selected_regions = st.sidebar.multiselect("Region", options=region_options, default=region_options)

device_options = (
    sorted(users_df["device_type"].dropna().unique().tolist())
    if "device_type" in users_df.columns
    else []
)
selected_devices = st.sidebar.multiselect("Device Type", options=device_options, default=device_options)

risk_threshold = st.sidebar.slider(
    "High Risk Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.05,
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "This dashboard provides real-time insights into customer churn risk, "
    "revenue impact, and ROI simulation for retention campaigns."
)

filtered_users = users_df.copy()

if selected_plan != "All":
    filtered_users = filtered_users[filtered_users["subscription_plan"] == selected_plan]

if selected_regions and "region" in filtered_users.columns:
    filtered_users = filtered_users[filtered_users["region"].isin(selected_regions)]

if selected_devices and "device_type" in filtered_users.columns:
    filtered_users = filtered_users[filtered_users["device_type"].isin(selected_devices)]

high_risk_users = filtered_users[filtered_users["churn_probability"] >= risk_threshold]

total_users = int(len(filtered_users))
high_risk_count = int(len(high_risk_users))
average_churn_probability = float(filtered_users["churn_probability"].mean()) if not filtered_users.empty else 0.0
avg_churn_probability_delta = average_churn_probability - float(users_df["churn_probability"].mean())
prediction_counts_df = build_prediction_label_counts(filtered_users, threshold=risk_threshold)
risk_segment_df = build_risk_segments(filtered_users)
drift_compare_df = build_drift_comparison_df(drift_info)

st.markdown("---")
st.subheader("Key Performance Indicators")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Users", f"{total_users:,}")
col2.metric("High-Risk Users", f"{high_risk_count:,}")
col3.metric("Avg Churn Probability", f"{average_churn_probability:.2%}", f"{avg_churn_probability_delta:+.2%}")
col4.metric("Monthly Revenue at Risk", format_naira(revenue_at_risk))

st.markdown("---")
st.subheader("Model Status and Monitoring")

model_col1, model_col2, model_col3 = st.columns(3)

with model_col1:
    model_version = model_info.get("version", "unknown")
    st.metric("Current Model Version", model_version)
    if model_info.get("promoted_at"):
        st.caption(f"Promoted: {str(model_info['promoted_at'])[:10]}")

with model_col2:
    metrics = model_info.get("baseline_metrics", {})
    model_f1 = metrics.get("f1", 0)
    accuracy = metrics.get("accuracy", 0)
    st.metric("Model F1 Score", f"{model_f1:.4f}")
    st.caption(f"Accuracy: {accuracy:.2%}")

with model_col3:
    if drift_info.get("drift_detected"):
        st.warning("DRIFT DETECTED")
        st.caption(f"Severity: {drift_info.get('severity', 'UNKNOWN')}")
    else:
        st.success("No Drift Detected")
    change_pct = drift_info.get("change_percentage", 0)
    st.caption(f"Change: {change_pct:.1f}%")

st.markdown("---")
st.subheader("At-Risk Revenue by Plan")

revenue_by_plan_df = load_revenue_at_risk_by_plan(threshold=risk_threshold)

if not revenue_by_plan_df.empty:
    plan_rev_col1, plan_rev_col2 = st.columns([2, 3])

    with plan_rev_col1:
        for _, row in revenue_by_plan_df.iterrows():
            st.metric(
                label=f"{row['subscription_plan']} - At-Risk Revenue",
                value=format_naira(float(row["revenue_at_risk"])),
                delta=f"{int(row['at_risk_users'])} users | {row['avg_churn_probability']:.0%} avg risk",
                delta_color="inverse",
            )

    with plan_rev_col2:
        plan_rev_fig = px.bar(
            revenue_by_plan_df,
            x="subscription_plan",
            y="revenue_at_risk",
            text="revenue_at_risk",
            title="Revenue at Risk by Subscription Plan",
        )
        plan_rev_fig.update_traces(texttemplate="N%{y:,.0f}", textposition="outside")
        plan_rev_fig.update_layout(
            xaxis_title="Subscription Plan",
            yaxis_title="Revenue at Risk",
            showlegend=False,
        )
        st.plotly_chart(plan_rev_fig, use_container_width=True)
else:
    st.info("No at-risk users found for the current threshold.")

st.markdown("---")
st.subheader("ROI Simulation")
st.caption("Estimate the return on investment for churn-prevention campaigns")

roi_col1, roi_col2 = st.columns([1, 2])

with roi_col1:
    st.markdown("**Intervention Parameters**")
    intervention_cost = st.number_input("Intervention cost per user (N)", min_value=0.0, value=500.0, step=50.0)
    retention_rate = st.slider("Expected retention rate", 0.0, 1.0, 0.30, 0.05)
    avg_ltv = st.number_input("Average LTV per user (N)", min_value=0.0, value=5000.0, step=100.0)
    run_roi = st.button("Simulate ROI", type="primary")

with roi_col2:
    if run_roi:
        if high_risk_users.empty:
            st.warning("No high-risk users match the current filters.")
        else:
            with st.spinner("Running ROI simulation..."):
                db = SessionLocal()
                try:
                    roi_result = simulate_roi(
                        db=db,
                        intervention_cost_per_user=intervention_cost,
                        expected_retention_rate=retention_rate,
                        avg_ltv=avg_ltv,
                        high_risk_users=high_risk_users["id"].tolist(),
                    )
                finally:
                    db.close()

            r1, r2, r3 = st.columns(3)
            positive = roi_result["net_roi"] >= 0
            cls = "roi-positive" if positive else "roi-negative"

            with r1:
                st.markdown(
                    f'<div class="{cls}"><div class="val">{format_naira(roi_result["saved_revenue"])}</div>'
                    f'<div class="lbl">Saved Revenue</div></div>',
                    unsafe_allow_html=True,
                )
            with r2:
                st.markdown(
                    f'<div class="{cls}"><div class="val">{format_naira(roi_result["net_roi"])}</div>'
                    f'<div class="lbl">Net ROI</div></div>',
                    unsafe_allow_html=True,
                )
            with r3:
                st.markdown(
                    f'<div class="{cls}"><div class="val">{roi_result["roi_pct"]:.1f}%</div>'
                    f'<div class="lbl">ROI Percentage</div></div>',
                    unsafe_allow_html=True,
                )

            st.write("")
            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Targeted Users", f"{roi_result['total_targeted_users']:,}")
            s2.metric("Expected Saved", f"{roi_result['expected_saved_users']:,.0f}")
            s3.metric("Campaign Cost", format_naira(roi_result["total_intervention_cost"]))
            s4.metric("Cost / Saved User", format_naira(roi_result["cost_per_saved_user"]))

            breakdown = roi_result.get("plan_breakdown", {})
            if breakdown:
                st.write("")
                st.markdown("**Per-Plan Breakdown**")
                bd_rows = []
                for plan, data in breakdown.items():
                    bd_rows.append(
                        {
                            "Plan": plan,
                            "Users": data["user_count"],
                            "Monthly Spend": format_naira(data["monthly_spend"]),
                            "Avg Churn": f"{data['avg_churn_probability']:.2%}",
                            "Est. Saved": data["estimated_saved"],
                            "Saved Revenue": format_naira(data["plan_saved_revenue"]),
                            "Cost": format_naira(data["plan_cost"]),
                            "Net ROI": format_naira(data["plan_net_roi"]),
                        }
                    )
                st.dataframe(pd.DataFrame(bd_rows), use_container_width=True, hide_index=True)
    else:
        st.info(
            "Configure the intervention parameters on the left and click "
            "**Simulate ROI** to see the projected return on your retention campaign."
        )

st.markdown("---")
st.subheader("Distribution Charts")

dist_col1, dist_col2 = st.columns(2)

with dist_col1:
    st.markdown("**Churn Probability Distribution**")
    hist_fig = px.histogram(
        filtered_users,
        x="churn_probability",
        nbins=20,
        title="Distribution of Churn Probabilities",
    )
    hist_fig.update_layout(
        xaxis_title="Churn Probability",
        yaxis_title="User Count",
        bargap=0.08,
    )
    st.plotly_chart(hist_fig, use_container_width=True)

with dist_col2:
    st.markdown("**Users by Subscription Plan**")
    if not filtered_users.empty and "subscription_plan" in filtered_users.columns:
        plan_counts = (
            filtered_users["subscription_plan"]
            .value_counts()
            .rename_axis("subscription_plan")
            .reset_index(name="user_count")
            .sort_values("user_count", ascending=False)
        )
        plan_fig = px.bar(
            plan_counts,
            x="subscription_plan",
            y="user_count",
            text="user_count",
            title="Users per Subscription Plan",
        )
        plan_fig.update_traces(textposition="outside")
        plan_fig.update_layout(
            xaxis_title="Subscription Plan",
            yaxis_title="Users",
            showlegend=False,
        )
        st.plotly_chart(plan_fig, use_container_width=True)
    else:
        st.info("No plan summary available.")

st.markdown("---")
st.subheader("Risk Overview")

risk_col1, risk_col2 = st.columns(2)

with risk_col1:
    st.markdown("**Risk Segment Breakdown**")
    if not risk_segment_df.empty:
        donut_fig = px.pie(
            risk_segment_df,
            names="risk_band",
            values="count",
            hole=0.55,
            title="Users by Risk Segment",
        )
        donut_fig.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(donut_fig, use_container_width=True)
    else:
        st.info("No risk segment data available.")

with risk_col2:
    st.markdown("**Predicted Churn vs Non-Churn**")
    if not prediction_counts_df.empty:
        pred_fig = px.bar(
            prediction_counts_df,
            x="prediction_label",
            y="count",
            text="count",
            title=f"Prediction Split at Threshold {risk_threshold:.2f}",
        )
        pred_fig.update_traces(textposition="outside")
        pred_fig.update_layout(
            xaxis_title="Prediction Label",
            yaxis_title="Count",
            showlegend=False,
        )
        st.plotly_chart(pred_fig, use_container_width=True)
    else:
        st.info("Prediction split unavailable.")

st.markdown("---")
st.subheader("Monitoring")

tab_pred, tab_drift, tab_feat = st.tabs(["Prediction Trends", "Drift Detection", "Feature Statistics"])

with tab_pred:
    mon_col1, mon_col2 = st.columns(2)

    with mon_col1:
        st.markdown("**Daily Prediction Count**")
        if not monitoring_daily_df.empty:
            count_fig = go.Figure()
            count_fig.add_trace(
                go.Scatter(
                    x=monitoring_daily_df["prediction_day"],
                    y=monitoring_daily_df["prediction_count"],
                    mode="lines+markers",
                    name="Daily Count",
                )
            )
            if "prediction_count_rolling_3" in monitoring_daily_df.columns:
                count_fig.add_trace(
                    go.Scatter(
                        x=monitoring_daily_df["prediction_day"],
                        y=monitoring_daily_df["prediction_count_rolling_3"],
                        mode="lines",
                        name="3-Day Rolling Avg",
                        line=dict(dash="dash"),
                    )
                )
            count_fig.update_layout(
                xaxis_title="Day",
                yaxis_title="Prediction Count",
                legend_title="Series",
            )
            st.plotly_chart(count_fig, use_container_width=True)
        else:
            st.info("No prediction logs yet.")

    with mon_col2:
        st.markdown("**Average Churn Probability Trend**")
        if not monitoring_daily_df.empty:
            prob_fig = go.Figure()
            prob_fig.add_trace(
                go.Scatter(
                    x=monitoring_daily_df["prediction_day"],
                    y=monitoring_daily_df["avg_churn_probability"],
                    mode="lines+markers",
                    name="Daily Avg",
                )
            )
            if "avg_churn_probability_rolling_3" in monitoring_daily_df.columns:
                prob_fig.add_trace(
                    go.Scatter(
                        x=monitoring_daily_df["prediction_day"],
                        y=monitoring_daily_df["avg_churn_probability_rolling_3"],
                        mode="lines",
                        name="3-Day Rolling Avg",
                        line=dict(dash="dash"),
                    )
                )
            prob_fig.update_layout(
                xaxis_title="Day",
                yaxis_title="Average Churn Probability",
                legend_title="Series",
            )
            st.plotly_chart(prob_fig, use_container_width=True)
        else:
            st.info("No monitoring averages available yet.")

with tab_drift:
    drift_col1, drift_col2 = st.columns([1, 2])

    with drift_col1:
        baseline_avg = drift_info.get("baseline_avg_probability", 0)
        recent_avg = drift_info.get("recent_avg_probability", 0)
        change = drift_info.get("change_percentage", 0)

        st.metric("Baseline Avg (30d)", f"{baseline_avg:.4f}")
        st.metric("Recent Avg (7d)", f"{recent_avg:.4f}")
        st.metric("Change %", f"{change:.2f}%")

        if drift_info.get("error"):
            st.caption(f"Drift note: {drift_info['error']}")

    with drift_col2:
        st.markdown("**Baseline vs Recent Drift Comparison**")
        drift_fig = px.bar(
            drift_compare_df,
            x="period",
            y="avg_probability",
            text="avg_probability",
            title="Average Churn Probability by Time Window",
        )
        drift_fig.update_traces(texttemplate="%{y:.4f}", textposition="outside")
        drift_fig.update_layout(
            xaxis_title="Time Window",
            yaxis_title="Average Churn Probability",
            showlegend=False,
        )
        st.plotly_chart(drift_fig, use_container_width=True)

with tab_feat:
    if feature_stats and "statistics" in feature_stats:
        stats_data = feature_stats["statistics"]
        stats_display = []
        for feature_name, fmetrics in stats_data.items():
            stats_display.append(
                {
                    "Feature": feature_name,
                    "Mean": f"{fmetrics['mean']:.2f}",
                    "Median": f"{fmetrics['median']:.2f}",
                    "Std Dev": f"{fmetrics['std']:.2f}",
                    "Min": f"{fmetrics['min']:.2f}",
                    "Max": f"{fmetrics['max']:.2f}",
                }
            )
        st.dataframe(pd.DataFrame(stats_display), use_container_width=True, hide_index=True)
    else:
        st.info("Feature statistics not available yet.")

st.markdown("---")
st.subheader("Feature Explorer")

numeric_feature_candidates = [
    c
    for c in ["tenure", "monthly_spend", "churn_probability", "balance", "age", "estimated_salary"]
    if c in filtered_users.columns and pd.api.types.is_numeric_dtype(filtered_users[c])
]

if numeric_feature_candidates:
    selected_feature = st.selectbox("Select a numeric feature", numeric_feature_candidates)
    feature_fig = px.histogram(
        filtered_users,
        x=selected_feature,
        nbins=25,
        title=f"Distribution of {selected_feature}",
    )
    feature_fig.update_layout(
        xaxis_title=selected_feature,
        yaxis_title="Count",
    )
    st.plotly_chart(feature_fig, use_container_width=True)
else:
    st.info("No numeric features available for exploration.")

st.markdown("---")
st.subheader("Data Tables")

tab_hr, tab_plan, tab_logs, tab_features = st.tabs(
    ["High-Risk Users", "Plan Summary", "Prediction Logs", "Feature Store"]
)

with tab_hr:
    display_hr = (
        high_risk_users.sort_values(["churn_probability", "monthly_spend"], ascending=[False, False])
        .head(20)
        .copy()
    )
    if not display_hr.empty:
        display_hr["risk_level"] = display_hr["churn_probability"].apply(
            lambda p: "HIGH" if p >= 0.7 else ("MEDIUM" if p >= 0.4 else "LOW")
        )
        cols = [
            "id",
            "name",
            "risk_level",
            "churn_probability",
            "monthly_spend",
            "subscription_plan",
            "email",
            "region",
            "device_type",
        ]
        cols = [c for c in cols if c in display_hr.columns]
        st.dataframe(display_hr[cols], use_container_width=True, hide_index=True)
    else:
        st.info("No high-risk users with current filters.")

with tab_plan:
    st.dataframe(plan_df, use_container_width=True, hide_index=True)

with tab_logs:
    display_logs = logs_df.head(50).copy()
    st.dataframe(display_logs, use_container_width=True, hide_index=True)

with tab_features:
    display_features = features_df.head(50).copy()
    st.dataframe(display_features, use_container_width=True, hide_index=True)

st.markdown("---")
st.caption(f"Last refreshed: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC")