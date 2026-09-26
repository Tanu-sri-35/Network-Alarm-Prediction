# pages/5_visualizations.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(page_title="Visualizations", page_icon="📈", layout="wide")

with open("assets/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

if 'data' not in st.session_state or st.session_state.data is None:
    st.warning("No data available. Please upload or load sample data from the Data Upload page.")
    st.markdown("[\u2190 Go to Data Upload](/pages/1_data_upload.py)")
    st.stop()

# ---- Column detection ----
def detect_column(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

df = st.session_state.data.copy()

# Parse timestamp columns
for col in ['First Time Detected', 'Last Time Cleared', 'Last Time Detected']:
    if col in df.columns:
        df[col + " Clean"] = pd.to_datetime(
            df[col].astype(str).str.extract(r'(^\d{4}/\d{2}/\d{2} \d{2}:\d{2}:\d{2})')[0],
            format="%Y/%m/%d %H:%M:%S",
            errors="coerce"
        )

# Use cleaned version
timestamp_col = detect_column(df, ["timestamp", "First Time Detected Clean"])
site_col = detect_column(df, ["site_id", "Site Name", "site"])
alarm_col = detect_column(df, ["alarm_type", "Alarm Type", "alarm"])
severity_col = detect_column(df, ["severity", "Severity"])
temperature_col = detect_column(df, ["temperature", "Temperature"])

if not (timestamp_col and site_col and alarm_col and severity_col):
    st.error(f"Dataset must contain columns for time, site, alarm type, and severity.\nColumns found: {list(df.columns)}")
    st.stop()

# Ensure timestamp is datetime
if not np.issubdtype(df[timestamp_col].dtype, np.datetime64):
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors='coerce')

valid_dates = df[timestamp_col].dropna()
min_date = valid_dates.min().date() if not valid_dates.empty else datetime.today().date()
max_date = valid_dates.max().date() if not valid_dates.empty else datetime.today().date()

st.markdown("""
    <div class="header">
        <div class="logo">📈</div>
        <div class="title-container">
            <h1 class="main-title">Advanced Visualizations</h1>
            <p class="subtitle">Interactive visual analysis of network alarm patterns</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Sidebar filters
st.sidebar.markdown("## 🔍 Visualization Options")
viz_type = st.sidebar.selectbox(
    "Select Visualization Type",
    ["Alarm Hotspot Map", "Time-based Analysis", "Site Comparison", "Severity Analysis", "Correlation Insights"]
)

# Filters
st.sidebar.markdown("### Data Filters")
start_date = st.sidebar.date_input("Start Date", min_date)
end_date = st.sidebar.date_input("End Date", max_date)
df = df[(df[timestamp_col].dt.date >= start_date) & (df[timestamp_col].dt.date <= end_date)]

all_sites = df[site_col].dropna().unique()
selected_sites = st.sidebar.multiselect("Select Sites", all_sites, default=all_sites[:5] if len(all_sites) > 5 else all_sites)
if selected_sites:
    df = df[df[site_col].isin(selected_sites)]

all_alarm_types = df[alarm_col].dropna().unique()
selected_alarm_types = st.sidebar.multiselect("Select Alarm Types", all_alarm_types)
if selected_alarm_types:
    df = df[df[alarm_col].isin(selected_alarm_types)]

# --- Visualizations ---
import plotly.express as px

if viz_type == "Alarm Hotspot Map":
    st.markdown("## 🌐 Network Alarm Hotspot Map")
    st.markdown("Visualizing alarm intensity across sites using a color density heatmap.")

    # Site-wise alarm count
    counts = df.groupby(site_col).size().reset_index(name='count')
    counts = counts.sort_values('count', ascending=False)

    # Assign a numerical index to sites (for heatmap axis)
    counts["site_index"] = range(len(counts))

    # Create a visually rich heatmap (like a leaderboard)
    fig = px.density_heatmap(
        counts,
        x="site_index",
        y="count",
        z="count",
        color_continuous_scale="thermal",
        labels={"site_index": "Site Rank", "count": "Alarm Count"},
        title="🔥 Heatmap of Alarm Intensity by Site (Top Sites)"
    )

    # Update x-axis ticks to display actual site names
    fig.update_layout(
        xaxis=dict(
            tickmode="array",
            tickvals=counts["site_index"],
            ticktext=counts[site_col].astype(str),
            title="Sites"
        ),
        yaxis_title="Number of Alarms"
    )

    st.plotly_chart(fig, use_container_width=True)

    # --- Add Top Site Summary Bar ---
    st.markdown("### 🏆 Top 10 Sites by Alarm Volume")
    top_sites = counts.head(10)
    fig2 = px.bar(
        top_sites,
        x=site_col,
        y="count",
        color="count",
        title="Top 10 Alarm-Heavy Sites",
        color_continuous_scale="bluered"
    )
    st.plotly_chart(fig2, use_container_width=True)


elif viz_type == "Time-based Analysis":
    st.markdown("## ⏱️ Time-based Alarm Analysis")
    df['date'] = df[timestamp_col].dt.date
    daily = df.groupby('date').size().reset_index(name='count')
    st.plotly_chart(px.line(daily, x='date', y='count', title="Alarms Over Time"), use_container_width=True)

    df['hour'] = df[timestamp_col].dt.hour
    hr = df.groupby('hour').size().reset_index(name='count')
    st.plotly_chart(px.bar(hr, x='hour', y='count', title="Alarms by Hour of Day"), use_container_width=True)

elif viz_type == "Site Comparison":
    st.markdown("## 🔄 Site Comparison Analysis")
    metrics = []
    for site in selected_sites:
        sub = df[df[site_col] == site]
        metrics.append({
            'Site': site,
            'Total Alarms': sub.shape[0],
            'Critical Alarms': sub[sub[severity_col].astype(str).str.lower() == 'critical'].shape[0],
            'Unique Alarms': sub[alarm_col].nunique(),
            'Avg Temperature': sub[temperature_col].mean() if temperature_col in sub else None
        })
    st.dataframe(pd.DataFrame(metrics))

elif viz_type == "Severity Analysis":
    st.markdown("## 🚨 Severity Analysis")
    st.plotly_chart(px.histogram(df, x=severity_col, title="Alarm Severity Distribution", color=severity_col), use_container_width=True)

    df['date'] = df[timestamp_col].dt.date
    trend = df.groupby(['date', severity_col]).size().reset_index(name='count')
    st.plotly_chart(px.line(trend, x='date', y='count', color=severity_col, title="Severity Trends Over Time"), use_container_width=True)

elif viz_type == "Correlation Insights":
    st.markdown("## 🔗 Correlation Insights")
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) < 2:
        st.warning("Not enough numeric features for correlation analysis.")
    else:
        st.plotly_chart(px.imshow(df[num_cols].corr(), text_auto=True, title="Feature Correlation Matrix"), use_container_width=True)

# --- Key metrics summary ---
st.markdown("---")
k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Alarms", df.shape[0])
k2.metric("Unique Sites", df[site_col].nunique())
if severity_col in df.columns:
    n_crit = df[df[severity_col].astype(str).str.lower() == 'critical'].shape[0]
    k3.metric("Critical Alarms", n_crit)
if timestamp_col in df.columns:
    duration = (df[timestamp_col].max() - df[timestamp_col].min()).days
    rate = df.shape[0] / max(1, duration)
    k4.metric("Alarms Per Day", f"{rate:.1f}")

# --- Navigation ---
st.markdown("---")
c1, c2 = st.columns([1, 1])
with c1:
    st.markdown("[← Back to Predictions](/pages/4_predictions.py)")
with c2:
    st.markdown("[Continue to Insights →](/pages/6_insights.py)")
