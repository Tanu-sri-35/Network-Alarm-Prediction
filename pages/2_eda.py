# pages/2_eda.py  –  RAW-CSV friendly EDA page
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime,timedelta

# ------------------------------------------------
# --------  1.  Page & style ----------------------
# ------------------------------------------------
st.set_page_config(page_title="Exploratory Data Analysis", page_icon="📈", layout="wide")
with open("assets/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ------------------------------------------------
# --------  2.  Guard clause ----------------------
# ------------------------------------------------
if "data" not in st.session_state or st.session_state.data is None:
    st.warning("No data available. Please upload or load sample data from the Data Upload page.")
    st.stop()

if "raw_data" in st.session_state:
    raw_df = st.session_state.raw_data.copy()  # always original
else:
    st.error("No raw data found. Please upload from the Data Upload page.")
    st.stop()


# ------------------------------------------------
# --------  3.  Helper functions ------------------
# ------------------------------------------------
def detect_column(df: pd.DataFrame, candidates: list[str]):
    """Return the first matching column name in df or None."""
    return next((c for c in candidates if c in df.columns), None)

def parse_timestamp_column(col: pd.Series) -> pd.Series:
    """
    Parse strings like 2024/11/29 05:30:14 000 +05:30 GMT to pandas datetime.
    Strategy: drop trailing ' GMT', get rid of the '000 +05:30' piece,
    then let pandas figure it out.
    """
    cleaned = (
        col.astype(str)
           .str.replace(r"\s+\d{3}\s+[+-]\d{2}:\d{2}\s+GMT$", "", regex=True)
           .str.replace(" GMT", "", regex=False)
           .str.strip()
    )
    dt = pd.to_datetime(cleaned, errors="coerce")
    return dt

def make_standard_df(df, site_col, alarm_col, sev_col, ts_col):
    """
    Return df with canonical column names expected by utils.visualization:
    site_id, alarm_type, severity, timestamp
    """
    rename_map = {}
    if site_col   and site_col   != "site_id":    rename_map[site_col]   = "site_id"
    if alarm_col  and alarm_col  != "alarm_type": rename_map[alarm_col]  = "alarm_type"
    if sev_col    and sev_col    != "severity":   rename_map[sev_col]    = "severity"
    if ts_col     and ts_col     != "timestamp":  rename_map[ts_col]     = "timestamp"
    df_std = df.rename(columns=rename_map)

    # --- derived time features for EDA ---
    if "timestamp" in df_std.columns:
        df_std["hour"]        = df_std["timestamp"].dt.hour
        df_std["day_of_week"] = df_std["timestamp"].dt.dayofweek
        df_std["month"]       = df_std["timestamp"].dt.month
    return df_std

# ------------------------------------------------
# --------  4.  Dynamic column detection ----------
# ------------------------------------------------
site_col       = detect_column(raw_df, ["site_id", "Site Name", "site", "Site"])
alarm_col      = detect_column(raw_df, ["alarm_type", "Alarm Type", "alarm", "Alarm Name"])
sev_col        = detect_column(raw_df, ["severity", "Severity", "Previous Severity"])
ts_col         = detect_column(raw_df, ["timestamp",
                                        "First Time detected", "First Time Detected",
                                        "First Time Detected Clean", "first_time_detected",
                                        "Last Time detected", "Last Time Detected"])

required_map = {"site": site_col, "alarm type": alarm_col, "severity": sev_col, "timestamp": ts_col}
for label, colname in required_map.items():
    if colname is None:
        st.error(f"Could not find a column for **{label}** in the data.\n\nAvailable columns: {list(raw_df.columns)}")
        st.stop()

# ------------------------------------------------
# --------  5.  Timestamp parsing -----------------
# ------------------------------------------------
if not pd.api.types.is_datetime64_any_dtype(raw_df[ts_col]):
    raw_df[ts_col] = parse_timestamp_column(raw_df[ts_col])

if raw_df[ts_col].isna().all():
    st.error(f"Failed to parse any valid dates from column **{ts_col}**. "
             "Please verify the timestamp format.")
    st.stop()

# ------------------------------------------------
# --------  6.  Sidebar filters -------------------
# ------------------------------------------------
st.sidebar.header("🔍 Data Filters")

min_date = raw_df[ts_col].min().date()
max_date = raw_df[ts_col].max().date()
start_date = st.sidebar.date_input("Start Date", min_date, key="start_date")
end_date   = st.sidebar.date_input("End Date",   max_date, key="end_date")

all_sites = raw_df[site_col].unique()
sel_sites = st.sidebar.multiselect("Sites", all_sites,
                                   default=all_sites[:5] if len(all_sites) > 5 else all_sites)

all_types = raw_df[alarm_col].unique()
sel_types = st.sidebar.multiselect("Alarm Types", all_types, default=all_types)

all_sev = raw_df[sev_col].unique()
sel_sev = st.sidebar.multiselect("Severity", all_sev, default=all_sev)

# -- apply filters
flt = raw_df[
    (raw_df[ts_col].dt.date.between(start_date, end_date)) &
    (raw_df[site_col].isin(sel_sites)) &
    (raw_df[alarm_col].isin(sel_types)) &
    (raw_df[sev_col].isin(sel_sev))
].copy()

std_df = make_standard_df(flt, site_col, alarm_col, sev_col, ts_col)

# ------------------------------------------------
# --------  7.  KPI summary -----------------------
# ------------------------------------------------
# st.markdown("## 📋 Data Overview")
# k1, k2, k3, k4 = st.columns(4)
# k1.metric("Total Alarms", len(std_df))
# k2.metric("Unique Sites", std_df["site_id"].nunique())
# k3.metric("Alarm Types",  std_df["alarm_type"].nunique())
# crit = std_df[std_df["severity"].str.lower().str.contains("critical", na=False)]
# pct  = 0 if len(std_df)==0 else len(crit)/len(std_df)*100
# k4.metric("Critical Alarms", f"{len(crit)} ({pct:.1f}%)")

from datetime import timedelta

def metric_with_delta(curr, prev, label, icon, color_up="lime", color_down="red"):
    # Calculate percentage change and render with arrow and color
    if prev == 0:
        pct = 0
    else:
        pct = ((curr - prev) / prev) * 100
    if pct > 0:
        arrow = "↑"
        color = color_up
        sign = "+"
    elif pct < 0:
        arrow = "↓"
        color = color_down
        sign = "-"
    else:
        arrow = ""
        color = "gray"
        sign = ""
    pct_str = f'<span style="color:{color};font-weight:bold;">{arrow} {abs(pct):.0f}%</span>'
    st.markdown(f"""
    <div style="padding:1.2em;background:#191c24;border-radius:1em;display:flex;flex-direction:column;align-items:left;min-height:120px;">
        <span style="color:#bfc7d5;font-size:1em;">{label}</span>
        <span style="font-size:2.5em;font-weight:bold;color:white;">{curr} <span style="font-size:1.1em;">{icon}</span></span>
        <span style="font-size:1em;">{pct_str} <span style="color:#bfc7d5;">vs last week</span></span>
    </div>
    """, unsafe_allow_html=True)

# -- For metrics, use std_df, which has standardized columns
# Calculate weeks
latest = std_df["timestamp"].max()
last_week = latest - timedelta(days=7)
week_before = last_week - timedelta(days=7)

# This week's and last week's DataFrames
curr_week = std_df[std_df["timestamp"] > last_week]
prev_week = std_df[(std_df["timestamp"] > week_before) & (std_df["timestamp"] <= last_week)]

# 1. Total Alarms
curr_total = len(curr_week)
prev_total = len(prev_week)

# 2. Critical Alarms
curr_crit = curr_week[curr_week["severity"].str.lower().str.contains("critical", na=False)]
prev_crit = prev_week[prev_week["severity"].str.lower().str.contains("critical", na=False)]
curr_crit_count = len(curr_crit)    
prev_crit_count = len(prev_crit)
# crit_pct=(curr_crit_count -prev_crit_count)/prev_crit_count*100

# 3. Active Alarms (for demo: any alarm not cleared; tweak as per your logic!)
# Let's say alarms in curr_week are "active" if no matching "cleared" in prev_week (customize as needed)
curr_active = curr_week  # (replace with your own active alarm logic, e.g., filter by a flag)
prev_active = prev_week  # (replace as above)
curr_active_count = len(curr_active)
prev_active_count = len(prev_active)

# 4. Most Affected Region (if you have region/zone column)
# region_col = None
# for col in ["Region", "Site Name", "zone", "Zone"]:
#     if col in curr_week.columns:
#         region_col = col
#         break
# if region_col:
#     top_region = curr_week[region_col].value_counts().idxmax()
# else:
#     top_region = "N/A"
top_region = raw_df[site_col].mode()[0]
st.markdown("# Network Alarm Dashboard")
c1, c2, c3, c4 = st.columns(4)

with c1:
    metric_with_delta(curr_total, prev_total, "Total Alarms", "📈")
with c2:
    metric_with_delta(curr_crit_count, prev_crit_count, "Critical Alarms", "🚨")
with c3:
    metric_with_delta(curr_active_count, prev_active_count, "Active Alarms", "⏰")
with c4:
    st.markdown(f"""
    <div style="padding:1.2em;background:#191c24;border-radius:1em;display:flex;flex-direction:column;align-items:left;min-height:120px;">
        <span style="color:#bfc7d5;font-size:1em;">Most Affected Site</span>
        <span style="font-size:2.5em;font-weight:bold;color:white;">{top_region} <span style="font-size:1.1em;">📍</span></span>
    </div>
    """, unsafe_allow_html=True)


# ------------------------------------------------
# --------  8.  Visual tabs -----------------------
# ------------------------------------------------
from utils.visualization import (
    plot_alarm_distribution, plot_alarm_severity, plot_alarm_trends,
    plot_site_heatmap, plot_correlation_matrix, plot_alarm_calendar
)

tab1, tab2, tab3, tab4 = st.tabs(
    ["Alarm Distribution", "Time Trends", "Site Analysis", "Correlations"]
)

with tab1:
    st.subheader("📊 Alarm Distribution")
    st.plotly_chart(plot_alarm_distribution(std_df), use_container_width=True)
    st.plotly_chart(plot_alarm_severity(std_df),     use_container_width=True)

with tab2:
    st.subheader("⏱️ Time Trend Analysis")
    st.plotly_chart(plot_alarm_trends(std_df),   use_container_width=True)
    st.plotly_chart(plot_alarm_calendar(std_df), use_container_width=True)

    st.markdown("#### Alarms by Hour of Day")
    hr_counts = std_df.groupby("hour").size().reset_index(name="count")
    st.plotly_chart(
        px.bar(hr_counts, x="hour", y="count",
               labels={"hour":"Hour (0-23)", "count":"Alarms"},
               title="Alarms per Hour"),
        use_container_width=True
    )

with tab3:
    st.subheader("🌐 Site Analysis")
    # simple bar
    bar_df = std_df.groupby("site_id").size().reset_index(name="count").sort_values("count", ascending=False)
    st.plotly_chart(px.bar(bar_df, x="site_id", y="count", color="count",
                           title="Number of Alarms per Site",
                           color_continuous_scale="Viridis"),
                    use_container_width=True)

    st.plotly_chart(plot_site_heatmap(std_df), use_container_width=True)

with tab4:
    st.subheader("🔗 Correlation Analysis")
    num_cols = std_df.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) > 1:
        st.plotly_chart(plot_correlation_matrix(std_df, num_cols), use_container_width=True)
        x_feat = st.selectbox("X-axis", num_cols, index=0, key="corr_x")
        y_feat = st.selectbox("Y-axis", [c for c in num_cols if c!=x_feat], index=0, key="corr_y")
        st.plotly_chart(
            px.scatter(std_df, x=x_feat, y=y_feat, color="alarm_type",
                       hover_data=["timestamp","site_id","severity"],
                       title=f"{x_feat} vs {y_feat}"),
            use_container_width=True
        )
    else:
        st.info("Not enough numeric columns for correlation heat-map.")

# ------------------------------------------------
# --------  9.  Nav buttons -----------------------
# ------------------------------------------------
st.markdown("---")
left, right = st.columns(2)
with left:
    st.markdown("[← Back to Data Upload](/pages/1_data_upload.py)")
with right:
    st.markdown("[Continue to ML Modeling →](/pages/3_modeling.py)")
