import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta
import random

# Page Config
st.set_page_config(page_title="Heatwave Crisis Simulation", layout="wide")

# Scenario Introduction
st.title("🔥 Summer Heatwave Crisis Simulation")
st.markdown("""
Due to a severe summer heatwave, temperature-related alarms are surging across critical network sites.
Monitor alarms, take preventive actions, raise tickets, and manage alerts proactively.
""")

# Synthetic Data Generation Function
def generate_alarm_data(num_records=200):
    sites = ["Mumbai_DC", "Delhi_DC", "Chennai_DC", "Kolkata_DC"]
    alarm_types = ["Temperature High", "Power Surge", "AC Failure", "Humidity Critical"]
    severities = ["Critical", "Major", "Minor"]
    causes = ["AC Malfunction", "Power Instability", "Ventilation Blockage", "Extreme Heat"]

    data = []
    now = datetime.now()
    for _ in range(num_records):
        site = random.choice(sites)
        alarm_type = random.choice(alarm_types)
        severity = random.choices(severities, weights=[0.6, 0.3, 0.1])[0]
        cause = random.choice(causes)
        timestamp = now - timedelta(minutes=random.randint(1, 720))
        acknowledged = random.choice([True, False])
        data.append({
            "Timestamp": timestamp,
            "Site": site,
            "Alarm Type": alarm_type,
            "Severity": severity,
            "Cause": cause,
            "Acknowledged": acknowledged
        })

    return pd.DataFrame(data)

# Load data
df = generate_alarm_data()

# Sidebar for controls
st.sidebar.header("🚨 Simulation Controls")
selected_site = st.sidebar.selectbox("Select Site", df["Site"].unique())
severity_filter = st.sidebar.multiselect("Select Severity", df["Severity"].unique(), default=df["Severity"].unique())
visualization_option = st.sidebar.selectbox("Choose Visualization", ["Scatter Plot", "Bar Chart", "Pie Chart"])

# Filtered Data
site_data = df[(df["Site"] == selected_site) & (df["Severity"].isin(severity_filter))]

# Real-time Visualization
st.header(f"Real-Time Alarm Visualization for {selected_site}")
if visualization_option == "Scatter Plot":
    fig = px.scatter(site_data, x="Timestamp", y="Alarm Type", color="Severity",
                     title=f"Alarm Events at {selected_site}", size_max=10, symbol="Acknowledged")
elif visualization_option == "Bar Chart":
    alarm_counts = site_data["Alarm Type"].value_counts().reset_index()
    alarm_counts.columns = ["Alarm Type", "Count"]
    fig = px.bar(alarm_counts, x="Alarm Type", y="Count", color="Alarm Type",
                 title=f"Alarm Counts at {selected_site}")
elif visualization_option == "Pie Chart":
    cause_counts = site_data["Cause"].value_counts().reset_index()
    cause_counts.columns = ["Cause", "Occurrences"]
    fig = px.pie(cause_counts, names="Cause", values="Occurrences",
                 title=f"Top Causes at {selected_site}")
st.plotly_chart(fig, use_container_width=True)

# Alarm Insights
st.header("📈 Alarm Insights")
col1, col2 = st.columns(2)
with col1:
    st.subheader("Alarm Counts by Type")
    alarm_counts = site_data["Alarm Type"].value_counts().reset_index()
    alarm_counts.columns = ["Alarm Type", "Count"]
    st.dataframe(alarm_counts)

with col2:
    st.subheader("Top Probable Causes")
    cause_counts = site_data["Cause"].value_counts().reset_index().head(3)
    cause_counts.columns = ["Cause", "Occurrences"]
    st.dataframe(cause_counts)

# Predictive Action & Preventive Recommendations
st.header("🛠️ Immediate Preventive Actions")
preventive_actions = {
    "Temperature High": "Initiate AC health check",
    "Power Surge": "Start power diagnostics",
    "AC Failure": "Activate backup cooling",
    "Humidity Critical": "Check ventilation system"
}

triggered_alarms = site_data[site_data["Acknowledged"] == False]["Alarm Type"].unique()
for alarm in triggered_alarms:
    action = preventive_actions.get(alarm, "No action defined")
    if st.button(f"Trigger: {action} for {alarm}"):
        st.success(f"✅ Action triggered: {action}")

# Ticket Generation
st.header("🎫 Raise Ticket")
if st.button("Generate Support Ticket"):
    ticket_info = {
        "Ticket ID": f"TKT-{random.randint(1000,9999)}",
        "Site": selected_site,
        "Raised On": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "Severity": ", ".join(severity_filter),
        "Status": "Open"
    }
    st.success(f"🎟️ Ticket Created: {ticket_info['Ticket ID']}")
    st.json(ticket_info)

# Alerting Authority
st.header("🔔 Alert Authorities")
alert_message = st.text_area("Customize Alert Message", f"Urgent alarms detected at {selected_site} with severity levels: {', '.join(severity_filter)}. Immediate attention required.")
if st.button("Send Alert Notification"):
    st.warning(alert_message)

# CSV Download
st.header("📥 Download Alarm Report")
csv = site_data.to_csv(index=False).encode('utf-8')
st.download_button(label="Download CSV Report", data=csv, file_name=f"alarm_report_{selected_site}.csv", mime='text/csv')
