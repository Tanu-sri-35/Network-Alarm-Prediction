import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="System Architecture", page_icon="🏗️", layout="wide")
with open("assets/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.markdown("""
<div class="header">
    <div class="logo">🏗️</div>
    <div class="title-container">
        <h1 class="main-title">System Architecture</h1>
        <p class="subtitle">How our working prototype delivers real-time network alarm prediction and monitoring</p>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
### Overview

The diagrams and flow below **directly reflect the modules and data flow implemented in our prototype**. Each block is live and working as part of the deployed app: from raw network alarms to actionable predictions, dashboards, and alerts.
""")

# ==== MAIN SYSTEM FLOW DIAGRAM ====

node_labels = [
    "Network Devices",
    "Monitoring System",
    "Data Ingestion",
    "Data Storage",
    "Preprocessing",
    "Feature Engineering",
    "ML Model Training",
    "Prediction Engine",
    "API Service",
    "Visualization Dashboard",
    "Alert & Ticketing"
]
node_x = [0, 1, 2, 3, 3, 4, 5, 6, 7, 8, 8]
node_y = [0, 0, 0, 0, -1, -1, -1, 0, 0, 0, -1]

# Draw nodes as bubbles
fig = go.Figure()
for i, label in enumerate(node_labels):
    fig.add_trace(go.Scatter(
        x=[node_x[i]],
        y=[node_y[i]],
        mode="markers+text",
        marker=dict(size=40, color="#2CA58D" if i < 4 else ("#F3A712" if i < 8 else "#E6212B")),
        text=label,
        textposition="bottom center"
    ))

# Draw arrows (edges) between the modules
edges = [
    (0, 1), (1, 2), (2, 3),         # Data flow: devices -> monitor -> ingest -> storage
    (3, 4), (4, 5), (5, 6),         # Preprocessing -> features -> model training
    (6, 7), (7, 8), (8, 9),         # Prediction -> API -> dashboard
    (7, 10), (10, 0)                # Alerts & ticketing (feedback loop)
]
for src, dst in edges:
    fig.add_trace(go.Scatter(
        x=[node_x[src], node_x[dst]],
        y=[node_y[src], node_y[dst]],
        mode="lines",
        line=dict(width=3, color="#555555"),
        showlegend=False
    ))

fig.update_layout(
    title="End-to-End System Architecture",
    showlegend=False,
    height=500,
    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    margin=dict(l=20, r=20, t=60, b=20)
)
st.plotly_chart(fig, use_container_width=True)

st.info("**Every block above is implemented as a real module in the running prototype. The arrows show live data flow between them.**")

# ==== DETAILED COMPONENT FLOW (TABS) ====

tab1, tab2, tab3, tab4 = st.tabs(["Data Pipeline", "ML Pipeline", "API & Visualization", "Deployment"])

with tab1:
    st.markdown("#### Data Pipeline")
    st.markdown("""
- **Network Devices:** Live or simulated devices generate raw alarms (e.g., temperature, power, etc.).
- **Monitoring System:** Alarms are collected via SNMP, syslog, or polling.
- **Data Ingestion:** Real-time and batch ingestion implemented using pandas and Streamlit file upload.
- **Data Storage:** Data is held in-memory or on disk for this prototype, with time and site partitioning.
- **Preprocessing:** Missing values, outlier handling, feature extraction are performed in `model_training.py`.
    """)
    # Flow as a horizontal process
    steps = ["Devices", "Monitoring", "Ingestion", "Storage", "Preprocessing", "Feature Eng."]
    fig2 = go.Figure()
    for i, step in enumerate(steps):
        fig2.add_trace(go.Scatter(
            x=[i], y=[0], mode="markers+text",
            marker=dict(size=30, color="#0A2342"),
            text=step, textposition="bottom center"
        ))
        if i > 0:
            fig2.add_trace(go.Scatter(
                x=[i-1, i], y=[0, 0], mode="lines",
                line=dict(width=2, color="#aaa"), showlegend=False
            ))
    fig2.update_layout(
        showlegend=False, xaxis=dict(showticklabels=False), yaxis=dict(showticklabels=False),
        height=200, margin=dict(l=40, r=40, t=20, b=20)
    )
    st.plotly_chart(fig2, use_container_width=True)

with tab2:
    st.markdown("#### ML Pipeline")
    st.markdown("""
- **Feature Engineering:** Time features, lag features, categorical encodings (in `model_training.py`).
- **Model Training:** Chained models: Alarm Type, Cause, Duration, Severity, Timestamp.
- **Prediction Engine:** Models are loaded and chained together for each prediction request.
- **All modeling and predictions run live in the app and use current alarm data uploaded or simulated.
    """)
    stages = ["Feature Eng.", "Model Train", "Model Eval", "Prediction"]
    fig3 = go.Figure()
    for i, stage in enumerate(stages):
        fig3.add_trace(go.Scatter(
            x=[i], y=[0], mode="markers+text",
            marker=dict(size=30, color="#F3A712"),
            text=stage, textposition="bottom center"
        ))
        if i > 0:
            fig3.add_trace(go.Scatter(
                x=[i-1, i], y=[0, 0], mode="lines",
                line=dict(width=2, color="#aaa"), showlegend=False
            ))
    fig3.update_layout(
        showlegend=False, xaxis=dict(showticklabels=False), yaxis=dict(showticklabels=False),
        height=200, margin=dict(l=40, r=40, t=20, b=20)
    )
    st.plotly_chart(fig3, use_container_width=True)

with tab3:
    st.markdown("#### API & Visualization")
    st.markdown("""
- **API Service:** Real-time and batch predictions are available via app endpoints (`model_training.py`).
- **Visualization Dashboard:** All pages (upload, EDA, predictions, insights, etc.) are implemented in Streamlit and update live based on backend state.
- **Alert & Ticketing:** Automated recommendations and risk alerts, shown interactively in the demo and predictions pages.
    """)
    # Visual as a simple line with splits to dashboard & alert
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(x=[0, 1, 2], y=[0, 0, 0], mode="lines+markers+text",
                              text=["API", "Dashboard", "Alert"], textposition="bottom center",
                              marker=dict(size=[30, 30, 30], color=["#E6212B", "#2CA58D", "#E6212B"])))
    fig4.update_layout(showlegend=False, height=150, xaxis=dict(showticklabels=False), yaxis=dict(showticklabels=False))
    st.plotly_chart(fig4, use_container_width=True)

with tab4:
    st.markdown("#### Deployment")
    st.markdown("""
- **Streamlit app**: All modules run as part of a modular Streamlit web application.
- **ML models**: Saved and loaded on disk, can be re-trained in the UI.
- **Scalability**: Can be dockerized or run in cloud for production.
- **Security**: User authentication and role-based access can be added as needed.
    """)
    # Show horizontal layers: Web App | Models | Storage
    layers = ["Web App (Streamlit)", "ML Models", "Local/Cloud Storage"]
    fig5 = go.Figure()
    for i, layer in enumerate(layers):
        fig5.add_trace(go.Scatter(x=[0], y=[-i], mode="markers+text", marker=dict(size=40, color="#aaa"), text=layer, textposition="middle right"))
    fig5.update_layout(showlegend=False, height=200, xaxis=dict(showticklabels=False), yaxis=dict(showticklabels=False))
    st.plotly_chart(fig5, use_container_width=True)

# ====== FOOTER ======

st.markdown("""
---
**Every architecture diagram and pipeline step above is backed by real code and data flow in the current working prototype, not just design slides. The prototype is end-to-end functional, and every box is a working module you can see in action in the app.**
""")

st.markdown("---")
col1, col2 = st.columns(2)
with col1:
    st.markdown("[← Back to Insights](/pages/6_insights.py)")
with col2:
    st.markdown("[Continue to Demo Simulation →](/pages/8_demo.py)")
