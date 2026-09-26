import streamlit as st
import pandas as pd
from datetime import datetime

# Configure page settings
st.set_page_config(
    page_title="Network Alarm Prediction Dashboard",
    page_icon="🔔",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import custom CSS
with open("assets/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Set up session state for data persistence across pages
if 'data' not in st.session_state:
    st.session_state.data = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'features' not in st.session_state:
    st.session_state.features = None



# App header with logo and title
st.markdown(
    """
    <div class="header">
        <div class="logo">🔔</div>
        <div class="title-container">
            <h1 class="main-title">Network Alarm Prediction System</h1>
            <p class="subtitle">Predict & prevent network issues before they occur</p>
        </div>
    </div>
    """, 
    unsafe_allow_html=True
)

# Main content
st.markdown("## 🏠 Welcome to Network Alarm Prediction System")

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    ### 🎯 Project Overview
    
    This application analyzes historical network alarm data to predict future alarms, 
    helping network operators proactively address potential issues before they impact service.
    
    ### 🔑 Key Features
    
    - **Data Analysis**: Upload and explore network alarm data
    - **Predictive Modeling**: Train ML models to forecast next likely alarms
    - **Visualization**: View alarm trends, patterns, and hotspots
    - **Simulation**: Test prediction capabilities with real-time simulation
    
    ### 📊 How It Works
    
    1. Upload historical network alarm data
    2. Explore data patterns and correlations
    3. Train prediction models on your data
    4. Visualize trends and hotspots
    5. Get proactive alerts for likely future alarms
    """)

with col2:
    st.markdown("""
    <div class="stat-container">
        <div class="stat-card">
            <h3>Prediction Accuracy</h3>
            <p class="stat-value">77%<span class="stat-label">avg</span></p>
        </div>
        <div class="stat-card">
            <h3>Response Time</h3>
            <p class="stat-value">-30%<span class="stat-label">improvement</span></p>
        </div>
        <div class="stat-card">
            <h3>Downtime Prevention</h3>
            <p class="stat-value">42%<span class="stat-label">reduction</span></p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Quick start button
    st.markdown("""
    <div class="action-container">
        <a href="?page=1_data_upload" class="start-button">
            Get Started →
        </a>
    </div>
    """, unsafe_allow_html=True)

# Sample data section
st.markdown("---")
st.markdown("### 🔍 Try with Sample Data")

col1, col2 = st.columns([3, 1])

with col1:
    st.markdown("""
    Don't have your own network alarm data? Use our sample dataset to explore the application's capabilities.
    This dataset contains synthetic network alarm records with timestamps, alarm types, sites, and severity levels.
    """)

with col2:
    if st.button("Load Sample Data", key="load_sample"):
        from utils.data_processing import load_sample_data
        
        # Load sample data
        st.session_state.data = load_sample_data()
        st.success("Sample data loaded successfully! Navigate to the Data Upload & EDA page to explore.")

# Show current date and app version at the bottom
st.markdown("---")
col1, col2 = st.columns([1, 1])
with col1:
    st.markdown(f"**Current Date:** {datetime.now().strftime('%Y-%m-%d')}")
with col2:
    st.markdown("**App Version:** 1.0.0")

st.markdown("---")
col3, = st.columns([1])
with col3:
    st.markdown(
        """
        <div style='text-align: center; font-size: 40px; font-weight: bold;'>
            Made with ❤️ from Team-1 Nokia Her in Tech
        </div>
        """,
        unsafe_allow_html=True
    )
