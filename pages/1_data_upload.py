import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import os

# Page config
st.set_page_config(page_title="Data Upload & EDA", page_icon="📊", layout="wide")

# Import custom CSS
with open("assets/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Set up session state
if 'data' not in st.session_state:
    st.session_state.data = None

# Page header
st.markdown(
    """
    <div class="header">
        <div class="logo">📊</div>
        <div class="title-container">
            <h1 class="main-title">Data Upload & Exploration</h1>
            <p class="subtitle">Upload and analyze your network alarm data</p>
        </div>
    </div>
    """, 
    unsafe_allow_html=True
)

# Data upload section
st.markdown("## 📤 Upload Your Data")

# File uploader
uploaded_file = st.file_uploader("Upload a CSV file with alarm data", type=["csv"])

# Sample data option
col1, col2 = st.columns([3, 1])

with col1:
    st.markdown("""
    If you don't have your own data, you can use our sample dataset to explore the application.
    The sample data contains synthetic network alarm records with timestamps, alarm types, sites,
    and severity levels to demonstrate the application's capabilities.
    """)

with col2:
    if st.button("Load Sample Data"):
        from utils.data_processing import load_sample_data
        sample = load_sample_data()
        st.session_state.data = sample.copy()
        st.session_state.raw_data = sample.copy()
        st.success("Sample data loaded successfully!")


# Process uploaded file
if uploaded_file is not None:
    try:
        # Load data
        data = pd.read_csv(uploaded_file)

        # Show available columns
        all_columns = data.columns.tolist()
        st.markdown("### ✅ Column Selection")
        selected_columns = st.multiselect("Select columns you want to keep", all_columns, default=all_columns)
        # Store untouched raw version for EDA
        if 'raw_data' not in st.session_state:
            st.session_state.raw_data = data.copy()
        else:
            st.session_state.raw_data = data.copy()

        if selected_columns:
            filtered_data = data[selected_columns]
            st.session_state.data = filtered_data
            st.success(f"Data uploaded successfully with {len(selected_columns)} columns selected!")
        else:
            st.warning("Please select at least one column to proceed.")


    except Exception as e:
        st.error(f"Error: {str(e)}")

# Display data preview if available
if st.session_state.data is not None:
    st.markdown("## 👁️ Data Preview")
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["Preview", "Summary Statistics", "Data Quality"])
    
    with tab1:
        # Display preview
        st.dataframe(st.session_state.data.head(10), use_container_width=True)
        
        # Display basic info
        st.markdown(f"**Dataset Shape:** {st.session_state.data.shape[0]} rows, {st.session_state.data.shape[1]} columns")
        st.markdown(f"**Columns:** {', '.join(st.session_state.data.columns)}")
        
        # Convert timestamp to datetime if needed
        data_copy = st.session_state.data.copy()
        if 'timestamp' in data_copy.columns and data_copy['timestamp'].dtype != 'datetime64[ns]':
            data_copy['timestamp'] = pd.to_datetime(data_copy['timestamp'])
        
        # Display time range
        if 'timestamp' in data_copy.columns:
            min_date = data_copy['timestamp'].min()
            max_date = data_copy['timestamp'].max()
            st.markdown(f"**Time Range:** {min_date} to {max_date}")
    
    with tab2:
        # Display summary statistics
        st.markdown("### Summary Statistics")
        
        numeric_cols = st.session_state.data.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            st.dataframe(st.session_state.data[numeric_cols].describe(), use_container_width=True)
        
        # Display categorical summaries
        st.markdown("### Categorical Columns")
        
        categorical_cols = st.session_state.data.select_dtypes(include=['object']).columns.tolist()
        
        if categorical_cols:
            for col in categorical_cols:
                st.markdown(f"**{col}**")
                st.write(st.session_state.data[col].value_counts())
                st.markdown("---")
    
    with tab3:
        from models.model_training import preprocess_alarm_data

        # Display missing values
        st.markdown("### Missing Values")
        missing_data = st.session_state.data.isnull().sum()
        st.dataframe(missing_data.to_frame('Missing Values'), use_container_width=True)

        # Display duplicate count
        st.markdown("### Duplicate Records")
        duplicates = st.session_state.data.duplicated().sum()
        st.markdown(f"Number of duplicate rows: **{duplicates}**")

        # Clean + preprocess
        if st.button("Clean and Preprocess Data"):
            # Save old columns BEFORE overwriting data
            old_cols = set(st.session_state.data.columns)

            # Preprocess
            cleaned_data, df_time, encoders = preprocess_alarm_data(st.session_state.data)

            # Update session state
            st.session_state.data = cleaned_data
            st.session_state.df_time = df_time
            st.session_state.encoders = encoders

            # Show feedback
            st.success(f"Data cleaned and preprocessed! New shape: {cleaned_data.shape}")

            # Show new features
            st.markdown("### Added Features")
            new_features = list(set(cleaned_data.columns) - old_cols)
            if new_features:
                st.write(f"New features: {', '.join(new_features)}")
                st.dataframe(cleaned_data[new_features].head(), use_container_width=True)
            dropped_cols = list(old_cols - set(cleaned_data.columns))
            if dropped_cols:
                st.markdown(f"Dropped columns: {', '.join(dropped_cols)}")



# Data export section
if st.session_state.data is not None:
    st.markdown("## 💾 Export Processed Data")
    
    if st.download_button(
        label="Download Processed Data as CSV",
        data=st.session_state.data.to_csv(index=False),
        file_name="processed_alarm_data.csv",
        mime="text/csv"
    ):
        st.success("Data downloaded successfully!")

# Navigation buttons
st.markdown("---")
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("[← Back to Home](/?page=)")

with col2:
    if st.session_state.data is not None:
        st.markdown("[Continue to EDA →](/pages/2_eda.py)")
    else:
        st.warning("Please upload or load sample data to continue.")