import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import streamlit as st
import os

def load_sample_data():
    """
    Generate sample network alarm data if not available
    """
    # Check if sample data exists
    if os.path.exists("data/sample_alarms.csv"):
        return pd.read_csv("data/sample_alarms.csv")
    
    # Create sample data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    # Generate synthetic data
    alarms = generate_synthetic_alarms()
    
    # Save to file
    alarms.to_csv("data/sample_alarms.csv", index=False)
    
    return alarms

def generate_synthetic_alarms(n_samples=1000):
    """Generate synthetic network alarm data"""
    # Define possible values
    alarm_types = ["Link Down", "High CPU", "Memory Overflow", "Power Issue", 
                  "Temperature Alert", "Packet Loss", "Latency Spike", 
                  "Authentication Failure", "Config Change", "Interface Flapping"]
    
    sites = [f"Site_{i}" for i in range(1, 21)]
    severities = ["Critical", "Major", "Minor", "Warning", "Info"]
    
    # Create base timestamp
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)
    
    # Generate timestamps
    timestamps = [start_date + (end_date - start_date) * random.random() for _ in range(n_samples)]
    timestamps.sort()
    
    # Generate data
    data = {
        "timestamp": timestamps,
        "alarm_type": [random.choice(alarm_types) for _ in range(n_samples)],
        "site_id": [random.choice(sites) for _ in range(n_samples)],
        "severity": [random.choice(severities) for _ in range(n_samples)],
        "temperature": [round(random.uniform(18.0, 35.0), 1) for _ in range(n_samples)],
        "duration_minutes": [random.randint(1, 180) for _ in range(n_samples)]
    }
    
    # Create dataframe
    df = pd.DataFrame(data)
    
    # Add some patterns and relationships
    # Sites with temperature issues
    hot_sites = random.sample(sites, 3)
    for site in hot_sites:
        idx = df[df['site_id'] == site].index
        df.loc[idx, 'temperature'] = df.loc[idx, 'temperature'] + 5
        
        # More temperature alarms for hot sites
        temp_idx = random.sample(list(idx), len(idx) // 3)
        df.loc[temp_idx, 'alarm_type'] = "Temperature Alert"
        df.loc[temp_idx, 'severity'] = "Major"
    
    # Add some sequential patterns (certain alarms follow others)
    for i in range(1, len(df)):
        # 20% chance of related alarms
        if random.random() < 0.2:
            if df.loc[i-1, 'alarm_type'] == "High CPU":
                df.loc[i, 'alarm_type'] = "Memory Overflow"
                df.loc[i, 'site_id'] = df.loc[i-1, 'site_id']
            elif df.loc[i-1, 'alarm_type'] == "Link Down":
                df.loc[i, 'alarm_type'] = "Packet Loss"
                df.loc[i, 'site_id'] = df.loc[i-1, 'site_id']
    
    # Convert timestamp to string for storage
    df['timestamp'] = df['timestamp'].astype(str)
    
    return df

def clean_and_preprocess_data(df):
    """Clean and preprocess the alarm data"""
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    # Convert timestamp to datetime if it's not already
    if df['timestamp'].dtype == 'object':
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Sort by timestamp
    df = df.sort_values('timestamp')
    
    # Check for and handle missing values
    missing_values = df.isnull().sum()
    
    # Fill missing values appropriately
    if 'temperature' in df.columns and df['temperature'].isnull().any():
        df['temperature'].fillna(df['temperature'].mean(), inplace=True)
    
    if 'duration_minutes' in df.columns and df['duration_minutes'].isnull().any():
        df['duration_minutes'].fillna(df['duration_minutes'].median(), inplace=True)
    
    # Drop rows with missing critical fields
    critical_fields = ['timestamp', 'alarm_type', 'site_id', 'severity']
    df = df.dropna(subset=critical_fields)
    
    # Extract time features
    df['hour'] = df['timestamp'].dt.hour
    df['day'] = df['timestamp'].dt.day
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    
    # Encode categorical variables
    if df['alarm_type'].dtype == 'object':
        df['alarm_type_code'] = df['alarm_type'].astype('category').cat.codes
    
    if df['site_id'].dtype == 'object':
        df['site_id_code'] = df['site_id'].astype('category').cat.codes
    
    if df['severity'].dtype == 'object':
        # Create an ordered category for severity
        severity_order = ["Info", "Warning", "Minor", "Major", "Critical"]
        df['severity'] = pd.Categorical(df['severity'], categories=severity_order, ordered=True)
        df['severity_code'] = df['severity'].cat.codes
    
    return df, missing_values

def get_feature_importance(model, feature_names):
    """Extract feature importance from the trained model"""
    try:
        # For models with feature_importances_ attribute (like Random Forest, XGBoost)
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            return pd.DataFrame({'feature': feature_names, 'importance': importances})
        
        # For linear models with coef_ attribute
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_[0])
            return pd.DataFrame({'feature': feature_names, 'importance': importances})
        
        # Fall back to permutation importance if available
        else:
            return None
    
    except:
        return None

def create_lag_features(df, target_col, lag_periods=[1, 2, 3], group_col=None):
    """
    Create lag features for time series prediction
    
    Parameters:
    -----------
    df : DataFrame
        The input DataFrame sorted by timestamp
    target_col : str
        The column to create lag features for
    lag_periods : list
        List of lag periods to create
    group_col : str, optional
        Column to group by before creating lags (e.g., site_id)
    
    Returns:
    --------
    DataFrame with added lag features
    """
    df_result = df.copy()
    
    if group_col:
        for lag in lag_periods:
            lag_col_name = f'{target_col}_lag_{lag}'
            df_result[lag_col_name] = df_result.groupby(group_col)[target_col].shift(lag)
    else:
        for lag in lag_periods:
            lag_col_name = f'{target_col}_lag_{lag}'
            df_result[lag_col_name] = df_result[target_col].shift(lag)
    
    # Drop rows with NaN values created by lag
    df_result = df_result.dropna()
    
    return df_result

def get_next_alarm_prediction_features(df, current_time=None, site_id=None):
    """
    Extract features for predicting the next alarm
    
    Parameters:
    -----------
    df : DataFrame
        The processed DataFrame with alarm data
    current_time : datetime, optional
        The current time to use as reference
    site_id : str, optional
        The site ID to filter data for
    
    Returns:
    --------
    DataFrame with features for prediction
    """
    if current_time is None:
        current_time = df['timestamp'].max()
    
    # Filter data
    if site_id:
        recent_data = df[(df['timestamp'] <= current_time) & (df['site_id'] == site_id)]
    else:
        recent_data = df[df['timestamp'] <= current_time]
    
    # Sort by timestamp
    recent_data = recent_data.sort_values('timestamp')
    
    # Get the most recent records
    recent_data = recent_data.tail(5)
    
    # Create features
    features = {
        'last_alarm_type': recent_data['alarm_type_code'].iloc[-1] if len(recent_data) > 0 else None,
        'last_severity': recent_data['severity_code'].iloc[-1] if len(recent_data) > 0 else None,
        'last_temperature': recent_data['temperature'].iloc[-1] if len(recent_data) > 0 else None,
        'hour_of_day': current_time.hour if hasattr(current_time, 'hour') else 0,
        'day_of_week': current_time.dayofweek if hasattr(current_time, 'dayofweek') else 0
    }
    
    # Add previous alarm types if available
    for i in range(min(3, len(recent_data))):
        features[f'prev_alarm_{i+1}'] = recent_data['alarm_type_code'].iloc[-(i+1)] if i < len(recent_data) else -1
    
    # Convert to DataFrame
    features_df = pd.DataFrame([features])
    
    return features_df