import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta
import time
import random

def generate_next_alarm(df, current_time=None, site_id=None, model=None, alarm_types=None):
    """
    Generate the next alarm for simulation
    
    Parameters:
    -----------
    df : DataFrame
        Historical alarm data
    current_time : datetime, optional
        Current simulation time
    site_id : str, optional
        Site to generate alarm for
    model : trained model, optional
        Model to use for prediction
    alarm_types : list or dict, optional
        Mapping from alarm code to alarm type
        
    Returns:
    --------
    dict
        Generated alarm
    """
    from utils.data_processing import get_next_alarm_prediction_features
    
    # Set current time if not provided
    if current_time is None:
        if df['timestamp'].dtype != 'datetime64[ns]':
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        current_time = df['timestamp'].max() + timedelta(minutes=random.randint(30, 120))
    
    # Select a random site if not provided
    if site_id is None:
        site_id = random.choice(df['site_id'].unique())
    
    # If we have a model, use it for prediction
    if model is not None and alarm_types is not None:
        from models.model_training import predict_next_alarm
        
        # Prepare features
        features = get_next_alarm_prediction_features(df, current_time, site_id)
        
        # Make prediction
        prediction = predict_next_alarm(model, features, alarm_types)
        
        # Create alarm
        alarm = {
            'timestamp': current_time,
            'alarm_type': prediction['alarm_type'],
            'site_id': site_id,
            'severity': random.choice(['Critical', 'Major', 'Minor', 'Warning', 'Info']),
            'temperature': round(random.uniform(18.0, 35.0), 1),
            'duration_minutes': random.randint(1, 180),
            'predicted': True,
            'probability': prediction['probability']
        }
    
    # Otherwise, generate a random alarm
    else:
        alarm_types_list = df['alarm_type'].unique()
        severity_list = df['severity'].unique()
        
        alarm = {
            'timestamp': current_time,
            'alarm_type': random.choice(alarm_types_list),
            'site_id': site_id,
            'severity': random.choice(severity_list),
            'temperature': round(random.uniform(18.0, 35.0), 1),
            'duration_minutes': random.randint(1, 180),
            'predicted': False,
            'probability': 0.0
        }
    
    return alarm

def run_simulation(df, model=None, alarm_types=None, duration_minutes=30, time_factor=10, callback=None):
    """
    Run a real-time simulation of alarm generation and prediction
    
    Parameters:
    -----------
    df : DataFrame
        Historical alarm data
    model : trained model, optional
        Model to use for prediction
    alarm_types : list or dict, optional
        Mapping from alarm code to alarm type
    duration_minutes : int
        Duration of simulation in minutes
    time_factor : int
        Factor to speed up simulation time
    callback : function, optional
        Function to call with each new alarm
        
    Returns:
    --------
    list
        List of generated alarms
    """
    from models.model_training import predict_next_alarm
    from utils.data_processing import get_next_alarm_prediction_features
    
    # Ensure timestamp is datetime
    if df['timestamp'].dtype != 'datetime64[ns]':
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Set up simulation
    start_time = datetime.now()
    end_time = start_time + timedelta(minutes=duration_minutes)
    
    current_time = start_time
    
    # List to store generated alarms
    generated_alarms = []
    predicted_alarms = []
    
    # Simulation loop
    while current_time < end_time:
        # Generate a random site
        site_id = random.choice(df['site_id'].unique())
        
        # Generate actual alarm
        actual_alarm = generate_next_alarm(df, current_time, site_id)
        
        # Make prediction if model is available
        if model is not None and alarm_types is not None:
            # Prepare features
            features = get_next_alarm_prediction_features(df, current_time, site_id)
            
            # Make prediction
            prediction = predict_next_alarm(model, features, alarm_types)
            
            # Create predicted alarm
            pred_time = current_time + timedelta(minutes=random.randint(5, 30))
            predicted_alarm = {
                'timestamp': pred_time,
                'alarm_type': prediction['alarm_type'],
                'site_id': site_id,
                'severity': actual_alarm['severity'],
                'temperature': actual_alarm['temperature'],
                'duration_minutes': actual_alarm['duration_minutes'],
                'predicted': True,
                'probability': prediction['probability']
            }
            
            predicted_alarms.append(predicted_alarm)
        
        # Add actual alarm to generated list
        generated_alarms.append(actual_alarm)
        
        # Call callback if provided
        if callback:
            callback(actual_alarm, predicted_alarms[-1] if predicted_alarms else None)
        
        # Advance time
        wait_time = random.randint(5, 15) / time_factor
        time.sleep(wait_time)
        current_time = datetime.now()
    
    return generated_alarms, predicted_alarms

def create_simulation_widget(df, model=None, alarm_types=None):
    """
    Create a simulation widget for Streamlit
    
    Parameters:
    -----------
    df : DataFrame
        Historical alarm data
    model : trained model, optional
        Model to use for prediction
    alarm_types : list or dict, optional
        Mapping from alarm code to alarm type
        
    Returns:
    --------
    None
    """
    from utils.visualization import plot_prediction_timeline
    
    # Set up containers
    simulation_container = st.empty()
    timeline_container = st.empty()
    metrics_container = st.empty()
    
    # List to store alarms
    all_alarms = []
    all_predictions = []
    
    # Function to update UI with new alarm
    def update_ui(actual_alarm, predicted_alarm=None):
        with simulation_container.container():
            # Display latest alarm
            st.subheader("Latest Network Alarm")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Alarm Type", actual_alarm['alarm_type'])
                
            with col2:
                st.metric("Site ID", actual_alarm['site_id'])
                
            with col3:
                st.metric("Severity", actual_alarm['severity'])
            
            # Add to lists
            all_alarms.append(actual_alarm)
            
            if predicted_alarm:
                all_predictions.append(predicted_alarm)
                
                # Display prediction
                st.subheader("Prediction for Next Alarm")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Predicted Alarm", predicted_alarm['alarm_type'])
                    
                with col2:
                    st.metric("Probability", f"{predicted_alarm['probability']:.2f}")
                    
                with col3:
                    st.metric("Expected Time", predicted_alarm['timestamp'].strftime("%H:%M:%S"))
        
        # Update timeline
        with timeline_container.container():
            # Convert alarms to format for timeline
            timeline_predictions = [
                {
                    'timestamp': p['timestamp'],
                    'alarm_type': p['alarm_type'],
                    'probability': p['probability'],
                    'color': {'Critical': 'red', 'Major': 'orange', 'Minor': 'yellow', 
                              'Warning': 'green', 'Info': 'blue'}.get(p['severity'], 'gray')
                }
                for p in all_predictions
            ]
            
            timeline_actuals = [
                {
                    'timestamp': a['timestamp'],
                    'alarm_type': a['alarm_type'],
                    'color': {'Critical': 'red', 'Major': 'orange', 'Minor': 'yellow', 
                              'Warning': 'green', 'Info': 'blue'}.get(a['severity'], 'gray')
                }
                for a in all_alarms
            ]
            
            # Plot timeline
            if timeline_predictions:
                st.subheader("Alarm Prediction Timeline")
                fig = plot_prediction_timeline(timeline_predictions, timeline_actuals)
                st.plotly_chart(fig, use_container_width=True)
        
        # Update metrics
        with metrics_container.container():
            if all_predictions and all_alarms:
                # Calculate accuracy
                correct_predictions = sum(
                    1 for p, a in zip(all_predictions[:len(all_alarms)], all_alarms) 
                    if p['alarm_type'] == a['alarm_type']
                )
                
                accuracy = correct_predictions / len(all_alarms) if all_alarms else 0
                
                # Display metrics
                st.subheader("Simulation Metrics")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Alarms Generated", len(all_alarms))
                    
                with col2:
                    st.metric("Prediction Accuracy", f"{accuracy:.2f}")
                    
                with col3:
                    avg_prob = sum(p['probability'] for p in all_predictions) / len(all_predictions) if all_predictions else 0
                    st.metric("Avg. Prediction Confidence", f"{avg_prob:.2f}")
    
    # Run simulation
    if st.button("Start Simulation"):
        with st.spinner("Running simulation..."):
            run_simulation(df, model, alarm_types, duration_minutes=5, time_factor=20, callback=update_ui)
        
        st.success("Simulation completed!")