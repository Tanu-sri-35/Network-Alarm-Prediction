import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_alarm_distribution(df):
    """Plot the distribution of alarms by type"""
    # Count alarms by type
    alarm_counts = df['alarm_type'].value_counts().reset_index()
    alarm_counts.columns = ['Alarm Type', 'Count']
    
    # Create bar plot
    fig = px.bar(
        alarm_counts, 
        x='Alarm Type', 
        y='Count',
        color='Count',
        color_continuous_scale='Teal',
        title='Distribution of Alarm Types'
    )
    
    fig.update_layout(
        xaxis_title='Alarm Type',
        yaxis_title='Count',
        coloraxis_showscale=False,
        height=500
    )
    
    return fig

def plot_alarm_severity(df):
    """Plot the distribution of alarms by severity"""
    # Count alarms by severity
    severity_counts = df['severity'].value_counts().reset_index()
    severity_counts.columns = ['Previous Severity', 'Count']
    
    # Create ordered list of severities
    severity_order = ["critical", "major", "minor", "warning", "indeterminate"]
    
    # Define colors for each severity
    severity_colors = {
        "Critical": "#FF5252",
        "Major": "#FF9800",
        "Minor": "#FFC107",
        "Warning": "#8BC34A",
        "Info": "#2196F3"
    }
    
    # Filter and order data
    severity_counts = severity_counts[severity_counts['Previous Severity'].isin(severity_order)]
    severity_counts['Previous Severity'] = pd.Categorical(
        severity_counts['Previous Severity'], 
        categories=severity_order, 
        ordered=True
    )
    severity_counts = severity_counts.sort_values('Previous Severity')
    
    # Create pie chart
    fig = px.pie(
        severity_counts,
        values='Count',
        names='Previous Severity',
        color='Previous Severity',
        color_discrete_map=severity_colors,
        title='Alarm Severity Distribution'
    )
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(height=500)
    
    return fig

def plot_alarm_trends(df):
    """Plot alarm trends over time"""
    # Convert timestamp to datetime if needed
    if df['timestamp'].dtype != 'datetime64[ns]':
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Resample by day
    df_resampled = df.copy()
    df_resampled['date'] = df_resampled['timestamp'].dt.date
    daily_counts = df_resampled.groupby(['date', 'alarm_type']).size().reset_index(name='count')
    
    # Create line plot
    fig = px.line(
        daily_counts,
        x='date',
        y='count',
        color='alarm_type',
        title='Alarm Trends Over Time',
        line_shape='spline'
    )
    
    fig.update_layout(
        xaxis_title='Date',
        yaxis_title='Number of Alarms',
        height=500
    )
    
    return fig

def plot_site_heatmap(df):
    """Create a heatmap of alarms by site and type"""
    # Create pivot table
    pivot_data = df.pivot_table(
        index='site_id',
        columns='alarm_type',
        values='timestamp',
        aggfunc='count',
        fill_value=0
    )
    
    # Create heatmap
    fig = px.imshow(
        pivot_data,
        color_continuous_scale='Viridis',
        title='Alarm Heatmap by Site and Type'
    )
    
    fig.update_layout(
        xaxis_title='Alarm Type',
        yaxis_title='Site ID',
        height=600
    )
    
    return fig

def plot_correlation_matrix(df, numeric_cols=None):
    """Plot correlation matrix of numeric features"""
    # Select numeric columns if not specified
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Calculate correlation matrix
    corr_matrix = df[numeric_cols].corr()
    
    # Create heatmap
    fig = px.imshow(
        corr_matrix,
        color_continuous_scale='RdBu_r',
        title='Feature Correlation Matrix',
        zmin=-1,
        zmax=1
    )
    
    fig.update_layout(height=600)
    
    return fig

def plot_model_performance(y_true, y_pred, labels=None):
    """Plot confusion matrix for model evaluation"""
    from sklearn.metrics import confusion_matrix
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Create heatmap
    fig = px.imshow(
        cm,
        labels=dict(x="Predicted", y="Actual"),
        x=labels if labels else list(range(len(cm))),
        y=labels if labels else list(range(len(cm))),
        text_auto=True,
        color_continuous_scale='Blues',
        title='Confusion Matrix'
    )
    
    fig.update_layout(height=600)
    
    return fig

def plot_feature_importance(importance_df, top_n=10):
    """Plot feature importance from model"""
    # Sort and select top features
    importance_df = importance_df.sort_values('importance', ascending=False).head(top_n)
    
    # Create bar plot
    fig = px.bar(
        importance_df,
        x='importance',
        y='feature',
        orientation='h',
        color='importance',
        color_continuous_scale='Viridis',
        title=f'Top {top_n} Feature Importance'
    )
    
    fig.update_layout(
        xaxis_title='Importance',
        yaxis_title='Feature',
        height=500
    )
    
    return fig

def plot_hotspot_map(df, size_col='count', color_col='severity_code'):
    """Create a scatter plot showing alarm hotspots"""
    # Generate synthetic coordinates for sites (for visualization)
    sites = df['site_id'].unique()
    np.random.seed(42)  # For reproducibility
    
    # Create mapping of sites to coordinates
    site_coords = {}
    for site in sites:
        site_coords[site] = (np.random.uniform(0, 100), np.random.uniform(0, 100))
    
    # Prepare data for plotting
    plot_data = df.groupby('site_id').agg({
        color_col: 'mean',
        'alarm_type': 'count'
    }).reset_index()
    
    plot_data['x'] = plot_data['site_id'].map(lambda x: site_coords[x][0])
    plot_data['y'] = plot_data['site_id'].map(lambda x: site_coords[x][1])
    plot_data['count'] = plot_data['alarm_type']
    
    # Create scatter plot
    fig = px.scatter(
        plot_data,
        x='x',
        y='y',
        size='count',
        color=color_col,
        hover_name='site_id',
        color_continuous_scale='Inferno',
        size_max=40,
        title='Network Alarm Hotspots'
    )
    
    fig.update_layout(
        xaxis_title='',
        yaxis_title='',
        xaxis=dict(showticklabels=False),
        yaxis=dict(showticklabels=False),
        height=600
    )
    
    # Add some styling to make it look like a network map
    fig.update_layout(
        plot_bgcolor='rgba(240, 240, 240, 0.5)',
        paper_bgcolor='white'
    )
    
    # Add some random "connections" between sites
    np.random.seed(42)
    num_connections = len(sites) * 2
    
    for _ in range(num_connections):
        site1, site2 = np.random.choice(list(site_coords.keys()), 2, replace=False)
        x0, y0 = site_coords[site1]
        x1, y1 = site_coords[site2]
        
        fig.add_shape(
            type="line",
            x0=x0, y0=y0,
            x1=x1, y1=y1,
            line=dict(color="rgba(100, 100, 100, 0.2)", width=1)
        )
    
    return fig

def plot_prediction_timeline(predictions, actual=None):
    """Plot timeline of predicted alarms with actual if available"""
    # Create figure
    fig = go.Figure()
    
    # Add predicted alarms
    for i, pred in enumerate(predictions):
        fig.add_trace(go.Scatter(
            x=[pred['timestamp']],
            y=[1],
            mode='markers',
            marker=dict(
                size=15,
                color=pred.get('color', 'blue'),
                symbol='circle'
            ),
            name=pred['alarm_type'],
            hovertext=f"Predicted: {pred['alarm_type']}<br>Probability: {pred['probability']:.2f}"
        ))
    
    # Add actual alarms if available
    if actual:
        for i, act in enumerate(actual):
            fig.add_trace(go.Scatter(
                x=[act['timestamp']],
                y=[0],
                mode='markers',
                marker=dict(
                    size=15,
                    color=act.get('color', 'red'),
                    symbol='square'
                ),
                name=act['alarm_type'],
                hovertext=f"Actual: {act['alarm_type']}"
            ))
    
    # Update layout
    fig.update_layout(
        title='Alarm Prediction Timeline',
        xaxis_title='Time',
        yaxis=dict(
            showticklabels=False,
            showgrid=False,
            zeroline=False
        ),
        height=300,
        showlegend=True
    )
    
    # Add annotation for axis labels
    fig.add_annotation(
        x=0.01,
        y=1,
        xref="paper",
        yref="y",
        text="Predicted",
        showarrow=False
    )
    
    if actual:
        fig.add_annotation(
            x=0.01,
            y=0,
            xref="paper",
            yref="y",
            text="Actual",
            showarrow=False
        )
    
    return fig

def plot_alarm_calendar(df):
    """Create a calendar heatmap of alarms by day"""
    # Ensure timestamp is datetime
    if df['timestamp'].dtype != 'datetime64[ns]':
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Extract date components
    df['date'] = df['timestamp'].dt.date
    df['day'] = df['timestamp'].dt.day
    df['month'] = df['timestamp'].dt.month
    
    # Count alarms by date
    daily_counts = df.groupby(['date', 'month', 'day']).size().reset_index(name='count')
    
    # Create a list of all possible days for the date range
    min_date = df['date'].min()
    max_date = df['date'].max()
    all_dates = pd.date_range(min_date, max_date).date
    all_days = pd.DataFrame({'date': all_dates})
    all_days['month'] = [d.month for d in all_days['date']]
    all_days['day'] = [d.day for d in all_days['date']]
    
    # Merge to include days with no alarms
    daily_counts = pd.merge(
        all_days, 
        daily_counts, 
        on=['date', 'month', 'day'], 
        how='left'
    ).fillna(0)
    
    # Create calendar heatmap
    fig = px.density_heatmap(
        daily_counts,
        x='day',
        y='month',
        z='count',
        color_continuous_scale='YlOrRd',
        title='Alarm Calendar Heatmap'
    )
    
    # Update layout
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    fig.update_layout(
        xaxis_title='Day of Month',
        yaxis_title='Month',
        yaxis=dict(
            tickmode='array',
            tickvals=list(range(1, 13)),
            ticktext=month_names
        ),
        height=500
    )
    
    return fig