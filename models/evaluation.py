import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)
import streamlit as st

def evaluate_classification_model(y_true, y_pred, class_names=None):
    """
    Evaluate a classification model and return metrics
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    class_names : list, optional
        Names of the classes
        
    Returns:
    --------
    dict
        Dictionary with evaluation metrics
    """
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    
    try:
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
    except:
        precision = 0
        recall = 0
        f1 = 0
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Generate classification report
    report = classification_report(y_true, y_pred, output_dict=True)
    
    # Return results
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm,
        'classification_report': report
    }

def plot_confusion_matrix(cm, class_names=None):
    """
    Plot confusion matrix
    
    Parameters:
    -----------
    cm : array-like
        Confusion matrix
    class_names : list, optional
        Names of the classes
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure with the confusion matrix plot
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot confusion matrix
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=class_names if class_names else 'auto',
        yticklabels=class_names if class_names else 'auto',
        ax=ax
    )
    
    # Set labels
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')
    
    # Return figure
    return fig

def plot_roc_curve(y_true, y_score, class_idx=None):
    """
    Plot ROC curve
    
    Parameters:
    -----------
    y_true : array-like
        True labels (one-hot encoded for multi-class)
    y_score : array-like
        Predicted probabilities
    class_idx : int, optional
        Index of the class to plot
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure with the ROC curve plot
    """
    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_score = np.array(y_score)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # For binary classification
    if len(y_score.shape) == 1 or y_score.shape[1] == 2:
        # Calculate ROC curve
        if len(y_score.shape) == 2:
            # Use probability of positive class
            y_score = y_score[:, 1]
        
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        
        # Plot ROC curve
        ax.plot(fpr, tpr, lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    
    # For multi-class classification
    else:
        # If class index is specified
        if class_idx is not None:
            # Calculate ROC curve for the specified class
            fpr, tpr, _ = roc_curve(
                (y_true == class_idx).astype(int), 
                y_score[:, class_idx]
            )
            roc_auc = auc(fpr, tpr)
            
            # Plot ROC curve
            ax.plot(fpr, tpr, lw=2, label=f'Class {class_idx} (area = {roc_auc:.2f})')
        
        # If class index is not specified, plot for all classes
        else:
            for i in range(y_score.shape[1]):
                # Calculate ROC curve for each class
                fpr, tpr, _ = roc_curve(
                    (y_true == i).astype(int), 
                    y_score[:, i]
                )
                roc_auc = auc(fpr, tpr)
                
                # Plot ROC curve
                ax.plot(fpr, tpr, lw=2, label=f'Class {i} (area = {roc_auc:.2f})')
    
    # Add diagonal line
    ax.plot([0, 1], [0, 1], 'k--', lw=2)
    
    # Set labels and title
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax.legend(loc="lower right")
    
    # Return figure
    return fig

def plot_metric_curves(train_metrics, val_metrics, metric_name='accuracy'):
    """
    Plot training and validation metrics
    
    Parameters:
    -----------
    train_metrics : list
        List of training metrics
    val_metrics : list
        List of validation metrics
    metric_name : str
        Name of the metric
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure with the metric curves plot
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot metrics
    epochs = range(1, len(train_metrics) + 1)
    ax.plot(epochs, train_metrics, 'b-', label=f'Training {metric_name}')
    ax.plot(epochs, val_metrics, 'r-', label=f'Validation {metric_name}')
    
    # Set labels and title
    ax.set_xlabel('Epochs')
    ax.set_ylabel(metric_name.capitalize())
    ax.set_title(f'Training and Validation {metric_name.capitalize()}')
    ax.legend()
    
    # Return figure
    return fig

def compare_models(model_results_list):
    """
    Compare multiple models based on their performance metrics
    
    Parameters:
    -----------
    model_results_list : list of dict
        List of dictionaries containing model results
        
    Returns:
    --------
    DataFrame
        DataFrame with model comparison
    """
    # Extract model names and metrics
    comparison_data = []
    
    for results in model_results_list:
        model_name = results.get('model_type', 'Unknown')
        
        # Extract metrics
        comparison_data.append({
            'Model': model_name,
            'Accuracy': results.get('accuracy', 0),
            'Precision': results.get('precision', 0),
            'Recall': results.get('recall', 0),
            'F1 Score': results.get('f1_score', 0)
        })
    
    # Create DataFrame
    comparison_df = pd.DataFrame(comparison_data)
    
    return comparison_df

def calculate_prediction_success_rate(predictions, actual):
    """
    Calculate success rate of predictions
    
    Parameters:
    -----------
    predictions : list of dict
        List of prediction dictionaries
    actual : list of dict
        List of actual alarm dictionaries
        
    Returns:
    --------
    float
        Success rate (0-1)
    """
    if not predictions or not actual:
        return 0.0
    
    # Count correct predictions
    correct_count = 0
    
    for pred, act in zip(predictions, actual):
        if pred.get('alarm_type') == act.get('alarm_type'):
            correct_count += 1
    
    # Calculate success rate
    success_rate = correct_count / len(predictions)
    
    return success_rate

def calculate_time_to_alarm(predictions, actual):
    """
    Calculate average time difference between prediction and actual alarm
    
    Parameters:
    -----------
    predictions : list of dict
        List of prediction dictionaries with timestamp
    actual : list of dict
        List of actual alarm dictionaries with timestamp
        
    Returns:
    --------
    float
        Average time difference in minutes
    """
    if not predictions or not actual:
        return 0.0
    
    # Calculate time differences
    time_diffs = []
    
    for pred, act in zip(predictions, actual):
        # Get timestamps
        pred_time = pd.to_datetime(pred.get('timestamp'))
        act_time = pd.to_datetime(act.get('timestamp'))
        
        # Calculate difference in minutes
        diff = (act_time - pred_time).total_seconds() / 60
        time_diffs.append(diff)
    
    # Calculate average
    avg_diff = sum(time_diffs) / len(time_diffs)
    
    return avg_diff