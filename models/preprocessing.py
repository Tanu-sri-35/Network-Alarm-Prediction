import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import os

def create_preprocessing_pipeline(categorical_features, numerical_features):
    """
    Create a preprocessing pipeline for the alarm data
    
    Parameters:
    -----------
    categorical_features : list
        List of categorical column names
    numerical_features : list
        List of numerical column names
        
    Returns:
    --------
    sklearn.pipeline.Pipeline
        Preprocessing pipeline
    """
    # Create transformers for different column types
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    numerical_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    # Combine transformers into a preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical_features),
            ('num', numerical_transformer, numerical_features)
        ],
        remainder='drop'  # Drop other columns
    )
    
    # Create preprocessing pipeline
    preprocessing_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor)
    ])
    
    return preprocessing_pipeline

def prepare_features_target(df, target_col='alarm_type', feature_cols=None):
    """
    Prepare features and target for model training
    
    Parameters:
    -----------
    df : DataFrame
        Input DataFrame with processed data
    target_col : str
        Name of the target column
    feature_cols : list, optional
        List of feature columns to use
        
    Returns:
    --------
    X : DataFrame
        Features for model training
    y : Series
        Target variable for model training
    """
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    # Set default feature columns if not provided
    if feature_cols is None:
        # Use all numeric columns and encoded categorical columns
        feature_cols = [col for col in df.columns if col.endswith('_code') 
                        or df[col].dtype in ['int64', 'float64']]
        
        # Remove the target column if it's in the feature columns
        if f"{target_col}_code" in feature_cols:
            feature_cols.remove(f"{target_col}_code")
            
        # Remove timestamp from features
        if 'timestamp' in feature_cols:
            feature_cols.remove('timestamp')
    
    # Extract features and target
    X = df[feature_cols]
    
    # Use encoded version of target if available
    if f"{target_col}_code" in df.columns:
        y = df[f"{target_col}_code"]
    else:
        y = df[target_col]
    
    return X, y

def prepare_sequence_features(df, sequence_length=5):
    """
    Prepare sequence features for time series prediction
    
    Parameters:
    -----------
    df : DataFrame
        Input DataFrame with processed data, sorted by timestamp
    sequence_length : int
        Number of previous alarms to use as features
        
    Returns:
    --------
    X : numpy.ndarray
        Sequence features for model training
    y : numpy.ndarray
        Target variable (next alarm type) for model training
    """
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    # Use alarm_type_code as target
    if 'alarm_type_code' in df.columns:
        feature_col = 'alarm_type_code'
    else:
        # Encode alarm_type if not already encoded
        df['alarm_type_code'] = df['alarm_type'].astype('category').cat.codes
        feature_col = 'alarm_type_code'
    
    # Create sequences
    sequences = []
    targets = []
    
    for i in range(len(df) - sequence_length):
        # Extract sequence and target
        seq = df[feature_col].iloc[i:i+sequence_length].values
        target = df[feature_col].iloc[i+sequence_length]
        
        sequences.append(seq)
        targets.append(target)
    
    # Convert to numpy arrays
    X = np.array(sequences)
    y = np.array(targets)
    
    return X, y

def save_preprocessor(preprocessor, model_dir='models'):
    """
    Save the preprocessor to disk
    
    Parameters:
    -----------
    preprocessor : sklearn.pipeline.Pipeline
        Preprocessing pipeline to save
    model_dir : str
        Directory to save the preprocessor
    """
    # Create directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    # Save preprocessor
    joblib.dump(preprocessor, os.path.join(model_dir, 'preprocessor.joblib'))

def load_preprocessor(model_dir='models'):
    """
    Load the preprocessor from disk
    
    Parameters:
    -----------
    model_dir : str
        Directory where the preprocessor is saved
        
    Returns:
    --------
    sklearn.pipeline.Pipeline
        Loaded preprocessing pipeline
    """
    # Check if preprocessor exists
    preprocessor_path = os.path.join(model_dir, 'preprocessor.joblib')
    if not os.path.exists(preprocessor_path):
        return None
    
    # Load preprocessor
    return joblib.load(preprocessor_path)

def extract_feature_names(preprocessor):
    """
    Extract feature names from the preprocessor
    
    Parameters:
    -----------
    preprocessor : sklearn.pipeline.Pipeline
        Preprocessing pipeline
        
    Returns:
    --------
    list
        List of feature names after preprocessing
    """
    # Extract transformer from pipeline
    transformer = preprocessor.named_steps['preprocessor']
    
    # Get column names for categorical features
    cat_transformer = transformer.transformers_[0][1]
    cat_encoder = cat_transformer.named_steps['onehot']
    cat_features = transformer.transformers_[0][2]
    cat_feature_names = cat_encoder.get_feature_names_out(cat_features).tolist()
    
    # Get column names for numerical features
    num_features = transformer.transformers_[1][2]
    
    # Combine feature names
    feature_names = cat_feature_names + num_features
    
    return feature_names