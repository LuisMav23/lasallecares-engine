"""
Risk Rating Classification Module

Uses pre-trained TensorFlow model to predict RiskRating for new student data.
"""

import os
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model

# Global cache for loaded models
_LOADED_MODELS = {}


def load_risk_rating_model(model_path='models/risk_rating'):
    """Load the pre-trained risk rating model and related objects."""
    global _LOADED_MODELS
    
    # Return cached models if already loaded
    if 'risk_rating' in _LOADED_MODELS:
        return _LOADED_MODELS['risk_rating']
    
    print(f"Loading risk rating model from {model_path}...")
    
    # Load TensorFlow model
    model = load_model(os.path.join(model_path, 'risk_rating_model.h5'))
    
    # Load scaler
    with open(os.path.join(model_path, 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    
    # Load label encoder
    with open(os.path.join(model_path, 'label_encoder.pkl'), 'rb') as f:
        label_encoder = pickle.load(f)
    
    # Load feature names
    with open(os.path.join(model_path, 'feature_names.pkl'), 'rb') as f:
        feature_names = pickle.load(f)
    
    models = {
        'model': model,
        'scaler': scaler,
        'label_encoder': label_encoder,
        'feature_names': feature_names
    }
    
    # Cache the loaded models
    _LOADED_MODELS['risk_rating'] = models
    
    print("Risk rating model loaded successfully!")
    return models


def predict_risk_rating(df: pd.DataFrame, form_type='ASSI-A') -> dict:
    """
    Predict risk rating for new student data using pre-trained model.
    
    Args:
        df: DataFrame with student data (including Name, Gender, GradeLevel, and question responses)
        form_type: Type of form ('ASSI-A' or 'ASSI-C')
    
    Returns:
        Dictionary with predictions and model info
    """
    print("Predicting risk ratings...")
    
    # Load pre-trained model
    models = load_risk_rating_model()
    model = models['model']
    scaler = models['scaler']
    label_encoder = models['label_encoder']
    feature_names = models['feature_names']
    
    # Prepare features - match training data format
    df_features = df.copy()
    
    # Standardize column names - Grade vs GradeLevel
    if 'Grade' in df_features.columns and 'GradeLevel' not in df_features.columns:
        df_features = df_features.rename(columns={'Grade': 'GradeLevel'})
    
    # Drop non-feature columns (Name is not used in model)
    cols_to_drop = ['Name', 'RiskRating']  # Also drop RiskRating if present (from labeled data)
    X = df_features.drop(columns=[col for col in cols_to_drop if col in df_features.columns])
    
    # Encode Gender if needed
    if 'Gender' in X.columns and X['Gender'].dtype == object:
        X['Gender'] = X['Gender'].map({'Female': 0, 'Male': 1})
    
    # Handle ASSI-C conversion if needed
    if form_type == 'ASSI-C':
        answer_map = {'Never': 0, 'Sometimes': 1, 'Often': 2}
        for col in X.columns:
            if col not in ['Gender', 'Grade', 'GradeLevel']:
                if X[col].dtype == object:
                    X[col] = X[col].map(answer_map)
    
    # Ensure columns match training data
    # Reorder columns to match feature_names
    missing_cols = [col for col in feature_names if col not in X.columns]
    if missing_cols:
        print(f"Warning: Missing columns: {missing_cols}")
        # Add missing columns with zeros
        for col in missing_cols:
            X[col] = 0
    
    # Select only the features used in training, in the correct order
    X = X[feature_names]
    
    # Scale features
    X_scaled = scaler.transform(X)
    
    # Predict
    predictions_prob = model.predict(X_scaled, verbose=0)
    predictions = np.argmax(predictions_prob, axis=1)
    
    # Decode predictions
    predictions_labels = label_encoder.inverse_transform(predictions)
    
    # Get prediction confidence (max probability)
    confidence = np.max(predictions_prob, axis=1)
    
    # Count predictions by risk level
    unique, counts = np.unique(predictions_labels, return_counts=True)
    risk_distribution = dict(zip(unique, counts.tolist()))
    
    print(f"Predictions complete. Risk distribution: {risk_distribution}")
    
    return {
        'model_name': 'Neural Network (Pre-trained)',
        'predictions': predictions_labels.tolist(),
        'confidence': confidence.tolist(),
        'risk_distribution': risk_distribution,
        'classes': label_encoder.classes_.tolist()
    }