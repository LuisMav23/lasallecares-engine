"""
Risk Rating Classification Module

Uses pre-trained TensorFlow model to predict RiskRating for new student data.
"""

import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Global cache for loaded models
_LOADED_MODELS = {}


def load_risk_rating_model(model_path='models/risk_rating_nn_model.h5'):
    """Load the pre-trained risk rating model."""
    global _LOADED_MODELS
    
    # Return cached model if already loaded
    if 'risk_rating' in _LOADED_MODELS:
        return _LOADED_MODELS['risk_rating']
    
    print(f"Loading risk rating model from {model_path}...")
    
    # Load TensorFlow model
    model = load_model(model_path)
    
    # Cache the loaded model
    _LOADED_MODELS['risk_rating'] = model
    
    print("Risk rating model loaded successfully!")
    return model


def predict_risk_rating(df: pd.DataFrame) -> dict:
    """
    Predict risk rating for new student data using pre-trained model.
    Only supports ASSI-A form type.
    
    Args:
        df: DataFrame with student data (including Gender, GradeLevel, and question responses)
    
    Returns:
        Dictionary with predictions and model info
    """
    print("Predicting risk ratings for ASSI-A data...")
    
    # Load pre-trained model
    model = load_risk_rating_model()
    
    # Prepare features - match training data preprocessing exactly
    # Drop 'StudentNumber' and 'RiskRating' (if present)
    X_new = df.drop(['StudentNumber', 'RiskRating'], axis=1, errors='ignore')
    
    # Standardize column names - Grade vs GradeLevel
    if 'Grade' in X_new.columns and 'GradeLevel' not in X_new.columns:
        X_new = X_new.rename(columns={'Grade': 'GradeLevel'})
    
    # Encode Gender using LabelEncoder
    label_encoder_gender = LabelEncoder()
    if 'Gender' in X_new.columns:
        X_new['Gender'] = label_encoder_gender.fit_transform(X_new['Gender'])
        print("Encoded Gender using LabelEncoder")
    
    # Scale features using StandardScaler
    scaler = StandardScaler()
    X_new_scaled = scaler.fit_transform(X_new)
    print(f"Scaled features using StandardScaler. Shape: {X_new_scaled.shape}")
    
    # Make predictions using the loaded Neural Network model
    predictions_proba = model.predict(X_new_scaled, verbose=0)
    predictions_encoded = np.argmax(predictions_proba, axis=1)
    
    # Map predictions to risk levels (assuming standard risk rating classes)
    # Since we don't have label_encoder_target, we'll use default mapping
    risk_levels = ['Low', 'Medium', 'High']
    predictions_labels = [risk_levels[pred] if pred < len(risk_levels) else f'Level_{pred}' for pred in predictions_encoded]
    
    # Get prediction confidence (max probability)
    confidence = np.max(predictions_proba, axis=1)
    
    # Count predictions by risk level
    unique, counts = np.unique(predictions_labels, return_counts=True)
    risk_distribution = dict(zip(unique, counts.tolist()))
    
    print(f"Predictions complete. Risk distribution: {risk_distribution}")
    
    return {
        'model_name': 'Neural Network (Pre-trained)',
        'predictions': predictions_labels,
        'confidence': confidence.tolist(),
        'risk_distribution': risk_distribution,
        'classes': risk_levels
    }