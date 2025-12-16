"""
Risk Rating Classification Module

Uses pre-trained TensorFlow model to predict RiskRating for new student data.
"""

import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder

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
        df: DataFrame with student data (including Name, Gender, GradeLevel, and question responses)
    
    Returns:
        Dictionary with predictions and model info
    """
    print("Predicting risk ratings for ASSI-A data...")
    
    # Load pre-trained model
    model = load_risk_rating_model()
    
    # Prepare features - match training data preprocessing exactly
    df_features = df.copy()
    
    # Standardize column names - Grade vs GradeLevel
    if 'Grade' in df_features.columns and 'GradeLevel' not in df_features.columns:
        df_features = df_features.rename(columns={'Grade': 'GradeLevel'})
    
    # Drop non-feature columns - match training preprocessing
    # Training drops: StudentNumber and RiskRating
    cols_to_drop = ['StudentNumber', 'RiskRating', 'Name']  # Drop Name if present (for compatibility)
    X = df_features.drop(columns=[col for col in cols_to_drop if col in df_features.columns])
    
    # Encode Gender using LabelEncoder (matching training preprocessing)
    if 'Gender' in X.columns and X['Gender'].dtype == object:
        label_encoder_gender = LabelEncoder()
        X['Gender'] = label_encoder_gender.fit_transform(X['Gender'])
    
    # Convert to numpy array for prediction
    X_array = X.values.astype(np.float32)
    
    # Predict
    predictions_prob = model.predict(X_array, verbose=0)
    predictions = np.argmax(predictions_prob, axis=1)
    
    # Get prediction confidence (max probability)
    confidence = np.max(predictions_prob, axis=1)
    
    # Map predictions to risk levels (assuming standard risk rating classes)
    # If model outputs numeric predictions, map them to labels
    risk_levels = ['Low', 'Medium', 'High']  # Default risk levels
    predictions_labels = [risk_levels[pred] if pred < len(risk_levels) else f'Level_{pred}' for pred in predictions]
    
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