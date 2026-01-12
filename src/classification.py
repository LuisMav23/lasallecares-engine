"""
Risk Rating Classification Module

Uses pre-trained TensorFlow model to predict RiskRating for new student data.
"""

import pandas as pd
import numpy as np
import joblib
import os
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Global cache for loaded models
_LOADED_MODELS = {}


def load_risk_rating_resources(model_path='models/risk_rating_nn_model.h5', transformers_dir='models/transform_models'):
    """Load the pre-trained risk rating model and transformers."""
    global _LOADED_MODELS
    
    # Return cached resources if already loaded
    if 'risk_rating_model' in _LOADED_MODELS and 'transformers' in _LOADED_MODELS:
        return _LOADED_MODELS['risk_rating_model'], _LOADED_MODELS['transformers']
    
    print(f"Loading risk rating model from {model_path}...")
    
    # Load TensorFlow model
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    model = load_model(model_path)
    _LOADED_MODELS['risk_rating_model'] = model
    print("Risk rating model loaded successfully!")

    print(f"Loading transformers from {transformers_dir}...")
    transformers = {}
    
    # Load transformers using joblib (sklearn saves with joblib internally)
    try:
        transformers['gender_encoder'] = joblib.load(os.path.join(transformers_dir, 'label_encoder_gender.pkl'))
        transformers['target_encoder'] = joblib.load(os.path.join(transformers_dir, 'label_encoder_target.pkl'))
        transformers['scaler'] = joblib.load(os.path.join(transformers_dir, 'scaler.pkl'))
            
        _LOADED_MODELS['transformers'] = transformers
        print("Transformers loaded successfully!")
    except FileNotFoundError as e:
        print(f"Error loading transformers: {e}")
        raise e

    return model, transformers


def predict_risk_rating(df: pd.DataFrame) -> dict:
    """
    Predict risk rating for new student data using pre-trained model and transformers.
    Only supports ASSI-A form type.
    
    Args:
        df: DataFrame with student data (including Gender, GradeLevel, and question responses)
    
    Returns:
        Dictionary with predictions and model info
    """
    print("Predicting risk ratings for ASSI-A data...")
    
    try:
        # Load pre-trained model and transformers
        model, transformers = load_risk_rating_resources()
        gender_encoder = transformers['gender_encoder']
        target_encoder = transformers['target_encoder']
        scaler = transformers['scaler']
        
        # Prepare features
        # Step 1: Drop StudentNumber and RiskRating (matching notebook)
        # We also drop 'Cluster' if present, as it's not part of the training data
        cols_to_drop = ['StudentNumber', 'RiskRating', 'Cluster', 'Name']
        X_new = df.drop([c for c in cols_to_drop if c in df.columns], axis=1)
        
        # Standardize column names - Grade vs GradeLevel
        if 'Grade' in X_new.columns and 'GradeLevel' not in X_new.columns:
            X_new = X_new.rename(columns={'Grade': 'GradeLevel'})
        
        print(f"Features shape before transformations: {X_new.shape}")
        
        # Step 2: Encode categorical variables (Gender) using LOADED encoder
        if 'Gender' in X_new.columns:
            X_new = X_new.copy()
            try:
                # Ensure input is what encoder expects (likely string)
                # If dataframe has numbers, convert to string? Or if encoder expects numbers?
                # Usually LabelEncoder fits on strings.
                X_new['Gender'] = gender_encoder.transform(X_new['Gender'])
                print("Encoded Gender using pre-trained LabelEncoder")
            except Exception as e:
                print(f"Error encoding Gender: {e}")
                # Try handling if it's already 0/1 but encoder expects 'Female'/'Male'
                # But typically we should receive the format that matches training.
                pass
        else:
            raise ValueError("Gender column is required but not found in the data")
            
        # Step 3: Scale features using LOADED scaler
        # We must ensure columns are in the same order as trained scaler
        if hasattr(scaler, 'feature_names_in_'):
            required_features = scaler.feature_names_in_
            
            # Add missing columns with 0
            for col in required_features:
                if col not in X_new.columns:
                    X_new[col] = 0
            
            # Reorder columns and drop extras
            X_new = X_new[required_features]
        
        X_new_scaled = scaler.transform(X_new)
        print(f"Scaled features using pre-trained StandardScaler. Shape: {X_new_scaled.shape}")
        
        # Step 4: Make predictions
        predictions_proba = model.predict(X_new_scaled, verbose=0)
        predictions_encoded = np.argmax(predictions_proba, axis=1)
        
        # Map predictions to risk levels using target_encoder
        predictions_labels = target_encoder.inverse_transform(predictions_encoded)
        
        # Get prediction confidence
        confidence = np.max(predictions_proba, axis=1)
        
        # Count predictions by risk level
        unique, counts = np.unique(predictions_labels, return_counts=True)
        risk_distribution = dict(zip(unique, counts.tolist()))
        
        print(f"Predictions complete. Risk distribution: {risk_distribution}")
        
        return {
            'model_name': 'Neural Network (Pre-trained)',
            'predictions': predictions_labels.tolist(),
            'confidence': confidence.tolist(),
            'risk_distribution': risk_distribution,
            'classes': target_encoder.classes_.tolist()
        }
        
    except Exception as e:
        print(f"Prediction error: {e}")
        import traceback
        traceback.print_exc()
        raise e