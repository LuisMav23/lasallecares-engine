"""
Model Training Script for Guidance System

This script trains and saves:
1. TensorFlow Neural Network for RiskRating classification
2. KMeans clustering model with PCA and StandardScaler

Run this script to generate pre-trained models from labeled data.
"""

import os
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def load_and_preprocess_labeled_data(file_path):
    """Load and preprocess the labeled dataset."""
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    
    print(f"Original shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Drop StudentNumber as it's just an ID
    if 'StudentNumber' in df.columns:
        df = df.drop(columns=['StudentNumber'])
    
    # Encode Gender to numeric
    df['Gender'] = df['Gender'].map({'Female': 0, 'Male': 1})
    
    # Remove any rows with missing values
    df = df.dropna()
    
    print(f"After preprocessing shape: {df.shape}")
    print(f"RiskRating distribution:\n{df['RiskRating'].value_counts()}")
    
    return df


def train_risk_rating_model(df, model_save_path):
    """Train TensorFlow model for RiskRating classification."""
    print("\n=== Training Risk Rating Classification Model ===")
    
    # Separate features and target
    X = df.drop(columns=['RiskRating'])
    y = df['RiskRating']
    
    # Encode target labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    num_classes = len(le.classes_)
    
    print(f"Classes: {le.classes_}")
    print(f"Number of classes: {num_classes}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert to categorical for multi-class
    y_train_cat = to_categorical(y_train, num_classes=num_classes)
    y_test_cat = to_categorical(y_test, num_classes=num_classes)
    
    # Build neural network
    model = Sequential([
        Dense(128, input_dim=X_train_scaled.shape[1], activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("Training model...")
    history = model.fit(
        X_train_scaled, y_train_cat,
        validation_split=0.2,
        epochs=100,
        batch_size=32,
        verbose=1
    )
    
    # Evaluate
    y_pred_prob = model.predict(X_test_scaled, verbose=0)
    y_pred = np.argmax(y_pred_prob, axis=1)
    
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
    matrix = confusion_matrix(y_test, y_pred)
    
    print(f"\nTest Accuracy: {accuracy:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    print(f"\nConfusion Matrix:\n{matrix}")
    
    # Save model and related objects
    print(f"\nSaving model to {model_save_path}...")
    os.makedirs(model_save_path, exist_ok=True)
    
    # Save TensorFlow model
    model.save(os.path.join(model_save_path, 'risk_rating_model.h5'))
    
    # Save scaler and label encoder
    with open(os.path.join(model_save_path, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    
    with open(os.path.join(model_save_path, 'label_encoder.pkl'), 'wb') as f:
        pickle.dump(le, f)
    
    # Save feature names for consistency
    with open(os.path.join(model_save_path, 'feature_names.pkl'), 'wb') as f:
        pickle.dump(X.columns.tolist(), f)
    
    print("Risk Rating model saved successfully!")
    
    return model, scaler, le, accuracy, report, matrix


def train_clustering_model(df, model_save_path):
    """Train KMeans clustering model with PCA."""
    print("\n=== Training Clustering Model ===")
    
    # Drop RiskRating for unsupervised clustering
    X = df.drop(columns=['RiskRating'])
    
    print(f"Features shape: {X.shape}")
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply PCA
    pca = PCA()
    pca.fit(X_scaled)
    eigenvalues = pca.explained_variance_
    optimal_pc = np.sum(eigenvalues > 1)  # Kaiser criterion
    
    print(f"Optimal number of principal components: {optimal_pc}")
    
    # Apply PCA with optimal components
    pca = PCA(n_components=optimal_pc)
    X_pca = pca.fit_transform(X_scaled)
    
    # Find optimal K using elbow method
    distortions = []
    K = range(1, 11)
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_pca)
        distortions.append(kmeans.inertia_)
    
    # Find elbow point
    optimal_k = np.argmax(np.diff(distortions, 2)) + 2
    
    print(f"Optimal number of clusters: {optimal_k}")
    
    # Train final KMeans model
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_pca)
    
    # Count items per cluster
    unique, counts = np.unique(clusters, return_counts=True)
    cluster_counts = dict(zip(unique, counts))
    print(f"Cluster distribution: {cluster_counts}")
    
    # Save models
    print(f"\nSaving clustering models to {model_save_path}...")
    os.makedirs(model_save_path, exist_ok=True)
    
    models_dict = {
        'scaler': scaler,
        'pca': pca,
        'kmeans': kmeans,
        'optimal_k': optimal_k,
        'optimal_pc': optimal_pc,
        'feature_names': X.columns.tolist()
    }
    
    with open(os.path.join(model_save_path, 'clustering_models.pkl'), 'wb') as f:
        pickle.dump(models_dict, f)
    
    print("Clustering models saved successfully!")
    
    return kmeans, pca, scaler, optimal_k, optimal_pc, cluster_counts


def main():
    """Main training function."""
    print("=" * 60)
    print("GUIDANCE SYSTEM - MODEL TRAINING")
    print("=" * 60)
    
    # Paths
    data_path = 'data/ASSI-A-Responses Labeled.csv'
    risk_model_path = 'models/risk_rating'
    clustering_model_path = 'models/clustering'
    
    # Check if data file exists
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        return
    
    # Load and preprocess data
    df = load_and_preprocess_labeled_data(data_path)
    
    # Train Risk Rating classification model
    risk_model, risk_scaler, label_encoder, accuracy, report, matrix = train_risk_rating_model(
        df, risk_model_path
    )
    
    # Train clustering model
    kmeans, pca, clustering_scaler, optimal_k, optimal_pc, cluster_counts = train_clustering_model(
        df, clustering_model_path
    )
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"Risk Rating Model Accuracy: {accuracy:.4f}")
    print(f"Optimal Clusters: {optimal_k}")
    print(f"Optimal Principal Components: {optimal_pc}")
    print("\nModels saved to:")
    print(f"  - {risk_model_path}/")
    print(f"  - {clustering_model_path}/")
    print("=" * 60)


if __name__ == '__main__':
    main()

