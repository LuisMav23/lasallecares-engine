# Model Training and Prediction System

## Overview

The system has been upgraded to use pre-trained models instead of training new models on each data upload. This significantly improves performance and consistency.

## Changes Made

### 1. Pre-trained Models
- **Risk Rating Model**: TensorFlow neural network trained on 5,377 labeled samples
  - Predicts: Low, Medium, or High risk
  - Accuracy: 92.01%
  - Located: `models/risk_rating/`

- **Clustering Model**: KMeans clustering with PCA
  - Optimal clusters: 2
  - Optimal principal components: 4
  - Located: `models/clustering/`

### 2. Code Changes

#### `src/train_models.py` (NEW)
- Script to train and save models using `data/ASSI-A-Responses Labeled.csv`
- Run this script whenever you need to retrain models with new labeled data
- Usage: `python src/train_models.py`

#### `src/classification.py`
- Removed: `svm_classification()` and `random_forest_classification()`
- Added: `predict_risk_rating()` - uses pre-trained TensorFlow model
- Model is loaded once and cached in memory for efficiency

#### `src/process.py`
- Added: `predict_clusters()` - uses pre-trained KMeans/PCA models
- Updated: `pca()` now uses pre-trained PCA transformation
- Old functions kept for backward compatibility

#### `app.py`
- Updated to use only TensorFlow predictions
- Results now include both:
  - `cluster_summary`: Clustering results
  - `risk_rating_summary`: Risk rating predictions with confidence scores

#### `src/db.py`
- Updated to include `RiskRating` and `RiskConfidence` in student data responses

## How It Works

### When New Data is Uploaded:

1. **Data Preprocessing**: Same as before
2. **Clustering Prediction**: Uses pre-trained KMeans model to assign clusters
3. **Risk Rating Prediction**: Uses pre-trained TensorFlow model to predict risk levels
4. **Results Storage**: Both cluster and risk rating are saved with student data

### Model Files Structure

```
models/
├── risk_rating/
│   ├── risk_rating_model.h5      # TensorFlow model
│   ├── scaler.pkl                # Feature scaler
│   ├── label_encoder.pkl         # Risk rating encoder
│   └── feature_names.pkl         # Feature column names
└── clustering/
    └── clustering_models.pkl     # KMeans, PCA, Scaler bundle
```

## Retraining Models

To retrain models with updated labeled data:

1. Update `data/ASSI-A-Responses Labeled.csv` with new labeled samples
2. Run: `python src/train_models.py`
3. Restart the Flask application

## API Response Changes

The `/api/data` POST endpoint now returns:

```json
{
  "id": "uuid",
  "user": "username",
  "type": "ASSI-A",
  "data_summary": {
    "answers_summary": {...},
    "pca_summary": {
      "optimal_pc": 4
    },
    "cluster_summary": {
      "optimal_k": 2,
      "cluster_count": {...}
    },
    "risk_rating_summary": {
      "model_name": "Neural Network (Pre-trained)",
      "risk_distribution": {
        "Low": 95,
        "Medium": 3,
        "High": 2
      },
      "classes": ["High", "Low", "Medium"]
    }
  }
}
```

## Student Data Format

Each student now has:
- `Name`: Student name
- `Grade`: Grade level
- `Gender`: Gender
- `Cluster`: Assigned cluster (0, 1, etc.)
- `RiskRating`: Predicted risk level (Low/Medium/High)
- `RiskConfidence`: Prediction confidence (0.0 - 1.0)
- `Questions`: All survey responses

## Benefits

1. **Performance**: No training on each upload = faster processing
2. **Consistency**: Same model used for all predictions
3. **Reliability**: Model trained on large labeled dataset (5,377 samples)
4. **Efficiency**: Models loaded once and cached in memory
5. **Maintainability**: Single source of truth for model training

## Notes

- The system maintains both clustering and risk rating predictions
- KMeans clustering is unsupervised (doesn't use RiskRating labels)
- Risk rating classification is supervised (trained on RiskRating labels)
- Both provide complementary insights into student data

