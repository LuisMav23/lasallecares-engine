# Quick Start Guide

## System Overview

Your guidance system now uses **pre-trained models** for fast and consistent predictions:

- **Risk Rating**: TensorFlow neural network (92% accuracy)
- **Clustering**: KMeans with PCA (2 optimal clusters)

## Running the Application

```bash
cd guidance-application
python app.py
```

The Flask server will start on `http://localhost:5000`

## What Happens When You Upload Data

1. **Data Upload**: CSV file uploaded via `/api/data` POST endpoint
2. **Preprocessing**: Data cleaned and standardized
3. **Clustering**: Pre-trained KMeans assigns clusters
4. **Risk Rating**: Pre-trained TensorFlow predicts Low/Medium/High risk
5. **Storage**: Results saved with both cluster and risk rating

## Key Features

### Dual Predictions
Each student gets:
- **Cluster Assignment**: Groups similar students (unsupervised)
- **Risk Rating**: Predicts risk level (supervised, with confidence score)

### Fast Processing
- No training on each upload
- Models loaded once and cached
- Predictions in seconds, not minutes

### Consistent Results
- Same model for all predictions
- Trained on 5,377 labeled samples
- Reproducible results

## Model Retraining

### When to Retrain
- You have new labeled data
- You want to improve model accuracy
- Data distribution has changed

### How to Retrain

1. Update `data/ASSI-A-Responses Labeled.csv` with new labeled data
2. Run training script:
   ```bash
   python src/train_models.py
   ```
3. Restart Flask application

## API Endpoints

### POST `/api/data`
Upload and process student data

**Request**:
- Form data with CSV file
- `datasetName`: Name for the dataset
- `kindOfData`: "ASSI-A" or "ASSI-C"
- `user`: Username

**Response**:
```json
{
  "message": "File uploaded and processed successfully",
  "data": {
    "id": "uuid",
    "data_summary": {
      "cluster_summary": {
        "optimal_k": 2,
        "cluster_count": {"Cluster 1": 50, "Cluster 2": 30}
      },
      "risk_rating_summary": {
        "model_name": "Neural Network (Pre-trained)",
        "risk_distribution": {"Low": 70, "Medium": 8, "High": 2}
      }
    }
  }
}
```

### GET `/api/student/data/:uuid/:form_type/:name`
Get individual student data including:
- Cluster assignment
- Risk rating
- Confidence score
- All survey responses

## Files Modified

- ✅ `src/classification.py` - Only TensorFlow prediction
- ✅ `src/process.py` - Pre-trained clustering
- ✅ `src/db.py` - Added RiskRating fields
- ✅ `app.py` - Updated to use predictions
- ✅ `src/train_models.py` - New training script

## Models Directory

```
models/
├── risk_rating/          # Classification model
│   ├── risk_rating_model.h5
│   ├── scaler.pkl
│   ├── label_encoder.pkl
│   └── feature_names.pkl
└── clustering/           # Clustering model
    └── clustering_models.pkl
```

**Important**: Keep the `models/` directory for the system to work!

## Troubleshooting

### Models Not Found
If you see "No such file or directory" for models:
```bash
python src/train_models.py
```

### Low Accuracy on Medium/High Risk
The model has class imbalance (mostly Low risk samples). To improve:
1. Collect more Medium/High risk labeled samples
2. Update the labeled CSV
3. Retrain with class weights or balanced sampling

### Column Name Issues
System automatically handles both:
- `Grade` → converted to `GradeLevel`
- `GradeLevel` → used directly

## Performance Improvements

| Metric | Before | After |
|--------|--------|-------|
| Processing Time | ~5-10 minutes | ~5-10 seconds |
| Training | Every upload | Once (pre-trained) |
| Consistency | Varies per upload | Same model always |
| Model Type | 3 models compete | 1 optimized model |

## Current Model Stats

### Risk Rating Model
- **Type**: TensorFlow Neural Network
- **Accuracy**: 92.01%
- **Training Samples**: 5,377
- **Classes**: Low (91%), Medium (7%), High (2%)

### Clustering Model  
- **Type**: KMeans
- **Clusters**: 2
- **PCA Components**: 4
- **Approach**: Unsupervised

## Support

For questions or issues:
1. Check `MODEL_TRAINING.md` for detailed documentation
2. Check `IMPLEMENTATION_SUMMARY.md` for implementation details
3. Review training output in console logs

