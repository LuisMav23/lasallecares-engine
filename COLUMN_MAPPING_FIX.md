# Column Mapping Fix - Handling Full Question Text

## Problem

The system was failing when uploading CSV files because:

1. **Column Name Mismatch**: Pre-trained models expect columns named `Q1, Q2, Q3...Q28`, but uploaded files have full question text as column names (e.g., "Because I need at least a high-school degree...").

2. **Missing Name Column**: Some uploaded files don't have a 'Name' column, causing KeyError.

3. **Feature Name Validation**: scikit-learn models validate that feature names match exactly what was used during training.

## Solution

### 1. Column Mapping Function

Added `map_question_columns_to_q_format()` that:
- Maps full question text to Q1-Q28 format
- Handles whitespace and newline variations
- Skips columns already in Q format
- Works for both ASSI-A and ASSI-C form types

### 2. Name Column Handling

Updated `upload_file()` to:
- Check if 'Name' column exists
- Create it from 'StudentNumber' if available
- Create it from index if neither exists
- Ensures all files have a Name column

### 3. Feature Order Matching

Updated `predict_clusters()` and `apply_pca()` to:
- Use feature names from saved models
- Reorder columns to match training data order
- Add missing features with zeros
- Use unscaled data with pre-trained scaler

### 4. Error Handling

Added error checks in `app.py`:
- Validates PCA transformation succeeded
- Validates cluster prediction succeeded
- Returns clear error messages if models fail

## Files Modified

1. **`src/process.py`**
   - Added `get_question_mapping()` function
   - Added `map_question_columns_to_q_format()` function
   - Updated `upload_file()` to handle missing Name column
   - Updated `load_data_and_preprocess()` to map columns
   - Updated `predict_clusters()` to use feature names
   - Updated `apply_pca()` to use feature names

2. **`app.py`**
   - Added error checking for None returns
   - Pass unscaled data to PCA function

3. **`src/classification.py`**
   - Updated to drop RiskRating column if present

## How It Works

### Upload Flow

```
1. User uploads CSV with full question text columns
   ↓
2. upload_file() ensures Name column exists
   ↓
3. load_data_and_preprocess() maps columns:
   - "Because I need..." → Q1
   - "Because I experience..." → Q2
   - etc.
   ↓
4. Models receive data in Q1-Q28 format
   ↓
5. Predictions work correctly!
```

### Column Mapping Example

**Input CSV:**
```csv
Name,Gender,Grade,"Because I need at least a high-school degree...","Because I experience pleasure..."
John,Male,10,7,6
```

**After Mapping:**
```csv
Name,Gender,GradeLevel,Q1,Q2
John,Male,10,7,6
```

## Testing

### What to Upload

**For Testing**: Upload the **unlabeled** CSV file with full question text as column names.

The system will:
1. Automatically map question columns to Q1-Q28
2. Create Name column if missing
3. Use pre-trained models for predictions
4. Return both cluster and risk rating results

### Expected Behavior

- ✅ Files with full question text → Automatically mapped
- ✅ Files with Q1-Q28 format → Used directly
- ✅ Files without Name column → Created automatically
- ✅ Files with StudentNumber → Name created from it

## Error Messages

If you see errors, check:

1. **"Feature names should match"**
   - Solution: Column mapping should handle this automatically
   - Check that uploaded file has all 28 questions

2. **"KeyError: 'Name'"**
   - Solution: Name column is now created automatically
   - Check that file has at least StudentNumber or row index

3. **"Failed to apply PCA transformation"**
   - Solution: Check that all Q1-Q28 columns are present after mapping
   - Verify data format matches expected structure

## Compatibility

The system now handles:
- ✅ Full question text columns (mapped to Q format)
- ✅ Q1-Q28 format columns (used directly)
- ✅ Files with or without Name column
- ✅ Files with StudentNumber column
- ✅ Grade vs GradeLevel column names
- ✅ Whitespace variations in question text

## Summary

**Before**: System failed when uploading files with full question text  
**After**: System automatically maps to Q format and handles missing columns

**Result**: You can now upload either format and the system will work correctly!

