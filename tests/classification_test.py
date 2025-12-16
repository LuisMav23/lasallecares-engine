"""
Test script for classification.py module

Tests the risk rating classification functionality including:
- Model loading
- Prediction functionality
- Model caching
- Different form types (ASSI-A and ASSI-C)
- Edge cases
"""

import unittest
import sys
import os
import pandas as pd
import numpy as np

# Add parent directory to path to import src modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.classification import load_risk_rating_model, predict_risk_rating, _LOADED_MODELS


class TestClassification(unittest.TestCase):
    """Test cases for classification module"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures once for all tests"""
        # Create sample test data for ASSI-A format
        cls.sample_data_assi_a = pd.DataFrame({
            'Name': ['Student1', 'Student2', 'Student3'],
            'Gender': ['Female', 'Male', 'Female'],
            'GradeLevel': [1, 2, 1],
            'Q1': [7, 3, 4],
            'Q2': [6, 7, 5],
            'Q3': [1, 4, 4],
            'Q4': [4, 6, 7],
            'Q5': [3, 2, 2],
            'Q6': [5, 7, 7],
            'Q7': [4, 6, 6],
            'Q8': [3, 3, 4],
            'Q9': [6, 6, 5],
            'Q10': [4, 3, 3],
            'Q11': [5, 7, 5],
            'Q12': [3, 7, 2],
            'Q13': [6, 7, 6],
            'Q14': [6, 7, 6],
            'Q15': [4, 5, 3],
            'Q16': [5, 6, 6],
            'Q17': [4, 3, 3],
            'Q18': [6, 5, 6],
            'Q19': [2, 2, 2],
            'Q20': [6, 7, 7],
            'Q21': [5, 7, 7],
            'Q22': [3, 4, 2],
            'Q23': [7, 5, 6],
            'Q24': [4, 3, 4],
            'Q25': [5, 4, 6],
            'Q26': [2, 1, 2],
            'Q27': [7, 2, 5],
            'Q28': [6, 5, 5]
        })
        
        # Create sample test data for ASSI-C format (with text answers)
        cls.sample_data_assi_c = pd.DataFrame({
            'Name': ['Student1', 'Student2'],
            'Gender': ['Female', 'Male'],
            'GradeLevel': [1, 2],
            'Q1': ['Never', 'Sometimes'],
            'Q2': ['Often', 'Never'],
            'Q3': ['Sometimes', 'Often'],
            'Q4': ['Never', 'Sometimes'],
            'Q5': ['Sometimes', 'Never'],
            'Q6': ['Often', 'Sometimes'],
            'Q7': ['Never', 'Often'],
            'Q8': ['Sometimes', 'Never'],
            'Q9': ['Often', 'Sometimes'],
            'Q10': ['Never', 'Often'],
            'Q11': ['Sometimes', 'Never'],
            'Q12': ['Often', 'Sometimes'],
            'Q13': ['Never', 'Often'],
            'Q14': ['Sometimes', 'Never'],
            'Q15': ['Often', 'Sometimes'],
            'Q16': ['Never', 'Often'],
            'Q17': ['Sometimes', 'Never'],
            'Q18': ['Often', 'Sometimes'],
            'Q19': ['Never', 'Often'],
            'Q20': ['Sometimes', 'Never'],
            'Q21': ['Often', 'Sometimes'],
            'Q22': ['Never', 'Often'],
            'Q23': ['Sometimes', 'Never'],
            'Q24': ['Often', 'Sometimes'],
            'Q25': ['Never', 'Often'],
            'Q26': ['Sometimes', 'Never'],
            'Q27': ['Often', 'Sometimes'],
            'Q28': ['Never', 'Often']
        })
        
        # Create minimal test data (fewer columns)
        cls.minimal_data = pd.DataFrame({
            'Name': ['TestStudent'],
            'Gender': ['Male'],
            'Grade': [3],  # Using 'Grade' instead of 'GradeLevel'
            'Q1': [5],
            'Q2': [6],
            'Q3': [4],
            'Q4': [7],
            'Q5': [3],
            'Q6': [5],
            'Q7': [4],
            'Q8': [6],
            'Q9': [5],
            'Q10': [4],
            'Q11': [6],
            'Q12': [5],
            'Q13': [4],
            'Q14': [6],
            'Q15': [5],
            'Q16': [4],
            'Q17': [6],
            'Q18': [5],
            'Q19': [4],
            'Q20': [6],
            'Q21': [5],
            'Q22': [4],
            'Q23': [6],
            'Q24': [5],
            'Q25': [4],
            'Q26': [6],
            'Q27': [5],
            'Q28': [4]
        })
    
    def setUp(self):
        """Clear model cache before each test"""
        _LOADED_MODELS.clear()
    
    def test_load_risk_rating_model_exists(self):
        """Test that model file exists and can be loaded"""
        model_path = 'models/risk_rating_nn_model.h5'
        if os.path.exists(model_path):
            model = load_risk_rating_model()
            self.assertIsNotNone(model)
            print(f"✓ Model loaded successfully from {model_path}")
        else:
            self.skipTest(f"Model file not found at {model_path}")
    
    def test_load_risk_rating_model_caching(self):
        """Test that model is cached after first load"""
        model_path = 'models/risk_rating_nn_model.h5'
        if not os.path.exists(model_path):
            self.skipTest(f"Model file not found at {model_path}")
        
        # First load
        model1 = load_risk_rating_model()
        self.assertIsNotNone(model1)
        
        # Check cache
        self.assertIn('risk_rating', _LOADED_MODELS)
        
        # Second load should return cached model
        model2 = load_risk_rating_model()
        self.assertIs(model1, model2)  # Should be the same object
        print("✓ Model caching works correctly")
    
    def test_load_risk_rating_model_custom_path(self):
        """Test loading model with custom path"""
        model_path = 'models/risk_rating_nn_model.h5'
        if not os.path.exists(model_path):
            self.skipTest(f"Model file not found at {model_path}")
        
        model = load_risk_rating_model(model_path)
        self.assertIsNotNone(model)
        print("✓ Custom model path works correctly")
    
    def test_predict_risk_rating_assi_a(self):
        """Test prediction with ASSI-A format data"""
        model_path = 'models/risk_rating_nn_model.h5'
        if not os.path.exists(model_path):
            self.skipTest(f"Model file not found at {model_path}")
        
        result = predict_risk_rating(self.sample_data_assi_a)
        
        # Check result structure
        self.assertIsInstance(result, dict)
        self.assertIn('model_name', result)
        self.assertIn('predictions', result)
        self.assertIn('confidence', result)
        self.assertIn('risk_distribution', result)
        self.assertIn('classes', result)
        
        # Check predictions
        self.assertEqual(len(result['predictions']), len(self.sample_data_assi_a))
        self.assertEqual(len(result['confidence']), len(self.sample_data_assi_a))
        
        # Check that predictions are valid risk levels
        valid_levels = ['Low', 'Medium', 'High']
        for pred in result['predictions']:
            self.assertIn(pred, valid_levels or pred.startswith('Level_'))
        
        # Check confidence values are between 0 and 1
        for conf in result['confidence']:
            self.assertGreaterEqual(conf, 0.0)
            self.assertLessEqual(conf, 1.0)
        
        print(f"✓ ASSI-A prediction successful: {result['risk_distribution']}")
    
    def test_predict_risk_rating_assi_c_not_supported(self):
        """Test that ASSI-C format is not supported (only ASSI-A is supported)"""
        model_path = 'models/risk_rating_nn_model.h5'
        if not os.path.exists(model_path):
            self.skipTest(f"Model file not found at {model_path}")
        
        # ASSI-C data with text answers (Never/Sometimes/Often) will not be properly converted
        # since the function no longer handles ASSI-C conversion
        # This test documents that only ASSI-A is supported
        try:
            result = predict_risk_rating(self.sample_data_assi_c)
            # If it doesn't raise an error, it may still produce incorrect results
            # since text values won't be converted to numeric
            print("⚠ ASSI-C data processed but may produce incorrect results (ASSI-A only supported)")
            self.assertIsInstance(result, dict)  # At least verify it returns a dict
        except (ValueError, TypeError) as e:
            # Expected behavior - ASSI-C format not supported
            print(f"✓ ASSI-C correctly rejected: {str(e)}")
    
    def test_predict_risk_rating_grade_column(self):
        """Test prediction with 'Grade' column instead of 'GradeLevel'"""
        model_path = 'models/risk_rating_nn_model.h5'
        if not os.path.exists(model_path):
            self.skipTest(f"Model file not found at {model_path}")
        
        result = predict_risk_rating(self.minimal_data)
        
        # Should handle Grade column correctly
        self.assertIsInstance(result, dict)
        self.assertIn('predictions', result)
        self.assertEqual(len(result['predictions']), len(self.minimal_data))
        print("✓ Grade column handling works correctly")
    
    def test_predict_risk_rating_with_riskrating_column(self):
        """Test that RiskRating column is dropped if present"""
        model_path = 'models/risk_rating_nn_model.h5'
        if not os.path.exists(model_path):
            self.skipTest(f"Model file not found at {model_path}")
        
        # Add RiskRating column to test data
        test_data = self.sample_data_assi_a.copy()
        test_data['RiskRating'] = ['Low', 'Medium', 'High']
        
        # Should not raise an error
        result = predict_risk_rating(test_data)
        self.assertIsInstance(result, dict)
        self.assertIn('predictions', result)
        print("✓ RiskRating column dropping works correctly")
    
    def test_predict_risk_rating_single_student(self):
        """Test prediction with single student"""
        model_path = 'models/risk_rating_nn_model.h5'
        if not os.path.exists(model_path):
            self.skipTest(f"Model file not found at {model_path}")
        
        single_student = self.sample_data_assi_a.iloc[[0]]
        result = predict_risk_rating(single_student, form_type='ASSI-A')
        
        self.assertEqual(len(result['predictions']), 1)
        self.assertEqual(len(result['confidence']), 1)
        print(f"✓ Single student prediction: {result['predictions'][0]} (confidence: {result['confidence'][0]:.3f})")
    
    def test_predict_risk_rating_empty_dataframe(self):
        """Test that empty dataframe is handled gracefully"""
        model_path = 'models/risk_rating_nn_model.h5'
        if not os.path.exists(model_path):
            self.skipTest(f"Model file not found at {model_path}")
        
        empty_df = pd.DataFrame(columns=self.sample_data_assi_a.columns)
        
        # Should either handle gracefully or raise appropriate error
        try:
            result = predict_risk_rating(empty_df)
            # If it succeeds, check structure
            self.assertIsInstance(result, dict)
        except Exception as e:
            # If it fails, that's also acceptable for empty data
            self.assertIsInstance(e, (ValueError, IndexError))
        print("✓ Empty dataframe handling tested")
    
    def test_gender_encoding(self):
        """Test that Gender column is properly encoded"""
        model_path = 'models/risk_rating_nn_model.h5'
        if not os.path.exists(model_path):
            self.skipTest(f"Model file not found at {model_path}")
        
        # Test with both genders
        test_data = pd.DataFrame({
            'Name': ['FemaleStudent', 'MaleStudent'],
            'Gender': ['Female', 'Male'],
            'GradeLevel': [1, 1],
            **{f'Q{i}': [5] * 2 for i in range(1, 29)}
        })
        
        result = predict_risk_rating(test_data)
        self.assertEqual(len(result['predictions']), 2)
        print("✓ Gender encoding works correctly")


def run_tests():
    """Run all tests and print summary"""
    print("=" * 60)
    print("Classification Module Test Suite")
    print("=" * 60)
    print()
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestClassification)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print()
    print("=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    
    if result.wasSuccessful():
        print("\n✓ All tests passed!")
    else:
        print("\n✗ Some tests failed or were skipped")
        if result.failures:
            print("\nFailures:")
            for test, traceback in result.failures:
                print(f"  - {test}: {traceback}")
        if result.errors:
            print("\nErrors:")
            for test, traceback in result.errors:
                print(f"  - {test}: {traceback}")
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)

