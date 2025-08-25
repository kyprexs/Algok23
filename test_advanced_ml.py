"""
Test script for Advanced ML Ensemble Pipeline

This script tests the new advanced ensemble models to ensure they're working
correctly and can train on synthetic financial data.
"""

import sys
import os
import asyncio
import logging
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, os.path.abspath('.'))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_imports():
    """Test all imports for the advanced ML pipeline"""
    print("🔍 Testing Advanced ML Pipeline Imports...")
    
    try:
        # Test core ML libraries
        import xgboost as xgb
        print(f"  ✅ XGBoost {xgb.__version__}")
        
        import lightgbm as lgb
        print(f"  ✅ LightGBM {lgb.__version__}")
        
        import torch
        print(f"  ✅ PyTorch {torch.__version__}")
        
        import sklearn
        print(f"  ✅ Scikit-learn {sklearn.__version__}")
        
        # Test our advanced ensemble
        from src.core.models.advanced_ensemble import (
            AdvancedEnsemble,
            AdvancedLSTM,
            TransformerPredictor,
            HyperparameterOptimizer
        )
        print("  ✅ Advanced Ensemble Models imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"  ❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"  ❌ Unexpected error: {e}")
        return False

def create_synthetic_data(n_samples: int = 1000, n_features: int = 20) -> pd.DataFrame:
    """Create synthetic financial time series data for testing"""
    print(f"📊 Creating synthetic data: {n_samples} samples, {n_features} features...")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create timestamps
    start_date = datetime.now() - timedelta(days=n_samples)
    timestamps = [start_date + timedelta(days=i) for i in range(n_samples)]
    
    # Create price data with trend and noise
    initial_price = 100.0
    returns = np.random.normal(0.0005, 0.02, n_samples)  # Daily returns
    prices = [initial_price]
    
    for i in range(1, n_samples):
        prices.append(prices[-1] * (1 + returns[i]))
    
    # Create OHLC data
    data = {
        'timestamp': timestamps,
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'close': prices,
        'volume': np.random.randint(1000, 100000, n_samples)
    }
    
    # Add technical features
    for i in range(n_features):
        # Various technical indicators (simulated)
        if i < 5:  # Moving averages
            window = 5 + i * 5
            data[f'ma_{window}'] = pd.Series(prices).rolling(window, min_periods=1).mean()
        elif i < 10:  # RSI-like oscillators
            data[f'rsi_{i}'] = np.random.uniform(20, 80, n_samples)
        elif i < 15:  # Volatility indicators
            data[f'vol_{i}'] = pd.Series(prices).rolling(10, min_periods=1).std()
        else:  # Other indicators
            data[f'feature_{i}'] = np.random.normal(0, 1, n_samples)
    
    df = pd.DataFrame(data)
    print(f"  ✅ Created DataFrame with shape: {df.shape}")
    return df

def test_ensemble_initialization():
    """Test ensemble model initialization"""
    print("🏗️ Testing Advanced Ensemble Initialization...")
    
    try:
        from src.core.models.advanced_ensemble import AdvancedEnsemble
        
        # Test with default parameters
        ensemble = AdvancedEnsemble()
        print(f"  ✅ Default ensemble created with sequence_length: {ensemble.sequence_length}")
        
        # Test with custom parameters
        custom_weights = {
            'xgboost': 0.3,
            'lightgbm': 0.3,
            'lstm': 0.2,
            'transformer': 0.2
        }
        
        custom_ensemble = AdvancedEnsemble(
            sequence_length=30,
            prediction_horizon=3,
            ensemble_weights=custom_weights
        )
        
        print(f"  ✅ Custom ensemble created with weights: {custom_ensemble.ensemble_weights}")
        return True, ensemble
        
    except Exception as e:
        print(f"  ❌ Ensemble initialization failed: {e}")
        return False, None

def test_data_preparation(ensemble, df):
    """Test data preparation for ML models"""
    print("📋 Testing Data Preparation...")
    
    try:
        feature_columns = [col for col in df.columns if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        print(f"  📊 Using {len(feature_columns)} feature columns")
        
        # Test feature preparation
        X_seq, X_tab, y = ensemble.prepare_features(df, feature_columns)
        
        print(f"  ✅ Sequence data shape: {X_seq.shape}")
        print(f"  ✅ Tabular data shape: {X_tab.shape}")
        print(f"  ✅ Target data shape: {y.shape}")
        print(f"  📈 Target distribution: {np.bincount(y)}")
        
        return True, X_seq, X_tab, y, feature_columns
        
    except Exception as e:
        print(f"  ❌ Data preparation failed: {e}")
        return False, None, None, None, None

def test_pytorch_models():
    """Test PyTorch model architectures"""
    print("🧠 Testing PyTorch Model Architectures...")
    
    try:
        from src.core.models.advanced_ensemble import AdvancedLSTM, TransformerPredictor
        import torch
        
        # Test LSTM
        input_size = 20
        sequence_length = 60
        batch_size = 32
        
        lstm_model = AdvancedLSTM(input_size=input_size)
        dummy_input = torch.randn(batch_size, sequence_length, input_size)
        
        with torch.no_grad():
            lstm_output = lstm_model(dummy_input)
        
        print(f"  ✅ LSTM output shape: {lstm_output.shape}")
        
        # Test Transformer
        transformer_model = TransformerPredictor(
            input_size=input_size,
            sequence_length=sequence_length
        )
        
        with torch.no_grad():
            transformer_output = transformer_model(dummy_input)
        
        print(f"  ✅ Transformer output shape: {transformer_output.shape}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ PyTorch models test failed: {e}")
        return False

def test_hyperparameter_optimizer():
    """Test hyperparameter optimization"""
    print("⚙️ Testing Hyperparameter Optimizer...")
    
    try:
        from src.core.models.advanced_ensemble import HyperparameterOptimizer
        from sklearn.datasets import make_classification
        from sklearn.model_selection import train_test_split
        
        # Create small test dataset
        X, y = make_classification(n_samples=500, n_features=10, n_classes=3, n_informative=8, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Test optimizer with fewer trials for speed
        optimizer = HyperparameterOptimizer(n_trials=5, timeout=60)
        
        # Test XGBoost optimization
        best_xgb_params = optimizer.optimize_xgboost(X_train, y_train, X_val, y_val)
        print(f"  ✅ XGBoost optimization completed: {len(best_xgb_params)} parameters")
        
        # Test LightGBM optimization
        best_lgb_params = optimizer.optimize_lightgbm(X_train, y_train, X_val, y_val)
        print(f"  ✅ LightGBM optimization completed: {len(best_lgb_params)} parameters")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Hyperparameter optimization failed: {e}")
        return False

async def test_ensemble_training():
    """Test full ensemble training (quick version)"""
    print("🎯 Testing Ensemble Training (Quick Version)...")
    
    try:
        from src.core.models.advanced_ensemble import AdvancedEnsemble
        
        # Create smaller dataset for faster training
        df = create_synthetic_data(n_samples=300, n_features=10)
        feature_columns = [col for col in df.columns if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        
        # Create ensemble with smaller models for testing
        ensemble = AdvancedEnsemble(sequence_length=20)
        
        # Reduce optimization trials for faster testing
        ensemble.optimizer = ensemble.optimizer.__class__(n_trials=2, timeout=30)
        
        # Train ensemble (this will be a quick test)
        print("  🚀 Starting training...")
        results = await ensemble.train_ensemble(df, feature_columns, validation_split=0.3)
        
        print(f"  ✅ Ensemble training completed!")
        print(f"  📊 Results: {results}")
        
        return True, ensemble, results
        
    except Exception as e:
        print(f"  ❌ Ensemble training failed: {e}")
        return False, None, None

def test_predictions(ensemble, df, feature_columns):
    """Test ensemble predictions"""
    print("🔮 Testing Ensemble Predictions...")
    
    try:
        # Prepare test data
        X_seq, X_tab, y = ensemble.prepare_features(df, feature_columns)
        
        # Test predictions on a small sample
        test_size = 10
        X_seq_test = X_seq[-test_size:]
        X_tab_test = X_tab[-test_size:]
        
        predictions = ensemble.predict_ensemble(X_seq_test, X_tab_test)
        
        print(f"  ✅ Generated {len(predictions)} predictions")
        print(f"  📊 Prediction distribution: {np.bincount(predictions)}")
        
        # Test feature importance
        importance = ensemble.get_feature_importance()
        print(f"  ✅ Feature importance extracted: {list(importance.keys())}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Predictions test failed: {e}")
        return False

async def main():
    """Main test function"""
    print("🚀 Starting Advanced ML Pipeline Tests")
    print("=" * 60)
    
    # Track test results
    tests_passed = 0
    total_tests = 8
    
    # Test 1: Imports
    if test_imports():
        tests_passed += 1
    
    # Test 2: Data creation
    df = create_synthetic_data(n_samples=500, n_features=15)
    if df is not None:
        tests_passed += 1
    
    # Test 3: Ensemble initialization
    success, ensemble = test_ensemble_initialization()
    if success:
        tests_passed += 1
    
    # Test 4: Data preparation
    if ensemble:
        success, X_seq, X_tab, y, feature_columns = test_data_preparation(ensemble, df)
        if success:
            tests_passed += 1
    
    # Test 5: PyTorch models
    if test_pytorch_models():
        tests_passed += 1
    
    # Test 6: Hyperparameter optimization
    if test_hyperparameter_optimizer():
        tests_passed += 1
    
    # Test 7: Ensemble training (quick version)
    success, trained_ensemble, results = await test_ensemble_training()
    if success:
        tests_passed += 1
        
        # Test 8: Predictions (use the same data as training)
        if trained_ensemble:
            # Create consistent test data
            test_df = create_synthetic_data(n_samples=300, n_features=10)
            test_feature_columns = [col for col in test_df.columns if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            if test_predictions(trained_ensemble, test_df, test_feature_columns):
                tests_passed += 1
    
    # Summary
    print("=" * 60)
    print(f"🎯 Test Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! Advanced ML Pipeline is ready!")
        print("🚀 Next steps: Integrate with main AgloK23 system")
    else:
        print(f"⚠️  {total_tests - tests_passed} tests failed. Check errors above.")
    
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())
